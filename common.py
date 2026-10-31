"""Shared helpers (v2): landmark extraction, time-based windows, camera.

RAW FRAME FORMAT (147 floats per frame). We store RAW landmarks, not derived features, so the
model code can change later without re-extracting your dataset.

  [  0:126]  2 hand slots x 21 landmarks x (x, y, z)   (slots ordered left-to-right in the image)
  [126:128]  hand-present flags (slot 0, slot 1)
  [128:146]  9 upper-body pose points x (x, y): nose, mouth L/R, shoulders L/R, elbows L/R, wrists L/R
  [146:147]  pose-present flag

All x and z values are multiplied by the image aspect ratio, so x, y and z share one scale
(units of image height). Everything uses the UNFLIPPED camera frame; only the preview is mirrored.

MediaPipe backends (chosen automatically; both give identical RAW frames):
  * "legacy": mp.solutions.hands / mp.solutions.pose  (mediapipe up to ~0.10.2x)
  * "tasks" : HandLandmarker / PoseLandmarker         (newer mediapipe; the .task model files are
                                                       downloaded once into ./mp_models/)
Force one with the environment variable SIGN_BACKEND=legacy|tasks.
"""
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:  # pragma: no cover
    raise SystemExit("mediapipe is not installed. Run: pip install -r requirements.txt") from e

SEQ_LEN = 30
FORMAT_VERSION = 2
POSE_IDX = (0, 9, 10, 11, 12, 13, 14, 15, 16)
N_POSE = len(POSE_IDX)
HANDS = slice(0, 126)
HFLAG = slice(126, 128)
POSE = slice(128, 128 + 2 * N_POSE)
PFLAG = slice(128 + 2 * N_POSE, 129 + 2 * N_POSE)
RAW_DIM = 129 + 2 * N_POSE  # 147

IDLE_NAMES = {"idle", "nothing", "none", "_idle", "background", "blank"}

HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
POSE_LINKS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

MODEL_DIR = Path(os.environ.get("SIGN_MP_MODELS", "mp_models"))
MODEL_URLS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}


# ------------------------------------------------------------------------------ packing
def pack_frame(hands, pose, aspect):
    """hands: list of (21,3) normalized arrays (x,y,z); pose: (33,3) array (x,y,visibility) or None.
    Returns (raw[RAW_DIM], n_hands). Shared by both backends."""
    raw = np.zeros(RAW_DIM, np.float32)
    n = 0
    if hands:
        sc = np.array([aspect, 1.0, aspect], np.float32)
        pts = [np.asarray(h, np.float32) * sc for h in hands[:2]]
        pts.sort(key=lambda a: a[0, 0])  # left-to-right by wrist x
        for i, a in enumerate(pts):
            raw[i * 63:(i + 1) * 63] = a.ravel()
            raw[126 + i] = 1.0
        n = len(pts)
    if pose is not None and pose[11, 2] >= 0.5 and pose[12, 2] >= 0.5:
        xy = np.asarray(pose, np.float32)[list(POSE_IDX), :2] * np.array([aspect, 1.0], np.float32)
        raw[POSE] = xy.ravel()
        raw[PFLAG] = 1.0
    return raw, n


# ------------------------------------------------------------------------------ backends
class _LegacyBackend:
    def __init__(self, use_pose, static, fast):
        cx = 0 if fast else 1
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static, max_num_hands=2, model_complexity=cx,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = None
        if use_pose:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=static, model_complexity=0 if fast else 1,
                smooth_landmarks=not static, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect(self, rgb):
        hres = self.hands.process(rgb)
        hands = [np.array([[p.x, p.y, p.z] for p in hl.landmark], np.float32)
                 for hl in (hres.multi_hand_landmarks or [])[:2]]
        pose = None
        if self.pose is not None:
            pres = self.pose.process(rgb)
            if pres.pose_landmarks:
                pose = np.array([[p.x, p.y, p.visibility] for p in pres.pose_landmarks.landmark], np.float32)
        return hands, pose

    def reset(self):
        for s in (self.hands, self.pose):
            if s is not None and hasattr(s, "reset"):
                s.reset()

    def close(self):
        self.hands.close()
        if self.pose is not None:
            self.pose.close()


def _download(url, dest, tries=3, timeout=30):
    """Download to a .part file with a per-read timeout, progress, retries and a size check; then rename atomically.
    A stalled or interrupted download can therefore never leave a half-written model behind."""
    tmp = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                total = int(r.headers.get("Content-Length") or 0)
                got, shown = 0, 0.0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - shown > 1.0:
                            pct = f" ({got * 100 // total}%)" if total else ""
                            print(f"\r  {got / 1e6:.1f} MB{pct}   ", end="", flush=True)
                            shown = time.time()
            print(f"\r  {got / 1e6:.1f} MB done        ", flush=True)
            if total and got != total:
                raise OSError(f"incomplete download ({got} of {total} bytes)")
            if got < 1_000_000:
                raise OSError(f"file too small ({got} bytes), probably an error page")
            tmp.replace(dest)
            return
        except BaseException as e:  # noqa: BLE001  (includes Ctrl+C: clean up, then re-raise it)
            tmp.unlink(missing_ok=True)
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            last = e
            print(f"\n  attempt {attempt}/{tries} failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(1.5)
    raise last


def ensure_model(name):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / name
    if not path.exists() or path.stat().st_size < 1_000_000:
        print(f"Downloading MediaPipe model {name} (one time, about 5-10 MB)...", flush=True)
        try:
            _download(MODEL_URLS[name], path)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Could not download {name}: {e}\n"
                f"Check your internet / VPN / firewall, then run   python download_models.py\n"
                f"or download it by hand:\n  curl -L -o {MODEL_DIR}/{name} {MODEL_URLS[name]}\n"
                f"(or open the link in a browser and put the file in the folder '{MODEL_DIR}/').")
    return str(path)


class _TasksBackend:
    def __init__(self, use_pose, static, fast):
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        mode = vision.RunningMode.IMAGE if static else vision.RunningMode.VIDEO
        self.static, self._ts = static, 0
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=ensure_model("hand_landmarker.task")),
            running_mode=mode, num_hands=2, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
        self.pose = None
        if use_pose:
            name = "pose_landmarker_lite.task" if fast else "pose_landmarker_full.task"
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=ensure_model(name)),
                running_mode=mode, num_poses=1, min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5, min_tracking_confidence=0.5))

    def detect(self, rgb):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(rgb, dtype=np.uint8, order="C"))
        if self.static:
            hres = self.hands.detect(img)
            pres = self.pose.detect(img) if self.pose is not None else None
        else:
            self._ts = max(self._ts + 1, int(time.perf_counter() * 1000))
            hres = self.hands.detect_for_video(img, self._ts)
            pres = self.pose.detect_for_video(img, self._ts) if self.pose is not None else None
        hands = [np.array([[p.x, p.y, p.z] for p in hl], np.float32) for hl in hres.hand_landmarks[:2]]
        pose = None
        if pres is not None and pres.pose_landmarks:
            pose = np.array([[p.x, p.y, 1.0 if p.visibility is None else p.visibility]
                             for p in pres.pose_landmarks[0]], np.float32)
        return hands, pose

    def reset(self):
        pass

    def close(self):
        self.hands.close()
        if self.pose is not None:
            self.pose.close()


def backend_name():
    forced = os.environ.get("SIGN_BACKEND", "").lower()
    if forced in ("legacy", "tasks"):
        return forced
    return "legacy" if hasattr(mp, "solutions") else "tasks"


class Extractor:
    """MediaPipe Hands (+ optional Pose) -> one RAW frame vector."""

    def __init__(self, use_pose=True, static=False, fast=False):
        self.backend_name = backend_name()
        cls = _LegacyBackend if self.backend_name == "legacy" else _TasksBackend
        self.backend = cls(use_pose, static, fast)

    def process(self, frame_bgr):
        """Return (raw[RAW_DIM], n_hands, detections) - detections is only used for drawing."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hands, pose = self.backend.detect(rgb)
        raw, n = pack_frame(hands, pose, w / float(h))
        return raw, n, (hands, pose)

    def reset(self):
        self.backend.reset()

    def close(self):
        self.backend.close()


def draw_landmarks(frame, det):
    """Draw hands (and upper-body pose) on a BGR frame (in place)."""
    if det is None:
        return
    hands, pose = det
    h, w = frame.shape[:2]
    for hd in hands:
        px = (hd[:, :2] * (w, h)).astype(int)
        for a, b in HAND_LINKS:
            cv2.line(frame, tuple(px[a]), tuple(px[b]), (255, 255, 255), 2)
        for p in px:
            cv2.circle(frame, tuple(p), 3, (0, 0, 255), -1)
    if pose is not None:
        px = (pose[:, :2] * (w, h)).astype(int)
        for a, b in POSE_LINKS:
            cv2.line(frame, tuple(px[a]), tuple(px[b]), (0, 200, 255), 2)
        for i in POSE_IDX:
            cv2.circle(frame, tuple(px[i]), 4, (0, 200, 255), -1)


# ---------------------------------------------------------------------------- sequences
def hands_present(seq):
    """Number of frames in a (T, RAW_DIM) array that contain at least one hand."""
    seq = np.asarray(seq)
    return int((seq[:, HFLAG].max(axis=1) > 0).sum())


def resample(seq, n=SEQ_LEN):
    """Evenly pick n frames (by index) from a variable-length sequence."""
    seq = np.asarray(seq, np.float32)
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return seq[idx]


def sample_window(times, frames, t_end, width, n=SEQ_LEN):
    """Take n frames spread evenly over the last `width` seconds (nearest frame in time).

    times: ascending 1-D array of frame timestamps; frames: (N, D). Returns (n, D) or None
    when there is not enough history yet. This makes live input independent of camera fps.
    """
    if len(times) < 2 or times[0] > t_end - width * 0.9:
        return None
    targets = np.linspace(t_end - width, t_end, n)
    idx = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
    left, right = times[idx - 1], times[idx]
    idx = np.where(np.abs(targets - left) <= np.abs(right - targets), idx - 1, idx)
    return frames[idx]


def pretty_label(name):
    """'12. hello_world' -> 'HELLO WORLD'."""
    s = re.sub(r"^\s*\d+[\.\-_)\s]+", "", name)
    s = s.replace("_", " ").strip()
    return (s or name).upper()


# -------------------------------------------------------------------------------- camera
def open_camera(index=0, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the camera. On macOS: System Settings > Privacy & Security > Camera, "
            "allow your Terminal / IDE, then reopen it. Try --camera 1 if you have several cameras.")
    return cap


class CameraStream:
    """Reads the camera in a background thread so processing never waits on the sensor
    and always works on the newest frame (lower latency than a plain read loop)."""

    def __init__(self, cap):
        self.cap = cap
        self.cond = threading.Condition()
        self.frame, self.seq, self.ok, self.closed = None, 0, True, False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.closed:
            ok, frame = self.cap.read()
            with self.cond:
                self.ok = ok
                if ok:
                    self.frame, self.seq = frame, self.seq + 1
                self.cond.notify_all()
            if not ok:
                break

    def read(self, last_seq, timeout=2.0):
        """Block until a frame newer than last_seq exists. Returns (frame, seq) or (None, last_seq)."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.ok, timeout)
            if not self.ok or self.frame is None or self.seq == last_seq:
                return None, last_seq
            return self.frame, self.seq

    def release(self):
        self.closed = True
        self.thread.join(timeout=1.0)
        self.cap.release()


def now():
    return time.perf_counter()


# ---------------------------------------------------------------------------- camera / window helpers
def frame_is_black(frame, thr=6.0):
    """True for an (almost) all-black image: no camera permission, lens covered, or the wrong camera selected."""
    return frame is None or float(frame.mean()) < thr


def show_message(win, text, size=(640, 480)):
    """Show text in the window straight away (so it is never blank while models load)."""
    img = np.zeros((size[1], size[0], 3), np.uint8)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(img, line, (20, 60 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(win, img)
    cv2.waitKey(1)

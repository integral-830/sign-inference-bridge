"""Convert a downloaded dataset (images and/or videos) into RAW landmark sequences, in parallel.

Label = the folder that directly contains the file:
  dataset/A/img001.jpg            (static letters/numbers -> the frame is repeated 30x)
  dataset/hello/clip01.mp4        (words -> sign is trimmed to when hands are visible, resampled to 30 frames)

Output: data/sequences/<label>/*.npy   shape (30, 147)

Examples:
  python prepare_dataset.py --src ~/Downloads/asl/asl_alphabet_train --max-per-class 300
  python prepare_dataset.py --src ~/Downloads/include --max-per-class 80
  python prepare_dataset.py --src ~/Downloads/isl_alphabet --flip      # if the images are mirrored selfies

Safe to re-run: files that already exist are skipped (--no-resume to redo everything).
Use --no-pose for hand-only data (e.g. close-up hand photos): faster, and avoids useless pose lookups.
"""
import argparse
import os
import random
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import SEQ_LEN, Extractor, backend_name, ensure_model, resample
from labels import load_map, resolve

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm", ".m4v"}
MAX_VIDEO_FRAMES = 60  # frames actually analysed per video

_ex = None


def _init(use_pose):
    global _ex
    cv2.setNumThreads(1)
    _ex = Extractor(use_pose=use_pose, static=True)


def safe(name):
    return re.sub(r"[^\w\-. ]+", "_", name).strip()


def shrink(frame, max_w=640):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (max_w, int(h * max_w / w))) if w > max_w else frame


def _work(task):
    src, dst, flip, resume = task
    if resume and os.path.exists(dst):
        return "kept"
    src = Path(src)
    try:
        seq = None
        if src.suffix.lower() in IMG_EXT:
            img = cv2.imread(str(src))
            if img is None:
                return "skipped"
            img = shrink(img)
            if flip:
                img = cv2.flip(img, 1)
            raw, n, _ = _ex.process(img)
            if n:
                seq = np.repeat(raw[None], SEQ_LEN, axis=0)
        else:
            cap = cv2.VideoCapture(str(src))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            stride = max(1, -(-total // MAX_VIDEO_FRAMES)) if total > 0 else 1
            frames, flags, i = [], [], 0
            while cap.grab():
                if i % stride == 0:
                    ok, fr = cap.retrieve()
                    if not ok:
                        break
                    fr = shrink(fr)
                    if flip:
                        fr = cv2.flip(fr, 1)
                    raw, n, _ = _ex.process(fr)
                    frames.append(raw)
                    flags.append(n > 0)
                i += 1
            cap.release()
            idx = [k for k, v in enumerate(flags) if v]
            if len(idx) >= 5:
                seq = resample(np.stack(frames[idx[0]: idx[-1] + 1]))
        if seq is None:
            return "skipped"
        np.save(dst, seq.astype(np.float32))
        return "ok"
    except Exception:
        return "skipped"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/sequences")
    ap.add_argument("--flip", action="store_true", help="mirror every frame horizontally")
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = no limit (random subset when set)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-pose", action="store_true", help="hands only")
    ap.add_argument("--no-resume", action="store_true", help="redo files that already exist")
    ap.add_argument("--label-map", default=None, help="JSON file folding variants into one sign (see labelmap_example.json)")
    ap.add_argument("--auto-labels", action="store_true", help="auto-merge obvious variants: E1/E2 -> E, hello_2 -> hello")
    ap.add_argument("--merge-case", action="store_true", help="treat 'Hello' and 'hello' as one sign")
    args = ap.parse_args()

    src, out = Path(args.src).expanduser(), Path(args.out)
    mapping = load_map(args.label_map) if args.label_map else None
    groups = defaultdict(list)
    n_folders = set()
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT | VID_EXT and not p.name.startswith("."):
            n_folders.add(p.parent.name)
            groups[resolve(safe(p.parent.name), mapping, args.auto_labels, args.merge_case)].append(p)
    if not groups:
        raise SystemExit(f"No images/videos found under {src}")
    print(f"{len(groups)} classes found" + (f" (merged from {len(n_folders)} folders)" if len(groups) < len(n_folders) else "")
          + f", {args.workers} workers, mediapipe backend: {backend_name()}")
    if backend_name() == "tasks":  # download model files once, before the workers start
        ensure_model("hand_landmarker.task")
        if not args.no_pose:
            ensure_model("pose_landmarker_full.task")

    tasks = []
    for label, files in groups.items():
        if args.max_per_class:
            random.Random(0).shuffle(files)
            files = files[: args.max_per_class]
        (out / label).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(files):
            tasks.append((str(p), str(out / label / f"{safe(p.stem)}_{i}.npy"), args.flip, not args.no_resume))

    stats = defaultdict(int)
    with Pool(args.workers, initializer=_init, initargs=(not args.no_pose,)) as pool:
        for r in tqdm(pool.imap_unordered(_work, tasks, chunksize=8), total=len(tasks)):
            stats[r] += 1
    print(f"done: new {stats['ok']}, already there {stats['kept']}, skipped {stats['skipped']} "
          f"(no hand detected / unreadable)")


if __name__ == "__main__":
    main()

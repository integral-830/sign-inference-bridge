"""Record your own signs from the webcam (v2: time-based clips, body pose included).

Each sample = `--seconds` of video resampled to 30 frames, saved as data/sequences/<label>/<ms>.npy
Because clips are time-based, they look the same whatever your camera / CPU speed is.

Examples:
  python collect_data.py --labels idle,A,B,C --samples 60 --seconds 1.0 --auto
  python collect_data.py --labels hello,thanks,please --samples 60 --seconds 1.5 --auto

Keys: SPACE start (2 s countdown)   u undo last sample   n next label   p previous label   q quit
Record an 'idle' class (hands down / resting / not signing) - it is what lets sentences end cleanly.
Tips: vary distance, angle and lighting a little; for moving signs do the WHOLE sign inside the clip.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from common import (SEQ_LEN, IDLE_NAMES, Extractor, draw_landmarks, frame_is_black, hands_present,
                    open_camera, resample, sample_window, show_message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="comma-separated labels")
    ap.add_argument("--samples", type=int, default=60, help="samples per label")
    ap.add_argument("--seconds", type=float, default=1.5, help="clip length in seconds")
    ap.add_argument("--out", default="data/sequences")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-pose", action="store_true", help="hands only (skip body pose)")
    ap.add_argument("--auto", action="store_true", help="keep recording back-to-back after the first SPACE")
    args = ap.parse_args()

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    out = Path(args.out)
    for l in labels:
        (out / l).mkdir(parents=True, exist_ok=True)
    counts = {l: len(list((out / l).glob("*.npy"))) for l in labels}

    cap = open_camera(args.camera)
    cv2.namedWindow("collect", cv2.WINDOW_NORMAL)
    show_message("collect", "Starting...\nLoading the hand/pose models (the first run downloads them).\n"
                            "If this stays black: see the messages in the Terminal.")
    ex = Extractor(use_pose=not args.no_pose, static=False)
    li, state, t_go, msg = 0, "idle", 0.0, ""
    times, frames, saved = [], [], []
    bad_reads, black_run, black_warned = 0, 0, False

    while True:
        ok, frame = cap.read()
        if not ok:
            bad_reads += 1
            if bad_reads > 40:   # ~2 s of failed reads
                print("The camera stopped giving frames. Another app may be using it, or it was unplugged/asleep. "
                      "Try:  python camera_test.py --list   and   --camera 1", flush=True)
                break
            time.sleep(0.05)
            cv2.waitKey(1)
            continue
        bad_reads = 0
        t = time.perf_counter()
        raw, n, det = ex.process(frame)
        label = labels[li]

        if state == "countdown" and t >= t_go:
            state, times, frames = "recording", [], []
        if state == "recording":
            times.append(t)
            frames.append(raw)
            if t - times[0] >= args.seconds:
                seq = sample_window(np.array(times), np.stack(frames), times[-1], times[-1] - times[0])
                if seq is None:
                    seq = resample(np.stack(frames))
                if label in IDLE_NAMES or hands_present(seq) >= SEQ_LEN * 0.4:
                    p = out / label / f"{int(time.time() * 1000)}.npy"
                    np.save(p, seq.astype(np.float32))
                    saved.append((label, p))
                    counts[label] += 1
                    msg = f"saved {counts[label]}/{args.samples}"
                else:
                    msg = "discarded: hands not visible enough"
                state = "countdown" if args.auto else "idle"
                t_go = t + 0.6
                if counts[label] >= args.samples:
                    state = "idle"
                    if li < len(labels) - 1:
                        li += 1
                        msg = f"'{label}' done -> next: '{labels[li]}'"
                    else:
                        msg = "all labels done - press q"

        draw_landmarks(frame, det)
        view = cv2.flip(frame, 1)
        label = labels[li]
        status = state.upper() + (f" {max(0.0, t_go - t):.1f}s" if state == "countdown" else "")
        cv2.putText(view, f"{label}  {counts[label]}/{args.samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(view, f"{status}  {msg}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(view, f"hands: {n}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if n else (0, 0, 255), 2)
        cv2.putText(view, "SPACE record | u undo | n/p label | q quit", (10, view.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        black_run = black_run + 1 if frame_is_black(frame) else 0
        if black_run >= 30:
            cv2.putText(view, "BLACK IMAGE: no camera permission or wrong camera", (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(view, "try --camera 1 (iPhone/other camera?)", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if not black_warned:
                print("The camera image is black. Fix: System Settings > Privacy & Security > Camera > allow Terminal, "
                      "then reopen Terminal; or try  --camera 1  (camera 0 may be your iPhone). "
                      "Run  python camera_test.py --list  to see which camera works.", flush=True)
                black_warned = True
        cv2.imshow("collect", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit with q.", flush=True)
            break
        elif key == ord(" ") and state == "idle":
            state, t_go, msg = "countdown", time.perf_counter() + 2.0, ""
        elif key == ord("u") and saved:
            l, p = saved.pop()
            p.unlink(missing_ok=True)
            counts[l] -= 1
            msg = f"undid last sample of '{l}'"
        elif key == ord("n") and li < len(labels) - 1:
            li, state = li + 1, "idle"
        elif key == ord("p") and li > 0:
            li, state = li - 1, "idle"

    cap.release()
    cv2.destroyAllWindows()
    ex.close()
    print({l: counts[l] for l in labels})


if __name__ == "__main__":
    main()

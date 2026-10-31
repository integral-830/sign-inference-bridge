"""Download the MediaPipe model files once (needed by the new MediaPipe API):   python download_models.py

  python download_models.py            # hand + body-pose models (what collect_data/live_demo use)
  python download_models.py --lite     # also the lighter pose model (used by --fast)
  python download_models.py --force    # delete and download again (use after an interrupted/corrupt download)

Files go into ./mp_models/.  Downloads have a timeout, retries and a size check; a half-finished file is never kept.
If your network blocks it, download by hand:  curl -L -o mp_models/<name> <url>   (the URLs are printed on failure).
"""
import argparse
import sys

from common import MODEL_DIR, MODEL_URLS, ensure_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    names = ["hand_landmarker.task", "pose_landmarker_full.task"] + (["pose_landmarker_lite.task"] if args.lite else [])
    for n in names:
        if args.force:
            (MODEL_DIR / n).unlink(missing_ok=True)
        ensure_model(n)
        print(f"OK  {MODEL_DIR / n}  ({(MODEL_DIR / n).stat().st_size / 1e6:.1f} MB)")
    print("All models ready.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Nothing half-downloaded was kept; run it again.")
        sys.exit(1)

"""Camera / window test (no MediaPipe, no model):   python camera_test.py   |   python camera_test.py --list

Shows what each camera index gives and opens a live window with fps and brightness.
Use it when a window is black/empty or the program exits straight away.

  --list          try camera indexes 0-4 and print which ones give a picture (black = no permission / wrong camera)
  --camera N      open camera N in a window (default 0)
  --mediapipe     also run hand/pose detection and draw it (tests MediaPipe on live frames)
Press q in the window to quit. Click the window first if keys do nothing.
"""
import argparse
import sys
import time

import cv2

from common import frame_is_black, open_camera, show_message


def probe(i):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(i)
    if not cap.isOpened():
        cap.release()
        return None
    frame = None
    for _ in range(15):                 # the first frames are often black while the sensor warms up
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.05)
    cap.release()
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mediapipe", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("Probing camera indexes 0-4 (a permission pop-up may appear: click OK)...")
        for i in range(5):
            f = probe(i)
            if f is None:
                print(f"  camera {i}: cannot open / no frames")
            else:
                print(f"  camera {i}: {f.shape[1]}x{f.shape[0]}, brightness {f.mean():.0f}"
                      + ("   <- BLACK: no permission, covered lens, or not a real camera" if frame_is_black(f) else "   <- looks OK"))
        print("Use the number that looks OK:  python collect_data.py --camera N ...   python run.py --camera N")
        return

    cap = open_camera(args.camera)
    cv2.namedWindow("camera test", cv2.WINDOW_NORMAL)
    show_message("camera test", "Starting camera...")
    ex = None
    if args.mediapipe:
        from common import Extractor, draw_landmarks
        show_message("camera test", "Loading MediaPipe (first run downloads models)...")
        ex = Extractor(use_pose=True, static=False)
    n, t0, fps, warned, bad = 0, time.time(), 0.0, False, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            bad += 1
            if bad > 40:
                print("No frames from the camera. Another app may be using it. Try --list and a different --camera.")
                break
            time.sleep(0.05)
            cv2.waitKey(1)
            continue
        bad = 0
        n += 1
        if time.time() - t0 >= 1.0:
            fps, n, t0 = n / (time.time() - t0), 0, time.time()
        hands = 0
        if ex is not None:
            raw, hands, det = ex.process(frame)
            draw_landmarks(frame, det)
        view = cv2.flip(frame, 1)
        bright = frame.mean()
        cv2.putText(view, f"{fps:.0f} fps  brightness {bright:.0f}" + (f"  hands {hands}" if ex else ""), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if bright < 6:
            cv2.putText(view, "BLACK IMAGE: permission / wrong camera", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not warned:
                print("Black image. Allow Terminal under System Settings > Privacy & Security > Camera (then reopen Terminal), "
                      "or try another --camera index (see --list).")
                warned = True
        cv2.imshow("camera test", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if ex is not None:
        ex.close()


if __name__ == "__main__":
    main()

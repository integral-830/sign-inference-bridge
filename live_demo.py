"""Live sign recognition + sentence transcription (webcam or video file).

  python live_demo.py                                   # one model (model/sign_model.pt)
  python live_demo.py --weights model/words/sign_model.pt model/letters/sign_model.pt   # two models
  python live_demo.py --translate llm                   # also turn each sentence into English (needs GROQ_API_KEY)
  python live_demo.py --push http://localhost:8000     # also send sentences to server.py (for the ESP)
  python live_demo.py --video clip.mp4                  # transcribe a video file
  python live_demo.py --no-window --speak               # terminal only, read sentences aloud (macOS)

Two models: a 'letters' model (single letters/digits) and a 'words' model. The role of each model is detected
from its classes. Each step both predict; the one that fits how much your hands move is preferred
(still hands -> letters, moving hands -> words). Press m to force a mode.

Sentences: signs are committed when stable; letters in a row become one word; a sentence ends after a short
pause with no hands (or the 'idle' pose). Sentences are saved to transcripts/session-<time>.txt.

Window keys: q quit | c clear sentence | BACKSPACE delete last word | ENTER end sentence now | m mode
"""
import argparse
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from common import (IDLE_NAMES, CameraStream, Extractor, draw_landmarks, hands_present, now, open_camera,
                    sample_window)
from fusion import arbitrate, detect_role, window_motion
from model import build_model
from push import Pusher
from sentence import SentenceBuilder
from translate import GlossTranslator


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    if ck.get("version") != 2:
        raise SystemExit("This model was made by the old v1 scripts. Re-extract your data with the v2 "
                         "prepare_dataset.py / collect_data.py and run train.py again.")
    if len(ck["classes"]) < 2:
        raise SystemExit(f"{path} has only {len(ck['classes'])} class {ck['classes']}: it can only ever answer that sign, "
                         f"always at 100%. Do not use it. Delete it (rm -rf {Path(path).parent}) and train again "
                         f"with at least 2 classes (see RUNBOOK Part 9), or run with only the other model.")
    if not any(c.lower() in IDLE_NAMES for c in ck["classes"]):
        print(f"[warn] {path} has no 'idle' class: it will name a sign even when you are not signing. "
              f"Record idle (collect_data.py --labels idle ...) and retrain.")
    model = build_model(ck["arch"], len(ck["classes"]), ck["features"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


class Transcript:
    def __init__(self, folder):
        self.folder, self.path, self.lock = Path(folder), None, threading.Lock()

    def add(self, text):
        with self.lock:
            if self.path is None:
                self.folder.mkdir(parents=True, exist_ok=True)
                self.path = self.folder / f"session-{datetime.now():%Y%m%d-%H%M%S}.txt"
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now():%H:%M:%S}] {text}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=["model/sign_model.pt"],
                    help="one or two model files (e.g. a words model and a letters model)")
    ap.add_argument("--mode", default="auto", choices=["auto", "letters", "words"],
                    help="which model to prefer when two are loaded (auto = by hand motion)")
    ap.add_argument("--motion-thr", type=float, default=0.25,
                    help="hand speed (image heights/s) above which a sign counts as a moving word")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default=None, help="transcribe a video file instead of the webcam")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.65, help="min confidence for a prediction to count")
    ap.add_argument("--scales", default="0.8,1.4,2.2",
                    help="window lengths in seconds tried every step (best one wins)")
    ap.add_argument("--stride", type=int, default=2, help="predict every N frames")
    ap.add_argument("--vote-window", type=int, default=7)
    ap.add_argument("--min-votes", type=int, default=4, help="agreeing predictions needed to commit a sign")
    ap.add_argument("--letter-gap", type=float, default=1.6, help="seconds between letters of one spelled word")
    ap.add_argument("--sentence-pause", type=float, default=2.5, help="seconds without hands that end a sentence")
    ap.add_argument("--wordlist", default=None, help="text file, one word per line, to correct spelled words")
    ap.add_argument("--translate", default="off", choices=["off", "simple", "llm"],
                    help="turn each finished sentence into English: llm = Groq API, simple = offline formatting")
    ap.add_argument("--translate-model", default="openai/gpt-oss-20b")
    ap.add_argument("--translate-context", default=os.environ.get("SIGN_CONTEXT", ""),
                    help="optional facts about the signer for the English step, e.g. \"Ravi, a student; talks about school, "
                         "family, food\" (or set SIGN_CONTEXT)")
    ap.add_argument("--push", default=None, metavar="URL",
                    help="send every finished sentence to server.py, e.g. http://localhost:8000 (for the ESP)")
    ap.add_argument("--push-token", default=None, help="shared secret if server.py was started with --token")
    ap.add_argument("--only-sentences", action="store_true", help="do not print individual signs")
    ap.add_argument("--speak", action="store_true", help="read finished sentences aloud (macOS 'say')")
    ap.add_argument("--save-dir", default="transcripts")
    ap.add_argument("--fast", action="store_true", help="lighter MediaPipe models (faster, slightly less accurate)")
    args = ap.parse_args()

    torch.set_num_threads(2)
    models = []
    for path in args.weights:
        model, ck = load_model(path)
        models.append({"model": model, "classes": ck["classes"], "ck": ck, "role": detect_role(ck["classes"])})
        print(f"Loaded {path}: {models[-1]['role']} model, {ck['arch']}, {ck['features']} features, "
              f"{len(ck['classes'])} classes, test acc {ck.get('test_acc', float('nan')):.3f}")
    modes = ["auto", "words", "letters"] if len({m["role"] for m in models}) > 1 else ["auto"]
    mode = args.mode if args.mode in modes else "auto"

    scales = sorted(float(s) for s in args.scales.split(","))
    wordlist = Path(args.wordlist).read_text(encoding="utf-8").split() if args.wordlist else None
    builder = SentenceBuilder(args.threshold, args.vote_window, args.min_votes, 3,
                              args.letter_gap, args.sentence_pause, wordlist)
    transcript = Transcript(args.save_dir)
    use_pose = any(m["ck"]["features"] == "full" for m in models)
    ex = Extractor(use_pose=use_pose, static=False, fast=args.fast)

    translator = executor = None
    if args.translate != "off":
        translator = GlossTranslator(args.translate, args.translate_model, user_context=args.translate_context)
        executor = ThreadPoolExecutor(max_workers=1)  # one at a time, keeps sentence order, never blocks the camera
        print(translator.describe())
    pusher = Pusher(args.push, args.push_token) if args.push else None
    if pusher:
        print(f"Pushing sentences to {args.push}")
    state = {"en": ""}

    video = args.video is not None
    if video:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stream = None
    else:
        cap = open_camera(args.camera)
        stream = CameraStream(cap)
    print("Sign naturally; pause to end a sentence. Ctrl+C / q to quit.\n")

    def stamp(t):
        return f"{int(t // 60):02d}:{t % 60:04.1f}" if video else time.strftime("%H:%M:%S")

    def say(text):
        if args.speak and sys.platform == "darwin":
            subprocess.Popen(["say", text])

    def translate_job(gloss):
        english = translator.translate(gloss)
        state["en"] = english
        print(f"   English: {english}\n", flush=True)
        transcript.add(f"EN: {english}   <= {gloss}")
        if pusher:
            pusher.send(gloss, english)   # one message with both the glosses and the English
        say(english)

    def handle(events, t):
        for ev in events:
            if ev[0] == "sign" and not args.only_sentences:
                print(f"[{stamp(t)}] {ev[1]}  ({ev[2] * 100:.0f}%)", flush=True)
            elif ev[0] == "word" and not args.only_sentences:
                print(f"[{stamp(t)}] spelled: {ev[1]}", flush=True)
            elif ev[0] == "sentence":
                print(f"\n>> {ev[1]}", flush=True)
                transcript.add(ev[1])
                if translator is not None:
                    executor.submit(translate_job, ev[1])
                else:
                    if pusher:
                        pusher.send(ev[1], "")
                    print(flush=True)
                    say(ev[1])

    times, frames = deque(), deque()
    last_seq, frame_i, last_hand_t = 0, 0, -1e9
    shown, ms_avg, t0, t = ("", 0.0, ""), 0.0, now(), 0.0

    try:
        while True:
            if video:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] > 640:
                    frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
                t = frame_i / fps
            else:
                frame, last_seq = stream.read(last_seq)
                if frame is None:
                    if not stream.ok:
                        break
                    continue
                t = now() - t0
            frame_i += 1

            tp = now()
            raw, n, det = ex.process(frame)
            times.append(t)
            frames.append(raw)
            while times[0] < t - scales[-1] - 0.5:
                times.popleft()
                frames.popleft()
            if n:
                last_hand_t = t
            hands_visible = (t - last_hand_t) < 0.4

            if frame_i % args.stride == 0:
                pairs = []
                if hands_visible:
                    t_arr, f_arr = np.array(times), np.stack(frames)
                    for w in scales:
                        win = sample_window(t_arr, f_arr, t, w)
                        if win is not None and hands_present(win) >= 6:
                            pairs.append((w, win))
                pick = None
                if pairs:
                    batch = torch.from_numpy(np.stack([w for _, w in pairs]))
                    cands = []
                    with torch.inference_mode():
                        for m in models:
                            p = torch.softmax(m["model"](batch), dim=-1)
                            conf, idx = p.max(dim=1)
                            k = int(conf.argmax())
                            cands.append((m["role"], m["classes"][int(idx[k])], float(conf[k])))
                    motion = window_motion(pairs[0][1], pairs[0][0])
                    pick = arbitrate(cands, motion, mode, args.motion_thr)
                if pick:
                    shown = (pick[1], pick[2], pick[0])
                    events = builder.update(pick[1], pick[2], t, hands_visible)
                else:
                    shown = ("", 0.0, "")
                    events = builder.update(None, 0.0, t, hands_visible)
            else:
                events = builder.tick(t)
            handle(events, t)
            if pusher:
                pusher.send_live(builder.current_text())   # words signed so far in the open sentence ("" when idle)
            ms_avg = 0.9 * ms_avg + 0.1 * (now() - tp) * 1000 if ms_avg else (now() - tp) * 1000

            if not args.no_window:
                draw_landmarks(frame, det)
                view = frame if video else cv2.flip(frame, 1)
                h, w = view.shape[:2]
                top = f"{shown[0]} {shown[1] * 100:.0f}%" if shown[0] else "..."
                if len(modes) > 1 and shown[0]:
                    top += f" [{shown[2]}]"
                cv2.putText(view, top, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                info = f"{1000 / max(ms_avg, 1):.0f} fps" + (f"  mode: {mode}" if len(modes) > 1 else "")
                cv2.putText(view, info, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                lines = textwrap.wrap(builder.current_text(), 40)[-3:] or ["(waiting for signs)"]
                if state["en"]:
                    lines = ["EN: " + state["en"][-40:]] + lines
                elif builder.history:
                    lines = ["last: " + builder.history[-1][-36:]] + lines
                box = 28 * len(lines) + 8
                cv2.rectangle(view, (0, h - box), (w, h), (0, 0, 0), -1)
                for i, ln in enumerate(lines):
                    cv2.putText(view, ln, (10, h - box + 24 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2)
                cv2.imshow("sign live", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    builder.clear()
                    state["en"] = ""
                elif key in (8, 127):
                    builder.delete_last()
                elif key in (10, 13):
                    handle(builder.finish(t), t)
                elif key == ord("m") and len(modes) > 1:
                    mode = modes[(modes.index(mode) + 1) % len(modes)]
    except KeyboardInterrupt:
        pass
    finally:
        handle(builder.finish(t), t)
        if executor is not None:
            executor.shutdown(wait=True)  # let a pending translation finish and print
        if pusher is not None:
            pusher.close()
        if stream is not None:
            stream.release()
        else:
            cap.release()
        ex.close()
        if not args.no_window:
            cv2.destroyAllWindows()
        print(f"\nStopped. {frame_i} frames, avg {ms_avg:.1f} ms/frame ({1000 / max(ms_avg, 1):.0f} fps max).")
        if transcript.path:
            print(f"Transcript saved to {transcript.path}")


if __name__ == "__main__":
    main()

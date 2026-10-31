"""Letters model + words model working together (pure NumPy / Python, no torch).

Two small models are better than one when you mix photo-based letters with video-based words:
the letters model uses hand-shape only, the words model also uses motion and body position.
At every step both models predict; we pick one using how much the hands MOVE in the window:
  * hands almost still  -> fingerspelling is far more likely -> prefer the letters model
  * hands moving        -> a word sign is far more likely    -> prefer the words model
The model that does not fit the motion keeps only `penalty` of its confidence, so a very
confident wrong-mode prediction can still win. Press 'm' in the live window to force a mode.
"""
import numpy as np

from common import HANDS, HFLAG, IDLE_NAMES


def is_letter_label(name):
    """A single letter or digit sign (fingerspelling)."""
    return len(name) == 1 and name.isalnum()


def detect_role(classes):
    """'letters' if every non-idle class is a single character, else 'words'."""
    real = [c for c in classes if c.lower() not in IDLE_NAMES]
    return "letters" if real and all(is_letter_label(c) for c in real) else "words"


def keep_class(name, subset):
    """Used by train.py --subset: idle is kept in both subsets."""
    if subset == "all" or name.lower() in IDLE_NAMES:
        return True
    return is_letter_label(name) == (subset == "letters")


def window_motion(win, duration):
    """Wrist path length of the more active hand per second, in image heights/second.
    win: (T, RAW_DIM). Resting hands are around 0.02, a moving sign is typically above 0.3."""
    win = np.asarray(win)
    flag = win[:, HFLAG] > 0
    wrist = win[:, HANDS].reshape(len(win), 2, 21, 3)[:, :, 0, :2]
    both = flag[1:] & flag[:-1]
    step = np.where(both, np.linalg.norm(wrist[1:] - wrist[:-1], axis=-1), 0.0)
    return float(step.sum(axis=0).max() / max(duration, 1e-3))


def arbitrate(cands, motion, mode="auto", motion_thr=0.25, penalty=0.6):
    """cands: list of (role, label, confidence). Returns the chosen (role, label, confidence) or None."""
    if not cands:
        return None
    if mode in ("letters", "words"):
        pool = [c for c in cands if c[0] == mode] or cands
        return max(pool, key=lambda c: c[2])
    dynamic = motion >= motion_thr

    def score(c):
        return c[2] * (1.0 if (c[0] == "words") == dynamic else penalty)

    return max(cands, key=score)

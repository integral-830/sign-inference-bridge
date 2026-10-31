"""Quick environment check:  python check_setup.py
Shows what is installed, which MediaPipe backend is used, whether the Apple GPU (MPS) is available,
whether the camera opens, and whether you already have data and a trained model.
(For a deeper test of the code itself run: python selftest.py)
"""
import sys
from pathlib import Path


def line(ok, text):
    print(("[ OK ] " if ok else "[FAIL] ") + text)


print("Python", sys.version.split()[0])
if sys.version_info[:2] >= (3, 13):
    print("[WARN] Python 3.13+ may lack wheels for some packages. Python 3.11 is the safest choice.")

for name in ["numpy", "cv2", "mediapipe", "torch", "sklearn", "tqdm"]:
    try:
        mod = __import__(name)
        line(True, f"{name} {getattr(mod, '__version__', '')}")
    except Exception as e:
        hint = ""
        if "multiarray" in str(e) or "numpy" in str(e):
            hint = ('\n         NumPy / OpenCV versions do not match. Fix:  pip install --force-reinstall "numpy<2" '
                    '&& pip install --force-reinstall -r requirements.txt')
        line(False, f"{name} not importable -> pip install -r requirements.txt  ({e}){hint}")

try:
    import groq  # noqa: F401
    import os
    line(bool(os.environ.get("GROQ_API_KEY")),
         "English translation (optional): groq installed" +
         ("" if os.environ.get("GROQ_API_KEY") else ", but GROQ_API_KEY is not set"))
except ImportError:
    print("[ -- ] English translation (optional): 'pip install groq' + GROQ_API_KEY to enable --translate llm")

try:
    import torch
    line(torch.backends.mps.is_available(), "Apple GPU (MPS) available (CPU still works if not)")
except Exception:
    pass

try:
    from common import backend_name, MODEL_DIR
    b = backend_name()
    line(True, f"MediaPipe backend: {b}" + ("" if b == "legacy" else
         f" (model files are downloaded once into ./{MODEL_DIR}/ on first use)"))
except BaseException as e:  # noqa: BLE001
    line(False, f"MediaPipe backend: {e}")

try:
    from common import open_camera
    cap = open_camera(0)
    ok, _ = cap.read()
    cap.release()
    line(ok, "camera opened and returned a frame" if ok else "camera opened but gave no frame")
except Exception as e:
    line(False, f"camera: {e}")

seq = Path("data/sequences")
n = len(list(seq.rglob("*.npy"))) if seq.exists() else 0
line(n > 0, f"training data: {n} sample files in data/sequences/")
line(Path("model/sign_model.pt").exists(), "trained model at model/sign_model.pt")

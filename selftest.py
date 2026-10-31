"""Self-test (about 30-60 s, no camera needed):  python selftest.py

Checks your install and the whole pipeline on synthetic data: landmark packing, time windows,
feature/model shapes, augmentation, a short training run + model reload, and the sentence builder.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
results = []


def check(name, fn):
    print(f"{name:<46}", end="", flush=True)
    try:
        msg = fn()
        print("OK", msg or "")
        results.append(True)
    except BaseException as e:  # noqa: BLE001
        print("FAIL:", repr(e)[:400])
        results.append(False)


def synth_raw(rng, k, n_classes):
    """One fake (30, 147) sample for class k: class 0 = idle (no hands), others = a hand shape + motion."""
    from common import RAW_DIM
    seq = np.zeros((30, RAW_DIM), np.float32)
    shoulders = np.array([[0.8, 0.9], [1.2, 0.9]], np.float32)
    for t in range(30):
        seq[t, 128 + 6:128 + 10] = (shoulders + rng.normal(0, 0.003, (2, 2))).ravel()  # idx 3,4 of pose points
        seq[t, 128 + 0:128 + 2] = [1.0, 0.5]
        seq[t, 146] = 1.0
    if k == 0:
        return seq
    trng = np.random.RandomState(k)
    template = trng.normal(0, 0.05, (21, 3)).astype(np.float32)
    template[0] = 0
    direction = trng.normal(0, 1, 2) * 0.004
    start = np.array([1.0, 1.2]) + rng.normal(0, 0.05, 2)
    for t in range(30):
        pts = template + rng.normal(0, 0.003, (21, 3))
        pts[:, :2] += start + direction * t
        seq[t, 0:63] = pts.ravel()
        seq[t, 126] = 1.0
    return seq


def t_imports():
    import cv2, mediapipe, sklearn, torch  # noqa: E401,F401
    from common import backend_name
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return f"torch {torch.__version__} ({dev}), opencv {cv2.__version__}, mediapipe {mediapipe.__version__} [{backend_name()} backend]"


def t_pack_window():
    from common import RAW_DIM, pack_frame, sample_window
    pose = np.zeros((33, 3), np.float32)
    pose[11] = [0.6, 0.5, 1]
    pose[12] = [0.4, 0.5, 1]
    hands = [np.random.rand(21, 3).astype(np.float32), np.random.rand(21, 3).astype(np.float32)]
    raw, n = pack_frame(hands, pose, 4 / 3)
    assert raw.shape == (RAW_DIM,) and n == 2 and raw[126] == 1 and raw[127] == 1 and raw[146] == 1
    assert raw[0] <= raw[63], "hand slots must be ordered left-to-right"
    raw, n = pack_frame([], None, 1.0)
    assert n == 0 and not raw.any()
    times = np.arange(0, 3, 1 / 30.0)
    frames = np.tile(times[:, None], (1, 5))
    w = sample_window(times, frames, times[-1], 1.0)
    assert w.shape == (30, 5) and abs(w[0, 0] - (times[-1] - 1.0)) < 0.04 and abs(w[-1, 0] - times[-1]) < 1e-6
    assert sample_window(times[:10], frames[:10], times[9], 1.0) is None


def t_models():
    import torch
    from model import FEATURE_DIMS, build_model, compute_features
    rng = np.random.RandomState(0)
    x = torch.from_numpy(np.stack([synth_raw(rng, k % 4, 4) for k in range(8)]))
    out = []
    for feats in ("hands", "full"):
        f = compute_features(x, feats)
        assert f.shape == (8, 30, FEATURE_DIMS[feats]) and torch.isfinite(f).all()
        for arch in ("tcn", "lstm", "transformer"):
            m = build_model(arch, 7, feats).eval()
            y = m(x)
            assert y.shape == (8, 7) and torch.isfinite(y).all()
        out.append(feats)
    return "hands+full x tcn/lstm/transformer"


def t_augment():
    import torch
    from common import HANDS, HFLAG, PFLAG
    from train import augment
    rng = np.random.RandomState(1)
    x = torch.from_numpy(np.stack([synth_raw(rng, k % 5, 5) for k in range(64)]))
    for mirror in (0.0, 1.0):
        a = augment(x, mirror, 0.5)
        assert a.shape == x.shape and torch.isfinite(a).all()
        hands = a[..., HANDS].reshape(64, 30, 2, 63)
        flag = a[..., HFLAG]
        assert (hands.abs().sum(-1)[flag == 0] == 0).all(), "absent hands must stay zero"
        assert set(flag.unique().tolist()) <= {0.0, 1.0}
        assert set(a[..., PFLAG].unique().tolist()) <= {0.0, 1.0}
    # mirrored single hand stays in slot 0
    assert (augment(x, 1.0, 0.0)[..., 126].sum() > 0)


def t_train_reload():
    import torch
    rng = np.random.RandomState(2)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        names = ["idle", "alpha", "beta", "gamma", "delta", "eps"]
        for k, nm in enumerate(names):
            (td / "seq" / nm).mkdir(parents=True)
            for i in range(40):
                np.save(td / "seq" / nm / f"{i}.npy", synth_raw(rng, k, len(names)))
        cmd = [sys.executable, str(HERE / "train.py"), "--data", str(td / "seq"), "--out", str(td / "model"),
               "--model", "tcn", "--epochs", "14", "--device", "cpu", "--patience", "50"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
        if r.returncode != 0:
            raise RuntimeError((r.stdout + r.stderr)[-800:])
        from live_demo import load_model
        model, ck = load_model(str(td / "model" / "sign_model.pt"))
        assert ck["classes"] == sorted(names) and ck["features"] in ("hands", "full")
        test = np.stack([synth_raw(rng, k, 6) for k in range(6) for _ in range(5)])
        truth = np.array([sorted(names).index(names[k]) for k in range(6) for _ in range(5)])
        with torch.inference_mode():
            pred = model(torch.from_numpy(test)).argmax(1).numpy()
        acc = float((pred == truth).mean())
        assert acc >= 0.8, f"reloaded model accuracy only {acc:.2f}"
        return f"reloaded model acc {acc:.2f} ({ck['features']} features)"


def t_sentences():
    from sentence import SentenceBuilder

    def run(builder, plan, dt=1 / 15):
        t, events = 0.0, []
        for label, steps, visible in plan:
            for _ in range(steps):
                t += dt
                events += builder.update(label, 0.9 if label else 0.0, t, visible)
        return events, t

    letters = lambda s: [x for ch in s for x in ((ch, 12, True), (None, 4, True))]
    plan = [("good", 15, True), (None, 5, False)] + letters("HELLO") + [("thanks", 15, True),
                                                                      (None, 60, False)]
    b = SentenceBuilder()
    events, _ = run(b, plan)
    sentences = [e[1] for e in events if e[0] == "sentence"]
    assert sentences == ["GOOD HELLO THANKS"], sentences
    signs = [e[1] for e in events if e[0] == "sign"]
    assert signs.count("L") == 2, f"double letter lost: {signs}"

    b2 = SentenceBuilder(wordlist=["hello", "world"])
    events, _ = run(b2, letters("HELO") + [(None, 60, False)])
    assert [e[1] for e in events if e[0] == "sentence"] == ["HELLO"], events

    # X -> Y transition must not commit X twice; idle ends the sentence
    b3 = SentenceBuilder()
    events, _ = run(b3, [("yes", 15, True), ("no", 15, True), ("idle", 75, True)])
    assert [e[1] for e in events if e[0] == "sign"] == ["YES", "NO"], events
    assert [e[1] for e in events if e[0] == "sentence"] == ["YES NO"]
    return "sentence + letters + idle ok"


def t_labels():
    from labels import auto_target, load_map, plan_labels, resolve
    cases = {"E1": "E", "E2": "E", "9a": "9", "9b": "9", "12. Hello": "Hello", "hello (2)": "hello",
             "hello_2": "hello", "A_2": "A", "10": "10", "21": "21", "hello": "hello", "A": "A", "I": "I"}
    for src, want in cases.items():
        assert auto_target(src) == want, (src, auto_target(src), want)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        uid = 0  # unique file names; on macOS 'Hello' and 'hello' are the SAME folder (case-insensitive disk)
        for name, n in [("E", 2), ("E1", 3), ("E2", 2), ("9a", 2), ("9b", 1), ("Hello", 2), ("hello", 1), ("thanks", 2), ("thank_you", 1)]:
            (td / "seq" / name).mkdir(parents=True, exist_ok=True)
            for i in range(n):
                uid += 1
                np.save(td / "seq" / name / f"{uid}_{i}.npy", np.zeros((30, 147), np.float32))
        (td / "map.json").write_text('{"thank_you": ["thanks"]}')
        mapping = load_map(td / "map.json")
        assert mapping == {"thanks": "thank_you"}
        plan = plan_labels(sorted(p.name for p in (td / "seq").iterdir()), mapping, True, False)
        assert plan["E"] == ["E", "E1", "E2"] and plan["9"] == ["9a", "9b"] and plan["thank_you"] == ["thank_you", "thanks"]
        assert resolve("Hello", None, False, True) == "hello" and resolve("e", None, False, True) == "E"
        cmd = [sys.executable, str(HERE / "merge_labels.py"), "--data", str(td / "seq"), "--map", str(td / "map.json"), "--auto"]
        dry = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
        assert dry.returncode == 0 and "Dry run" in dry.stdout and (td / "seq" / "E1").exists(), dry.stdout + dry.stderr
        done = subprocess.run(cmd + ["--apply", "--merge-case"], capture_output=True, text=True, cwd=str(HERE))
        assert done.returncode == 0, done.stdout + done.stderr
        left = sorted(p.name for p in (td / "seq").iterdir())
        assert left == ["9", "E", "hello", "thank_you"], left
        assert len(list((td / "seq" / "E").glob("*.npy"))) == 7 and len(list((td / "seq" / "hello").glob("*.npy"))) == 3
        assert len(list((td / "seq" / "thank_you").glob("*.npy"))) == 3
    return "E1/E2->E, 9a/9b->9, case + map merges ok"


def t_organize():
    import json
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vids = td / "videos" / "nested"
        vids.mkdir(parents=True)
        for i in range(6):
            (vids / f"clip{i}.mp4").write_bytes(b"x")
        (td / "d.csv").write_text("uid;Video file;Gloss;split\n"
                                  "1;clip0.mp4;hello;train\n2;clip1.mp4;hello;train\n3;clip2;thanks;val\n"
                                  "4;nested/clip3.mp4;thanks;train\n5;clip4.mp4;go/come;train\n6;missing.mp4;hello;train\n")
        base = [sys.executable, str(HERE / "organize_videos.py"), "--videos", str(td / "videos")]
        r = subprocess.run(base + ["--csv", str(td / "d.csv"), "--out", str(td / "out"), "--splits", "train,val",
                                   "--split-col", "split"], capture_output=True, text=True, cwd=str(HERE))
        assert r.returncode == 0, r.stdout + r.stderr
        out = td / "out"
        assert sorted(p.name for p in out.iterdir()) == ["go_come", "hello", "thanks"], list(out.iterdir())
        assert len(list((out / "hello").iterdir())) == 2 and len(list((out / "thanks").iterdir())) == 2
        assert "1 rows had no matching video" in r.stdout, r.stdout
        r = subprocess.run(base + ["--csv", str(td / "d.csv"), "--list-columns"], capture_output=True, text=True, cwd=str(HERE))
        assert "Video file" in r.stdout and "Gloss" in r.stdout, r.stdout
        (td / "w.json").write_text(json.dumps([{"gloss": "book", "instances": [{"video_id": "clip0", "split": "train"},
                                                                              {"video_id": "clip1", "split": "test"}]},
                                               {"gloss": "cat", "instances": [{"video_id": "clip2", "split": "train"}]}]))
        r = subprocess.run(base + ["--wlasl-json", str(td / "w.json"), "--out", str(td / "out2"), "--splits", "train"],
                           capture_output=True, text=True, cwd=str(HERE))
        assert r.returncode == 0 and sorted(p.name for p in (td / "out2").iterdir()) == ["book", "cat"], r.stdout + r.stderr
        assert len(list((td / "out2" / "book").iterdir())) == 1
    return "csv (guess columns, split filter, ids, missing) + WLASL json ok"


def t_fusion():
    from common import RAW_DIM
    from fusion import arbitrate, detect_role, keep_class, window_motion
    still = np.zeros((30, RAW_DIM), np.float32)
    still[:, 126] = 1
    still[:, 0:2] = [1.0, 1.0]
    moving = still.copy()
    moving[:, 0] = np.linspace(1.0, 1.6, 30)
    assert window_motion(still, 0.8) < 0.05 and window_motion(moving, 0.8) > 0.5
    assert detect_role(["idle", "A", "B", "9"]) == "letters" and detect_role(["idle", "hello", "A"]) == "words"
    assert keep_class("A", "letters") and not keep_class("hello", "letters") and keep_class("idle", "letters")
    assert keep_class("hello", "words") and not keep_class("A", "words") and keep_class("idle", "words")
    cands = [("words", "hello", 0.80), ("letters", "B", 0.90)]
    assert arbitrate(cands, 0.02)[1] == "B" and arbitrate(cands, 0.8)[1] == "hello"
    assert arbitrate([("words", "idle", 0.95), ("letters", "A", 0.85)], 0.02)[1] == "A"
    assert arbitrate([("words", "hello", 0.7), ("letters", "idle", 0.95)], 0.9)[1] == "hello"
    assert arbitrate(cands, 0.02, mode="words")[1] == "hello" and arbitrate(cands, 0.9, mode="letters")[1] == "B"
    assert arbitrate([("words", "x", 0.9)], 0.0, mode="letters")[1] == "x" and arbitrate([], 0.1) is None
    return "motion + role + arbitration ok"


def t_translate():
    from translate import GlossTranslator, SYSTEM, build_system, clean_reply, simple_english
    assert simple_english("ME SCHOOL GO") == "Me school go." and simple_english("I LOVE YOU") == "I love you."
    assert clean_reply('  "I am going to school."  ') == "I am going to school."
    assert clean_reply("English: Thank you.\nNote: blah") == "Thank you." and clean_reply("") == "" and clean_reply("x" * 400) == ""
    for must in ("Indian Sign Language", "cannot speak", "recogniser", "finger-spelled", "(unclear)", "first person"):
        assert must in SYSTEM, must

    class Msg:
        def __init__(self, text):
            self.content = text

    class Choice:
        def __init__(self, text):
            self.message = Msg(text)

    class Resp:
        def __init__(self, text):
            self.choices = [Choice(text)]

    class Fake:
        def __init__(self):
            self.calls, self.fail, self.reject_reasoning, self.empty_first = [], False, False, False
            self.chat = self
            self.completions = self

        def create(self, **kw):
            self.calls.append(kw)
            if self.fail:
                raise RuntimeError("offline")
            if self.reject_reasoning and "reasoning_effort" in kw:
                raise TypeError("unexpected keyword argument 'reasoning_effort'")
            if self.empty_first and "reasoning_effort" in kw:
                return Resp("")          # reasoning used up the token budget
            return Resp("I am going to school.")

    fake = Fake()
    tr = GlossTranslator("llm", client=fake, user_context="Ravi, a student;  talks about school")
    assert tr.translate("ME SCHOOL GO") == "I am going to school."
    assert tr.translate("TOMORROW ME GO") == "I am going to school."
    first, second = fake.calls[0]["messages"], fake.calls[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[1]["content"] == "Signs: ME SCHOOL GO" and second[3]["content"] == "Signs: TOMORROW ME GO"
    assert "Indian Sign Language" in first[0]["content"] and "Ravi, a student; talks about school" in first[0]["content"]
    assert build_system("") == SYSTEM
    assert fake.calls[0]["max_completion_tokens"] >= 500 and fake.calls[0]["reasoning_effort"] == "low"   # gpt-oss settings
    fake.reject_reasoning = True                    # old SDK / model without reasoning_effort: retry with plain settings
    n = len(fake.calls)
    assert tr.translate("A B") == "I am going to school." and len(fake.calls) == n + 2 and "max_tokens" in fake.calls[-1]
    fake.reject_reasoning, fake.empty_first = False, True   # empty answer on the first try: retry
    n = len(fake.calls)
    assert tr.translate("C D") == "I am going to school." and len(fake.calls) == n + 2
    fake.empty_first, fake.fail = False, True
    assert tr.translate("MY NAME RAM") == "My name ram."
    fake.fail = False
    assert tr.translate("E F") == "I am going to school."  # recovered after a failure
    other = GlossTranslator("llm", model="llama-3.1-8b-instant", client=Fake())
    other.translate("X")
    assert "reasoning_effort" not in other.client.calls[0]   # only for gpt-oss models
    assert GlossTranslator("simple").translate("YOU NAME WHAT") == "You name what."
    return "context prompt, history, signer context, parameter retry, empty-answer retry, failure fallback ok"


def t_extractor():
    from common import Extractor
    ex = Extractor(use_pose=True, static=True)
    raw, n, det = ex.process(np.zeros((480, 640, 3), np.uint8))
    ex.close()
    assert n == 0 and raw.shape[0] == 147


if __name__ == "__main__":
    print("Sign-recognition v2 self-test\n")
    check("imports / versions", t_imports)
    check("landmark packing + time windows", t_pack_window)
    check("features + models (6 combos)", t_models)
    check("augmentation", t_augment)
    check("short training + reload", t_train_reload)
    check("sentence builder", t_sentences)
    check("label merging (E1/E2 ...)", t_labels)
    check("csv / json video sorter", t_organize)
    check("two-model fusion logic", t_fusion)
    check("English translation step", t_translate)
    check("mediapipe on a blank image", t_extractor)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)

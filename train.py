"""Train the sign classifier (PyTorch; uses the Apple GPU via MPS on an M2).

  python train.py                      # trains TCN, LSTM and Transformer, keeps the best
  python train.py --model tcn          # one model only (fastest)
  python train.py --features hands     # force hand-only features (default: auto)
  python train.py --device cpu         # if MPS gives any error
  python train.py --subset letters     # letters/digits only  -> model/letters/sign_model.pt
  python train.py --subset words       # everything else      -> model/words/sign_model.pt

Outputs: model/sign_model.pt (weights + class names + feature mode), model/labels.json, model/report.txt
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from common import SEQ_LEN, RAW_DIM, HANDS, HFLAG, POSE, PFLAG, N_POSE, FORMAT_VERSION
from common import IDLE_NAMES
from fusion import keep_class
from model import build_model

MIRROR_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7]  # swap left/right pose points when mirroring
LR = {"tcn": 3e-3, "lstm": 2e-3, "transformer": 6e-4}


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------------------------------------------------------------------ data
def load_data(root, subset="all"):
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"{root} not found. Run prepare_dataset.py or collect_data.py first.")
    classes = sorted(p.name for p in root.iterdir()
                     if p.is_dir() and any(p.glob("*.npy")) and keep_class(p.name, subset))
    X, y, old = [], [], 0
    for i, c in enumerate(classes):
        for f in sorted((root / c).glob("*.npy")):
            a = np.load(f)
            if a.shape == (SEQ_LEN, RAW_DIM):
                X.append(a)
                y.append(i)
            else:
                old += 1
    if old:
        print(f"NOTE: ignored {old} old-format files (v1). Re-run prepare_dataset.py / collect_data.py for them.")
    if not X:
        raise SystemExit("No usable v2 samples found. Re-run prepare_dataset.py / collect_data.py.")
    y = np.array(y)
    keep = sorted(set(y.tolist()))
    remap = {k: i for i, k in enumerate(keep)}
    return np.stack(X).astype(np.float32), np.array([remap[v] for v in y], np.int64), [classes[k] for k in keep]


def choose_features(X, y, classes, requested):
    """'full' uses body pose too; it is only safe when pose is present for (nearly) every class equally.
    If some classes come from close-up photos (no body) and others from webcam/video (body visible),
    the model would learn 'body visible => not that class'. In that case use hand-only features."""
    pose_on = X[:, :, PFLAG].mean(axis=(1, 2)) > 0.5
    rate = np.array([pose_on[y == k].mean() for k in range(len(classes))])
    if requested != "auto":
        return requested
    if rate.max() - rate.min() > 0.5:
        low = [classes[k] for k in range(len(classes)) if rate[k] < 0.25][:12]
        print(f"features: hands-only (auto) - body pose is missing for some classes ({', '.join(low)} ...) "
              f"but present for others; mixing would bias the model.")
        return "hands"
    if rate.mean() >= 0.5:
        print("features: full (hands + body pose, auto)")
        return "full"
    print("features: hands-only (auto) - body pose rarely detected in this data")
    return "hands"


def split(X, y):
    def tts(a, b, size):
        try:
            return train_test_split(a, b, test_size=size, stratify=b, random_state=0)
        except ValueError:  # a class has too few samples to stratify
            return train_test_split(a, b, test_size=size, random_state=0)

    Xtr, Xtmp, ytr, ytmp = tts(X, y, 0.3)
    Xva, Xte, yva, yte = tts(Xtmp, ytmp, 0.5)
    return Xtr, ytr, Xva, yva, Xte, yte


# ------------------------------------------------------------------------------ augmentation
def augment(x, mirror_p=0.3, pose_drop=0.3):
    """Augment a batch of RAW windows (B, T, RAW_DIM) on-device. Simulates what the live camera does:
    different signing speed, tilt, left-handed/mirrored signers, jitter, missed detections, and windows
    that catch a sign only partly (onset/offset)."""
    B, T, _ = x.shape
    dev = x.device

    # 1) time warp (speed 0.8x-1.25x + small shift)
    s = torch.empty(B, 1, device=dev).uniform_(0.8, 1.25)
    shift = torch.empty(B, 1, device=dev).uniform_(-0.12, 0.12) * T
    c0 = (T - 1) / 2
    base = torch.arange(T, device=dev, dtype=torch.float32).unsqueeze(0)
    idx = ((base - c0) * s + c0 + shift).round().clamp(0, T - 1).long()
    x = x[torch.arange(B, device=dev).unsqueeze(1), idx]

    hands = x[..., HANDS].reshape(B, T, 2, 21, 3)
    hflag = x[..., HFLAG]
    pose = x[..., POSE].reshape(B, T, N_POSE, 2)
    pflag = x[..., PFLAG]

    # 2) small in-plane rotation (+-12 degrees) of everything
    ang = (torch.rand(B, device=dev) - 0.5) * 2 * math.radians(12)
    cs, sn = torch.cos(ang), torch.sin(ang)
    c4, s4 = cs.view(B, 1, 1, 1), sn.view(B, 1, 1, 1)
    hands = torch.stack([c4 * hands[..., 0] - s4 * hands[..., 1],
                         s4 * hands[..., 0] + c4 * hands[..., 1], hands[..., 2]], dim=-1)
    c3, s3 = cs.view(B, 1, 1), sn.view(B, 1, 1)
    pose = torch.stack([c3 * pose[..., 0] - s3 * pose[..., 1],
                        s3 * pose[..., 0] + c3 * pose[..., 1]], dim=-1)

    # 3) mirror (left-handed signers / mirrored datasets)
    if mirror_p > 0:
        flip = torch.rand(B, device=dev) < mirror_p
        hm = hands * torch.tensor([-1.0, 1.0, 1.0], device=dev)
        both = (hflag[..., 0] * hflag[..., 1]).view(B, T, 1, 1, 1) > 0
        hm = torch.where(both, hm.flip(2), hm)
        hands = torch.where(flip.view(B, 1, 1, 1, 1), hm, hands)
        pm = (pose * torch.tensor([-1.0, 1.0], device=dev))[:, :, MIRROR_PERM]
        pose = torch.where(flip.view(B, 1, 1, 1), pm, pose)

    # 4) landmark jitter
    hands = hands + torch.randn_like(hands) * 0.003 * (hflag > 0).view(B, T, 2, 1, 1)
    pose = pose + torch.randn_like(pose) * 0.006 * (pflag > 0).view(B, T, 1, 1)

    # 5) missed detections (5% of frames) + hidden onset/offset (25% of samples)
    keep = (torch.rand(B, T, 1, device=dev) >= 0.05).float()
    n_hide = (torch.empty(B, 1, device=dev).uniform_(0.1, 0.5) * T).long()
    t_idx = torch.arange(T, device=dev).view(1, T)
    prefix = (torch.rand(B, 1, device=dev) < 0.5)
    hide = torch.where(prefix, t_idx < n_hide, t_idx >= (T - n_hide)) & (torch.rand(B, 1, device=dev) < 0.25)
    keep = keep * (~hide).float().unsqueeze(-1)
    hands = hands * keep.view(B, T, 1, 1, 1)
    hflag = hflag * keep

    # 6) body-pose dropout for whole windows
    if pose_drop > 0:
        pk = (torch.rand(B, 1, 1, device=dev) >= pose_drop).float()
        pose = pose * pk.view(B, 1, 1, 1)
        pflag = pflag * pk

    return torch.cat([hands.reshape(B, T, 126), hflag, pose.reshape(B, T, 2 * N_POSE), pflag], dim=-1)


# ---------------------------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, X, y, bs=512):
    model.eval()
    logits = torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])
    loss = nn.functional.cross_entropy(logits, y).item()
    preds = logits.argmax(1)
    return (preds == y).float().mean().item(), loss, preds.cpu().numpy()


def train_one(arch, features, n_classes, Xtr, ytr, Xva, yva, weights, device, args):
    model = build_model(arch, n_classes, features).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    lr = LR[arch]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps_per = math.ceil(len(Xtr) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=args.epochs * steps_per)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    pose_drop = 0.3 if features == "full" else 0.0
    best_acc, best_loss, best_state, bad = -1.0, 1e9, None, 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        total = 0.0
        for i in range(0, len(Xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:
                continue
            xb = augment(Xtr[idx], args.mirror_aug, pose_drop)
            loss = crit(model(xb), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        acc, vloss, _ = evaluate(model, Xva, yva)
        if acc > best_acc + 1e-6 or (abs(acc - best_acc) <= 1e-6 and vloss < best_loss):
            best_acc, best_loss, bad = acc, vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep == 1 or ep % 5 == 0:
            print(f"[{arch}] epoch {ep:3d}/{args.epochs}  loss {total / len(Xtr):.4f}  "
                  f"val_acc {acc:.4f}  ({time.time() - t0:.0f}s)")
        if ep > args.epochs // 3 and bad >= args.patience:
            print(f"[{arch}] early stop at epoch {ep}")
            break
    print(f"[{arch}] best val_acc {best_acc:.4f}  params {n_params / 1e3:.0f}k  time {time.time() - t0:.0f}s")
    return best_acc, best_loss, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sequences")
    ap.add_argument("--model", default="all", choices=["all", "tcn", "lstm", "transformer"])
    ap.add_argument("--features", default="auto", choices=["auto", "hands", "full"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=25, help="early-stop patience (epochs)")
    ap.add_argument("--mirror-aug", type=float, default=0.3,
                    help="probability of mirroring a sample (0 = off; use 0 if direction left/right matters)")
    ap.add_argument("--device", default="auto", help="auto | mps | cpu | cuda")
    ap.add_argument("--subset", default="all", choices=["all", "letters", "words"],
                    help="train only single-letter/digit signs ('letters') or only the rest ('words'); idle goes in both")
    ap.add_argument("--out", default="model")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    print("device:", device)

    if args.subset != "all" and args.out == "model":
        args.out = f"model/{args.subset}"
    X, y, classes = load_data(args.data, args.subset)
    if len(classes) < 2:
        raise SystemExit(
            f"Only {len(classes)} class found for subset '{args.subset}': {classes}\n"
            f"A model needs at least 2 classes (with 1 class it would answer that sign, always at 100%).\n"
            f"  - Single-character labels count as letters/digits. If '{classes[0] if classes else 'I'}' is the pronoun 'I' from INCLUDE, rename it first:\n"
            f"      python merge_labels.py --map labelmap_include.json --apply\n"
            f"  - For real letters: convert a letters dataset (RUNBOOK Part 5) or record them with collect_data.py, and record 'idle'.\n"
            f"  - If you only need words, skip the letters model:  python train.py --subset words   and delete model/letters")
    if not any(c.lower() in IDLE_NAMES for c in classes):
        print("\n*** WARNING: no 'idle' class in this data. The model will name a sign even when you are NOT signing.\n"
              "    Record it:  python collect_data.py --labels idle --samples 80 --seconds 1.5 --auto   then train again. ***\n")
    print(f"classes ({len(classes)}): " + ", ".join(classes[:30]) + (" ..." if len(classes) > 30 else ""))
    cnt = Counter(y.tolist())
    print(f"{len(X)} samples, {len(classes)} classes; smallest class {min(cnt.values())}, largest {max(cnt.values())}")
    features = choose_features(X, y, classes, args.features)

    Xtr, ytr, Xva, yva, Xte, yte = split(X, y)
    f32 = lambda a: torch.from_numpy(a).to(device=device, dtype=torch.float32)
    i64 = lambda a: torch.from_numpy(a).to(device=device, dtype=torch.long)
    Xtr_t, Xva_t, Xte_t, ytr_t, yva_t, yte_t = f32(Xtr), f32(Xva), f32(Xte), i64(ytr), i64(yva), i64(yte)

    counts = np.bincount(ytr, minlength=len(classes)).clip(min=1).astype(np.float32)
    w = (counts.sum() / (len(classes) * counts)) ** 0.5
    weights = f32(w / w.mean())

    archs = ["tcn", "lstm", "transformer"] if args.model == "all" else [args.model]
    results = {a: train_one(a, features, len(classes), Xtr_t, ytr_t, Xva_t, yva_t, weights, device, args)
               for a in archs}
    best_arch = max(results, key=lambda a: (results[a][0], -results[a][1]))
    print(f"\nBest model: {best_arch} (val_acc {results[best_arch][0]:.4f})")

    model = build_model(best_arch, len(classes), features).to(device)
    model.load_state_dict(results[best_arch][2])
    test_acc, _, pred = evaluate(model, Xte_t, yte_t)
    print(f"TEST accuracy: {test_acc:.4f}\n")

    n = len(classes)
    report = classification_report(yte, pred, labels=list(range(n)), target_names=classes, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=list(range(n)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(report)
    if n <= 40:
        print(report)
    else:
        support = cm.sum(1)
        recall = np.where(support > 0, cm.diagonal() / np.maximum(support, 1), 1.0)
        weak = np.argsort(recall)[:10]
        print("Weakest classes (recall):", ", ".join(f"{classes[k]} {recall[k]:.2f}" for k in weak))
        print("(full per-class report: model/report.txt)")
    off = cm.copy()
    np.fill_diagonal(off, 0)
    worst = sorted(((off[i, j], classes[i], classes[j]) for i in range(n) for j in range(n) if off[i, j]),
                   reverse=True)[:8]
    if worst:
        print("\nMost confused (count, true -> predicted):")
        for c, a, b in worst:
            print(f"  {c}  {a} -> {b}")

    torch.save({"version": FORMAT_VERSION, "arch": best_arch, "features": features,
                "state_dict": results[best_arch][2], "classes": classes, "seq_len": SEQ_LEN,
                "val_acc": results[best_arch][0], "test_acc": test_acc}, out / "sign_model.pt")
    (out / "labels.json").write_text(json.dumps(classes, indent=2, ensure_ascii=False))
    print(f"\nSaved {out / 'sign_model.pt'}")


if __name__ == "__main__":
    main()

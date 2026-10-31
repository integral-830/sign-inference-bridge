"""Features + small sequence models (v2). All run fast on an M2 (MPS) and on CPU.

compute_features turns RAW landmark frames (see common.py) into model inputs:
  per hand: 21x3 shape relative to the wrist and scaled by hand size, wrist velocity, present flag
  both hands: wrist-to-wrist vector
  'full' adds the body frame: wrist position relative to the shoulders, its velocity, and the
  upper-body pose points (so signs made at the forehead vs chest differ).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import SEQ_LEN, HANDS, HFLAG, POSE, PFLAG, N_POSE

FEATURE_DIMS = {"hands": 134, "full": 161}
VEL_GAIN = 10.0


def _delta(x, valid):
    """Frame-to-frame difference of x (..., T, S, C); zero unless both frames are valid (..., T, S)."""
    d = torch.zeros_like(x)
    both = (valid[..., 1:, :] * valid[..., :-1, :]).unsqueeze(-1)
    d[..., 1:, :, :] = (x[..., 1:, :, :] - x[..., :-1, :, :]) * both
    return d


def compute_features(raw, mode="full"):
    """raw: (..., T, RAW_DIM) float tensor -> (..., T, FEATURE_DIMS[mode])."""
    lead = raw.shape[:-1]
    hands = raw[..., HANDS].reshape(*lead, 2, 21, 3)
    hflag = raw[..., HFLAG]
    wrist = hands[..., 0, :]
    rel = hands - wrist.unsqueeze(-2)
    scale = rel[..., :2].abs().amax(dim=(-2, -1)).clamp_min(1e-3)  # (..., T, 2)
    shape = (rel / scale[..., None, None]) * hflag[..., None, None]
    wxy = wrist[..., :2]
    vel = _delta(wxy, hflag) * VEL_GAIN
    both = (hflag[..., 0] * hflag[..., 1]).unsqueeze(-1)
    rel_hands = (wxy[..., 1, :] - wxy[..., 0, :]) / scale[..., 0:1] * both

    feats = [shape.flatten(-3), vel.flatten(-2), hflag, rel_hands]
    if mode == "full":
        pose = raw[..., POSE].reshape(*lead, N_POSE, 2)
        pflag = raw[..., PFLAG]
        c = (pose[..., 3, :] + pose[..., 4, :]) / 2
        wd = (pose[..., 3, :] - pose[..., 4, :]).norm(dim=-1, keepdim=True).clamp_min(1e-3)
        pose_rel = (pose - c.unsqueeze(-2)) / wd.unsqueeze(-1) * pflag.unsqueeze(-1)
        wb = (wxy - c.unsqueeze(-2)) / wd.unsqueeze(-1) * (hflag * pflag).unsqueeze(-1)
        wbv = _delta(wb, hflag * pflag) * VEL_GAIN
        feats += [wb.flatten(-2), wbv.flatten(-2), pose_rel.flatten(-2), pflag]
    return torch.cat(feats, dim=-1)


# ---------------------------------------------------------------- backbones: (B, T, d) -> logits
class _Block(nn.Module):
    def __init__(self, c, k, dil, p):
        super().__init__()
        self.dw = nn.Conv1d(c, c, k, padding=dil * (k - 1) // 2, dilation=dil, groups=c)
        self.norm = nn.GroupNorm(1, c)
        self.pw1 = nn.Conv1d(c, c * 2, 1)
        self.pw2 = nn.Conv1d(c * 2, c, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):  # (B, C, T)
        y = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))
        return x + self.drop(y)


class TCN(nn.Module):
    """Dilated depthwise-separable temporal convolutions + mean/max pooling. Fastest, strong on small data."""

    def __init__(self, d, n, width=128, dropout=0.25):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d, width), nn.LayerNorm(width), nn.GELU())
        self.blocks = nn.Sequential(*[_Block(width, 5, dil, dropout) for dil in (1, 2, 4)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 2, n))

    def forward(self, x):
        x = self.inp(x).transpose(1, 2)
        x = self.blocks(x)
        return self.head(torch.cat([x.mean(-1), x.amax(-1)], dim=1))


class BiLSTM(nn.Module):
    def __init__(self, d, n, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lstm = nn.LSTM(d, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 4, n))

    def forward(self, x):
        out, _ = self.lstm(self.norm(x))
        return self.head(torch.cat([out.mean(1), out.amax(1)], dim=1))


class SmallTransformer(nn.Module):
    def __init__(self, d, n, d_model=128, heads=4, layers=2, dropout=0.2, seq_len=SEQ_LEN):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, d_model), nn.LayerNorm(d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 2, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n))

    def forward(self, x):
        x = self.encoder(self.proj(x) + self.pos)
        return self.head(x.mean(1))


BACKBONES = {"tcn": TCN, "lstm": BiLSTM, "transformer": SmallTransformer}


class SignNet(nn.Module):
    """RAW landmark windows (B, T, RAW_DIM) -> class logits. Feature computation is inside the model,
    so training, saving and live inference all use exactly the same pipeline."""

    def __init__(self, arch, num_classes, features="full"):
        super().__init__()
        self.arch, self.features = arch, features
        self.backbone = BACKBONES[arch](FEATURE_DIMS[features], num_classes)

    def forward(self, raw):
        return self.backbone(compute_features(raw, self.features))


def build_model(arch, num_classes, features="full"):
    if arch not in BACKBONES:
        raise ValueError(f"unknown arch {arch!r}, choose from {list(BACKBONES)}")
    return SignNet(arch, num_classes, features)

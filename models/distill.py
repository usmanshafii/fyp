"""Cross-modal privileged distillation (spec 7.2).

    L = L_det + lambda_r * L_resp + lambda_f * L_feat

L_resp  response distillation from the frozen teacher (RGB + LWIR + SWIR):
        * class scores: BCE between the student's and the teacher's temperature-softened sigmoids
          (T = 2), x T^2, at every location, weighted by teacher confidence + a floor so the
          teacher's "nothing here" also transfers;
        * box distributions: KL between the teacher's and the student's softened bin
          distributions (x T^2) at teacher-positive locations - the anchors the label assigner
          picks from the teacher's own predictions.
L_feat  masked feature distillation at the P2 and P3 neck outputs: a 1x1 adapter maps the student
        feature to the teacher's channels, both are standardised per channel (over the image), and
        the loss is an MSE weighted by M = sum_i w_i G_i + 0.05. G_i is a Gaussian on object i
        (sigma = max(1 cell, half the box size in cells), per axis); the 0.05 floor transfers the
        teacher's features on clutter. The adapters are discarded after training.
w_i     CNR-gap weight 1 + beta * clip((C_T,i - C_S,i) / 5, 0, 1), beta = 2: C_T,i is the largest
        per-band contrast-to-noise of object i over the teacher's bands, C_S,i over the bands the
        student received in this batch (loader CNR after noise, magnitude). It puts the effort on
        objects the teacher sees much better than the student - the LWIR-weak / SWIR-strong case.

Nothing here runs at test time: the student is an ordinary detector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BAND_INDEX = {"rgb": 0, "lwir": 1, "swir": 2}      # columns of the loader's per-object CNR array


# ============================================================================ response distillation
def response_kd(s, t, t_pos, temperature: float = 2.0, floor: float = 0.05):
    """s, t: Flat outputs of student and teacher; t_pos [B, A] teacher-positive locations."""
    T = temperature
    with torch.no_grad():
        t_conf = t.cls.sigmoid().amax(-1)
        w = t_conf + floor
        pt = torch.sigmoid(t.cls / T)
    bce = F.binary_cross_entropy_with_logits(s.cls / T, pt, reduction="none").sum(-1)
    l_cls = (bce * w).sum() / w.sum() * T * T
    if t_pos.any():
        lt = F.log_softmax(t.box_logits[t_pos] / T, -1)
        ls = F.log_softmax(s.box_logits[t_pos] / T, -1)
        kl = (lt.exp() * (lt - ls)).sum(-1).mean(-1)                       # [P]
        wp = t_conf[t_pos]
        l_box = (kl * wp).sum() / wp.sum().clamp(min=1e-6) * T * T
    else:
        l_box = s.cls.sum() * 0.0
    return l_cls + l_box, dict(resp_cls=float(l_cls), resp_box=float(l_box))


# ============================================================================ object weights and masks
def cnr_gap_weights(cnr, teacher_bands, student_bands, beta: float = 2.0, scale: float = 5.0):
    """cnr: [N, 3] per-object contrast-to-noise after noise (rgb, lwir, swir; NaN = not loaded)."""
    c = torch.as_tensor(cnr, dtype=torch.float32).abs().nan_to_num(0.0)
    if c.numel() == 0:
        return torch.zeros(0)
    ct = c[:, [BAND_INDEX[b] for b in teacher_bands]].amax(1)
    cs = c[:, [BAND_INDEX[b] for b in student_bands]].amax(1)
    return 1.0 + beta * ((ct - cs) / scale).clamp(0.0, 1.0)


def gaussian_mask(boxes, weights, shape, stride: float, floor: float = 0.05):
    """M = sum_i w_i G_i + floor on an H x W grid of the given stride. boxes: [N, 4] xyxy, pixels."""
    H, W = shape
    dev = boxes.device if torch.is_tensor(boxes) else "cpu"
    m = torch.full((H, W), float(floor), device=dev)
    if len(boxes) == 0:
        return m
    b = torch.as_tensor(boxes, dtype=torch.float32, device=dev) / stride
    ys = torch.arange(H, device=dev, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=dev, dtype=torch.float32) + 0.5
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    sx = ((b[:, 2] - b[:, 0]) / 2).clamp(min=1.0)
    sy = ((b[:, 3] - b[:, 1]) / 2).clamp(min=1.0)
    gx = torch.exp(-0.5 * ((xs[None, :] - cx[:, None]) / sx[:, None]) ** 2)     # [N, W]
    gy = torch.exp(-0.5 * ((ys[None, :] - cy[:, None]) / sy[:, None]) ** 2)     # [N, H]
    w = torch.as_tensor(weights, dtype=torch.float32, device=dev)
    return m + torch.einsum("n,nh,nw->hw", w, gy, gx)


def _standardise(f, eps: float = 1e-5):
    mu = f.mean((2, 3), keepdim=True)
    sd = f.std((2, 3), keepdim=True)
    return (f - mu) / (sd + eps)


class FeatureDistiller(nn.Module):
    """1x1 adapters (student -> teacher channels) and the masked, standardised MSE."""

    def __init__(self, student_channels: dict, teacher_channels: dict, levels=("p2", "p3")):
        super().__init__()
        self.levels = [lv for lv in levels if lv in student_channels and lv in teacher_channels]
        self.adapters = nn.ModuleDict({lv: nn.Conv2d(student_channels[lv], teacher_channels[lv], 1)
                                       for lv in self.levels})

    def forward(self, s_feats: dict, t_feats: dict, masks: dict):
        total, parts = 0.0, {}
        for lv in self.levels:
            s = _standardise(self.adapters[lv](s_feats[lv]))
            t = _standardise(t_feats[lv].detach())
            m = masks[lv]                                            # [B, H, W]
            err = ((s - t) ** 2).mean(1)                             # [B, H, W]
            loss = (err * m).sum() / m.sum().clamp(min=1e-6)
            parts[f"feat_{lv}"] = float(loss)
            total = total + loss
        return total / max(1, len(self.levels)), parts


def feature_masks(targets, feats: dict, strides: dict, weights, floor: float = 0.05):
    """Per-level [B, H, W] masks for a batch. weights: list of [N_b] tensors (None = all ones)."""
    out = {}
    for lv, f in feats.items():
        H, W = f.shape[2:]
        ms = []
        for b, tg in enumerate(targets):
            bx = tg["boxes"].to(f.device).float()
            w = weights[b] if weights is not None else torch.ones(len(bx))
            ms.append(gaussian_mask(bx, w.to(f.device), (H, W), strides[lv], floor))
        out[lv] = torch.stack(ms)
    return out

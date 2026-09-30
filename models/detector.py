"""Lightweight one-stage multispectral detector (spec 7.1).

    per-band stems (stride 2-8)  ->  mid-level concat fusion + presence flags at strides 4 and 8
    ->  shared backbone (strides 16, 32 + SPPF)  ->  PAN neck (P2 optional, P3, P4, P5)
    ->  decoupled anchor-free heads

The stride-4 (P2) level exists because YOLO's usual strides 8/16/32 make the finest grid cell larger
than a 2-8 px target; spec 7.1 asks for one baseline run without it to document the gap.

Box head: each grid point predicts four quantities - the box centre's offset from the grid point
(x, y, in strides) and log2 of the box width and height in strides - each as a discrete distribution
over `bins` values (distribution focal loss style). The decoded value is the expectation. Centre
offsets rather than YOLOv8's distances to the four edges, because the positive anchor of a 2-px box
usually lies outside the box, where edge distances would have to be negative. The distributions
are what spec 7.2's response distillation transfers ("box distributions").
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn

from models.fusion import ModalityFusion
from models.stems import SPPF, BandStem, C2f, Conv

BAND_CHANNELS = {"rgb": 3, "lwir": 1, "swir": 1}


class Flat(NamedTuple):
    cls: torch.Tensor          # [B, A, nc] class logits
    box_logits: torch.Tensor   # [B, A, 4, K] logits of the four box distributions
    params: torch.Tensor       # [B, A, 4] expected (dx, dy, log2 w, log2 h), strides
    boxes: torch.Tensor        # [B, A, 4] xyxy, pixels
    anchors: torch.Tensor      # [A, 2] grid-point centres, pixels
    strides: torch.Tensor      # [A]


class Head(nn.Module):
    def __init__(self, c, nc, hidden, bins):
        super().__init__()
        self.cls = nn.Sequential(Conv(c, hidden, 3), Conv(hidden, hidden, 3), nn.Conv2d(hidden, nc, 1))
        self.box = nn.Sequential(Conv(c, hidden, 3), Conv(hidden, hidden, 3), nn.Conv2d(hidden, 4 * bins, 1))
        nn.init.zeros_(self.box[-1].bias)
        nn.init.normal_(self.box[-1].weight, std=0.01)

    def forward(self, x):
        return self.cls(x), self.box(x)


class Detector(nn.Module):
    def __init__(self, bands, num_classes: int = 7, width=(16, 32, 64, 128, 256), depth=(1, 2, 2, 1),
                 p2: bool = True, se: bool = False, bins: int = 17, offset_range: float = 2.5,
                 log2_size_range=(-3.0, 5.0)):
        super().__init__()
        self.bands = list(bands)
        self.nc = num_classes
        self.p2 = p2
        self.bins = int(bins)
        c0, c1, c2, c3, c4 = width
        self.stems = nn.ModuleDict({b: BandStem(BAND_CHANNELS[b], c0, c1, c2, depth[0]) for b in self.bands})
        self.fusion = ModalityFusion(self.bands, c1, c2, se)
        self.p3c = C2f(c2, c2, depth[1])
        self.s16 = nn.Sequential(Conv(c2, c3, 3, 2), C2f(c3, c3, depth[2]))
        self.s32 = nn.Sequential(Conv(c3, c4, 3, 2), C2f(c4, c4, depth[3]), SPPF(c4, c4))
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.td4 = C2f(c4 + c3, c3, 1, False)
        self.td3 = C2f(c3 + c2, c2, 1, False)
        if p2:
            self.td2 = C2f(c2 + c1, c1, 1, False)
            self.d2 = Conv(c1, c1, 3, 2)
            self.bu3 = C2f(c1 + c2, c2, 1, False)
        self.d3 = Conv(c2, c2, 3, 2)
        self.bu4 = C2f(c2 + c3, c3, 1, False)
        self.d4 = Conv(c3, c3, 3, 2)
        self.bu5 = C2f(c3 + c4, c4, 1, False)
        chans = ([c1] if p2 else []) + [c2, c3, c4]
        self.feat_channels = dict(zip((["p2"] if p2 else []) + ["p3", "p4", "p5"], chans))
        self.strides = ([4] if p2 else []) + [8, 16, 32]
        self.heads = nn.ModuleList(Head(c, num_classes, max(c, 48), self.bins) for c in chans)
        for h, s in zip(self.heads, self.strides):        # YOLOv8 prior: ~5 objects per 640x512 frame
            nn.init.constant_(h.cls[-1].bias, math.log(5.0 / num_classes / ((640 / s) * (512 / s))))
        lo, hi = log2_size_range
        vals = torch.stack([torch.linspace(-offset_range, offset_range, self.bins)] * 2
                           + [torch.linspace(lo, hi, self.bins)] * 2)
        self.register_buffer("bin_values", vals, persistent=False)          # [4, K]
        self._grid_cache = {}

    # ------------------------------------------------------------------ forward
    def forward(self, images: dict, return_feats: bool = False):
        """images: {band: [B, C, H, W] in [0, 1]}; a missing or None band is absent (zeros + flag)."""
        feats = {b: (self.stems[b](images[b]) if images.get(b) is not None else None) for b in self.bands}
        f4, f8 = self.fusion(feats)
        p3 = self.p3c(f8)
        p4 = self.s16(p3)
        p5 = self.s32(p4)
        n4 = self.td4(torch.cat([self.up(p5), p4], 1))
        n3 = self.td3(torch.cat([self.up(n4), p3], 1))
        outs, names = [], []
        if self.p2:
            n2 = self.td2(torch.cat([self.up(n3), f4], 1))
            n3 = self.bu3(torch.cat([self.d2(n2), n3], 1))
            outs.append(n2)
            names.append("p2")
        o4 = self.bu4(torch.cat([self.d3(n3), n4], 1))
        o5 = self.bu5(torch.cat([self.d4(o4), p5], 1))
        outs += [n3, o4, o5]
        names += ["p3", "p4", "p5"]
        raw = [h(x) for h, x in zip(self.heads, outs)]
        if return_feats:
            return raw, dict(zip(names, outs))
        return raw

    # ------------------------------------------------------------------ decoding
    def grid(self, raw, device):
        key = tuple((r[0].shape[2], r[0].shape[3]) for r in raw) + (str(device),)
        if key not in self._grid_cache:
            anchors, strides = [], []
            for (c, _), s in zip(raw, self.strides):
                h, w = c.shape[2], c.shape[3]
                ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
                anchors.append(torch.stack([(xs.flatten() + 0.5) * s, (ys.flatten() + 0.5) * s], 1).float())
                strides.append(torch.full((h * w,), float(s), device=device))
            self._grid_cache = {key: (torch.cat(anchors), torch.cat(strides))}
        return self._grid_cache[key]

    def decode(self, params, anchors, strides):
        s = strides.view(1, -1, 1)
        ctr = anchors.unsqueeze(0) + params[..., :2] * s
        wh = torch.exp2(params[..., 2:]) * s
        return torch.cat([ctr - wh / 2, ctr + wh / 2], -1)

    def flatten(self, raw) -> Flat:
        B = raw[0][0].shape[0]
        cls = torch.cat([c.flatten(2) for c, _ in raw], 2).transpose(1, 2)
        logits = torch.cat([b.flatten(2) for _, b in raw], 2).transpose(1, 2).reshape(B, -1, 4, self.bins)
        anchors, strides = self.grid(raw, cls.device)
        params = (logits.softmax(-1) * self.bin_values).sum(-1)
        return Flat(cls, logits, params, self.decode(params, anchors, strides), anchors, strides)

    @torch.no_grad()
    def predict(self, images: dict, conf: float = 0.05, iou: float = 0.5, max_det: int = 100, centre_px: float = 2.0):
        from models.loss import nms
        f = self.flatten(self(images))
        scores = f.cls.sigmoid()
        out = []
        for b in range(scores.shape[0]):
            s, lab = scores[b].max(1)
            keep = s > conf
            bx, s, lab = f.boxes[b][keep], s[keep], lab[keep]
            if len(s) > 3000:
                top = s.topk(3000).indices
                bx, s, lab = bx[top], s[top], lab[top]
            k = nms(bx, s, lab, iou, centre_px)[:max_det]
            out.append(dict(boxes=bx[k], scores=s[k], labels=lab[k]))
        return out


def build_detector(cfg: dict, bands) -> Detector:
    d = cfg["detector"]
    return Detector(bands, num_classes=len(cfg["classes"]), width=tuple(d["width"]), depth=tuple(d["depth"]),
                    p2=bool(d["p2_head"]), se=bool(d.get("se_weighting", False)), bins=int(d.get("box_bins", 17)),
                    offset_range=float(d.get("offset_range_strides", 2.5)),
                    log2_size_range=tuple(d.get("log2_size_range", (-3.0, 5.0))))

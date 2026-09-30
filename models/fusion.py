"""Mid-level fusion (spec 7.1, 7.2): concatenate the per-band features at strides 4 and 8 with one
presence-flag channel per band, then a 1x1 conv.

A band that is absent - dropped by the student's per-batch modality dropout during training, or
simply not available at test time (LWIR-only, RGB-only) - enters the fusion block as zeros and its
presence flag, broadcast as a constant channel, is 0. One set of weights therefore serves every
input subset, and the network knows which bands it is looking at.

Optional SE-style adaptive modality weighting: a small MLP on the pooled stride-8 features and the
flags predicts one sigmoid weight per band and sample.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.stems import Conv


class ModalityFusion(nn.Module):
    def __init__(self, bands, c4: int, c8: int, se: bool = False):
        super().__init__()
        self.bands = list(bands)
        nb = len(self.bands)
        self.fuse4 = Conv(c4 * nb + nb, c4, 1)
        self.fuse8 = Conv(c8 * nb + nb, c8, 1)
        self.se = nn.Sequential(nn.Linear(c8 * nb + nb, max(8, c8 // 2)), nn.SiLU(),
                                nn.Linear(max(8, c8 // 2), nb)) if se else None

    def forward(self, feats: dict):
        """feats: {band: (f4, f8) or None}; bands missing from the dict count as absent."""
        ref = next((v for v in feats.values() if v is not None), None)
        if ref is None:
            raise ValueError("at least one band must be present")
        B = ref[0].shape[0]
        present = [feats.get(b) is not None for b in self.bands]
        flags = torch.tensor(present, dtype=ref[0].dtype, device=ref[0].device).expand(B, -1)   # [B, nb]
        f4s = [feats[b][0] if p else torch.zeros_like(ref[0]) for b, p in zip(self.bands, present)]
        f8s = [feats[b][1] if p else torch.zeros_like(ref[1]) for b, p in zip(self.bands, present)]
        if self.se is not None:
            g = torch.cat([f.mean((2, 3)) for f in f8s] + [flags], 1)
            a = torch.sigmoid(self.se(g)) * flags
            f4s = [f * a[:, i].view(B, 1, 1, 1) for i, f in enumerate(f4s)]
            f8s = [f * a[:, i].view(B, 1, 1, 1) for i, f in enumerate(f8s)]

        def with_flags(fs):
            h, w = fs[0].shape[2:]
            return torch.cat(fs + [flags.view(B, -1, 1, 1).expand(B, len(self.bands), h, w)], 1)
        return self.fuse4(with_flags(f4s)), self.fuse8(with_flags(f8s))

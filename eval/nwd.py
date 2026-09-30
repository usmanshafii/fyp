"""Tiny-box matching criteria (spec 7): IoU, centre distance and Normalized Wasserstein Distance.

IoU is unstable for tiny boxes: shifting a 3x3 box one pixel diagonally drops IoU to ~0.29.
NWD (Wang et al. 2021, arXiv:2110.13389; Xu et al. 2022, arXiv:2206.13996) models a box as the 2-D
Gaussian N((cx, cy), diag(w^2/4, h^2/4)); W2^2 = ||(cx, cy, w/2, h/2)_a - (cx, cy, w/2, h/2)_b||^2
and NWD = exp(-sqrt(W2^2) / C), C a size constant (12.8 px in Xu et al.).
"""
from __future__ import annotations

import numpy as np


def _cwh(b):
    b = np.asarray(b, float).reshape(-1, 4)
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, (b[:, 2] - b[:, 0]) / 2, (b[:, 3] - b[:, 1]) / 2]


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    aa = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)
    ab = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(-1)
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def nwd_matrix(a, b, C: float = 12.8):
    ca, cb = _cwh(a), _cwh(b)
    return np.exp(-np.sqrt(((ca[:, None] - cb[None]) ** 2).sum(-1)) / C)


def centre_distance_matrix(a, b):
    ca, cb = _cwh(a), _cwh(b)
    return np.sqrt(((ca[:, None, :2] - cb[None, :, :2]) ** 2).sum(-1))


def centre_ok_matrix(pred, gt, abs_px: float = 2.0, rel: float = 0.5):
    """Centre within max(abs_px, rel * larger side of the ground-truth box)."""
    gt = np.asarray(gt, float).reshape(-1, 4)
    tol = np.maximum(abs_px, rel * np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]))
    return centre_distance_matrix(pred, gt) <= tol[None, :]

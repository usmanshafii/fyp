"""LWIR sensor model, applied in the data loader (spec 3.2, Phase 3 brief). Pure NumPy.

Radiance -> 14-bit counts with a random responsivity (counts per kelvin near ambient) and offset
-> column fixed-pattern noise + residual pixel non-uniformity + Gaussian temporal noise at an
NETD of 30-80 mK (He et al., Applied Optics 2018 style stripe model) -> clipping and rounding
-> automatic gain control: a linear stretch between robust percentiles, like the 8-bit video an
uncooled core outputs.
"""
from __future__ import annotations

import numpy as np

from thermal.planck_lut import lwir_lut


def sample_params(rng, cfg: dict) -> dict:
    u = lambda k: float(rng.uniform(*cfg[k]))  # noqa: E731
    return dict(netd_k=u("netd_mk") / 1000.0, counts_per_k=u("counts_per_k"), offset=u("offset_counts"),
                col_k=u("column_fpn_k"), pix_k=u("pixel_fpn_k"), agc_lo=u("agc_low_pct"), agc_hi=u("agc_high_pct"))


def to_counts(L: np.ndarray, T_air_k: float, p: dict) -> np.ndarray:
    lut = lwir_lut()
    gain = p["counts_per_k"] / float(lut.dL_dT(T_air_k))           # counts per W m^-2 sr^-1
    return p["offset"] + gain * (np.asarray(L, np.float32) - float(lut.radiance(T_air_k)))


def apply(L: np.ndarray, rng, cfg: dict, T_air_k: float, params: dict | None = None) -> tuple[np.ndarray, dict]:
    """L: HxW radiance [W m^-2 sr^-1]. Returns (AGC image in [0, 1], params incl. the raw counts)."""
    p = params or sample_params(rng, cfg)
    c = to_counts(L, T_air_k, p)
    h, w = c.shape
    cpk = p["counts_per_k"]
    c = c + (rng.standard_normal(w) * p["col_k"] * cpk)[None, :]
    c = c + rng.standard_normal((h, w)) * p["pix_k"] * cpk
    c = c + rng.standard_normal((h, w)) * p["netd_k"] * cpk
    top = 2 ** int(cfg["bits"]) - 1
    c = np.clip(np.round(c), 0, top).astype(np.float32)
    lo, hi = np.percentile(c, [p["agc_lo"], p["agc_hi"]])
    out = np.clip((c - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    return out.astype(np.float32), dict(p, agc_range=[float(lo), float(hi)])


def nominal_sigma(T_air_k: float, cfg: dict) -> float:
    """Temporal + column noise sigma in radiance units (for the CNR)."""
    n = cfg["nominal"]
    k = float(np.hypot(n["netd_mk"] / 1000.0, n["column_fpn_k"]))
    return k * float(lwir_lut().dL_dT(T_air_k))

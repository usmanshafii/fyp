"""SWIR (InGaAs) sensor model, applied in the data loader (spec 3.3, Phase 4 brief). Pure NumPy.

Noise components follow the categories used in synthetic SWIR noise modelling (Jiang, Wang,
Zheng, ICCVW 2025 style): photon shot noise, dark-current shot noise, read noise, row and column
fixed-pattern offsets, per-pixel gain non-uniformity (PRNU) and stuck / hot pixels, then 14-bit
quantisation and an automatic-gain display stretch. The parameter ranges are generic InGaAs
modelling assumptions (configs/sensors.yaml), not the paper's fitted values.
"""
from __future__ import annotations

import numpy as np


def sample_params(rng, cfg: dict) -> dict:
    u = lambda k: float(rng.uniform(*cfg[k]))  # noqa: E731
    return dict(e_per_unit_ms=u("electrons_per_unit_ms"), dark_e_ms=u("dark_current_e_per_ms"), read_e=u("read_noise_e"),
                full_well=u("full_well_e"), col=u("column_fpn_frac"), row=u("row_fpn_frac"), prnu=u("prnu_frac"),
                bad=u("bad_pixel_frac"), ae_target=u("ae_target"), agc_lo=u("agc_low_pct"), agc_hi=u("agc_high_pct"))


def _light_factor(median_signal_e: float, dark_e: float, full_well: float, ae_target: float) -> float:
    """Aperture / ND at a fixed exposure time: cut the light when the frame would exceed the target."""
    target = ae_target * full_well
    total = median_signal_e + dark_e
    return float(max(target - dark_e, 0.05 * target) / median_signal_e) if total > target and median_signal_e > 0 else 1.0


def apply(L: np.ndarray, rng, cfg: dict, exposure_ms: float, params: dict | None = None) -> tuple[np.ndarray, dict]:
    """L: HxW linear SWIR radiance (render units). Returns (display image in [0, 1], params)."""
    p = params or sample_params(rng, cfg)
    h, w = L.shape
    sig = np.clip(np.asarray(L, np.float32), 0, None) * (p["e_per_unit_ms"] * exposure_ms)
    dark = p["dark_e_ms"] * exposure_ms
    sig = sig * _light_factor(float(np.median(sig)), dark, p["full_well"], p["ae_target"])
    sig = sig * (1.0 + p["prnu"] * rng.standard_normal((h, w))).astype(np.float32)
    e_mean = sig + dark
    e = e_mean + np.sqrt(e_mean + p["read_e"] ** 2) * rng.standard_normal((h, w)).astype(np.float32)
    level = float(e_mean.mean())
    e = e + (rng.standard_normal(w) * p["col"] * level)[None, :] + (rng.standard_normal(h) * p["row"] * level)[:, None]
    n_bad = int(p["bad"] * h * w)
    if n_bad:
        idx = rng.choice(h * w, n_bad, replace=False)
        e.flat[idx] = rng.choice([0.0, p["full_well"]], n_bad)
    e = np.clip(e, 0, p["full_well"])
    med = float(np.median(e)) + 1e-6
    gain = float(np.clip(p["ae_target"] * p["full_well"] / med, 1.0, cfg["max_gain"]))   # then gain up to the target
    top = 2 ** int(cfg["bits"]) - 1
    dn = np.clip(np.round(e * gain / p["full_well"] * top), 0, top).astype(np.float32)
    lo, hi = np.percentile(dn, [p["agc_lo"], p["agc_hi"]])
    out = np.clip((dn - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    return out.astype(np.float32), dict(p, gain=gain, agc_range=[float(lo), float(hi)])


def nominal_sigma(L_level: float, exposure_ms: float, cfg: dict) -> float:
    n = cfg["nominal"]
    k = n["electrons_per_unit_ms"] * exposure_ms
    dark = n["dark_current_e_per_ms"] * exposure_ms
    k *= _light_factor(max(L_level, 0.0) * k, dark, n["full_well_e"], n["ae_target"])
    var = max(L_level, 0.0) * k + dark + n["read_noise_e"] ** 2
    return float(np.sqrt(var) / k)

"""RGB sensor model, applied in the data loader (spec 3.1). Pure NumPy.

Linear render radiance -> photo-electrons (radiance x exposure x conversion) -> heteroscedastic
Gaussian shot + read noise -> auto-exposure gain -> clipping and quantisation -> gamma.

This is the Brooks et al. (CVPR 2019) noise model: y ~ N(x, lambda_read + lambda_shot x) on linear
raw values x in [0, 1], with lambda_shot = gain / full_well and lambda_read = (gain sigma_read /
full_well)^2. Deriving both from a light level instead of drawing them independently makes night
and dusk frames noisier in a physically consistent way. Mosaicking / demosaicking is not modelled.
"""
from __future__ import annotations

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])


def luminance(img):
    img = np.asarray(img, np.float32)
    return img @ LUMA.astype(np.float32) if img.ndim == 3 and img.shape[-1] == 3 else img


def sample_params(rng, cfg: dict) -> dict:
    u = lambda k: float(rng.uniform(*cfg[k]))  # noqa: E731
    return dict(e_per_unit_ms=u("electrons_per_unit_ms"), read_e=u("read_noise_e"), full_well=u("full_well_e"),
                ae_target=u("ae_target"), wb=rng.uniform(0.96, 1.04, 3).tolist())


def auto_exposure(median_e: float, full_well: float, ae_target: float, max_gain: float) -> tuple[float, float]:
    """The exposure time is fixed by the scene (it also sets the rendered motion blur), so the camera
    meets its target level with the aperture / ND (light reduction, never more light than wide
    open) and, when that is not enough, with gain. Returns (light factor <= 1, gain >= 1)."""
    target = ae_target * full_well
    if median_e > target:
        return target / median_e, 1.0
    return 1.0, float(min(target / max(median_e, 1e-6), max_gain))


def apply(L: np.ndarray, rng, cfg: dict, exposure_ms: float, params: dict | None = None) -> tuple[np.ndarray, dict]:
    """L: HxWx3 linear radiance (render units). Returns display-referred HxWx3 in [0, 1]."""
    p = params or sample_params(rng, cfg)
    e_open = np.clip(np.asarray(L, np.float32), 0, None) * (p["e_per_unit_ms"] * exposure_ms)
    e_open = e_open * np.asarray(p["wb"], np.float32)
    light, gain = auto_exposure(float(np.median(e_open.mean(-1))), p["full_well"], p["ae_target"], cfg["max_gain"])
    e_mean = e_open * light
    noisy = e_mean + np.sqrt(e_mean + p["read_e"] ** 2) * rng.standard_normal(e_mean.shape).astype(np.float32)
    noisy = np.minimum(noisy, p["full_well"])                           # pixel saturation
    raw = np.clip(noisy * gain / p["full_well"], 0.0, 1.0)
    q = 2 ** int(cfg["bits"]) - 1
    raw = np.round(raw * q) / q
    out = raw ** (1.0 / cfg["gamma"])
    return out.astype(np.float32), dict(p, gain=gain, light=light, lambda_shot=gain / p["full_well"],
                                        lambda_read=(gain * p["read_e"] / p["full_well"]) ** 2)


def nominal_sigma(L_level: float, exposure_ms: float, cfg: dict) -> float:
    """Noise sigma in radiance units at radiance L_level with the nominal sensor (for the CNR),
    with the same aperture / ND logic as apply() evaluated at that level."""
    n = cfg["nominal"]
    k = n["electrons_per_unit_ms"] * exposure_ms
    light, _ = auto_exposure(max(L_level, 0.0) * k, n["full_well_e"], n["ae_target"], cfg["max_gain"])
    k *= light
    return float(np.sqrt(max(L_level, 0.0) * k + n["read_noise_e"] ** 2) / k)

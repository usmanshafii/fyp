"""Point-spread function and supersample integration (spec 4 step 1). Pure NumPy.

Render at s x the output size with a 0.01 px film filter (each supersample is a point sample),
convolve with a Gaussian PSF whose sigma is given in OUTPUT pixels (sigma * s supersamples),
then box-average s x s blocks to the 640x512 grid. The box average is the pixel's area
integration; the Gaussian is the optics. Doing the blur before decimation keeps a target's
sub-pixel position, which matters most in the 2-8 px bins.

The kernel is Lindeberg's discrete Gaussian (e^-t I_n(t), t = sigma^2). Its variance is exactly
sigma^2 at any sigma, unlike a sampled Gaussian, which collapses to a delta below ~0.5 px - the
loader's extra blur at output resolution is often that small.
"""
from __future__ import annotations

import math

import numpy as np


def _bessel_i_scaled(n: int, t: float, terms: int = 60) -> float:
    """e^-t * I_n(t) by its power series (t <= ~50 is plenty here)."""
    if t == 0.0:
        return 1.0 if n == 0 else 0.0
    s, term = 0.0, (t / 2) ** n / math.factorial(n)
    for k in range(terms):
        s += term
        term *= (t / 2) ** 2 / ((k + 1) * (k + 1 + n))
        if term < 1e-17 * s:
            break
    return s * math.exp(-t)


def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """1-D discrete Gaussian with variance sigma^2, normalised to sum 1."""
    if sigma <= 1e-6:
        return np.array([1.0])
    t = sigma * sigma
    r = int(math.ceil(truncate * sigma)) + 2           # e^-t I_n(t) has heavier tails than a sampled Gaussian
    if t > 40.0:            # series loses precision; a sampled Gaussian is exact enough here
        x = np.arange(-r, r + 1)
        k = np.exp(-0.5 * x * x / t)
    else:
        k = np.array([_bessel_i_scaled(abs(n), t) for n in range(-r, r + 1)])
    return k / k.sum()


def _conv1d(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    r = len(k) // 2
    if r == 0:
        return img
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="reflect" if img.shape[axis] > r else "edge")
    n = img.shape[axis]
    out = np.zeros_like(img, dtype=np.float64 if img.dtype == np.float64 else np.float32)
    for i, w in enumerate(k):
        sl = [slice(None)] * img.ndim
        sl[axis] = slice(i, i + n)
        out += w * p[tuple(sl)]
    return out


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable blur of an HxW or HxWxC array (sigma in the array's own pixels)."""
    if sigma <= 1e-6:
        return np.asarray(img, np.float32)
    k = gaussian_kernel(sigma)
    a = np.asarray(img, np.float32)
    return _conv1d(_conv1d(a, k, 0), k, 1)


def box_downsample(img: np.ndarray, s: int) -> np.ndarray:
    """Average s x s blocks. Height and width must be multiples of s."""
    if s == 1:
        return np.asarray(img, np.float32)
    h, w = img.shape[:2]
    if h % s or w % s:
        raise ValueError(f"shape {img.shape[:2]} is not a multiple of {s}")
    a = np.asarray(img, np.float32)
    return a.reshape(h // s, s, w // s, s, *a.shape[2:]).mean(axis=(1, 3))


def psf_integrate(img_ss: np.ndarray, sigma_out: float, s: int) -> np.ndarray:
    """Supersampled render -> output pixels: Gaussian PSF (sigma in output px), then box average."""
    return box_downsample(gaussian_blur(img_ss, sigma_out * s), s)


def extra_sigma(sigma_total: float, sigma_applied: float) -> float:
    """Gaussian sigma still to apply so that sigma_applied (+) extra = sigma_total."""
    return math.sqrt(max(0.0, sigma_total ** 2 - sigma_applied ** 2))

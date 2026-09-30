"""In-band blackbody radiance lookup tables (spec 4, LWIR recipe).

    L(T) = integral over the band of 2hc^2 / lambda^5 / (exp(hc / (lambda k T)) - 1) d lambda   [W m^-2 sr^-1]

integrated once with NumPy (1,000 wavelength steps, Simpson's rule) on a 0.01 K grid, then
interpolated. Emissivity is applied by the caller: L_surface = emissivity * L(T).

    python thermal/planck_lut.py          # prints a few values and writes the cached tables
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

H = 6.62607015e-34        # J s
C = 2.99792458e8          # m / s
K = 1.380649e-23          # J / K
C1 = 2 * H * C ** 2       # W m^2 / sr
C2 = H * C / K            # m K

LWIR_BAND_UM = (8.0, 14.0)
SWIR_BAND_UM = (0.9, 1.7)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def planck_spectral(lam_m, T):
    """Spectral radiance B_lambda(T) in W m^-2 sr^-1 m^-1 (broadcasts)."""
    lam_m = np.asarray(lam_m, float)
    T = np.asarray(T, float)
    x = C2 / (lam_m * T)
    with np.errstate(over="ignore"):
        return C1 / lam_m ** 5 / np.expm1(x)


def band_radiance_direct(T, band_um=LWIR_BAND_UM, steps: int = 1000) -> np.ndarray:
    """Direct numerical integration (Simpson, `steps` intervals). Vectorised over T."""
    steps += steps % 2
    lam = np.linspace(band_um[0], band_um[1], steps + 1) * 1e-6
    T = np.atleast_1d(np.asarray(T, float))
    w = np.ones(steps + 1)
    w[1:-1:2], w[2:-1:2] = 4, 2
    dl = (lam[-1] - lam[0]) / steps
    out = np.empty(len(T))
    for i0 in range(0, len(T), 4096):                  # chunked: 4096 x 1001 doubles at a time
        B = planck_spectral(lam[None, :], T[i0:i0 + 4096, None])
        out[i0:i0 + 4096] = (B * w).sum(1) * dl / 3
    return out


class BandLUT:
    """L(T) and its inverse on a uniform temperature grid."""

    def __init__(self, band_um=LWIR_BAND_UM, t_min=200.0, t_max=800.0, dt=0.01, steps=1000, cache=True):
        self.band_um, self.t_min, self.t_max, self.dt = tuple(band_um), t_min, t_max, dt
        n = int(round((t_max - t_min) / dt)) + 1
        self.T = t_min + dt * np.arange(n)
        path = os.path.join(CACHE, f"planck_{band_um[0]:g}-{band_um[1]:g}um_{t_min:g}-{t_max:g}K_{dt:g}.npy")
        L = None
        if cache and os.path.exists(path):
            L = np.load(path)
            if L.shape != self.T.shape:
                L = None
        if L is None:
            L = band_radiance_direct(self.T, band_um, steps)
            if cache:
                try:
                    os.makedirs(CACHE, exist_ok=True)
                    np.save(path, L)
                except OSError:
                    pass
        self.L = L

    def radiance(self, T_kelvin):
        """In-band radiance [W m^-2 sr^-1] of a blackbody at T (linear interpolation, 0.01 K grid)."""
        T = np.asarray(T_kelvin, float)
        if np.any(T < self.t_min) or np.any(T > self.t_max):
            raise ValueError(f"temperature outside the table {self.t_min}-{self.t_max} K")
        return np.interp(T, self.T, self.L)

    def temperature(self, L):
        """Brightness temperature [K] for an in-band radiance (inverse table)."""
        return np.interp(np.asarray(L, float), self.L, self.T)

    def dL_dT(self, T_kelvin):
        """Radiance change per kelvin, used to convert NETD to radiance noise."""
        T = np.asarray(T_kelvin, float)
        return (self.radiance(np.minimum(T + 0.05, self.t_max)) - self.radiance(np.maximum(T - 0.05, self.t_min))) / 0.1


@lru_cache(maxsize=None)
def lwir_lut() -> BandLUT:
    return BandLUT(LWIR_BAND_UM, 200.0, 800.0, 0.01)


@lru_cache(maxsize=None)
def swir_lut() -> BandLUT:
    return BandLUT(SWIR_BAND_UM, 200.0, 1200.0, 0.05)


def celsius(tc):
    return np.asarray(tc, float) + 273.15


if __name__ == "__main__":
    lut = lwir_lut()
    for t in (250.0, 273.15, 300.0, 350.0, 500.0, 750.0):
        print(f"LWIR 8-14 um  T={t:7.2f} K  L={float(lut.radiance(t)):9.3f} W/m2/sr  dL/dT={float(lut.dL_dT(t)):.4f}")
    s = swir_lut()
    for t in (300.0, 523.15, 700.0, 900.0):
        print(f"SWIR 0.9-1.7 um T={t:7.2f} K  L={float(s.radiance(t)):.3e} W/m2/sr")

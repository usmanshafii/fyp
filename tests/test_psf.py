"""PSF and supersample integration (spec 4 step 1)."""
import numpy as np
import pytest

from render import psf


@pytest.mark.parametrize("sigma", [0.15, 0.3, 0.6, 1.3, 3.6])
def test_kernel_normalised_with_exact_variance(sigma):
    k = psf.gaussian_kernel(sigma)
    x = np.arange(len(k)) - len(k) // 2
    assert k.sum() == pytest.approx(1.0)
    assert (k * x * x).sum() == pytest.approx(sigma ** 2, rel=1e-3)


def test_integration_conserves_energy_and_subpixel_position():
    s = 4
    img = np.zeros((64 * s, 64 * s), np.float32)
    img[130, 141] = 16.0                                   # point source at output (35.375, 32.625)
    out = psf.psf_integrate(img, 0.5, s)
    assert out.shape == (64, 64)
    assert out.sum() * s * s == pytest.approx(16.0, rel=1e-4)
    yy, xx = np.mgrid[:64, :64]
    cx, cy = (out * (xx + 0.5)).sum() / out.sum(), (out * (yy + 0.5)).sum() / out.sum()
    assert cx == pytest.approx((141 + 0.5) / s, abs=0.02) and cy == pytest.approx((130 + 0.5) / s, abs=0.02)


def test_extra_sigma_composes():
    assert psf.extra_sigma(0.5, 0.3) == pytest.approx(0.4)
    assert psf.extra_sigma(0.3, 0.3) == 0.0

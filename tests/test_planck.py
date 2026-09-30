"""In-band Planck lookup tables (spec 4)."""
import numpy as np
import pytest

from thermal.planck_lut import band_radiance_direct, lwir_lut, swir_lut

SIGMA = 5.670374419e-8


def test_lut_matches_direct_integration():
    lut = lwir_lut()
    T = np.array([213.37, 255.0, 288.15, 301.234, 420.0, 777.7])
    assert np.allclose(lut.radiance(T), band_radiance_direct(T), rtol=2e-6)


def test_full_spectrum_equals_stefan_boltzmann():
    T = 300.0
    total = band_radiance_direct(np.array([T]), (0.2, 1000.0), 200000)[0]
    assert total == pytest.approx(SIGMA * T ** 4 / np.pi, rel=1e-4)


def test_known_lwir_value_and_monotonic():
    lut = lwir_lut()
    assert float(lut.radiance(300.0)) == pytest.approx(54.93, abs=0.05)   # 8-14 um, 300 K
    assert np.all(np.diff(lut.L) > 0)
    T = np.array([250.0, 300.0, 500.0])
    assert np.allclose(lut.temperature(lut.radiance(T)), T, atol=0.01)
    assert np.all(lut.dL_dT(T) > 0)


def test_swir_self_emission_only_matters_when_hot():
    s = swir_lut()
    assert float(s.radiance(300.0)) < 1e-6 < 0.05 < float(s.radiance(523.15))
    with pytest.raises(ValueError):
        lwir_lut().radiance(150.0)

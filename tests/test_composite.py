"""Haze and layer compositing (FYP_Sky_and_Background.md)."""
import numpy as np
import pytest

from data import composite as cp


def test_contrast_retention_example_from_sky_doc():
    k = cp.extinction(10.0)                                    # V = 10 km
    assert float(cp.transmittance(3700.0, k)) == pytest.approx(0.235, abs=0.002)   # "only 24%"


def test_scale_height_only_affects_slant_paths():
    k = cp.extinction(10.0)
    assert float(cp.optical_depth(3000.0, k, 0.0, 1500.0)) == pytest.approx(k * 3000.0)
    assert float(cp.optical_depth(3000.0, k, 30.0, 1500.0)) < k * 3000.0
    assert float(cp.optical_depth(3000.0, k, 30.0, None)) == pytest.approx(k * 3000.0)


def _layer(H=40, W=50, C=3):
    sky = np.linspace(2.0, 4.0, H)[:, None, None] * np.ones((H, W, C), np.float32)
    return dict(sky=sky.astype(np.float32))


def test_object_contrast_against_sky_scales_by_transmittance():
    layer = _layer()
    A = np.zeros((8, 8), np.float32)
    A[3:5, 3:5] = 1.0
    P = A[..., None] * np.array([0.5, 0.5, 0.5], np.float32)          # dark object
    obj = dict(name="t", crop=[20, 10, 28, 18], P=P, A=A, range_m=2500.0, elevation_deg=0.0)
    clear = cp.composite(layer, [obj], 0.0)
    k = cp.extinction(8.0)
    hazy = cp.composite(layer, [obj], k)
    t = float(cp.transmittance(2500.0, k))
    d_clear = clear - layer["sky"]
    d_hazy = hazy - layer["sky"]
    assert np.allclose(d_hazy, t * d_clear, atol=1e-5)


def test_ground_fades_to_horizon():
    H, W = 30, 40
    layer = _layer(H, W)
    layer["sky"][15:] = 0.0
    layer["ground"] = np.zeros((H, W, 3), np.float32)
    layer["ground"][15:] = 1.0
    layer["galpha"] = np.zeros((H, W), np.float32)
    layer["galpha"][15:] = 1.0
    layer["gdepth"] = np.zeros((H, W), np.float32)
    layer["gdepth"][15:] = 50000.0
    layer["horizon"] = np.full((W, 3), 3.0, np.float32)
    out = cp.composite(layer, [], cp.extinction(5.0))
    assert np.allclose(out[20], 3.0, atol=1e-3)                    # far ground -> horizon radiance
    assert np.allclose(out[:15], layer["sky"][:15])                # sky untouched


def test_save_load_roundtrip(tmp_path):
    layer = _layer()
    layer["sky"] *= 1e-5                                          # night-level radiance keeps precision
    layer["psf_sigma0"] = 0.3
    A = np.random.default_rng(0).random((6, 7)).astype(np.float32)
    obj = dict(name="neg0", crop=[1, 2, 8, 8], P=A[..., None] * np.ones(3, np.float32), A=A, range_m=123.0, elevation_deg=4.0)
    p = str(tmp_path / "l.npz")
    cp.save_layers(p, layer, [obj])
    l2, o2 = cp.load_layers(p)
    assert np.allclose(l2["sky"], layer["sky"], rtol=2e-3)
    assert o2[0]["crop"] == [1, 2, 8, 8] and o2[0]["range_m"] == 123.0
    assert np.allclose(o2[0]["A"], A, rtol=2e-3, atol=1e-3)

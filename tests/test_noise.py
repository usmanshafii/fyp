"""Sensor models applied in the loader."""
import numpy as np

from noise import lwir_fpn, rgb_noise, swir_noise
from render.sampling import load_configs
from thermal.planck_lut import lwir_lut

S = load_configs()["sensors"]


def test_rgb_noise_range_and_low_light_penalty():
    rng = np.random.default_rng(0)
    day = np.full((64, 64, 3), 10.0, np.float32)
    out, p = rgb_noise.apply(day, rng, S["rgb_noise"], 3.0)
    assert out.shape == day.shape and 0 <= out.min() and out.max() <= 1
    night = day * 1e-5
    out_n, pn = rgb_noise.apply(night, np.random.default_rng(0), S["rgb_noise"], 40.0, params=p)
    assert out_n.std() / out_n.mean() > 3 * out.std() / out.mean()          # dim scenes are noisier


def test_lwir_counts_per_kelvin():
    lut = lwir_lut()
    p = dict(netd_k=0.0, counts_per_k=50.0, offset=5000.0, col_k=0.0, pix_k=0.0, agc_lo=1.0, agc_hi=99.0)
    Ta = 300.0
    c = lwir_fpn.to_counts(np.array([lut.radiance(Ta), lut.radiance(Ta + 1.0)]), Ta, p)
    assert abs((c[1] - c[0]) - 50.0) < 0.5 and abs(c[0] - 5000.0) < 1e-3
    img = np.full((32, 48), float(lut.radiance(Ta)), np.float32)
    img[10:12, 20:22] = lut.radiance(Ta + 80)
    out, _ = lwir_fpn.apply(img, np.random.default_rng(1), S["lwir_noise"], Ta)
    assert out.shape == img.shape and out[10:12, 20:22].mean() > 0.95


def test_swir_output_range():
    out, _ = swir_noise.apply(np.full((40, 40), 0.3, np.float32), np.random.default_rng(2), S["swir_noise"], 5.0)
    assert out.shape == (40, 40) and 0 <= out.min() <= out.max() <= 1

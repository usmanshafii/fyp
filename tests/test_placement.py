"""Camera model, projection and bin-targeted placement (spec 4 step 2, spec 5)."""
import math

import numpy as np
import pytest

from render import placement as pl

EDGES = [(2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 256)]


def test_focal_lengths_match_spec():
    assert pl.focal_px(640, 50) == pytest.approx(686.24, abs=0.01)
    assert pl.focal_px(640, 10) == pytest.approx(3657.6, abs=0.1)
    assert pl.focal_mm(36, 50) == pytest.approx(38.6, abs=0.01)
    assert pl.focal_mm(36, 10) == pytest.approx(205.7, abs=0.05)


def test_camera_axes_and_roundtrip():
    cam = pl.Camera(640, 512, 50, (1.0, 2.0, 3.0), azimuth_deg=0, elevation_deg=0)
    assert np.allclose(cam.forward, [0, 1, 0], atol=1e-12)          # azimuth 0 = north = +Y
    cam = pl.Camera(640, 512, 10, (0, 0, 2), azimuth_deg=90, elevation_deg=20)
    assert np.allclose(cam.forward, pl.direction(90, 20), atol=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(20):
        u, v = rng.uniform(0, 640), rng.uniform(0, 512)
        P = cam.C + cam.ray(u, v) * rng.uniform(10, 5000)
        pu, pv, z = cam.project(P)
        assert pu[0] == pytest.approx(u, abs=1e-6) and pv[0] == pytest.approx(v, abs=1e-6) and z[0] > 0
    # image y grows downward: a point above the optical axis projects to a smaller row
    up = cam.C + cam.forward * 100 + cam.R[:, 1] * 5
    assert cam.project(up)[1][0] < 256


def test_object_heading_convention():
    R = pl.object_rotation(90.0)                        # nose (+X body) toward east
    assert np.allclose(R @ [1, 0, 0], [1, 0, 0], atol=1e-12)
    R = pl.object_rotation(0.0, pitch_deg=10)           # north, nose up
    nose = R @ [1, 0, 0]
    assert nose[1] > 0.98 and nose[2] == pytest.approx(math.sin(math.radians(10)))


def test_bins():
    assert pl.bin_index(2.0, EDGES) == 0 and pl.bin_index(3.999, EDGES) == 0 and pl.bin_index(4.0, EDGES) == 1
    assert pl.bin_index(300.0, EDGES) == 5 and pl.bin_index(1.9, EDGES) == -1


@pytest.mark.parametrize("hfov", [50.0, 10.0])
def test_solve_distance_hits_requested_size(hfov):
    rng = np.random.default_rng(1)
    V = rng.normal(size=(200, 3)) * [1.5, 1.1, 0.3]
    cam = pl.Camera(640, 512, hfov, (0, 0, 2), 30.0, 25.0)
    for p in (2.5, 7.0, 40.0, 120.0):
        d = cam.ray(rng.uniform(150, 490), rng.uniform(150, 360))
        dist, b = pl.solve_distance(cam, V, pl.object_rotation(rng.uniform(0, 360)), d, p)
        assert pl.box_size(b) == pytest.approx(p, rel=1e-3)


def test_place_lands_in_bin_and_frame():
    rng = np.random.default_rng(2)
    V = rng.normal(size=(300, 3)) * [1.5, 1.1, 0.3]
    cam = pl.Camera(640, 512, 10.0, (0, 0, 2), 0.0, 30.0)
    pose = lambda r: (pl.object_rotation(r.uniform(0, 360)), {})  # noqa: E731
    for k in range(6):
        p = pl.place(cam, V, rng, EDGES, k, pose)
        assert pl.bin_index(p.size_px, EDGES) == k
        assert pl.box_inside(p.bbox, 640, 512, 1.0)


def test_blur_length_example_from_spec():
    f = pl.focal_px(640, 10)
    assert pl.blur_length_px(f, 44, 0.005, 1000) == pytest.approx(0.8, abs=0.01)      # spec 3.1
    assert pl.blur_length_px(f, 44, 0.050, 1000) == pytest.approx(8.0, abs=0.1)

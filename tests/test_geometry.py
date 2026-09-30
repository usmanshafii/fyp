"""pytest tests/ -q   (pure NumPy; no Blender needed)"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "target"))
import kal_geometry as kg  # noqa: E402
import kal_measure as km  # noqa: E402

CFG = kg.load_config()


def test_defaults_valid_and_closed():
    mesh = kg.build_mesh(kg.defaults(CFG))
    assert kg.check(mesh["derived"]) == []
    assert kg.edge_report(mesh) == {}          # every part closed, consistently wound


@pytest.mark.parametrize("seed", range(30))
def test_variants_valid_and_closed(seed):
    p = kg.sample_variant(CFG, seed)
    mesh = kg.build_mesh(p)
    g = mesh["derived"]
    lo, hi = g["le_sweep_limits_deg"]
    assert lo <= g["sweep_deg"] <= hi
    assert g["fin_root_chord"] <= g["tip_chord"]
    assert kg.edge_report(mesh) == {}


def test_variants_stay_inside_ranges():
    rg = kg.ranges(CFG)
    for seed in range(50):
        p = kg.sample_variant(CFG, seed)
        for k, (a, b) in rg.items():
            assert a <= p[k] <= b, k


def test_sampling_is_deterministic():
    assert kg.sample_variant(CFG, 7) == kg.sample_variant(CFG, 7)


def test_output_frame_and_size():
    mesh = kg.build_mesh(kg.defaults(CFG))
    g = mesh["derived"]
    V = kg.to_output_frame(mesh)
    # +X forward: nose tip (plus probe) is the max-X point; span along Y; origin mid-length
    assert np.isclose(V[:, 0].max(), (g["x_mid"] - g["nose_x"] + g["probe_len"]) * g["span_m"], atol=1e-6)
    assert np.isclose(np.ptp(V[:, 1]), (1 + g["fin_thickness"]) * g["span_m"], atol=0.005)  # + fin decal offset
    # dimensions consistent with the photo evidence (spec section 2.2)
    s = kg.summary(g)
    assert 2.8 <= s["length_m (nose-hub)"] <= 3.6
    assert 0.28 <= s["forebody_dia_m"] <= 0.36
    assert 0.60 <= s["fin_height_m"] <= 0.80        # P8 rear view: fins ~0.30 of span


def test_all_slots_present():
    mesh = kg.build_mesh(kg.defaults(CFG))
    used = {kg.SLOTS[s] for s in mesh["face_slots"]}
    assert used == set(kg.SLOTS) - {"nose_window"}


def test_yamlite_matches_pyyaml():
    yaml = pytest.importorskip("yaml")
    import _yamlite
    with open(kg.DEFAULT_YAML) as f:
        assert _yamlite.loads(f.read()) == yaml.safe_load(open(kg.DEFAULT_YAML))


@pytest.mark.parametrize("seed", [None, 0, 1, 2])
def test_mesh_matches_parameters(seed):
    p = kg.defaults(CFG) if seed is None else kg.sample_variant(CFG, seed)
    assert km.validate(kg.build_mesh(p)) == []


def test_exhaust_hidden_from_level_front():
    mesh = kg.build_mesh(kg.defaults(CFG))
    front, rear, below = (km.direction(0, 0), km.direction(180, 0), km.direction(90, -45))
    vis = km.hot_visibility(mesh, np.array([front, rear, below]))
    assert vis[0] == 0.0 and vis[1] > 0 and vis[2] > 0

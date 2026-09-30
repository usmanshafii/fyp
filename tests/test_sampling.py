"""Scene sampler: determinism, balanced bins, placements inside the frame, splits."""
import json

import numpy as np
import pytest

from render import sampling

CFGS = sampling.load_configs()


@pytest.mark.parametrize("idx", [0, 1, 4, 5, 11, 17])
def test_scene_is_deterministic_and_valid(idx):
    a, _ = sampling.sample_scene(idx, CFGS, 720)
    b, objs = sampling.sample_scene(idx, CFGS, 720)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    t = a["objects"][0]
    assert t["size_bin"] == CFGS["render"]["size_bins"]["names"][idx % 6]      # bins balanced by index
    bb = t["bbox_px"]
    assert bb["x"] >= 0 and bb["y"] >= 0 and bb["x"] + bb["w"] <= 640 and bb["y"] + bb["h"] <= 512
    if a["family"] == "horizon":
        assert t["agl_m"] >= CFGS["render"]["target"]["min_agl_m"] - 1e-6
    for o in a["objects"]:
        x0, y0, x1, y1 = o["crop_px"]
        assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 512
    json.dumps(a)                                                                 # serialisable


def test_splits_are_by_scene_and_proportional():
    s = sampling.assign_splits(720, CFGS["render"]["dataset"]["splits"], 0)
    assert s.count("train") == 504 and s.count("val") == 108 and s.count("test") == 108

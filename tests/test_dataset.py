"""Writer -> layers -> loader -> COCO, end to end without Blender (synthetic layers stand in for renders)."""
import json
import os

import numpy as np
import pytest

from data import writer
from data.coco_export import export
from data.dataset import SceneDataset, gt_objects
from render import sampling

CFGS = sampling.load_configs()


def fake_layers(spec):
    H, W = spec["camera"]["height"], spec["camera"]["width"]
    out = {}
    for band, C in (("rgb", 3), ("lwir", 1), ("swir", 1)):
        base = {"rgb": 8.0, "lwir": 30.0, "swir": 0.5}[band]
        sky = (base * np.linspace(0.8, 1.2, H)[:, None, None] * np.ones((H, W, C))).astype(np.float32)
        layer = dict(sky=sky, psf_sigma0=spec["psf"][band]["sigma0"])
        objs = []
        for o in spec["objects"]:
            x0, y0, x1, y1 = o["crop_px"]
            yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
            b = o["bbox_px"]
            cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
            A = np.exp(-((xx - cx) ** 2 / (0.5 * max(b["w"], 1) ** 2)
                         + (yy - cy) ** 2 / (0.5 * max(b["h"], 1) ** 2))).astype(np.float32)
            val = {"rgb": 1.0, "lwir": 60.0, "swir": 0.2}[band]
            objs.append(dict(name=o["name"], crop=[x0, y0, x1, y1], P=A[..., None] * val * np.ones(C, np.float32), A=A,
                             range_m=o["range_m"], elevation_deg=o["elevation_deg"]))
        out[band] = (layer, objs)
    return out


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = str(tmp_path_factory.mktemp("ds"))
    for idx in range(6):
        spec, _ = sampling.sample_scene(idx, CFGS, 6)
        writer.write_scene(r, spec, fake_layers(spec), CFGS, exr=True, png=True)
    return r


def test_metadata_follows_spec_schema(root):
    m = json.load(open(os.path.join(root, "meta", "s000000.json")))
    for k in ("scene_id", "split", "seed", "camera", "target", "conditions", "modalities", "negatives"):
        assert k in m
    for k in ("variant", "yaw_deg", "range_m", "elevation_deg", "airspeed_mps", "projected_size_m", "bbox_px", "size_bin"):
        assert k in m["target"]
    assert set(m["modalities"]) == {"rgb", "lwir", "swir"} and "cnr" in m["modalities"]["lwir"]
    assert os.path.exists(os.path.join(root, m["target"]["variant"]["params_file"]))
    assert m["modalities"]["lwir"]["cnr"] > 0                      # hot synthetic blob


@pytest.mark.parametrize("train", [False, True])
def test_loader(root, train):
    ds = SceneDataset(root, None, ["rgb", "lwir", "swir"], CFGS, train=train,
                      aug=dict(hflip=0.5, scale=[0.75, 1.6], scale_prob=1.0, min_obj_px=1.5))
    for i in range(len(ds)):
        s = ds.load(i)
        assert s["images"]["rgb"].shape == (512, 640, 3) and s["images"]["lwir"].shape == (512, 640, 1)
        for v in s["images"].values():
            assert np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1
        if len(s["boxes"]):
            b = s["boxes"]
            assert (b[:, :2] >= 0).all() and (b[:, 2] <= 640).all() and (b[:, 3] <= 512).all()
            assert (np.max(b[:, 2:] - b[:, :2], 1) >= 1.0).all()


def test_eval_loader_is_deterministic(root):
    ds = SceneDataset(root, None, ["rgb", "lwir"], CFGS, train=False)
    a, b = ds.load(0), ds.load(0)
    assert np.array_equal(a["images"]["lwir"], b["images"]["lwir"])
    assert gt_objects(a["meta"])[0]["label"] == 0


def test_affine_uses_box_pixel_convention():
    from data.dataset import _affine
    yy, xx = np.mgrid[:64, :80] + 0.5                       # pixel centres in box coordinates
    img = np.exp(-((xx - 31.0) ** 2 + (yy - 21.0) ** 2) / (2 * 1.5 ** 2))[..., None].astype(np.float32)
    s, tx, ty = 1.6, -7.3, 4.1                              # blob centre (31, 21) must map to s * c + t
    out = _affine(img, np.array([[s, 0, tx], [0, s, ty]]), (64, 80))[..., 0]
    yy, xx = np.mgrid[:64, :80] + 0.5
    cx, cy = (out * xx).sum() / out.sum(), (out * yy).sum() / out.sum()
    assert cx == pytest.approx(s * 31 + tx, abs=0.02) and cy == pytest.approx(s * 21 + ty, abs=0.02)


def test_coco_export(root):
    counts = export(root)
    assert sum(n for n, _ in counts.values()) == 6
    split = next(iter(counts))
    c = json.load(open(os.path.join(root, "coco", f"{split}.json")))
    assert c["categories"][0]["name"] == "target" and all(len(a["bbox"]) == 4 for a in c["annotations"])

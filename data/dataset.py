"""PyTorch dataset over the rendered scenes (spec 4, 6, 7).

Per sample and band: load the layers -> composite with haze (nominal visibility, or a fresh draw in
training) -> the rest of the PSF (sigma drawn in the spec range; base sigma0 was applied at render
time) -> registration jitter for LWIR / SWIR against RGB (training) -> sensor noise model -> [0, 1].
Then one spatial augmentation shared by all bands (training): horizontal flip and a random scale
whose lower end is capped so the smallest object never drops below `min_obj_px` (spec 7), with
scale-up allowed. Evaluation is deterministic per scene: the noise generator is seeded from the
scene index, so every model sees identical test images.

Boxes are COCO-style in metadata and returned as xyxy in output pixels.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

from data import composite as cp
from noise import lwir_fpn, rgb_noise, swir_noise
from render import psf

CLASSES = ["target", "bird", "kite", "light_aircraft", "helicopter", "small_uav", "warm_clutter"]
BAND_CHANNELS = {"rgb": 3, "lwir": 1, "swir": 1}


def load_metas(root: str, split: str | None = None) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(root, "meta", "*.json"))):
        with open(p) as f:
            m = json.load(f)
        if split is None or m["split"] == split:
            out.append(m)
    return out


def gt_objects(meta: dict, use_blur_box: bool = False) -> list[dict]:
    """Target first, then the hard negatives, with everything the evaluation bins by."""
    objs = meta["extra"]["objects"]
    out = []
    for o in objs:
        b = o["bbox_blur_px"] if use_blur_box else o["bbox_px"]
        out.append(dict(box=[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], label=CLASSES.index(o["cls"]),
                        size_px=o["size_px"], size_bin=o["size_bin"], range_m=o["range_m"], name=o["name"]))
    return out


def _affine(img: np.ndarray, M: np.ndarray, out_hw) -> np.ndarray:
    """out(p) = img(M^-1 p) with bilinear interpolation. M (2x3) maps input to output in continuous
    pixel coordinates (pixel i spans [i, i+1), the convention of the boxes); SciPy indexes pixel
    centres, so the translation is shifted by A h - h with h = (0.5, 0.5)."""
    from scipy.ndimage import affine_transform
    A = np.vstack([M, [0, 0, 1]]).astype(float)
    h = np.array([0.5, 0.5])
    A[:2, 2] = A[:2, 2] + A[:2, :2] @ h - h
    Ai = np.linalg.inv(A)
    # scipy works in (row, col) = (y, x)
    R = np.array([[Ai[1, 1], Ai[1, 0]], [Ai[0, 1], Ai[0, 0]]])
    off = np.array([Ai[1, 2], Ai[0, 2]])
    chans = [affine_transform(img[..., c], R, off, output_shape=out_hw, order=1, mode="nearest") for c in range(img.shape[-1])]
    return np.stack(chans, -1)


class SceneDataset:
    def __init__(self, root: str, split: str | None, bands, cfgs: dict, train: bool = False, seed: int = 0,
                 aug: dict | None = None, use_blur_box: bool = False, metas: list | None = None, cache_layers: int = 0):
        self.root, self.bands, self.cfgs = root, list(bands), cfgs
        self.train, self.seed, self.epoch = train, seed, 0
        self.aug = aug or {}
        self.use_blur_box = use_blur_box
        self.metas = metas if metas is not None else load_metas(root, split)
        self.W, self.H = cfgs["sensors"]["image"]["width"], cfgs["sensors"]["image"]["height"]
        self.cache_max = int(cache_layers)          # decoded layer sets kept per worker (0 = off; ~5-10 MB each)
        self._cache = {}

    def _layers(self, path: str):
        if self.cache_max <= 0:
            return cp.load_layers(path)
        if path not in self._cache:
            if len(self._cache) >= self.cache_max:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = cp.load_layers(path)
        return self._cache[path]

    def __len__(self):
        return len(self.metas)

    def set_epoch(self, e: int):
        self.epoch = e

    # ------------------------------------------------------------------ conditions
    def _weather(self, meta, rng):
        w = meta["extra"]["weather"]
        wc = self.cfgs["render"]["weather"]
        if self.train and rng.uniform() < wc.get("resample_in_training", 0.0):
            keys = list(wc["probs"])
            p = np.array([wc["probs"][k] for k in keys], float)
            kind = keys[int(rng.choice(len(keys), p=p / p.sum()))]
            lo, hi = wc["visibility_km"][kind]
            vis = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            ratio = {b: (float(rng.uniform(*v)) if isinstance(v, list) else float(v)) for b, v in wc["band_ratio"][kind].items()}
            return vis, ratio
        return w["visibility_km"], w["band_ratio"]

    def band_image(self, meta: dict, band: str, rng, vis_km: float, ratio: dict, reg: bool) -> np.ndarray:
        S = self.cfgs["sensors"]
        layer, objects = self._layers(os.path.join(self.root, meta["modalities"][band]["layers"]))
        k = cp.extinction(vis_km, ratio[band])
        img = cp.composite(layer, objects, k, meta["extra"]["weather"].get("aerosol_scale_height_m"))
        s0 = layer["psf_sigma0"]
        sig = float(rng.uniform(*S["psf_sigma_px"][band])) if self.train else meta["extra"]["psf"][band]["nominal"]
        img = psf.gaussian_blur(img, psf.extra_sigma(sig, s0))
        if reg and band in S["registration"]:
            r = S["registration"][band]
            th = math.radians(rng.uniform(-r["rot_deg"], r["rot_deg"]))
            rad, ang = rng.uniform(0, r["shift_px"]), rng.uniform(0, 2 * math.pi)
            cx, cy = self.W / 2, self.H / 2
            c, s = math.cos(th), math.sin(th)
            M = np.array([[c, -s, cx - c * cx + s * cy + rad * math.cos(ang)],
                          [s, c, cy - s * cx - c * cy + rad * math.sin(ang)]])
            img = _affine(img, M, (self.H, self.W))
        exp_ms = meta["camera"]["exposure_ms"]
        if band == "rgb":
            out, _ = rgb_noise.apply(img, rng, S["rgb_noise"], exp_ms)
        elif band == "lwir":
            out, _ = lwir_fpn.apply(img[..., 0], rng, S["lwir_noise"], meta["conditions"]["air_temp_c"] + 273.15)
            out = out[..., None]
        else:
            out, _ = swir_noise.apply(img[..., 0], rng, S["swir_noise"], exp_ms)
            out = out[..., None]
        return out.astype(np.float32)

    # ------------------------------------------------------------------ sample
    def _rng(self, i, band_salt=0):
        idx = int(self.metas[i]["scene_id"][1:])
        if self.train:
            return np.random.default_rng([self.seed, self.epoch, idx, band_salt])
        return np.random.default_rng([self.seed + 7919, idx, band_salt])

    def load(self, i: int) -> dict:
        meta = self.metas[i]
        rng = self._rng(i)
        vis, ratio = self._weather(meta, rng)
        # registration jitter models LWIR / SWIR misalignment against RGB, the label reference; a model
        # without RGB (lwir_only) sees its own band in label coordinates, so it gets no jitter
        reg = self.train and "rgb" in self.bands
        # noise stream salted by band NAME, not list position: an LWIR image is identical whether a
        # model loads [lwir] or [rgb, lwir] (identical test scenes; consistent contrast buckets)
        salt = {"rgb": 1, "lwir": 2, "swir": 3}
        imgs = {b: self.band_image(meta, b, self._rng(i, salt[b]), vis, ratio, reg=reg and b != "rgb")
                for b in self.bands}
        objs = gt_objects(meta, self.use_blur_box)
        boxes = np.array([o["box"] for o in objs], np.float32).reshape(-1, 4)
        labels = np.array([o["label"] for o in objs], np.int64)
        if self.train:
            imgs, boxes, labels = self._augment(imgs, boxes, labels, rng)
        return dict(images=imgs, boxes=boxes, labels=labels, scene_id=meta["scene_id"], meta=meta, objects=objs,
                    visibility_km=vis)

    def _augment(self, imgs, boxes, labels, rng):
        a = self.aug
        if rng.uniform() < a.get("hflip", 0.5):
            imgs = {b: v[:, ::-1].copy() for b, v in imgs.items()}
            if len(boxes):
                boxes = np.c_[self.W - boxes[:, 2], boxes[:, 1], self.W - boxes[:, 0], boxes[:, 3]].astype(np.float32)
        lo, hi = a.get("scale", [1.0, 1.0])
        if len(boxes):
            smallest = float(np.max(boxes[:, 2:] - boxes[:, :2], axis=1).min())
            lo = max(lo, a.get("min_obj_px", 1.5) / max(smallest, 1e-3))      # spec 7: never below ~1 px
        if hi > lo and rng.uniform() < a.get("scale_prob", 0.5):
            s = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            if len(boxes):                  # keep one object (random) in view after scaling
                j = int(rng.integers(len(boxes)))
                px, py = (boxes[j, 0] + boxes[j, 2]) / 2, (boxes[j, 1] + boxes[j, 3]) / 2
            else:
                px, py = self.W / 2, self.H / 2

            def offset(size, p, m=8.0):
                # out = s * in + t. Zoom in: the frame stays covered (t in [size - s size, 0]) and the
                # chosen point stays m px inside. Zoom out: the image stays inside the frame.
                if s >= 1:
                    a0, a1 = max(size - s * size, m - s * p), min(0.0, size - m - s * p)
                else:
                    a0, a1 = 0.0, size - s * size
                return float(rng.uniform(a0, a1)) if a1 > a0 else float(np.clip(size / 2 - s * p, min(a0, a1), max(a0, a1)))
            tx, ty = offset(self.W, px), offset(self.H, py)
            M = np.array([[s, 0, tx], [0, s, ty]])
            imgs = {b: _affine(v, M, (self.H, self.W)) for b, v in imgs.items()}
            if len(boxes):
                boxes = boxes * s + np.array([tx, ty, tx, ty], np.float32)
                cb = np.c_[np.clip(boxes[:, [0, 2]], 0, self.W), np.clip(boxes[:, [1, 3]], 0, self.H)][:, [0, 2, 1, 3]]
                keep = ((cb[:, 2] - cb[:, 0]) * (cb[:, 3] - cb[:, 1])) > 0.5 * ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
                boxes, labels = cb[keep].astype(np.float32), labels[keep]
        return imgs, boxes, labels


# ============================================================================ torch glue
class TorchSceneDataset:
    """Map-style torch dataset around SceneDataset. Module-level so DataLoader workers can pickle it
    (Windows and macOS start workers by spawning)."""

    def __init__(self, ds: SceneDataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        import torch
        s = self.ds.load(i)
        imgs = {b: torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1))) for b, v in s["images"].items()}
        return imgs, dict(boxes=torch.from_numpy(s["boxes"]), labels=torch.from_numpy(s["labels"]),
                          scene_id=s["scene_id"], index=i)


def torch_dataset(ds: SceneDataset) -> TorchSceneDataset:
    return TorchSceneDataset(ds)


def collate(batch):
    import torch
    bands = batch[0][0].keys()
    imgs = {b: torch.stack([x[0][b] for x in batch]) for b in bands}
    return imgs, [x[1] for x in batch]

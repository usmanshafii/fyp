"""Per-scene outputs (spec 6). Pure NumPy, so it runs inside Blender right after the renders.

    <out>/layers/<band>/<scene>.npz    layers for the loader (data/composite.py)
    <out>/<band>/<scene>.exr           linear float preview at the scene's nominal visibility and PSF, no noise
    <out>/<band>/<scene>.png           8-bit tone-mapped preview of the same image
    <out>/meta/<scene>.json            metadata in the spec 6 schema (+ "extra" with everything else)
    <out>/params/<scene>.json          the full sample_variant() dict, so the target can be rebuilt exactly

Contrast-to-noise (spec 4 step 3): (target mean - local background) / noise sigma, per band, on the
noise-free preview, with the nominal sensor of configs/sensors.yaml. Target mean is taken over the
ground-truth box (rounded outward), background over a ring around it that excludes other objects.
Signed: negative means the target is darker than its surroundings.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from data import composite as cp
from data.imgio import write_exr, write_png
from noise import lwir_fpn, rgb_noise, swir_noise
from render import psf

BANDS = ("rgb", "lwir", "swir")


def nominal_image(layer: dict, objects: list, spec: dict, band: str) -> np.ndarray:
    w = spec["weather"]
    k = cp.extinction(w["visibility_km"], w["band_ratio"][band])
    img = cp.composite(layer, objects, k, w.get("aerosol_scale_height_m"))
    extra = psf.extra_sigma(spec["psf"][band]["nominal"], spec["psf"][band]["sigma0"])
    return psf.gaussian_blur(img, extra)


def noise_sigma(band: str, level: float, spec: dict, sensors: dict) -> float:
    if band == "rgb":
        return rgb_noise.nominal_sigma(level, spec["camera"]["exposure_ms"], sensors["rgb_noise"])
    if band == "lwir":
        return lwir_fpn.nominal_sigma(spec["air_temp_c"] + 273.15, sensors["lwir_noise"])
    return swir_noise.nominal_sigma(level, spec["camera"]["exposure_ms"], sensors["swir_noise"])


def _ibox(b, W, H, pad=0):
    x0, y0 = max(0, int(math.floor(b["x"])) - pad), max(0, int(math.floor(b["y"])) - pad)
    x1 = min(W, int(math.ceil(b["x"] + b["w"])) + pad)
    y1 = min(H, int(math.ceil(b["y"] + b["h"])) + pad)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def contrast_to_noise(img: np.ndarray, box: dict, others: list, band: str, spec: dict, sensors: dict) -> dict:
    Y = rgb_noise.luminance(img) if img.ndim == 3 and img.shape[-1] == 3 else img.reshape(img.shape[:2])
    H, W = Y.shape
    x0, y0, x1, y1 = _ibox(box, W, H)
    r = max(3, int(math.ceil(max(box["w"], box["h"]))))
    X0, Y0, X1, Y1 = _ibox(box, W, H, r)
    ring = np.zeros((H, W), bool)
    ring[Y0:Y1, X0:X1] = True
    ring[y0:y1, x0:x1] = False
    for o in others:
        a0, b0, a1, b1 = _ibox(o, W, H, 2)
        ring[b0:b1, a0:a1] = False
    t = float(Y[y0:y1, x0:x1].mean())
    b = float(Y[ring].mean()) if ring.any() else float(Y.mean())
    s = noise_sigma(band, b, spec, sensors)
    return dict(cnr=(t - b) / s, target_mean=t, background_mean=b, noise_sigma=s)


def _srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def tonemap(img: np.ndarray, band: str) -> np.ndarray:
    if band == "rgb":
        s = 0.18 / max(float(np.percentile(rgb_noise.luminance(img), 50)), 1e-12)
        return (_srgb(img * s) * 255 + 0.5).astype(np.uint8)
    a = img.reshape(img.shape[:2])
    lo, hi = np.percentile(a, [0.5, 99.8])
    return (np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1) * 255 + 0.5).astype(np.uint8)


def write_scene(root: str, spec: dict, layers: dict, cfgs: dict, calib: dict | None = None,
                timings: dict | None = None, exr: bool = True, png: bool = True) -> dict:
    sid = spec["scene_id"]
    objs = spec["objects"]
    target = objs[0]
    modalities = {}
    for band, (layer, objects) in layers.items():
        for d in (f"layers/{band}", band):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        lpath = f"layers/{band}/{sid}.npz"
        cp.save_layers(os.path.join(root, lpath), layer, objects, half=(band != "lwir"))
        img = nominal_image(layer, objects, spec, band)
        rel = f"{band}/{sid}.exr"
        if exr:
            chans = {"R": img[..., 0], "G": img[..., 1], "B": img[..., 2]} if img.shape[-1] == 3 else {"Y": img[..., 0]}
            write_exr(os.path.join(root, rel), chans)
        if png:
            write_png(os.path.join(root, f"{band}/{sid}.png"), tonemap(img, band))
        c = contrast_to_noise(img, target["bbox_px"], [o["bbox_px"] for o in objs[1:]], band, spec, cfgs["sensors"])
        modalities[band] = dict(path=rel, png=f"{band}/{sid}.png", layers=lpath, **c)

    os.makedirs(os.path.join(root, "params"), exist_ok=True)
    params_rel = f"params/{sid}.json"
    with open(os.path.join(root, params_rel), "w") as f:
        json.dump(spec["target_variant"]["params"], f, indent=1)
    tv = {k: v for k, v in spec["target_variant"].items() if k != "params"}
    cam = spec["camera"]
    meta = dict(
        scene_id=sid, split=spec["split"], seed=spec["seed"],
        camera=dict(hfov_deg=cam["hfov_deg"], width=cam["width"], height=cam["height"], f_px=cam["f_px"],
                    supersample=target["supersample"], supersample_background=cam["supersample_background"],
                    exposure_ms=cam["exposure_ms"]),
        target=dict(variant=dict(tv, params_file=params_rel),
                    yaw_deg=target["pose"]["heading_deg"], pitch_deg=target["pose"]["pitch_deg"],
                    roll_deg=target["pose"]["roll_deg"], range_m=target["range_m"], elevation_deg=target["elevation_deg"],
                    airspeed_mps=target["airspeed_mps"], wind_mps=target["wind_mps"],
                    projected_size_m=target["projected_size_m"], bbox_px=target["bbox_px"], size_bin=target["size_bin"]),
        conditions=dict(lighting=spec["lighting"], weather=spec["weather"]["kind"],
                        visibility_km=spec["weather"]["visibility_km"],
                        background="open_sky" if spec["family"] == "open_sky" else "horizon_clutter",
                        thermal_state=spec["thermal_state"], air_temp_c=spec["air_temp_c"],
                        warm_clutter=any(o["cls"] == "warm_clutter" for o in objs[1:]),
                        clouds=spec["clouds"] is not None),
        modalities=modalities,
        negatives=[{"class": o["cls"], "bbox_px": o["bbox_px"], "range_m": o["range_m"], "size_bin": o["size_bin"]}
                   for o in objs[1:]],
        extra=dict(objects=objs, camera=cam, sun=spec["sun"], moon=spec["moon"], atmosphere=spec["atmosphere"],
                   clouds=spec["clouds"], weather=spec["weather"], psf=spec["psf"], family=spec["family"],
                   swir_hot_emission=spec["swir_hot_emission"], signatures=spec["signatures"],
                   terrain=spec["terrain"], structures=spec["structures"], resample=spec.get("resample"),
                   ground_texture=spec.get("ground_texture"), sun_calibration=calib, timings_s=timings),
    )
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)
    with open(os.path.join(root, "meta", f"{sid}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta

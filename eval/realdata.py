"""Zero-shot real-sensor check (spec 8, Phase 7): Anti-UAV and Halmstad-style data.

These datasets contain consumer UAVs, birds and aircraft - never the KAL - so the numbers here
describe sensor-domain behaviour (does a model trained on synthetic frames fire on real RGB / IR
video at all, per apparent size, and what does it confuse with birds and aircraft?). They never
establish KAL-specific performance.

The real RGB and IR streams are not pixel-registered (different optics and resolutions), so each
stream is evaluated alone: the fused models run with the other band missing (zeroed at the
fusion, which modality dropout trains for), single-band models run on their own band.

Supported layouts
  anti_uav   <root>/<sequence>/ with {infrared,visible}.json or {IR,RGB}_label.json ("exist" and
             "gt_rect" [x, y, w, h] per frame) and the matching .mp4 or frame folder (Anti-UAV
             releases use both namings; the adapter fails loudly if neither is found).
  yolo       <root>/images/*.png|jpg + <root>/labels/*.txt ("cls cx cy w h", normalised) with a
             --class-map, e.g. "0:small_uav,1:bird,2:light_aircraft,3:helicopter" (Halmstad-style
             labels exported to YOLO format).
Frames are resized (letterboxed) to 640x512 and converted to the model's [0, 1] input.

    python eval/realdata.py --layout anti_uav --root D:/Anti-UAV/test --stream infrared \
        --runs out/runs/student out/runs/lwir_only --every 10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]

import numpy as np  # noqa: E402

from data.dataset import CLASSES  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval.bins import bin_of  # noqa: E402

W, H = 640, 512


def letterbox(img: np.ndarray):
    """Resize to fit 640x512 keeping aspect, pad with the mean. Returns image, scale, offset."""
    from PIL import Image
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    im = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    out = np.full((H, W) + im.shape[2:], im.mean(axis=(0, 1)), dtype=np.float32)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = im
    return out, s, (ox, oy)


def _frames_video(path, every):
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("reading .mp4 needs opencv-python (pip install opencv-python), "
                           "or extract frames to a folder named like the video") from e
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % every == 0:
            yield i, fr[..., ::-1] if fr.ndim == 3 else fr
        i += 1


def _frames_folder(folder, every):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    for i, f in enumerate(files):
        if i % every == 0:
            yield i, np.asarray(Image.open(f))


ANTI_UAV_NAMES = {"infrared": ("infrared", "IR"), "visible": ("visible", "RGB")}


def _first(paths):
    return next((p for p in paths if os.path.exists(p)), None)


def anti_uav_samples(root, stream, every):
    """Accepts both published layouts: {infrared,visible}.{json,mp4} and {IR,RGB}_label.json +
    {IR,RGB}.mp4 (or a frame folder of the same name)."""
    names = ANTI_UAV_NAMES[stream]
    seqs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    found = 0
    for seq in seqs:
        ann = _first([os.path.join(seq, f"{n}.json") for n in names] + [os.path.join(seq, f"{n}_label.json") for n in names])
        if ann is None:
            continue
        found += 1
        with open(ann) as f:
            a = json.load(f)
        vid = _first([os.path.join(seq, f"{n}.mp4") for n in names])
        folder = _first([os.path.join(seq, n) for n in names])
        if vid is None and folder is None:
            raise FileNotFoundError(f"{seq}: no {stream} video or frame folder")
        src = _frames_video(vid, every) if vid else _frames_folder(folder, every)
        for i, fr in src:
            if i >= len(a["exist"]):
                break
            boxes = [a["gt_rect"][i]] if a["exist"][i] and len(a["gt_rect"][i]) == 4 else []
            yield f"{os.path.basename(seq)}:{i}", fr, [(b, "small_uav") for b in boxes]
    if not found:
        raise FileNotFoundError(f"no Anti-UAV sequences with {stream} labels under {root}")


def yolo_samples(root, class_map, every):
    from PIL import Image
    imgs = sorted(glob.glob(os.path.join(root, "images", "*.*")))
    for i, p in enumerate(imgs):
        if i % every:
            continue
        im = np.asarray(Image.open(p))
        h, w = im.shape[:2]
        lab = os.path.join(root, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
        objs = []
        if os.path.exists(lab):
            for line in open(lab):
                v = line.split()
                if len(v) >= 5 and int(v[0]) in class_map:
                    cx, cy, bw, bh = (float(x) for x in v[1:5])
                    objs.append(([(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h], class_map[int(v[0])]))
        yield os.path.basename(p), im, objs


def to_band(img: np.ndarray, band: str) -> np.ndarray:
    x = img.astype(np.float32)
    x = x / (65535.0 if img.dtype == np.uint16 else 255.0)
    if band == "rgb":
        x = np.repeat(x[..., None], 3, -1) if x.ndim == 2 else x[..., :3]
    else:
        x = x.mean(-1) if x.ndim == 3 else x
        lo, hi = np.percentile(x, [1.0, 99.5])      # same display stretch as the synthetic LWIR / SWIR
        x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)[..., None]
    return x


def run(model, samples, band, cfg, device, drop_other=True):
    import torch
    recs = []
    for sid, fr, objs in samples:
        x, s, (ox, oy) = letterbox(to_band(fr, band))
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(device)
        imgs = {b: (t if b == band else None) for b in model.bands}
        if all(v is None for v in imgs.values()):
            raise ValueError(f"model bands {model.bands} do not include {band}")
        p = model.predict(imgs, cfg["conf"], cfg["nms_iou"], cfg["max_det"], cfg["nms_centre_px"])[0]
        gts = []
        for (x0, y0, bw, bh), cls in objs:
            box = [x0 * s + ox, y0 * s + oy, (x0 + bw) * s + ox, (y0 + bh) * s + oy]
            size = max(box[2] - box[0], box[3] - box[1])
            # real UAVs stand in for the target class: this measures domain transfer only
            label = 0 if cls in ("small_uav", "target") else CLASSES.index(cls)
            gts.append(dict(box=box, label=label, size_px=size, size_bin=bin_of(size) or "<2", range_m=float("nan"),
                            name=cls, cls_name=cls))
        recs.append(dict(scene_id=sid, preds={k: v.cpu().numpy() for k, v in p.items()}, gts=gts,
                         conditions={}, camera=dict(hfov_deg=float("nan"))))
    return recs


def main():
    from train.engine import device_of, load_all_configs, load_ckpt
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["anti_uav", "yolo"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--stream", default="infrared", help="anti_uav: visible | infrared")
    ap.add_argument("--band", default=None, help="model band to feed (default: lwir for infrared, rgb for visible)")
    ap.add_argument("--class-map", default="0:small_uav,1:bird,2:light_aircraft,3:helicopter")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "eval_real"))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    cfgs = load_all_configs()
    E = cfgs["train"]["eval"]
    dev = device_of(a.device)
    band = a.band or ("lwir" if a.stream == "infrared" else "rgb")
    cmap = {int(k): v for k, v in (kv.split(":") for kv in a.class_map.split(","))}
    os.makedirs(a.out, exist_ok=True)
    report = {}
    for run_ in a.runs:
        ck = run_ if run_.endswith(".pt") else os.path.join(run_, "best.pt")
        model, _ = load_ckpt(ck, dev)
        name = os.path.basename(os.path.dirname(ck))
        if band not in model.bands:
            print(f"{name}: skipped (bands {model.bands} lack {band})")
            continue
        samples = anti_uav_samples(a.root, a.stream, a.every) if a.layout == "anti_uav" else yolo_samples(a.root, cmap, a.every)
        recs = run(model, samples, band, E, dev)
        rep = dict(n_frames=len(recs), ap50=M.ap(recs, "iou", 0.5, cfg=E), ap_centre=M.ap(recs, "centre", 0.0, cfg=E),
                   op=M.operating_point(recs, E, "centre", 0.0),
                   bins={b: dict(ap_centre=M.ap(recs, "centre", 0.0, b, cfg=E),
                                 op=M.operating_point(recs, E, "centre", 0.0, bin_name=b))
                         for b in ["2-4", "4-8", "8-16", "16-32", "32-64", "64+"]})
        report[name] = rep
        print(f"{name} on {a.layout}/{a.stream} as {band}: AP50 {rep['ap50']:.3f}  AP(centre) {rep['ap_centre']:.3f}  "
              f"FPPI {rep['op']['fppi']:.3f} {rep['op']['fppi_by_class']}")
    with open(os.path.join(a.out, f"real_{a.layout}_{a.stream}.json"), "w") as f:
        json.dump(report, f, indent=1, default=lambda x: None)
    print("note: real UAVs are consumer drones; this is domain validation, not KAL performance.")


if __name__ == "__main__":
    main()

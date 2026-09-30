"""Visual QA: what the detector will see. One row per scene: RGB, LWIR and SWIR as produced by the
loader (haze, PSF, registration jitter, sensor noise), full frame with boxes plus a zoom on the target.

    python tools/preview_dataset.py --data out/dataset --split train --n 8 --out out/preview.png [--train-aug]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data.dataset import CLASSES, SceneDataset  # noqa: E402
from render.sampling import load_configs  # noqa: E402

BANDS = ["rgb", "lwir", "swir"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "out", "dataset"))
    ap.add_argument("--split", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "preview.png"))
    ap.add_argument("--train-aug", action="store_true", help="show training-time randomisation")
    a = ap.parse_args()
    cfgs = load_configs()
    ds = SceneDataset(a.data, a.split, BANDS, cfgs, train=a.train_aug,
                      aug=dict(hflip=0.5, scale=[0.75, 1.6], scale_prob=0.5, min_obj_px=1.5))
    n = min(a.n, len(ds))
    fig, axes = plt.subplots(n, 6, figsize=(24, 3.4 * n), squeeze=False)
    for r in range(n):
        s = ds.load(r)
        m = s["meta"]
        t = s["boxes"][0] if len(s["boxes"]) else np.array([320, 256, 330, 266.0])
        cx, cy = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
        half = max(12, 2.5 * max(t[2] - t[0], t[3] - t[1]))
        for k, b in enumerate(BANDS):
            img = s["images"][b]
            show = img if img.shape[-1] == 3 else img[..., 0]
            ax = axes[r, 2 * k]
            ax.imshow(show, cmap=None if img.shape[-1] == 3 else "gray", vmin=0, vmax=1)
            for box, lab in zip(s["boxes"], s["labels"]):
                ax.add_patch(mpatches.Rectangle((box[0] - 3, box[1] - 3), box[2] - box[0] + 6, box[3] - box[1] + 6,
                                                fill=False, ec="lime" if lab == 0 else "orange", lw=0.8))
            ax.set_title(f"{b}  {m['scene_id']} {m['conditions']['lighting']} {m['conditions']['background']} "
                         f"V={s['visibility_km']:.0f} km", fontsize=7)
            ax.axis("off")
            z = axes[r, 2 * k + 1]
            z.imshow(show, cmap=None if img.shape[-1] == 3 else "gray", vmin=0, vmax=1, interpolation="nearest")
            z.set_xlim(cx - half, cx + half)
            z.set_ylim(cy + half, cy - half)
            z.set_title(f"target {m['target']['size_bin']} px, {m['target']['range_m']:.0f} m, "
                        f"CNR {m['modalities'][b]['cnr']:+.1f}", fontsize=7)
            z.axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out, "| negatives:", ", ".join(CLASSES[1:]))


if __name__ == "__main__":
    main()

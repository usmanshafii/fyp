"""COCO annotations per split from the per-scene metadata (spec 6).

    python data/coco_export.py --root out/dataset          -> out/dataset/coco/{train,val,test}.json

Boxes come from projected mesh vertices (bbox_px, spec 4 step 3), COCO-style [x, y, w, h] with
(x, y) the top-left corner. category_id uses the spec's class ids (0 target ... 6 warm_clutter).
file_name points at the RGB preview; the per-band layers are listed under "fyp_modalities".
Extra per-annotation fields (size_bin, size_px, range_m) let per-bin evaluation run on COCO files.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from data.dataset import CLASSES, load_metas  # noqa: E402


def export(root: str, out_dir: str | None = None, use_blur_box: bool = False) -> dict:
    out_dir = out_dir or os.path.join(root, "coco")
    os.makedirs(out_dir, exist_ok=True)
    metas = load_metas(root)
    cats = [dict(id=i, name=n, supercategory="target" if i == 0 else "hard_negative") for i, n in enumerate(CLASSES)]
    counts = {}
    for split in sorted({m["split"] for m in metas}):
        images, anns = [], []
        for m in (x for x in metas if x["split"] == split):
            iid = int(m["scene_id"][1:])
            images.append(dict(id=iid, file_name=m["modalities"]["rgb"]["png"], width=m["camera"]["width"],
                               height=m["camera"]["height"], scene_id=m["scene_id"],
                               fyp_modalities={b: v["layers"] for b, v in m["modalities"].items()},
                               conditions=m["conditions"], hfov_deg=m["camera"]["hfov_deg"]))
            for o in m["extra"]["objects"]:
                b = o["bbox_blur_px"] if use_blur_box else o["bbox_px"]
                anns.append(dict(id=len(anns) + 1, image_id=iid, category_id=CLASSES.index(o["cls"]),
                                 bbox=[b["x"], b["y"], b["w"], b["h"]], area=b["w"] * b["h"], iscrowd=0,
                                 size_bin=o["size_bin"], size_px=o["size_px"], range_m=o["range_m"]))
        with open(os.path.join(out_dir, f"{split}.json"), "w") as f:
            json.dump(dict(images=images, annotations=anns, categories=cats,
                           info=dict(description="KAL multispectral synthetic dataset", box_source="projected mesh vertices")), f)
        counts[split] = (len(images), len(anns))
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/dataset")
    ap.add_argument("--blur-box", action="store_true", help="use the motion-blur union box instead of the mid-exposure box")
    a = ap.parse_args()
    for split, (ni, na) in export(a.root, use_blur_box=a.blur_box).items():
        print(f"{split}: {ni} images, {na} annotations")

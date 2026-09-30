"""Refresh the contrast-to-noise values in every scene's metadata from the stored layers.

    python tools/recompute_cnr.py --data out/dataset

CNR depends on configs/sensors.yaml (nominal sensor) and on the nominal visibility and PSF, none
of which needs a re-render: the layers hold everything. Run after changing the sensor models.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]

from data import composite as cp  # noqa: E402
from data import writer  # noqa: E402
from render.sampling import load_configs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "out", "dataset"))
    a = ap.parse_args()
    cfgs = load_configs()
    n = 0
    for path in sorted(glob.glob(os.path.join(a.data, "meta", "*.json"))):
        with open(path) as f:
            m = json.load(f)
        ex = m["extra"]
        spec = dict(weather=ex["weather"], psf=ex["psf"], camera=ex["camera"], air_temp_c=m["conditions"]["air_temp_c"])
        objs = ex["objects"]
        for band, mod in m["modalities"].items():
            layer, objects = cp.load_layers(os.path.join(a.data, mod["layers"]))
            img = writer.nominal_image(layer, objects, spec, band)
            mod.update(writer.contrast_to_noise(img, objs[0]["bbox_px"], [o["bbox_px"] for o in objs[1:]], band, spec,
                                                cfgs["sensors"]))
        with open(path, "w") as f:
            json.dump(m, f, indent=1)
        n += 1
        print(m["scene_id"], {b: round(v["cnr"], 2) for b, v in m["modalities"].items()})
    print(f"updated {n} scenes")


if __name__ == "__main__":
    main()

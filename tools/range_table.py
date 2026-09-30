"""Recompute the spec 5 range table from target.yaml and the variant sampler (reporting only).

    python tools/range_table.py [--variants 2000]

Range at which the target reaches each bin's lower edge, p = N W / (2 Z tan(HFOV/2)), for the
span-limited case (head-on / tail-on, span range) and the length-limited case (broadside /
planform, 5th-95th percentile of nose-to-hub length across sampled variants). Pure geometry, no
atmospheric loss. Placement always uses the actual projected box, so this is for reporting.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]

import numpy as np  # noqa: E402

import kal_geometry as kg  # noqa: E402
from eval.bins import NAMES, f_px  # noqa: E402

LOWER = [2, 4, 8, 16, 32, 64]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=2000)
    a = ap.parse_args()
    cfg = kg.load_config()
    d = kg.defaults(cfg)
    g0 = kg.derive(d)
    span_lo, span_hi = cfg["span_m"]["range"]
    span_def = cfg["span_m"]["default"]
    L = []
    for s in range(a.variants):
        g = kg.derive(kg.sample_variant(cfg, s))
        L.append(g["length"] * g["span_m"])
    L = np.array(L)
    l_lo, l_hi = np.percentile(L, [5, 95])
    l_def = g0["length"] * g0["span_m"]
    print(f"length nose-hub: default {l_def:.2f} m, variants {L.min():.2f}-{L.max():.2f} m, 5-95 % {l_lo:.2f}-{l_hi:.2f} m")
    print(f"span: default {span_def:.2f} m, range {span_lo:.2f}-{span_hi:.2f} m\n")
    hdr = "| Pixel bin | 50 deg, span-limited | 50 deg, length-limited | 10 deg, span-limited | 10 deg, length-limited |"
    print(hdr)
    print("| --- | --- | --- | --- | --- |")
    f50, f10 = f_px(640, 50), f_px(640, 10)
    for name, p in zip(NAMES, LOWER):
        cells = []
        for f in (f50, f10):
            for lo, hi, de in ((span_lo, span_hi, span_def), (l_lo, l_hi, l_def)):
                cells.append(f"{f * lo / p:,.0f}-{f * hi / p:,.0f} ({f * de / p:,.0f})")
        print(f"| {name} px | " + " | ".join([cells[0], cells[1], cells[2], cells[3]]) + " |")
    print(f"\nf_px = {f50:.1f} at 50 deg and {f10:.1f} at 10 deg (N_x = 640)")


if __name__ == "__main__":
    main()

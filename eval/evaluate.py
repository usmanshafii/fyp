"""Evaluate trained models on identical test scenes (spec 8).

    python eval/evaluate.py --data out/dataset --split test \
        --runs out/runs/rgb_only out/runs/lwir_only out/runs/rgb_lwir out/runs/teacher out/runs/student

Writes <out>/report.json (per model: overall / per bin / per condition metrics and Pd curves), the
paired-bootstrap comparisons that answer the research question, and the plots.

Negative results are results: if the teacher does not beat the RGB + LWIR baseline, or the
student does not beat it, the report says SWIR gave no transferable benefit under these
assumptions (spec 8).
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.dataset import SceneDataset  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval import plots  # noqa: E402
from train.engine import device_of, load_all_configs, load_ckpt  # noqa: E402

COMPARE = [("teacher", "rgb_lwir"), ("student", "rgb_lwir"), ("rgb_lwir", "rgb_only"), ("rgb_lwir", "lwir_only")]


def _json(x):
    if isinstance(x, dict):
        return {k: _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def evaluate(runs, data, split="test", out=os.path.join(ROOT, "out", "eval"), device="auto", batch=4, cfgs=None):
    cfgs = cfgs or load_all_configs()
    E = cfgs["train"]["eval"]
    dev = device_of(device)
    os.makedirs(out, exist_ok=True)
    reports, records = {}, {}
    for run in runs:
        ck = run if run.endswith(".pt") else os.path.join(run, "best.pt")
        model, meta = load_ckpt(ck, dev)
        name = os.path.basename(os.path.dirname(ck)) if not run.endswith(".pt") else os.path.splitext(os.path.basename(run))[0]
        ds = SceneDataset(data, split, model.bands, cfgs, train=False, seed=cfgs["train"]["train"]["seed"])
        print(f"{name}: {len(ds)} {split} scenes, bands {model.bands}")
        rec = M.predict_records(model, ds, model.bands, E, dev, batch)
        records[name] = rec
        reports[name] = M.summary(rec, E)
        o = reports[name]["overall"]
        print(f"  AP50 {o['ap50']:.3f}  AP50:95 {o['ap50_95']:.3f}  AP(centre) {o.get('ap_centre', float('nan')):.3f}  "
              f"FPPI {o['op_iou50']['fppi']:.3f}")
        for b, v in reports[name]["bins"].items():
            print(f"    {b:>6s} px  AP50 {v['ap50']:.3f}  AP(centre) {v.get('ap_centre', float('nan')):.3f}  "
                  f"recall {v['op_iou50']['recall']:.3f}  n {v['op_iou50']['n_gt']}")
    comparisons = {}
    for a_, b_ in COMPARE:
        if a_ in records and b_ in records:
            c = {}
            for key, fn in (("ap50", lambda r: M.ap(r, "iou", 0.5, cfg=E)),
                            ("ap_centre", lambda r: M.ap(r, "centre", 0.0, cfg=E))):
                c[key] = M.bootstrap_diff(records[a_], records[b_], fn, min(E["bootstrap"], 300))
            for b in E["tiny_bins"]:
                c[f"ap_centre_{b}"] = M.bootstrap_diff(records[a_], records[b_],
                                                       lambda r, b=b: M.ap(r, "centre", 0.0, b, cfg=E), min(E["bootstrap"], 300))
            comparisons[f"{a_}_vs_{b_}"] = c
    verdict = []
    for key in ("teacher_vs_rgb_lwir", "student_vs_rgb_lwir"):
        if key in comparisons:
            d = comparisons[key]["ap_centre"]
            if d["significant"] and d["diff"] > 0:
                verdict.append(f"{key}: +{d['diff']:.3f} AP(centre), 95% CI [{d['lo']:.3f}, {d['hi']:.3f}] - benefit")
            else:
                verdict.append(f"{key}: {d['diff']:+.3f} AP(centre), 95% CI [{d['lo']:.3f}, {d['hi']:.3f}] - "
                               "no significant transferable SWIR benefit under these assumptions")
    for v in verdict:
        print(v)
    report = _json(dict(split=split, models=reports, comparisons=comparisons, verdict=verdict))
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    plots.all_plots(reports, out, tuple(E["span_range_m"]))
    torch.save(records, os.path.join(out, "records.pt"))
    print("wrote", out)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "out", "dataset"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", nargs="+", required=True, help="run folders (containing best.pt) or checkpoint files")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "eval"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    evaluate(a.runs, a.data, a.split, a.out, a.device, a.batch)


if __name__ == "__main__":
    main()

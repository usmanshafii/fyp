"""End-to-end ML smoke test on a small rendered dataset: every model trains for a few iterations,
the student distils from the teacher, all are evaluated on the same test scenes, and the student's
CPU latency is measured. It checks the plumbing, not accuracy.

    python tools/smoke_ml.py --data out/smoke --out out/smoke_runs --iters 3
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]

from eval.evaluate import evaluate  # noqa: E402
from train.engine import load_all_configs, train_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "out", "smoke"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "smoke_runs"))
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    cfgs = load_all_configs()
    tc = cfgs["train"]["train"]
    tc.update(batch=a.batch, workers=a.workers, device=a.device, eval_every=1)
    runs, times = {}, {}
    for name in ("rgb_only", "lwir_only", "rgb_lwir", "teacher", "student"):
        t = time.time()
        teacher = runs.get("teacher") if name == "student" else None
        runs[name] = train_model(name, a.data, a.out, cfgs, teacher, max_iters=a.iters, epochs=1)
        times[name] = time.time() - t
        print(f"== {name}: {times[name]:.0f} s -> {runs[name]}")
    evaluate([os.path.dirname(p) for p in runs.values()], a.data, "test", os.path.join(a.out, "eval"), a.device, 2, cfgs)
    subprocess.run([sys.executable, os.path.join(ROOT, "eval", "latency.py"), "--run", os.path.dirname(runs["student"]), "--n", "5"],
                   check=False)
    print("train seconds:", {k: round(v) for k, v in times.items()})


if __name__ == "__main__":
    main()

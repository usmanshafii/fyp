"""Dataset driver (Phase 5): render scenes in Blender in chunks, then export COCO and a summary.

    python tools/generate_dataset.py --out out/dataset --n 720 --jobs 1 --chunk 20
    python tools/generate_dataset.py --out out/dataset --n 720 --summary-only

Each chunk runs `blender -b -P render/scene.py -- --start i --count chunk`; finished scenes are
skipped, so the command can be stopped and resumed. --jobs > 1 runs chunks in parallel (set
--threads so jobs x threads <= CPU cores). Blender is found via --blender, $BLENDER, or the
standard install folders.
"""
import argparse
import collections
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]


def find_blender(explicit=None):
    cands = [explicit, os.environ.get("BLENDER"), shutil.which("blender")]
    cands += sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender 4.*\blender.exe"), reverse=True)
    cands += ["/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender"]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("Blender not found: pass --blender or set $BLENDER")


def run_chunk(blender, out, start, count, n, extra, log_dir):
    cmd = [blender, "-b", "--factory-startup", "-P", os.path.join(ROOT, "render", "scene.py"), "--",
           "--start", str(start), "--count", str(count), "--out", out, "--n-scenes", str(n)] + extra
    log = os.path.join(log_dir, f"chunk_{start:06d}.log")
    t = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return start, count, r.returncode, time.time() - t, log


def summary(out):
    from data.dataset import load_metas
    metas = load_metas(out)
    c = collections.Counter()
    t = collections.defaultdict(list)
    for m in metas:
        c[("split", m["split"])] += 1
        c[("bin", m["target"]["size_bin"])] += 1
        c[("family", m["conditions"]["background"])] += 1
        c[("lighting", m["conditions"]["lighting"])] += 1
        c[("weather", m["conditions"]["weather"])] += 1
        c[("hfov", m["camera"]["hfov_deg"])] += 1
        c[("bin x family", (m["target"]["size_bin"], m["conditions"]["background"]))] += 1
        for n in m["negatives"]:
            c[("negative", n["class"])] += 1
        tm = m["extra"].get("timings_s") or {}
        t["total"].append(sum(tm.values()))
    s = dict(n_scenes=len(metas), counts={f"{k[0]}: {k[1]}": v for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))},
             mean_scene_seconds=float(sum(t["total"]) / max(1, len(t["total"]))))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "dataset"))
    ap.add_argument("--n", type=int, default=None, help="number of scenes (default render.yaml n_scenes)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--blender", default=None)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    from render.sampling import load_configs
    n = a.n or load_configs()["render"]["dataset"]["n_scenes"]
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    if not a.summary_only:
        blender = find_blender(a.blender)
        log_dir = os.path.join(out, "logs")
        os.makedirs(log_dir, exist_ok=True)
        extra = (["--threads", str(a.threads)] if a.threads else []) + (["--samples", str(a.samples)] if a.samples else [])
        chunks = [(s, min(a.chunk, n - s)) for s in range(a.start, n, a.chunk)]
        print(f"{len(chunks)} chunks of {a.chunk} scenes with {blender}, {a.jobs} parallel job(s)")
        with ThreadPoolExecutor(a.jobs) as ex:
            for start, count, rc, dt, log in ex.map(lambda c: run_chunk(blender, out, c[0], c[1], n, extra, log_dir), chunks):
                print(f"scenes {start}-{start + count - 1}: exit {rc}, {dt / 60:.1f} min, log {log}")
    from data.coco_export import export
    for split, (ni, na) in export(out).items():
        print(f"COCO {split}: {ni} images, {na} annotations")
    s = summary(out)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()

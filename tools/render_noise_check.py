"""Is the background render clean enough? Compares Cycles noise with the sensor noise it will be
buried under, and measures what fewer bounces change.

    blender -b -P tools/render_noise_check.py -- --scene 0 --band rgb --samples 256 --bounces 4 1

For each bounce setting the target-free background of the scene is rendered twice with different
Cycles seeds. After the PSF and the 2x box average:
    render sigma = std(a - b) / sqrt(2) over ground pixels, reported relative to the nominal sensor
    sigma at the same radiance (configs/sensors.yaml). Below ~0.35 the render adds < 6 % to the
    noise variance the detector sees.
    bias = mean ground radiance relative to the first (reference) setting.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]
import bpy  # noqa: E402
import numpy as np  # noqa: E402

from data import writer  # noqa: E402
from render import bpy_util as bu  # noqa: E402
from render import psf, sampling  # noqa: E402
from render import scene as sc_mod  # noqa: E402


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--band", default="rgb", choices=["rgb", "swir"])
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--bounces", type=int, nargs="+", default=[4, 1])
    a = ap.parse_args(argv)
    cfgs = sampling.load_configs()
    rc = cfgs["render"]["render"]
    rc["engine_samples"] = a.samples
    if a.threshold:
        rc["adaptive_threshold"] = a.threshold
    spec, objs = sampling.sample_scene(a.scene, cfgs)
    if spec["family"] != "horizon":
        print("note: open-sky scene; the background is sky only and has no Monte Carlo noise")
    calib = sc_mod.calibrate(spec, cfgs)
    h = sc_mod.build(spec, objs, cfgs, calib)
    sc_mod.apply_band(a.band, h, cfgs)
    sc_mod._visibility(h, labelled_visible=False)
    W, H = spec["camera"]["width"], spec["camera"]["height"]
    s = int(spec["camera"]["supersample_background"])
    s0 = spec["psf"][a.band]["sigma0"]
    tmp = os.path.join(ROOT, "out", "tmp", "noise_check")
    os.makedirs(tmp, exist_ok=True)
    grey = a.band != "rgb"
    ref = None
    for nb in a.bounces:
        bu.set_bounces(nb, rc)
        bu.set_samples(a.samples)
        imgs, times = [], []
        for seed in (0, 1):
            bpy.context.scene.cycles.seed = seed
            t = time.time()
            p = bu.render(os.path.join(tmp, f"bg_{nb}_{seed}.exr"), (W * s, H * s))
            times.append(time.time() - t)
            d = bu.read_passes(p, grey)
            bu.remove_file(p)
            imgs.append((psf.psf_integrate(d["combined"], s0, s), psf.psf_integrate(d["alpha"], s0, s)))
        (a0, al), (a1, _) = imgs
        m = al > 0.999
        if not m.any():
            print("no ground in frame")
            return
        y0, y1 = [writer.rgb_noise.luminance(x) if x.shape[-1] == 3 else x[..., 0] for x in (a0, a1)]
        sig_r = float(np.std((y0 - y1)[m]) / np.sqrt(2))
        level = float(y0[m].mean())
        sig_s = writer.noise_sigma(a.band, level, spec, cfgs["sensors"])
        ref = level if ref is None else ref
        print(f"bounces {nb}: {np.mean(times):6.1f} s/render  ground mean {level:.4g} (bias {100 * (level / ref - 1):+.2f} %)  "
              f"render sigma {sig_r:.3g} = {sig_r / sig_s:.2f} x sensor sigma ({sig_s:.3g})")


if __name__ == "__main__":
    main()

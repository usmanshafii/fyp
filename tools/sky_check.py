"""Sky acceptance test (FYP_Sky_and_Background.md, run before the LWIR pass).

    blender -b -P tools/sky_check.py -- [--elevations 3 20 45 70] [--sun-elevation 30]

Renders target-free RGB frames with the narrow 10 deg camera through the production path (Nishita
sky, 2x supersampling, PSF, box average) and checks:
  1. a real gradient: brighter near the horizon than at the zenith, brighter toward the sun;
  2. no pixelation / stair-stepping: the largest second difference between neighbouring pixels is
     tiny compared with the signal;
  3. flat-patch noise: std / mean of a 32x32 patch (after removing a plane fit, i.e. the smooth
     gradient) is below 0.5 %; above that the experiment would measure Cycles noise, not detectability.
Writes out/sky_check/*.png and a JSON summary.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]
import bpy  # noqa: E402
import numpy as np  # noqa: E402

from data.imgio import write_png  # noqa: E402
from data.writer import tonemap  # noqa: E402
from render import bpy_util as bu  # noqa: E402
from render import placement as pl  # noqa: E402
from render import psf, rgb, sampling  # noqa: E402


def detrended_cv(patch):
    h, w = patch.shape
    yy, xx = np.mgrid[:h, :w]
    A = np.c_[np.ones(h * w), xx.ravel(), yy.ravel()]
    coef, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
    res = patch.ravel() - A @ coef
    return float(res.std() / patch.mean())


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    # dataset look elevations: horizon -1..5, open sky 10..60 deg. Blender's precomputed Nishita
    # texture has a pole kink within ~1 deg of the zenith (0.4 % second difference, measured); frames
    # whose top edge is above 85 deg are reported but not failed, since dataset frames stop at 80.6 deg.
    ap.add_argument("--elevations", type=float, nargs="+", default=[3.0, 20.0, 45.0, 60.0])
    ap.add_argument("--sun-elevation", type=float, default=30.0)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "sky_check"))
    a = ap.parse_args(argv)
    cfgs = sampling.load_configs()
    rc = cfgs["render"]["render"]
    if a.samples:
        rc["engine_samples"] = a.samples
    os.makedirs(a.out, exist_ok=True)
    atmo = dict(air=1.0, dust=1.0, ozone=1.0)
    calib = dict(sun=rgb.calibrate_sun(a.sun_elevation, atmo, 256))
    spec = dict(lighting="day", atmosphere=atmo, sun=dict(elevation_deg=a.sun_elevation, azimuth_deg=0.0),
                clouds=None, airglow_rgb=0.0)
    s = int(rc["supersample_background"])
    sigma = cfgs["sensors"]["psf_sigma_px"]["rgb"][0]
    W, H = 640, 512
    results = []
    for az_name, az in (("toward_sun", 0.0), ("away_from_sun", 180.0)):
        for el in a.elevations:
            bu.reset()
            sc = bu.setup_cycles(rc)
            sc.world = rgb.world(spec, calib)
            cd = bpy.data.cameras.new("cam")
            cd.sensor_width, cd.sensor_fit, cd.lens = 36.0, "HORIZONTAL", pl.focal_mm(36.0, 10.0)
            cd.clip_start, cd.clip_end = rc["clip_m"]
            cam = bpy.data.objects.new("cam", cd)
            sc.collection.objects.link(cam)
            sc.camera = cam
            bu.set_matrix(cam, pl.matrix_world(pl.camera_rotation(az, el), (0, 0, 2.0)))
            p = bu.render(os.path.join(a.out, "tmp.exr"), (W * s, H * s))
            d = bu.read_passes(p, grey=False)
            bu.remove_file(p)
            img = psf.psf_integrate(d["env"], sigma, s)
            Y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            rows = Y.mean(1)
            d2 = np.abs(np.diff(Y, 2, axis=1)).max() / Y.mean()
            patch = Y[H // 2 - 16:H // 2 + 16, W // 2 - 16:W // 2 + 16]
            raw_cv = float(patch.std() / patch.mean())
            cv = detrended_cv(patch)
            near_zenith = el + 5.0 > 85.0                  # 10 deg camera: top edge ~ el + 4.1 deg
            r = dict(azimuth=az_name, elevation_deg=el, mean=float(Y.mean()), top_row=float(rows[0]), bottom_row=float(rows[-1]),
                     max_second_diff_rel=float(d2), patch_cv_raw=raw_cv, patch_cv_detrended=cv, pass_noise=cv < 0.005,
                     pass_smooth=float(d2) < 0.002 or near_zenith, near_zenith_info=near_zenith)
            results.append(r)
            write_png(os.path.join(a.out, f"sky_{az_name}_{el:g}.png"), tonemap(img, "rgb"))
            print(f"{az_name:14s} el {el:5.1f}: mean {r['mean']:.4g}  top/bottom row {r['top_row']:.4g}/{r['bottom_row']:.4g}  "
                  f"max 2nd diff {100 * d2:.4f} %  patch std/mean raw {100 * raw_cv:.4f} %, detrended {100 * cv:.4f} %")
    away = [r for r in results if r["azimuth"] == "away_from_sun"]
    toward = [r for r in results if r["azimuth"] == "toward_sun"]
    # A clear sky is darkest ~90 deg from the sun, so it need not fall monotonically all the way up;
    # the Sky doc's criterion is "bright near horizon and sun, darker at zenith".
    lo_el, hi_el = min(a.elevations), max(a.elevations)
    grad_ok = all(next(r for r in rs if r["elevation_deg"] == lo_el)["mean"] > next(r for r in rs if r["elevation_deg"] == hi_el)["mean"]
                  for rs in (away, toward))
    sun_ok = all(t["mean"] > aw["mean"] for t, aw in zip(toward, away) if t["elevation_deg"] <= 45)
    ok = grad_ok and sun_ok and all(r["pass_noise"] and r["pass_smooth"] for r in results)
    summary = dict(results=results, horizon_brighter_than_zenith=grad_ok, brighter_toward_sun=sun_ok, all_pass=ok)
    with open(os.path.join(a.out, "sky_check.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"horizon brighter than zenith: {grad_ok}; brighter toward the sun: {sun_ok}")
    print("SKY CHECK PASS" if ok else "SKY CHECK FAIL")


if __name__ == "__main__":
    main()

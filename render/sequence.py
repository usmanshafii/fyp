try: import blenderproc as bproc  # noqa: E401,E701  first code line, as in render/scene.py
except Exception: bproc = None  # noqa: E701
__doc__ = """Fixed-camera flight sequences for the live demo video (tools/demo_video.py).

    blender -b -P render/sequence.py -- --out out/demo --name day --seq-id 901 --lighting day

One scene is sampled exactly as for the dataset (render/sampling.py: sun, sky, weather, target
variant, per-band signatures), without hard negatives. The camera does not move, so each band's
background is rendered once and only the target crop is rendered per frame. Every frame is written
as an ordinary scene (meta/<id>.json, layers/<band>/<id>.npz) whose layers point at the shared
background (data/composite.py, background_ref), so it is hazed and loaded like any other scene.

The KAL flies straight and level at 50 m/s = 180 km/h, the middle of the published cruise band of
160-200 km/h [B: press report, Indian Masterminds, 21 Sep 2026, which also gives an operating
altitude of up to 5,000 m]. The path crosses the view obliquely, so the drone closes in while it
drifts across the frame; its altitude follows from the camera geometry and is recorded per frame.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(ROOT, "target"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bpy  # noqa: E402,F401
import numpy as np  # noqa: E402

from data import composite as cp  # noqa: E402
from render import bpy_util as bu  # noqa: E402
from render import placement as pl  # noqa: E402
from render import sampling  # noqa: E402
from render import scene as rs  # noqa: E402


def _log(msg):
    print(msg, flush=True)


def find_scene(cfgs, a):
    """First scene index >= a.start with the requested lighting and clear air: open sky, narrow lens,
    the camera tilted to a.cam_el (a steeper look puts the drone higher, and while it closes in it
    climbs out of the 8 deg vertical view), sun at least 25 deg off the line of sight."""
    for idx in range(a.start, a.start + 1000):
        try:
            spec, objs = sampling._sample_scene_once(idx, cfgs, None, 0, {"hfov": 10.0, "family": "open_sky"})
        except RuntimeError:
            continue
        if spec["lighting"] != a.lighting or spec["weather"]["visibility_km"] < 15.0:
            continue
        c = objs["camera"]
        cam = pl.Camera(c.W, c.H, c.hfov, c.C, c.azimuth, a.cam_el, c.roll)
        sun = pl.direction(spec["sun"]["azimuth_deg"], spec["sun"]["elevation_deg"])
        if spec["lighting"] != "night" and float(sun @ cam.forward) > np.cos(np.radians(25.0)):
            continue
        objs["camera"] = cam
        spec["camera"].update(elevation_deg=a.cam_el, matrix_world=pl.matrix_world(cam.R, cam.C).tolist())
        spec["objects"] = spec["objects"][:1]             # the target only
        spec["signatures"]["negatives"] = []
        objs["negatives"] = []
        return idx, spec, objs
    raise RuntimeError("no scene matches the requested conditions")


def frame_entry(cfgs, spec, objs, pos, R, vel, pose, t):
    """Object entry in the sampler's schema (render/sampling.py obj_entry) for one target pose."""
    Rc, S = cfgs["render"], cfgs["sensors"]
    cam, Vt, terrain = objs["camera"], objs["target_verts"], objs["terrain"]
    b = cam.bbox(Vt @ R.T + pos)
    if b is None:
        return None
    edges = [tuple(e) for e in Rc["size_bins"]["edges_px"]]
    size = pl.box_size(b)
    bi = pl.bin_index(size, edges)
    p = pl.Placement(position=np.asarray(pos, float), R=R, distance=float(np.linalg.norm(pos - cam.C)), bbox=b,
                     size_px=size, bin=bi, pose=pose, velocity=np.asarray(vel, float))
    p.bbox_blur = pl.motion_boxes(cam, Vt, p, spec["camera"]["exposure_ms"] / 1000.0)
    margin = int(math.ceil(3 * max(S["psf_sigma_px"][k][1] for k in S["psf_sigma_px"]))) + int(Rc["render"]["crop_margin_px"])
    bb = p.bbox_blur
    crop = [max(0, int(math.floor(bb[0])) - margin), max(0, int(math.floor(bb[1])) - margin),
            min(cam.W, int(math.ceil(bb[2])) + margin), min(cam.H, int(math.ceil(bb[3])) + margin)]
    rc = Rc["render"]
    ss = int(rc["supersample_small_object"] if size < rc["small_object_px"] else rc["supersample_object"])
    az, el = pl.azimuth_elevation(p.position - cam.C)
    gz = float(terrain.ground_z(p.position[0], p.position[1]))
    xywh = lambda q: dict(x=float(q[0]), y=float(q[1]), w=float(q[2] - q[0]), h=float(q[3] - q[1]))  # noqa: E731
    return dict(name="target", cls="target", class_id=0, t_s=t, position=[float(x) for x in p.position],
                matrix_world=pl.matrix_world(R, p.position).tolist(), velocity_mps=[float(x) for x in p.velocity],
                speed_mps=float(np.linalg.norm(p.velocity)), range_m=p.distance, ray_azimuth_deg=az, elevation_deg=el,
                altitude_m=float(p.position[2]), agl_m=float(p.position[2] - gz), bbox_px=xywh(b), bbox_blur_px=xywh(bb),
                size_px=float(size), size_bin=Rc["size_bins"]["names"][bi] if bi >= 0 else "<2", crop_px=crop,
                supersample=ss, pose=pose, **p.aspect(cam))


def flight_frames(cfgs, spec, objs, a):
    """Level flight at constant speed through the view; the crossing angle shrinks until it fits."""
    cam = objs["camera"]
    n = int(round(a.duration * a.fps))
    beta = 14.0
    for _ in range(12):
        for v_frac in (0.6, 0.5, 0.7, 0.4):
            mid = cam.C + cam.ray(cam.W / 2, cam.H * v_frac) * a.range_m
            g = cam.C - mid
            g[2] = 0.0
            g /= np.linalg.norm(g)
            b = math.radians(beta)
            vdir = np.array([g[0] * math.cos(b) - g[1] * math.sin(b), g[0] * math.sin(b) + g[1] * math.cos(b), 0.0])
            heading = pl.azimuth_elevation(vdir)[0]
            R = pl.object_rotation(heading, 0.0, 0.0)
            pose = dict(heading_deg=heading, pitch_deg=0.0, roll_deg=0.0)
            frames = []
            for k in range(n):
                t = k / a.fps
                e = frame_entry(cfgs, spec, objs, mid + a.speed * vdir * (t - a.duration / 2), R, a.speed * vdir, pose, t)
                bx = e and e["bbox_blur_px"]
                if (e is None or bx["x"] < 8 or bx["y"] < 8 or bx["x"] + bx["w"] > cam.W - 8
                        or bx["y"] + bx["h"] > cam.H - 8 or e["agl_m"] < 60.0):
                    frames = None
                    break
                frames.append(e)
            if frames:
                return frames
        beta *= 0.8
    raise RuntimeError("the flight does not stay in view")


def frame_meta(spec, fr, fid, a, bands, k):
    cam = spec["camera"]
    return dict(
        scene_id=fid, split="test", seed=spec["seed"],
        sequence=dict(name=a.name, index=k, t_s=fr["t_s"], fps=a.fps, base_scene=spec["idx"]),
        camera=dict(hfov_deg=cam["hfov_deg"], width=cam["width"], height=cam["height"], f_px=cam["f_px"],
                    supersample=fr["supersample"], supersample_background=cam["supersample_background"],
                    exposure_ms=cam["exposure_ms"]),
        target=dict(range_m=fr["range_m"], elevation_deg=fr["elevation_deg"], agl_m=fr["agl_m"],
                    speed_mps=fr["speed_mps"], heading_deg=fr["pose"]["heading_deg"], bbox_px=fr["bbox_px"],
                    size_bin=fr["size_bin"]),
        conditions=dict(lighting=spec["lighting"], weather=spec["weather"]["kind"],
                        visibility_km=spec["weather"]["visibility_km"], background="open_sky",
                        thermal_state=spec["thermal_state"], air_temp_c=spec["air_temp_c"], warm_clutter=False,
                        clouds=spec["clouds"] is not None),
        modalities={b: dict(layers=f"layers/{b}/{fid}.npz") for b in bands},
        negatives=[],
        extra=dict(objects=[fr], camera=cam, sun=spec["sun"], moon=spec["moon"], weather=spec["weather"],
                   psf=spec["psf"], family=spec["family"]),
    )


def render_sequence(cfgs, spec, objs, frames, a, bands):
    t0 = time.time()
    calib = rs.calibrate(spec, cfgs)
    spec["objects"] = [frames[0]]                    # the target is built at the first pose
    h = rs.build(spec, objs, cfgs, calib)
    tobj = h.labelled[0][1]
    exp_s = spec["camera"]["exposure_ms"] / 1000.0
    tmp = os.path.join(a.out, "tmp", a.name)
    os.makedirs(tmp, exist_ok=True)
    ids = [f"s{a.seq_id:03d}{k:04d}" for k in range(len(frames))]
    for band in bands:
        rs.apply_band(band, h, cfgs)
        layer = rs.render_background(band, spec, h, cfgs, tmp, _log)
        bg_rel = f"bg/{band}/{a.name}.npz"
        for d in (f"bg/{band}", f"layers/{band}"):
            os.makedirs(os.path.join(a.out, d), exist_ok=True)
        cp.save_layers(os.path.join(a.out, bg_rel), layer, [], half=(band != "lwir"))
        tb = time.time()
        for k, fr in enumerate(frames):
            bu.set_matrix(tobj, fr["matrix_world"])
            bu.linear_motion(tobj, fr["matrix_world"], fr["velocity_mps"], exp_s)
            objects = rs.render_objects(band, spec, h, cfgs, tmp, lambda m: None, [fr])
            lpath = os.path.join(a.out, f"layers/{band}/{ids[k]}.npz")
            ref = os.path.relpath(os.path.join(a.out, bg_rel), os.path.dirname(lpath))
            cp.save_layers(lpath, dict(psf_sigma0=layer["psf_sigma0"]), objects, half=(band != "lwir"), background_ref=ref)
            if k % 25 == 0 or k == len(frames) - 1:
                _log(f"  [{a.name}] {band} frame {k + 1}/{len(frames)} ({fr['size_px']:.1f} px @ {fr['range_m']:.0f} m) "
                     f"{(time.time() - tb) / (k + 1):.2f} s/frame")
    os.makedirs(os.path.join(a.out, "meta"), exist_ok=True)
    for k, fr in enumerate(frames):
        with open(os.path.join(a.out, "meta", f"{ids[k]}.json"), "w") as f:
            json.dump(frame_meta(spec, fr, ids[k], a, bands, k), f, indent=1)
    os.makedirs(os.path.join(a.out, "sequences"), exist_ok=True)
    tv = spec["target_variant"]
    summary = dict(name=a.name, seq_id=a.seq_id, base_scene=spec["idx"], bands=list(bands), fps=a.fps,
                   lighting=spec["lighting"], thermal_state=spec["thermal_state"], weather=spec["weather"],
                   air_temp_c=spec["air_temp_c"], camera=spec["camera"], sun=spec["sun"], moon=spec["moon"],
                   variant=dict(span_m=tv["span_m"], length_m=tv["length_m"], sweep_deg=tv["sweep_deg"]),
                   signatures=dict(target=spec["signatures"]["target"]), frame_ids=ids,
                   frames=[dict(id=i, t_s=fr["t_s"], range_m=fr["range_m"], agl_m=fr["agl_m"], speed_mps=fr["speed_mps"],
                                heading_deg=fr["pose"]["heading_deg"], size_px=fr["size_px"], bbox_px=fr["bbox_px"])
                           for i, fr in zip(ids, frames)],
                   seconds=time.time() - t0)
    with open(os.path.join(a.out, "sequences", f"{a.name}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    _log(f"[{a.name}] {len(frames)} frames, {'/'.join(bands)}, done in {time.time() - t0:.0f}s")


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description="fixed-camera flight sequence for the demo video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seq-id", type=int, required=True, help="3-digit id; frame ids are s<seq><frame:04d>")
    ap.add_argument("--lighting", choices=("day", "dusk", "night"), required=True)
    ap.add_argument("--start", type=int, default=5000, help="first scene index searched for the base scene")
    ap.add_argument("--cam-el", type=float, default=11.0, help="camera tilt above the horizon, deg")
    ap.add_argument("--range-m", type=float, default=900.0, help="slant range at mid-clip")
    ap.add_argument("--speed", type=float, default=50.0, help="ground speed, m/s (50 = 180 km/h)")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--bands", default="rgb,lwir")
    return ap.parse_args(argv)


def main():
    a = _args()
    a.out = os.path.abspath(a.out)
    cfgs = sampling.load_configs()
    if bproc is not None:
        bproc.init()
    bands = tuple(b for b in a.bands.split(",") if b)
    idx, spec, objs = find_scene(cfgs, a)
    _log(f"[{a.name}] base scene {idx}: {spec['lighting']} cam el {spec['camera']['elevation_deg']:.1f} deg, "
         f"vis {spec['weather']['visibility_km']:.1f} km, air {spec['air_temp_c']:.1f} C, "
         f"exposure {spec['camera']['exposure_ms']:.1f} ms")
    frames = flight_frames(cfgs, spec, objs, a)
    f0, f1 = frames[0], frames[-1]
    _log(f"[{a.name}] flight: {len(frames)} frames, {f0['range_m']:.0f} -> {f1['range_m']:.0f} m, "
         f"{f0['size_px']:.1f} -> {f1['size_px']:.1f} px, AGL {f0['agl_m']:.0f} m, {a.speed * 3.6:.0f} km/h")
    render_sequence(cfgs, spec, objs, frames, a, bands)


if __name__ == "__main__":
    main()

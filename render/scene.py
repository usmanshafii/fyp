try: import blenderproc as bproc  # noqa: E401,E701  first code line, required by `blenderproc run`
except Exception: bproc = None  # noqa: E701  plain Blender: blender -b -P render/scene.py -- ...
__doc__ = """Scene generator for Phases 2-5: pixel-aligned RGB, LWIR and SWIR renders of one scene.

    blender -b -P render/scene.py -- --start 0 --count 10 --out out/dataset
    blenderproc run render/scene.py --start 0 --count 10 --out out/dataset     (optional)

Per scene (render/sampling.py decides everything before any pixel is rendered):
  1. calibrate the sun (and moon) lamp against the Nishita sun disc for the scene's atmosphere;
  2. build camera, terrain / structures, the KAL target (target/build_target.py) and the hard
     negatives, with per-band materials, worlds and lamps;
  3. for each band, with the same scene and camera (so the three bands are pixel-aligned):
       background   full frame at 2x, labelled objects hidden: Env (sky), Combined (terrain),
                    Mist (ray distance)
       horizon      a thin probe just above the horizon (haze path radiance), horizon family only
       objects      one border render per labelled object at 2x (4x under 8 px), with motion blur,
                    every other object and the terrain as holdouts;
     each render goes through the Gaussian PSF at supersampled resolution and a box average;
  4. data/writer.py stores the layers, previews, contrast-to-noise and metadata.
Haze, the rest of the PSF, registration offsets and sensor noise are applied by the loader.
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

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import build_target as bt  # noqa: E402
import kal_geometry as kg  # noqa: E402
from data import writer  # noqa: E402
from render import backgrounds as bg  # noqa: E402
from render import bpy_util as bu  # noqa: E402
from render import lwir, rgb, sampling, swir  # noqa: E402
from render import placement as pl  # noqa: E402
from render import psf  # noqa: E402
from render.negatives import NEG_SLOTS  # noqa: E402

BANDS = ("rgb", "lwir", "swir")
K0 = 273.15


# ============================================================================ build
def calibrate(spec: dict, cfgs: dict) -> dict:
    sk = cfgs["render"]["sky"]
    n = int(sk.get("sun_calibration_samples", 256))
    out = {}
    if spec["lighting"] != "night":
        out["sun"] = rgb.calibrate_sun(spec["sun"]["elevation_deg"], spec["atmosphere"], n)
    if spec.get("moon"):
        out["moon"] = rgb.calibrate_sun(spec["moon"]["elevation_deg"], spec["atmosphere"], n)
    return out


class Handles:
    def __init__(self):
        self.cam = None
        self.probe = None
        self.env_objs = []          # terrain, structures
        self.labelled = []          # (name, object) in spec["objects"] order
        self.materials = {b: {} for b in BANDS}
        self.worlds = {}
        self.lamps = {b: [] for b in BANDS}


def _camera(name, cam_spec, cfgs, W=None, H=None, hfov=None, matrix=None):
    cd = bpy.data.cameras.new(name)
    cd.sensor_width = cam_spec["sensor_width_mm"]
    cd.sensor_fit = "HORIZONTAL"
    cd.lens = pl.focal_mm(cam_spec["sensor_width_mm"], hfov or cam_spec["hfov_deg"])
    cd.clip_start, cd.clip_end = cfgs["render"]["render"]["clip_m"]
    obj = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(obj)
    bu.set_matrix(obj, matrix if matrix is not None else cam_spec["matrix_world"])
    return obj


def build(spec: dict, objs: dict, cfgs: dict, calib: dict) -> Handles:
    R, SIG = cfgs["render"], cfgs["signatures"]
    bu.reset()
    sc = bu.setup_cycles(R["render"])
    world_mist = R["render"]["mist_depth_m"]
    h = Handles()
    cam_spec = spec["camera"]
    h.cam = _camera("camera", cam_spec, cfgs)
    sc.camera = h.cam
    if spec["family"] == "horizon":
        C = pl.Camera(64, 8, cam_spec["hfov_deg"], cam_spec["position"], cam_spec["azimuth_deg"], 1.0)
        h.probe = _camera("horizon_probe", cam_spec, cfgs, matrix=pl.matrix_world(C.R, C.C))

    # ---------------- environment
    terr = bg.build_terrain_object(objs["terrain"])
    h.env_objs.append(terr)
    struct = bg.build_structures_object(objs["blocks"])
    if struct is not None:
        h.env_objs.append(struct)

    # ---------------- target and negatives
    exp_s = cam_spec["exposure_ms"] / 1000.0
    tspec = spec["objects"][0]
    tobj = bt.build_target(spec["target_variant"]["params"], name="KAL")
    bu.set_matrix(tobj, tspec["matrix_world"])
    bu.linear_motion(tobj, tspec["matrix_world"], tspec["velocity_mps"], exp_s)
    if bproc is not None:
        bproc.object.convert_to_meshes([tobj])      # BlenderProc wrapper (spec Phase 1 step 4)
    h.labelled.append(("target", tobj))
    for k, n in enumerate(objs["negatives"]):
        o = spec["objects"][k + 1]
        m = n["mesh"]
        nobj = bu.mesh_object(o["name"], m["vertices"], m["faces"], m["face_slots"], len(NEG_SLOTS), smooth=False)
        bu.set_matrix(nobj, o["matrix_world"])
        if np.linalg.norm(o["velocity_mps"]) > 0:
            bu.linear_motion(nobj, o["matrix_world"], o["velocity_mps"], exp_s)
        h.labelled.append((o["name"], nobj))

    # ---------------- materials, worlds, lamps
    sigs = spec["signatures"]
    env = sigs["environment"]
    lw_env = env["lwir"]
    envr = lwir.environment_radiance(spec, lw_env)
    spec["_lwir_env"] = envr
    refl = SIG["lwir"].get("reflection_term", True)
    Ta = spec["air_temp_c"] + K0
    gt = spec["ground_texture"]
    unused = {b: _unused(b) for b in BANDS}

    # target
    ts = sigs["target"]
    for band in BANDS:
        mats = []
        for slot in kg.SLOTS:
            p = ts[band][slot]
            name = f"{band}.target.{slot}"
            if band == "rgb":
                mats.append(rgb.target_material(name, slot, p))
            elif band == "lwir":
                mats.append(lwir.target_material(name, slot, p, spec, envr, refl, ts["rgb"][slot]))
            else:
                T = Ta + p["emit_dT"]           # SWIR's own temperature draw (spec 7.2 rule 4)
                mats.append(swir.target_material(name, slot, p, swir.hot_emission(p, T, spec, SIG)))
        h.materials[band][tobj.name] = mats
    # negatives
    for k, (nname, nobj) in enumerate(h.labelled[1:]):
        ns = sigs["negatives"][k]
        for band in BANDS:
            mats = []
            for slot in NEG_SLOTS:
                if slot not in ns[band]:
                    mats.append(unused[band])
                    continue
                p = ns[band][slot]
                name = f"{band}.{nname}.{slot}"
                if band == "rgb":
                    mats.append(rgb.negative_material(name, slot, p))
                elif band == "lwir":
                    mats.append(lwir.negative_material(name, slot, p, spec, envr, refl, ns["rgb"].get(slot)))
                else:
                    T = Ta + p.get("emit_dT", 0.0)
                    mats.append(swir.negative_material(name, slot, p, swir.hot_emission(p, T, spec, SIG)))
            h.materials[band][nobj.name] = mats
    # environment
    h.materials["rgb"][terr.name] = [rgb.environment_material("rgb.ground", "ground", env["rgb"]["ground"], gt["rgb"])]
    h.materials["lwir"][terr.name] = [lwir.ground_material("lwir.ground", lw_env["ground"], spec, envr, gt["lwir"], refl)]
    h.materials["swir"][terr.name] = [swir.environment_material("swir.ground", env["swir"]["ground"], gt["swir"])]
    if struct is not None:
        h.materials["rgb"][struct.name] = [rgb.environment_material(f"rgb.{k}", k, env["rgb"][k]) for k in ("wall", "roof")]
        h.materials["lwir"][struct.name] = [lwir.slot_material(f"lwir.{k}", lw_env[k], spec, envr, refl) for k in ("wall", "roof")]
        h.materials["swir"][struct.name] = [swir.environment_material(f"swir.{k}", env["swir"][k]) for k in ("wall", "roof")]

    h.worlds["rgb"] = rgb.world(spec, calib)
    h.worlds["lwir"] = lwir.world(spec, lw_env, envr)
    h.worlds["swir"] = swir.world(spec, env, calib)
    for w in h.worlds.values():
        w.mist_settings.start = 0.0
        w.mist_settings.depth = world_mist
        w.mist_settings.falloff = "LINEAR"
    h.lamps["rgb"] = rgb.lamps(spec, calib)
    h.lamps["swir"] = swir.lamps(spec, calib)
    return h


def _unused(band):
    m, nt, out = bu.new_material(f"{band}.unused")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(tr.outputs[0], out.inputs["Surface"])
    return m


def apply_band(band: str, h: Handles, cfgs: dict):
    sc = bpy.context.scene
    sc.world = h.worlds[band]
    for (name, obj) in h.labelled:
        bu.assign_materials(obj, h.materials[band][obj.name])
    for obj in h.env_objs:
        bu.assign_materials(obj, h.materials[band][obj.name])
    for b, lamps in h.lamps.items():
        for lamp in lamps:
            lamp.hide_render = b != band
    rc = cfgs["render"]["render"]
    bu.set_bounces(0 if band == "lwir" else rc["light_paths"]["total"], rc)


# ============================================================================ render
def _visibility(h: Handles, labelled_visible: bool, active: str | None = None, env_holdout: bool = False,
                env_hidden: bool = False):
    for name, obj in h.labelled:
        obj.hide_render = not labelled_visible
        obj.is_holdout = labelled_visible and active is not None and name != active
    for obj in h.env_objs:
        obj.hide_render = env_hidden
        obj.is_holdout = env_holdout


def render_band(band: str, spec: dict, h: Handles, cfgs: dict, tmp: str, log) -> tuple:
    apply_band(band, h, cfgs)
    layer = render_background(band, spec, h, cfgs, tmp, log)
    return layer, render_objects(band, spec, h, cfgs, tmp, log)


def render_background(band: str, spec: dict, h: Handles, cfgs: dict, tmp: str, log) -> dict:
    """Background layers of the current band (apply_band first): sky, ground, horizon probe."""
    rc = cfgs["render"]["render"]
    W, H = spec["camera"]["width"], spec["camera"]["height"]
    grey = band != "rgb"
    s0 = spec["psf"][band]["sigma0"]
    sc = bpy.context.scene
    layer = dict(psf_sigma0=s0)

    # ---------------- background: labelled objects hidden
    sb = int(spec["camera"]["supersample_background"])
    _visibility(h, labelled_visible=False)
    bu.set_samples(rc["lwir_background_samples"] if band == "lwir" else rc["engine_samples"])
    if band != "lwir":
        bu.set_bounces(int(rc.get("background_bounces", rc["light_paths"]["total"])), rc)
    sc.camera = h.cam
    t = time.time()
    path = bu.render(os.path.join(tmp, f"{band}_bg.exr"), (W * sb, H * sb))
    d = bu.read_passes(path, grey)
    bu.remove_file(path)
    layer["sky"] = psf.psf_integrate(d["env"], s0, sb)
    a = d["alpha"]
    if a.max() > 1e-6:
        ga = psf.psf_integrate(a, s0, sb)
        dist = d["mist"] * rc["mist_depth_m"]
        gd = psf.psf_integrate(a * dist, s0, sb)
        layer["ground"] = psf.psf_integrate(d["combined"], s0, sb)
        layer["galpha"] = ga
        layer["gdepth"] = np.where(ga > 1e-4, gd / np.maximum(ga, 1e-4), 0.0).astype(np.float32)
    log(f"    {band} background {W * sb}x{H * sb}: {time.time() - t:.1f}s")

    # ---------------- horizon radiance (path radiance for hazed terrain)
    if "ground" in layer:
        if band == "lwir":
            layer["horizon"] = np.full((W, 1), spec["_lwir_env"]["L_air"], np.float32)
        else:
            layer["horizon"] = _horizon_probe(band, spec, h, cfgs, tmp, grey)
    return layer


def render_objects(band: str, spec: dict, h: Handles, cfgs: dict, tmp: str, log, objects_spec=None) -> list:
    """One crop per labelled object (default: all of spec["objects"]), every other object and the
    terrain held out. Objects are matched to their Blender object by name."""
    rc = cfgs["render"]["render"]
    W, H = spec["camera"]["width"], spec["camera"]["height"]
    grey = band != "rgb"
    s0 = spec["psf"][band]["sigma0"]
    objects = []
    bu.set_bounces(0 if band == "lwir" else rc["light_paths"]["total"], rc)
    for o in (objects_spec if objects_spec is not None else spec["objects"]):
        s = int(o["supersample"])
        x0, y0, x1, y1 = o["crop_px"]
        _visibility(h, labelled_visible=True, active=o["name"], env_holdout=True)
        bu.set_samples(rc["engine_samples"])
        t = time.time()
        moving = float(np.linalg.norm(o["velocity_mps"])) > 0
        path = bu.render(os.path.join(tmp, f"{band}_{o['name']}.exr"), (W * s, H * s),
                         border=(x0 * s, y0 * s, x1 * s, y1 * s), motion_blur=moving)
        d = bu.read_passes(path, grey)
        bu.remove_file(path)
        exp = ((y1 - y0) * s, (x1 - x0) * s)
        if d["alpha"].shape != exp:
            raise RuntimeError(f"crop {d['alpha'].shape} != expected {exp} for {o['name']}")
        objects.append(dict(name=o["name"], crop=[x0, y0, x1, y1], range_m=o["range_m"], elevation_deg=o["elevation_deg"],
                            P=psf.psf_integrate(d["combined"], s0, s), A=psf.psf_integrate(d["alpha"], s0, s)))
        log(f"    {band} {o['name']:7s} crop {x1 - x0}x{y1 - y0} px @{s}x: {time.time() - t:.1f}s")
    _visibility(h, labelled_visible=False)
    return objects


def _horizon_probe(band, spec, h, cfgs, tmp, grey):
    """Sky radiance just above the geometric horizon, per image column (terrain hidden)."""
    sc = bpy.context.scene
    W = spec["camera"]["width"]
    _visibility(h, labelled_visible=False, env_hidden=True)
    bu.set_samples(8, adaptive=False)
    sc.camera = h.probe
    path = bu.render(os.path.join(tmp, f"{band}_probe.exr"), (64, 8))
    d = bu.read_passes(path, grey)
    bu.remove_file(path)
    sc.camera = h.cam
    _visibility(h, labelled_visible=False)
    C = pl.Camera(64, 8, spec["camera"]["hfov_deg"], spec["camera"]["position"], spec["camera"]["azimuth_deg"], 1.0)
    rows = [r for r in range(8) if pl.azimuth_elevation(C.ray(32, r + 0.5))[1] > 0.2]
    prof = d["env"][rows].mean(0)                                    # (64, C)
    x = (np.arange(W) + 0.5) / W * 64 - 0.5
    return np.stack([np.interp(x, np.arange(64), prof[:, c]) for c in range(prof.shape[1])], -1).astype(np.float32)


# ============================================================================ driver
def _log(msg):
    print(msg, flush=True)


def generate(idx: int, cfgs: dict, out: str, n_scenes: int, bands=BANDS, keep_tmp=False, exr=True, png=True, log=_log):
    t0 = time.time()
    spec, objs = sampling.sample_scene(idx, cfgs, n_scenes)
    sid = spec["scene_id"]
    tg = spec["objects"][0]
    log(f"[{sid}] {spec['split']} {spec['family']} {spec['lighting']} hfov {spec['camera']['hfov_deg']:g} "
        f"bin {tg['size_bin']} ({tg['size_px']:.1f} px @ {tg['range_m']:.0f} m) negatives "
        f"{[o['cls'] for o in spec['objects'][1:]]}")
    timings = {}
    t = time.time()
    calib = calibrate(spec, cfgs)
    timings["calibration"] = time.time() - t
    t = time.time()
    h = build(spec, objs, cfgs, calib)
    timings["build"] = time.time() - t
    tmp = os.path.join(out, "tmp", sid)
    os.makedirs(tmp, exist_ok=True)
    layers = {}
    for band in bands:
        t = time.time()
        layers[band] = render_band(band, spec, h, cfgs, tmp, log)
        timings[band] = time.time() - t
    spec.pop("_lwir_env", None)
    t = time.time()
    writer.write_scene(out, spec, layers, cfgs, calib, timings, exr=exr, png=png)
    timings["write"] = time.time() - t
    if not keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"[{sid}] done in {time.time() - t0:.1f}s  " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items()))
    return spec


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--indices", type=str, default=None, help="comma-separated scene indices (overrides start/count)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "dataset"))
    ap.add_argument("--n-scenes", type=int, default=None, help="dataset size used for the split (default: render.yaml)")
    ap.add_argument("--samples", type=int, default=None, help="override render.engine_samples")
    ap.add_argument("--bands", default="rgb,lwir,swir")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--no-exr", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="Cycles CPU threads (0 = all)")
    return ap.parse_args(argv)


def main():
    a = _args()
    cfgs = sampling.load_configs()
    if a.samples:
        cfgs["render"]["render"]["engine_samples"] = a.samples
        cfgs["render"]["render"]["adaptive_min_samples"] = min(cfgs["render"]["render"]["adaptive_min_samples"], a.samples)
    if a.threads:
        cfgs["render"]["render"]["threads"] = a.threads
    n = a.n_scenes or cfgs["render"]["dataset"]["n_scenes"]
    a.out = os.path.abspath(a.out)          # Blender resolves relative render paths differently
    os.makedirs(a.out, exist_ok=True)
    if bproc is not None:
        bproc.init()
    bands = tuple(b for b in a.bands.split(",") if b)
    failures = []
    todo = [int(x) for x in a.indices.split(",")] if a.indices else range(a.start, a.start + a.count)
    for idx in todo:
        if not a.overwrite and os.path.exists(os.path.join(a.out, "meta", f"{sampling.scene_id(idx)}.json")):
            _log(f"[{sampling.scene_id(idx)}] exists, skipped")
            continue
        try:
            generate(idx, cfgs, a.out, n, bands, a.keep_tmp, not a.no_exr, not a.no_png)
        except Exception as e:  # keep going; the failure is logged and the scene can be rerun
            import traceback
            traceback.print_exc()
            failures.append((idx, repr(e)))
    tag = f"{min(todo)}_{max(todo) + 1}" if len(todo) else "none"
    fail_path = os.path.join(a.out, f"failures_{tag}.json")
    if failures:
        with open(fail_path, "w") as f:
            json.dump(failures, f, indent=1)
        _log(f"FAILED: {failures}")
    elif os.path.exists(fail_path):
        os.remove(fail_path)                     # a clean rerun clears the old failure list


if __name__ == "__main__":
    main()

"""Scene sampler (pure NumPy): everything about a scene except the pixels.

sample_scene(idx, cfgs, n_scenes) returns
    spec   JSON-serialisable description (camera, sun, weather, signatures, placements, ...)
    objs   in-memory geometry for the renderer (target mesh, negative meshes, terrain, structures)

The same function runs inside Blender (render/scene.py) and in plain Python (tests, tools), so the
placement that decides a scene's pixel bin can be checked without rendering.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (ROOT, os.path.join(ROOT, "target")):
    if p not in sys.path:
        sys.path.insert(0, p)

import kal_geometry as kg  # noqa: E402
import kal_measure as km  # noqa: E402
from render import backgrounds as bg  # noqa: E402
from render import negatives as neg  # noqa: E402
from render import placement as pl  # noqa: E402
from render import signatures as sg  # noqa: E402

BANDS = ("rgb", "lwir", "swir")


# ============================================================================ configs
def load_yaml(path):
    """PyYAML if available, else target/_yamlite.py (Blender's bundled Python has no PyYAML)."""
    try:
        import yaml
    except ImportError:
        import _yamlite
        return _yamlite.load(path)
    with open(path) as f:
        return yaml.safe_load(f)


def load_configs(root: str = ROOT) -> dict:
    c = os.path.join(root, "configs")
    return dict(render=load_yaml(os.path.join(c, "render.yaml")),
                sensors=load_yaml(os.path.join(c, "sensors.yaml")),
                signatures=load_yaml(os.path.join(c, "signatures.yaml")),
                target=kg.load_config(os.path.join(c, "target.yaml")))


def scene_id(idx: int) -> str:
    return f"s{idx:06d}"


def assign_splits(n: int, fractions: dict, seed: int = 0) -> list[str]:
    """Deterministic scene-level split: a fixed permutation cut 70/15/15 (spec 6)."""
    order = np.random.default_rng(seed + 991).permutation(n)
    out = [""] * n
    names = list(fractions)
    cuts = np.cumsum([fractions[k] for k in names])
    cuts = np.round(cuts / cuts[-1] * n).astype(int)
    start = 0
    for name, stop in zip(names, cuts):
        for i in order[start:stop]:
            out[int(i)] = name
        start = stop
    return out


def _choice(rng, probs: dict):
    keys = list(probs)
    p = np.array([probs[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _logu(rng, a, b):
    return float(math.exp(rng.uniform(math.log(a), math.log(b))))


# ============================================================================ poses
def _target_pose_fn(tcfg):
    def f(rng):
        h = float(rng.uniform(*tcfg["heading_deg"]))
        if rng.uniform() < tcfg["level_flight_prob"]:
            p = r = 0.0
        else:
            p, r = float(rng.uniform(*tcfg["pitch_deg"])), float(rng.uniform(*tcfg["roll_deg"]))
        return pl.object_rotation(h, p, r), dict(heading_deg=h, pitch_deg=p, roll_deg=r)
    return f


def _uv_above(cam: pl.Camera, char_len: float, min_agl: float):
    """Ray sampler for the horizon family: only rows whose ray can clear `min_agl` at the distance
    implied by the requested size (distance ~ f * char_len / p). Rows below that would put the
    object underground and would only waste placement tries."""
    def f(rng, p_t):
        m = p_t / 2 + 2
        dist = cam.f * char_len / p_t
        s = (min_agl - cam.C[2]) / dist
        if s >= 0.999:
            return None
        el_min = math.degrees(math.asin(max(-0.999, s)))
        v_max = cam.H / 2 - cam.f * math.tan(math.radians(el_min - cam.elevation))
        v_hi = min(cam.H - m, v_max)
        if v_hi <= m:
            return None
        return rng.uniform(m, cam.W - m), rng.uniform(m, v_hi)
    return f


def _neg_pose_fn(cls, wind_dir_deg):
    lim = dict(bird=(20, 35), kite=(0, 0), light_aircraft=(8, 25), helicopter=(6, 8), small_uav=(12, 15),
               warm_clutter=(0, 0))[cls]

    def f(rng):
        h = float(rng.uniform(0, 360)) if cls != "kite" else float((wind_dir_deg + 180) % 360)
        p, r = float(rng.uniform(-lim[0], lim[0])), float(rng.uniform(-lim[1], lim[1]))
        return pl.object_rotation(h, p, r), dict(heading_deg=h, pitch_deg=p, roll_deg=r)
    return f


# ============================================================================ scene
def sample_scene(idx: int, cfgs: dict, n_scenes: int | None = None, attempts: int = 8):
    """Deterministic in idx. Geometry can make a bin impossible for a draw (a 64+ px KAL is only
    20-40 m from the wide camera, so at >= 20 m AGL it cannot appear in a horizon-looking frame).
    Such scenes are redrawn: fresh draws first, then the narrow camera, and as a last resort the
    open-sky family. `spec["resample"]` records what happened, so the balance can be audited."""
    last = None
    for attempt in range(attempts):
        force = {}
        if attempt >= attempts // 2:
            force["hfov"] = min(cfgs["sensors"]["hfov_deg"])
        if attempt >= attempts - 2:
            force["family"] = "open_sky"
        try:
            spec, objs = _sample_scene_once(idx, cfgs, n_scenes, attempt, force)
            spec["resample"] = dict(attempt=attempt, forced=force)
            return spec, objs
        except RuntimeError as e:
            last = e
    raise RuntimeError(f"scene {idx}: {last}")


def _sample_scene_once(idx: int, cfgs: dict, n_scenes, attempt: int, force: dict):
    R, S, SIG, TGT = cfgs["render"], cfgs["sensors"], cfgs["signatures"], cfgs["target"]
    ds = R["dataset"]
    n_scenes = n_scenes or ds["n_scenes"]
    seed = int(ds.get("seed0", 0)) + idx
    rng = np.random.default_rng(seed if attempt == 0 else [seed, attempt])
    edges = [tuple(e) for e in R["size_bins"]["edges_px"]]
    names = R["size_bins"]["names"]
    W, H = S["image"]["width"], S["image"]["height"]

    # ---- strata: bins balanced by index (spec 6), the rest drawn
    bin_idx = idx % len(edges)
    family = _choice(rng, ds["families"])
    lighting = _choice(rng, ds["lighting"])
    hfov = float(rng.choice(S["hfov_deg"], p=np.array(S["hfov_probs"]) / np.sum(S["hfov_probs"])))
    family = force.get("family", family)
    hfov = float(force.get("hfov", hfov))
    state = SIG["lwir"]["states"][lighting]

    # ---- camera
    cc = R["camera"]
    f_px = pl.focal_px(W, hfov)
    vfov = math.degrees(2 * math.atan(H / 2 / f_px))
    lo, hi = cc["look_elevation_deg"][family]
    if family == "open_sky":
        lo = max(lo, vfov / 2 + cc["open_sky_edge_margin_deg"])
    cam_el = float(rng.uniform(lo, hi))
    cam_az = float(rng.uniform(0, 360))
    cam_h = float(rng.uniform(*cc["height_m"]))
    cam = pl.Camera(W, H, hfov, (0.0, 0.0, cam_h), cam_az, cam_el, cc.get("roll_deg", 0.0))

    # ---- sun, moon, atmosphere, clouds
    sk = R["sky"]
    near_los = lighting != "night" and rng.uniform() < sk["sun_near_los_prob"]
    s_lo, s_hi = sk["sun_elevation_deg"][lighting]
    if near_los:
        cone = sk["sun_near_los_cone_deg"]
        sun_az = (cam_az + rng.uniform(-cone, cone)) % 360
        sun_el = float(np.clip(cam_el + rng.uniform(-cone, cone), s_lo, s_hi))
    else:
        sun_az, sun_el = float(rng.uniform(0, 360)), float(rng.uniform(s_lo, s_hi))
    atmo = dict(air=float(rng.uniform(*sk["air"])), dust=float(rng.uniform(*sk["dust"])),
                ozone=float(rng.uniform(*sk["ozone"])))
    moon = None
    if lighting == "night" and rng.uniform() < sk["night"]["moon_prob"]:
        moon = dict(elevation_deg=float(rng.uniform(*sk["night"]["moon_elevation_deg"])),
                    azimuth_deg=float(rng.uniform(0, 360)),
                    ratio=_logu(rng, *sk["night"]["moon_to_sun_ratio"]))
    clouds = None
    if rng.uniform() < sk["clouds"]["prob"]:
        c = sk["clouds"]
        clouds = dict(coverage=float(rng.uniform(*c["coverage"])), scale=float(rng.uniform(*c["scale"])),
                      softness=float(rng.uniform(*c["softness"])), brightness=float(rng.uniform(*c["brightness"])),
                      offset=[float(x) for x in rng.uniform(-100, 100, 3)])
    airglow_rgb = _logu(rng, *sk["night"]["airglow_rgb"])
    gtex = {b: dict(scale_m=_logu(rng, *R["ground"]["texture_scale_m"]),      # own pattern per band (spec 7.2)
                    contrast=float(rng.uniform(*R["ground"]["texture_contrast"])), seed=float(rng.uniform(0, 100)))
            for b in BANDS}

    # ---- air temperature, exposure, weather, PSF
    at = R["air_temp_c"]
    if lighting == "day" and rng.uniform() < at["desert_day_prob"]:
        air_c = float(rng.uniform(*at["desert_day"]))
    else:
        air_c = float(rng.uniform(*at["default"]))
    exposure_ms = float(rng.uniform(*S["exposure_ms"][lighting]))
    wc = R["weather"]
    wkind = _choice(rng, wc["probs"])
    vis_km = _logu(rng, *wc["visibility_km"][wkind])
    ratios = {b: float(sg.draw(rng, wc["band_ratio"][wkind][b])) for b in BANDS}
    psf = {b: dict(sigma0=float(S["psf_sigma_px"][b][0]), nominal=float(rng.uniform(*S["psf_sigma_px"][b])))
           for b in BANDS}

    # ---- terrain and structures
    gcfg = R["ground"]
    if family == "horizon":
        terrain = bg.Terrain(np.random.default_rng(seed + 17), gcfg, (0.0, 0.0))
        blocks = bg.sample_structures(np.random.default_rng(seed + 23), R["structures"], terrain,
                                      (0.0, 0.0), cam_az, hfov)
    else:
        terrain = bg.flat_terrain(gcfg, (0.0, 0.0))
        blocks = []

    # ---- target
    variant = kg.sample_variant(TGT, seed)
    tmesh = kg.build_mesh({**kg.defaults(TGT), **variant})
    Vt = kg.to_output_frame(tmesh)
    tc = R["target"]
    air_rng = tc["airspeed_mps"]["range"]
    airspeed = float(rng.uniform(*air_rng))
    wind_speed = float(rng.uniform(*tc["wind_mps"]))
    wind_dir = float(rng.uniform(0, 360))
    wind = wind_speed * pl.direction(wind_dir, 0.0)

    def valid_air(min_agl, max_agl=None, max_alt=None, others=()):
        def f(pos, dist, box):
            gz = float(terrain.ground_z(pos[0], pos[1]))
            if pos[2] < gz + min_agl:
                return False
            if max_agl is not None and pos[2] > gz + max_agl:
                return False
            if max_alt is not None and pos[2] > max_alt:
                return False
            if any(pl.boxes_overlap(box, o, R["negatives"]["min_gap_px"]) for o in others):
                return False
            return family != "horizon" or bg.los_clear(cam.C, pos, terrain, blocks)
        return f

    t_agl = tc["min_agl_m"] if family == "horizon" else 0.0
    t_char = 0.8 * float(np.ptp(Vt, axis=0).max())
    tpl = pl.place(cam, Vt, rng, edges, bin_idx, _target_pose_fn(tc),
                   valid_air(t_agl, max_alt=tc.get("max_altitude_m")),
                   uv_fn=_uv_above(cam, t_char, t_agl + 5.0) if family == "horizon" else None,
                   sampling=R["size_bins"]["sampling"], tries=tc["place_tries"])
    tpl.velocity = airspeed * (tpl.R @ np.array([1.0, 0, 0])) + wind
    exp_s = exposure_ms / 1000.0
    tpl.bbox_blur = pl.motion_boxes(cam, Vt, tpl, exp_s)

    # ---- hard negatives
    nc = R["negatives"]
    negs = []
    boxes = [tpl.bbox_blur]
    n_neg = int(rng.integers(nc["count"][0], nc["count"][1] + 1))
    for k in range(n_neg):
        probs = dict(nc["class_probs"])
        for cname, ccfg in nc["classes"].items():
            if family not in ccfg.get("families", ["open_sky", "horizon"]):
                probs.pop(cname, None)
        cls = _choice(rng, probs)
        ccfg = nc["classes"][cls]
        nb = int(rng.choice(len(edges), p=np.array(nc["bin_probs"]) / np.sum(nc["bin_probs"])))
        span = float(rng.uniform(*ccfg["span_m"]))
        mrng = np.random.default_rng(seed * 101 + k)
        mesh = neg.build(cls, mrng, span)
        Vn = mesh["vertices"]
        try:
            if cls == "warm_clutter":
                npl = _place_clutter(rng, cam, Vn, edges, nb, terrain, blocks, boxes, nc, R)
            else:
                agl = ccfg.get("min_agl_m", 0.0)
                npl = pl.place(cam, Vn, rng, edges, nb, _neg_pose_fn(cls, wind_dir),
                               valid_air(agl, ccfg.get("max_agl_m"), others=boxes),
                               uv_fn=_uv_above(cam, 0.8 * float(np.ptp(Vn, axis=0).max()), agl + 2.0)
                               if family == "horizon" else None,
                               sampling=R["size_bins"]["sampling"], tries=150)
        except RuntimeError:
            continue
        speed = float(rng.uniform(*ccfg["speed_mps"]))
        npl.velocity = speed * (npl.R @ np.array([1.0, 0, 0]))
        npl.bbox_blur = pl.motion_boxes(cam, Vn, npl, exp_s)
        boxes.append(npl.bbox_blur)
        negs.append(dict(cls=cls, span_m=span, mesh=mesh, placement=npl))

    # ---- signatures
    sigs = dict(target=sg.sample_target(rng, SIG, state),
                negatives=[sg.sample_negative(rng, SIG, n["cls"], state) for n in negs],
                environment=sg.sample_environment(rng, SIG, state))
    swir_hot = rng.uniform() < SIG["swir"]["hot_emission"]["enabled_prob"][lighting]

    # ---- crops (output px): union of the shutter boxes + 3 sigma of the widest PSF + margin
    sig_max = max(S["psf_sigma_px"][b][1] for b in BANDS)
    margin = int(math.ceil(3 * sig_max)) + int(R["render"]["crop_margin_px"])

    def crop(b):
        x0, y0 = max(0, int(math.floor(b[0])) - margin), max(0, int(math.floor(b[1])) - margin)
        x1, y1 = min(W, int(math.ceil(b[2])) + margin), min(H, int(math.ceil(b[3])) + margin)
        return [x0, y0, x1, y1]

    rc = R["render"]

    def ss_for(size):
        return int(rc["supersample_small_object"] if size < rc["small_object_px"] else rc["supersample_object"])

    def obj_entry(name, cls, p: pl.Placement, extra):
        az, el = pl.azimuth_elevation(p.position - cam.C)
        gz = float(terrain.ground_z(p.position[0], p.position[1]))
        return dict(name=name, cls=cls, class_id=neg.CLASSES.index(cls),
                    position=[float(x) for x in p.position], matrix_world=pl.matrix_world(p.R, p.position).tolist(),
                    velocity_mps=[float(x) for x in p.velocity], range_m=float(p.distance),
                    ray_azimuth_deg=az, elevation_deg=el, altitude_m=float(p.position[2]), agl_m=float(p.position[2] - gz),
                    bbox_px=_xywh(p.bbox), bbox_blur_px=_xywh(p.bbox_blur), size_px=float(p.size_px),
                    size_bin=names[p.bin], crop_px=crop(p.bbox_blur), supersample=ss_for(p.size_px),
                    pose=p.pose, **p.aspect(cam), **extra)

    objects = [obj_entry("target", "target", tpl, dict(
        airspeed_mps=airspeed, wind_mps=wind_speed, wind_dir_deg=wind_dir,
        projected_size_m=_projected_size(cam, Vt, tpl), hot_visible_cm2=_hot_visible(tmesh, cam, tpl)))]
    for k, n in enumerate(negs):
        objects.append(obj_entry(f"neg{k}", n["cls"], n["placement"], dict(span_m=n["span_m"],
                                                                         extra=n["mesh"]["extra"])))

    g = kg.derive({**kg.defaults(TGT), **variant})
    split = assign_splits(n_scenes, ds["splits"], int(ds.get("seed0", 0)))[idx] if idx < n_scenes else "extra"
    spec = dict(
        scene_id=scene_id(idx), idx=idx, seed=seed, split=split,
        family=family, lighting=lighting, thermal_state=state,
        camera=dict(hfov_deg=hfov, width=W, height=H, f_px=f_px, sensor_width_mm=S["sensor_width_mm"],
                    focal_mm=pl.focal_mm(S["sensor_width_mm"], hfov), position=[float(x) for x in cam.C],
                    azimuth_deg=cam_az, elevation_deg=cam_el, roll_deg=cam.roll, vfov_deg=vfov,
                    exposure_ms=exposure_ms, supersample_background=int(rc["supersample_background"]),
                    matrix_world=pl.matrix_world(cam.R, cam.C).tolist()),
        sun=dict(elevation_deg=sun_el, azimuth_deg=sun_az, near_line_of_sight=bool(near_los)),
        moon=moon, atmosphere=atmo, clouds=clouds, airglow_rgb=airglow_rgb, ground_texture=gtex,
        air_temp_c=air_c, weather=dict(kind=wkind, visibility_km=vis_km, band_ratio=ratios,
                                       aerosol_scale_height_m=wc.get("aerosol_scale_height_m")),
        psf=psf, swir_hot_emission=bool(swir_hot),
        target_variant=dict(seed=seed, variant_index=variant["variant_index"], variant_seed=variant["variant_seed"],
                            params=variant, span_m=g["span_m"], root_chord=g["root_chord"],
                            tip_over_root=g["tip_over_root"], nose_over_root=g["nose_over_root"],
                            sweep_deg=g["sweep_deg"], forebody_dia=g["forebody_dia"],
                            fin_up_height=g["fin_up_height"], fin_down_height=g["fin_down_height"],
                            prop_dia=g["prop_dia"], length_m=g["length"] * g["span_m"]),
        objects=objects, signatures=_jsonable(sigs),
        terrain=dict(hill_amplitude_m=terrain.A, hill_wavelength_m=terrain.lam, edge_ridge_m=terrain.ridge,
                     grid=terrain.grid, size_m=terrain.size) if family == "horizon" else None,
        structures=[b.__dict__ for b in blocks],
    )
    objs = dict(camera=cam, terrain=terrain, blocks=blocks, target_mesh=tmesh, target_verts=Vt,
                negatives=negs, target_placement=tpl)
    return spec, objs


def _place_clutter(rng, cam, Vn, edges, nb, terrain, blocks, boxes, nc, R):
    """Warm clutter on a roof (when a building is in view) or on the terrain."""
    def pose(r):
        h = float(r.uniform(0, 360))
        return pl.object_rotation(h), dict(heading_deg=h, pitch_deg=0.0, roll_deg=0.0)

    def valid(pos, dist, box):
        if any(pl.boxes_overlap(box, o, nc["min_gap_px"]) for o in boxes):
            return False
        top = pos + np.array([0, 0, np.ptp(Vn[:, 2]) * 0.5])
        others = [b for b in blocks if not (abs(b.cx - pos[0]) < b.w and abs(b.cy - pos[1]) < b.d)]
        return bg.los_clear(cam.C, top, terrain, others, clearance=0.2)

    buildings = [b for b in blocks if b.kind == "building"]
    if buildings and rng.uniform() < 0.5:
        for _ in range(40):
            b = buildings[int(rng.integers(len(buildings)))]
            P = b.roof_point(rng)
            R_, pose_d = pose(rng)
            Vw = Vn @ R_.T
            pos = P - np.array([0, 0, Vw[:, 2].min()])
            box = cam.bbox(Vw + pos)
            if box is None or not pl.box_inside(box, cam.W, cam.H, 1.0):
                continue
            p = pl.box_size(box)
            if pl.bin_index(p, edges) != nb or not valid(pos, 0, box):
                continue
            return pl.Placement(position=pos, R=R_, distance=float(np.linalg.norm(pos - cam.C)), bbox=box,
                                size_px=p, bin=nb, pose=pose_d)
    return pl.place_on_ground(cam, Vn, rng, edges, nb, terrain.ground_z, pose, valid,
                              sampling=R["size_bins"]["sampling"], tries=150)


def _hot_visible(mesh: dict, cam: pl.Camera, p: pl.Placement) -> dict:
    """Exhaust and engine area visible from this camera (ray cast on the target mesh, kal_measure):
    explains LWIR-weak targets, e.g. the forebody hides the exhaust head-on (spec 3.2)."""
    to_cam = cam.C - p.position
    d = p.R.T @ (to_cam / np.linalg.norm(to_cam))          # body frame = kal_measure's output frame
    ex = km.hot_visibility(mesh, d[None], ("exhaust_outlet",))[0]
    en = km.hot_visibility(mesh, d[None], ("engine_region",))[0]
    return dict(exhaust=float(ex * 1e4), engine=float(en * 1e4))


def _projected_size(cam, V, p: pl.Placement) -> dict:
    """Extent of the target perpendicular to the ray, in metres, along image x and y."""
    Vw = V @ p.R.T
    return dict(w=float(np.ptp(Vw @ cam.R[:, 0])), h=float(np.ptp(Vw @ cam.R[:, 1])))


def _xywh(b) -> dict:
    return dict(x=float(b[0]), y=float(b[1]), w=float(b[2] - b[0]), h=float(b[3] - b[1]))


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x

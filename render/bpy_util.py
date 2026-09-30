"""Blender helpers shared by the band modules and the scene driver (tested with Blender 4.2 LTS).

Conventions checked on this renderer (see README, "Renderer facts"):
  * world Texture Coordinate > Generated is the unit view direction;
  * Nishita sun_rotation is the compass azimuth (clockwise from north, +Y);
  * the Mist pass stores ray distance (not z-depth); the Env pass holds the world as seen by
    camera rays even with a transparent film;
  * border renders with use_crop_to_border are pixel-exact when the border is offset by +0.25 px.
"""
from __future__ import annotations

import math
import os

import bpy
import numpy as np
from mathutils import Matrix

from data.imgio import read_exr


# ============================================================================ scene / render
def reset():
    """Empty scene. Under BlenderProc the factory reset would discard its setup, so data-blocks are
    removed one by one instead."""
    import sys
    if "blenderproc" not in sys.modules:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        return
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
                 bpy.data.worlds, bpy.data.images, bpy.data.actions, bpy.data.node_groups):
        for block in list(coll):
            coll.remove(block)


def setup_cycles(rc: dict):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = int(rc["engine_samples"])
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = float(rc["adaptive_threshold"])
    sc.cycles.adaptive_min_samples = int(rc["adaptive_min_samples"])
    sc.cycles.use_denoising = False
    sc.cycles.filter_width = float(rc["filter_width_px"])
    lp = rc["light_paths"]
    sc.cycles.max_bounces = lp["total"]
    sc.cycles.diffuse_bounces = lp["diffuse"]
    sc.cycles.glossy_bounces = lp["glossy"]
    sc.cycles.transmission_bounces = lp["transmission"]
    sc.cycles.volume_bounces = lp["volume"]
    sc.cycles.transparent_max_bounces = lp["transparent"]
    sc.cycles.sample_clamp_direct = 0.0
    sc.cycles.sample_clamp_indirect = 0.0         # unbiased: sunlit ground bounce exceeds the default clamp of 10
    sc.cycles.seed = 0
    sc.render.use_persistent_data = bool(rc.get("persistent_data", True))
    if rc.get("threads"):
        sc.render.threads_mode = "FIXED"
        sc.render.threads = int(rc["threads"])
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.render.use_compositing = False
    sc.render.use_sequencer = False
    im = sc.render.image_settings
    im.file_format = "OPEN_EXR_MULTILAYER"
    im.color_depth = "32"
    im.exr_codec = rc.get("exr_codec", "ZIP")
    vl = sc.view_layers[0]
    vl.use_pass_combined = True
    vl.use_pass_mist = True
    vl.use_pass_environment = True
    vl.use_pass_z = False
    sc.render.motion_blur_shutter = 1.0
    sc.render.motion_blur_position = "CENTER"
    sc.frame_set(1)
    return sc


def set_samples(n: int, adaptive: bool = True):
    sc = bpy.context.scene
    sc.cycles.samples = int(n)
    sc.cycles.use_adaptive_sampling = adaptive


def set_bounces(total: int, rc: dict):
    sc = bpy.context.scene
    lp = rc["light_paths"]
    sc.cycles.max_bounces = total
    for k, key in (("diffuse", "diffuse_bounces"), ("glossy", "glossy_bounces"), ("transmission", "transmission_bounces")):
        setattr(sc.cycles, key, min(total, lp[k]))


def render(path: str, res, border=None, motion_blur: bool = False) -> str:
    """Render to a multilayer EXR. border = (x0, y0, x1, y1) in pixels of `res`, y down."""
    sc = bpy.context.scene
    W, H = int(res[0]), int(res[1])
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.resolution_percentage = 100
    if border is None:
        sc.render.use_border = False
    else:
        x0, y0, x1, y1 = border
        sc.render.use_border = True
        sc.render.use_crop_to_border = True
        sc.render.border_min_x = (x0 + 0.25) / W
        sc.render.border_max_x = (x1 + 0.25) / W
        sc.render.border_min_y = (H - y1 + 0.25) / H
        sc.render.border_max_y = (H - y0 + 0.25) / H
    sc.render.use_motion_blur = bool(motion_blur)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def read_passes(path: str, grey: bool) -> dict:
    """Combined (premultiplied), alpha, Env and Mist from a multilayer EXR; grey keeps one channel."""
    d = read_exr(path)

    def ch(tail):
        for k, v in d.items():
            if k.endswith(tail):
                return v
        return None
    rgb = ("R",) if grey else ("R", "G", "B")
    out = dict(combined=np.stack([ch(f"Combined.{c}") for c in rgb], -1), alpha=ch("Combined.A"))
    env = [ch(f"Env.{c}") for c in rgb]
    if env[0] is not None:
        out["env"] = np.stack(env, -1)
    if ch("Mist.Z") is not None:
        out["mist"] = ch("Mist.Z")
    return out


# ============================================================================ objects
def mesh_object(name: str, V, F, face_slot_idx=None, n_slots: int = 1, smooth=False, collection=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(V, float).tolist(), [], [list(map(int, f)) for f in F])
    for _ in range(n_slots):
        me.materials.append(None)
    if face_slot_idx is not None:
        me.polygons.foreach_set("material_index", np.asarray(face_slot_idx, np.int32))
    if smooth is not False:
        vals = [bool(smooth)] * len(F) if isinstance(smooth, bool) else list(map(bool, smooth))
        me.polygons.foreach_set("use_smooth", vals)
    me.validate(clean_customdata=False)
    me.update()
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def set_matrix(obj, M):
    obj.matrix_world = Matrix(np.asarray(M, float).tolist())


def linear_motion(obj, M_mid, velocity, exposure_s: float):
    """Keyframe a straight-line move so a 1-frame shutter centred on frame 1 spans the exposure."""
    M_mid = np.asarray(M_mid, float)
    v = np.asarray(velocity, float)
    for fr, sgn in ((0, -1.0), (2, 1.0)):
        M = M_mid.copy()
        M[:3, 3] += sgn * v * exposure_s
        set_matrix(obj, M)
        obj.keyframe_insert("location", frame=fr)
    for fc in obj.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        fc.extrapolation = "LINEAR"
    bpy.context.scene.frame_set(1)


def assign_materials(obj, mats: list):
    for i, m in enumerate(mats):
        obj.data.materials[i] = m


# ============================================================================ node helpers
def new_material(name: str):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return m, nt, out


def math_node(nt, op, a=None, b=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = float(v)
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def principled(nt, color, roughness, metallic=0.0, coat=0.0, alpha=1.0):
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    if isinstance(color, (list, tuple, np.ndarray)):
        c = list(color) + [1.0] if len(color) == 3 else list(color)
        b.inputs["Base Color"].default_value = c
    else:
        nt.links.new(color, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = float(roughness)
    b.inputs["Metallic"].default_value = float(metallic)
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = float(coat)
    if isinstance(alpha, (int, float)):
        b.inputs["Alpha"].default_value = float(alpha)
    else:
        nt.links.new(alpha, b.inputs["Alpha"])
    return b


def texture_factor(nt, scale_m: float, contrast: float, seed: float = 0.0):
    """Multiplicative 1 +- contrast pattern in object coordinates (metres): terrain variation."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["W"].default_value = float(seed)
    noise.inputs["Scale"].default_value = 1.0 / float(scale_m)
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.55
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.3
    mr.inputs["From Max"].default_value = 0.7
    mr.inputs["To Min"].default_value = 1.0 - contrast
    mr.inputs["To Max"].default_value = 1.0 + contrast
    nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
    return mr.outputs["Result"]


def world_direction(nt):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    return tc.outputs["Generated"], sep.outputs["Z"]


def cloud_mask(nt, clouds: dict | None):
    """0..1 cloud cover as a function of view direction (procedural: resolution independent)."""
    if not clouds:
        return None
    vec, z = world_direction(nt)
    add = nt.nodes.new("ShaderNodeVectorMath")
    add.operation = "MULTIPLY_ADD"
    add.inputs[1].default_value = (clouds["scale"],) * 3
    add.inputs[2].default_value = tuple(clouds["offset"])
    nt.links.new(vec, add.inputs[0])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6
    nt.links.new(add.outputs[0], noise.inputs["Vector"])
    thr = 0.62 - 0.24 * clouds["coverage"]            # noise Fac is centred on 0.5
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = thr
    mr.inputs["From Max"].default_value = thr + clouds["softness"]
    nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
    fade = nt.nodes.new("ShaderNodeMapRange")          # thin out toward the horizon, none below it
    fade.interpolation_type = "SMOOTHSTEP"
    fade.inputs["From Min"].default_value = 0.0
    fade.inputs["From Max"].default_value = 0.12
    nt.links.new(z, fade.inputs["Value"])
    return math_node(nt, "MULTIPLY", mr.outputs["Result"], fade.outputs["Result"])


def mix_rgb(nt, fac, a, b):
    m = nt.nodes.new("ShaderNodeMix")
    m.data_type = "RGBA"
    if fac is None:
        m.inputs["Factor"].default_value = 0.0
    elif isinstance(fac, (int, float)):
        m.inputs["Factor"].default_value = float(fac)
    else:
        nt.links.new(fac, m.inputs["Factor"])
    for sock, v in ((m.inputs["A"], a), (m.inputs["B"], b)):
        if isinstance(v, (list, tuple, np.ndarray)):
            sock.default_value = list(v)[:3] + [1.0]
        else:
            nt.links.new(v, sock)
    return m.outputs["Result"]


def world(name: str):
    w = bpy.data.worlds.new(name)
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bgn = nt.nodes.new("ShaderNodeBackground")
    bgn.inputs["Strength"].default_value = 1.0
    nt.links.new(bgn.outputs["Background"], out.inputs["Surface"])
    return w, nt, bgn


def sky_node(nt, elevation_deg: float, azimuth_deg: float, atmo: dict):
    s = nt.nodes.new("ShaderNodeTexSky")
    types = [i.identifier for i in s.bl_rna.properties["sky_type"].enum_items]
    s.sky_type = "NISHITA" if "NISHITA" in types else "SINGLE_SCATTERING"   # renamed in Blender 5.x
    s.sun_disc = False
    s.sun_elevation = math.radians(elevation_deg)
    s.sun_rotation = math.radians(azimuth_deg % 360.0)      # compass azimuth (checked on this renderer)
    s.air_density = atmo["air"]
    s.dust_density = atmo["dust"]
    s.ozone_density = atmo["ozone"]
    return s


def sun_lamp(name: str, elevation_deg: float, azimuth_deg: float, strength: float, color=(1, 1, 1), angle_deg=0.526):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = float(strength)
    ld.color = tuple(float(c) for c in color)
    ld.angle = math.radians(angle_deg)
    obj = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(obj)
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    to_sun = np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])
    z = to_sun                                  # lamp shines along its local -Z
    x = np.cross([0.0, 0.0, 1.0], z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-9 else np.array([1.0, 0, 0])
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = x, y, z
    set_matrix(obj, M)
    return obj


def float_image(name: str, values: np.ndarray):
    """1-row float image (RGBA) used as a lookup table in shaders."""
    v = np.asarray(values, np.float32)
    if v.ndim == 1:
        v = np.repeat(v[:, None], 3, 1)
    n = len(v)
    img = bpy.data.images.new(name, n, 1, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    px = np.ones((n, 4), np.float32)
    px[:, :3] = v
    img.pixels.foreach_set(px.ravel())
    img.update()
    return img


def remove_file(path: str):
    try:
        os.remove(path)
    except OSError:
        pass

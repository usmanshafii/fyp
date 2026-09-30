"""build_target(params) -> Blender object for the KAL target.

Geometry comes from kal_geometry.py (pure NumPy), so what Blender renders is the
same mesh the reference drawing and the photo check use.

Run inside Blender:   blender -b -P target/build_target.py -- --variants 5 --out out/phase1
Or with the bpy module:   python target/build_target.py --variants 5 --out out/phase1

Output frame: metres, +X forward, +Y port, +Z up, origin on the fuselage axis midway
between the nose tip and the propeller hub. Tested with Blender/bpy 4.2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kal_geometry as kg  # noqa: E402

ASSETS = os.path.join(HERE, "..", "assets", "decals")

# Neutral RGB preview values. Band-specific values (RGB/LWIR/SWIR) are assigned per
# slot by the Phase 2-4 pipelines from signatures.yaml; only slot names matter here.
PREVIEW = {
    "skin":            dict(color=(0.035, 0.037, 0.040), rough=0.45, coat=0.3),
    "control_surface": dict(color=(0.035, 0.037, 0.040), rough=0.45, coat=0.3),
    "fin":             dict(color=(0.035, 0.037, 0.040), rough=0.45, coat=0.3),
    "fuselage":        dict(color=(0.035, 0.037, 0.040), rough=0.40, coat=0.4),
    "tail_cone":       dict(color=(0.035, 0.037, 0.040), rough=0.40, coat=0.4),
    "markings":        dict(color=(0.80, 0.80, 0.80), rough=0.5),
    "decal":           dict(color=(0.80, 0.80, 0.80), rough=0.5, texture=True),
    "engine_region":   dict(color=(0.30, 0.30, 0.31), rough=0.35, metal=0.8),
    "exhaust_outlet":  dict(color=(0.12, 0.11, 0.10), rough=0.6, metal=0.6),
    "propeller":       dict(color=(0.03, 0.03, 0.03), rough=0.4),
    "propeller_disc":  dict(color=(0.03, 0.03, 0.03), rough=0.4, alpha=0.25),
    "antenna":         dict(color=(0.03, 0.03, 0.03), rough=0.5),
    "nose_window":     dict(color=(0.02, 0.02, 0.03), rough=0.05),
}


def _material(slot: str) -> bpy.types.Material:
    name = f"KAL_{slot}"
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    spec = PREVIEW[slot]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*spec["color"], 1.0)
    bsdf.inputs["Roughness"].default_value = spec["rough"]
    bsdf.inputs["Metallic"].default_value = spec.get("metal", 0.0)
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = spec.get("coat", 0.0)
    if "alpha" in spec:
        bsdf.inputs["Alpha"].default_value = spec["alpha"]
        mat.blend_method = "BLEND"
    if spec.get("texture"):
        atlas = os.path.join(ASSETS, "decal_atlas.png")
        mat.blend_method = "BLEND"
        if os.path.exists(atlas):
            tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
            tex.image = bpy.data.images.load(atlas, check_existing=True)
            tex.image.pack()          # saved .blend files stay self-contained
            tex.interpolation = "Cubic"
            mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
            mat.node_tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
        else:  # never render a blank white rectangle
            bsdf.inputs["Alpha"].default_value = 0.0
    mat["kal_slot"] = slot
    return mat


def _atlas_regions() -> dict:
    path = os.path.join(ASSETS, "decal_atlas.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def build_target(params: dict | None = None, cfg_path: str | None = None, name: str = "KAL",
                 collection: bpy.types.Collection | None = None) -> bpy.types.Object:
    """Build the target from YAML defaults overridden by `params` (e.g. a sampled variant)."""
    cfg = kg.load_config(cfg_path or kg.DEFAULT_YAML)
    p = {**kg.defaults(cfg), **(params or {})}
    mesh = kg.build_mesh(p)
    V = kg.to_output_frame(mesh)

    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], [list(f) for f in mesh["faces"]])
    for slot in kg.SLOTS:
        me.materials.append(_material(slot))
    slots = mesh["face_slots"].astype(np.int32)
    me.polygons.foreach_set("material_index", slots)
    me.polygons.foreach_set("use_smooth", [kg.SLOTS[s] in kg.SMOOTH_SLOTS for s in slots])
    if hasattr(me, "set_sharp_from_angle"):
        me.set_sharp_from_angle(angle=math.radians(40))

    # UVs: decal quads map into their atlas region; every other face gets (0, 0)
    uv = me.uv_layers.new(name="UVMap")
    regions = _atlas_regions()
    kinds = {"wing_text_stbd": "wing_text", "wing_text_port": "wing_text",
             "fin_text_stbd": "fin_text", "fin_text_port": "fin_text", "flag": "flag"}
    for fi, corners in mesh["decal_uv"].items():
        part = mesh["face_parts"][fi]
        u0, v0, u1, v1 = regions.get(kinds.get(part, ""), (0, 0, 1, 1))
        poly = me.polygons[fi]
        for k, li in enumerate(poly.loop_indices):
            cu, cv = corners[k]
            uv.data[li].uv = (u0 + cu * (u1 - u0), v0 + cv * (v1 - v0))
    me.validate(clean_customdata=False)
    me.update()

    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(obj)
    obj["kal_params"] = json.dumps({k: v for k, v in p.items()}, default=float)
    obj["kal_summary"] = json.dumps(kg.summary(mesh["derived"]))
    return obj


# ============================================================================ previews
def _clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _preview_scene(res=(1200, 700)):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 32
    sc.cycles.use_denoising = False
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = False
    world = bpy.data.worlds.new("preview")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.9, 0.92, 0.95, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    sc.world = world
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(35), math.radians(-20), math.radians(30))
    sc.collection.objects.link(sun)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    cam.data.clip_end = 100
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def render_previews(obj, out_dir, tag):
    """Orthographic top / side / front / rear previews at one common scale."""
    os.makedirs(out_dir, exist_ok=True)
    cam = bpy.context.scene.camera
    r = bpy.context.scene.render
    aspect = r.resolution_y / r.resolution_x
    bb = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
    ext = np.ptp(bb, axis=0)
    # one common scale for all three views: the largest of length, span/aspect, height/aspect
    cam.data.ortho_scale = float(max(ext[0], ext[1], ext[1] / aspect, ext[2] / aspect)) * 1.08
    views = {  # name: (location, rotation)
        "top":   ((0, 0, 20), (0, 0, math.pi)),              # nose left, starboard up
        "side":  ((0, 20, 0), (math.radians(90), 0, math.pi)),  # from port, nose left
        "front": ((20, 0, 0), (math.radians(90), 0, math.radians(90))),  # from ahead, port on the right
        "rear":  ((-20, 0, 0), (math.radians(90), 0, -math.radians(90))),  # from behind, starboard on the right
    }
    for view, (loc, rot) in views.items():
        cam.location, cam.rotation_euler = loc, rot
        bpy.context.scene.render.filepath = os.path.join(out_dir, f"{tag}_{view}.png")
        bpy.ops.render.render(write_still=True)


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="out/phase1")
    ap.add_argument("--no-render", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = _args()
    cfg = kg.load_config()
    os.makedirs(a.out, exist_ok=True)
    jobs = [("canonical", None)] + [(f"variant_{a.seed0 + i:03d}", kg.sample_variant(cfg, a.seed0 + i)) for i in range(a.variants)]
    for tag, params in jobs:
        _clear_scene()
        _preview_scene()
        obj = build_target(params, name=f"KAL_{tag}")
        print(tag, obj["kal_summary"])
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(a.out, f"{tag}.blend")))
        if not a.no_render:
            render_previews(obj, a.out, tag)

"""SWIR band (Phase 4, teacher only), 0.9-1.7 um: greyscale reflectance pass (spec 4 step 5).

Each material's base colour is its SWIR reflectance. The sun is kept (the grey lamp uses the red
channel of the calibrated Nishita sun, the longest visible wavelength); the sky is the red channel
of the same Nishita sky divided by 3-30. At night a dim uniform airglow replaces the sun.
Optional self-emission for parts above ~250 C: Planck radiance in 0.9-1.7 um with emissivity
1 - reflectance, converted to render units with the in-band solar irradiance (signatures.yaml).
"""
from __future__ import annotations

import numpy as np

from render import bpy_util as bu
from render.rgb import ATLAS, ATLAS_WHITE
from thermal.planck_lut import swir_lut

K0 = 273.15


def hot_emission(p_swir: dict, T_kelvin: float, spec: dict, sig: dict) -> float:
    he = sig["swir"]["hot_emission"]
    if not spec.get("swir_hot_emission") or T_kelvin < he["min_temp_c"] + K0:
        return 0.0
    eps = 1.0 - p_swir["reflectance"]
    units = he["nishita_sun_units"] / he["inband_solar_irradiance"]
    return float(eps * swir_lut().radiance(min(T_kelvin, 1200.0)) * units)


def _surface(nt, p, alpha=None, emission=0.0):
    import bpy  # noqa: F401
    b = bu.principled(nt, [p["reflectance"]] * 3, p.get("roughness", 0.5), p.get("metallic", 0.0), p.get("coat", 0.0),
                      1.0 if alpha is None else alpha)
    sh = b.outputs[0]
    if emission > 0:
        e = nt.nodes.new("ShaderNodeEmission")
        e.inputs["Strength"].default_value = emission
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(sh, add.inputs[0])
        nt.links.new(e.outputs[0], add.inputs[1])
        sh = add.outputs[0]
    return sh


def target_material(name: str, slot: str, p: dict, emission: float = 0.0):
    import bpy
    m, nt, out = bu.new_material(name)
    if slot == "decal":
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(ATLAS, check_existing=True)
        tex.interpolation = "Cubic"
        sc = nt.nodes.new("ShaderNodeVectorMath")
        sc.operation = "SCALE"
        sc.inputs["Scale"].default_value = p["reflectance"] / ATLAS_WHITE
        nt.links.new(tex.outputs["Color"], sc.inputs[0])
        b = bu.principled(nt, sc.outputs[0], p.get("roughness", 0.5), alpha=tex.outputs["Alpha"])
        sh = b.outputs[0]
    else:
        sh = _surface(nt, p, p.get("alpha") if slot == "propeller_disc" else None, emission)
    nt.links.new(sh, out.inputs["Surface"])
    return m


def negative_material(name: str, slot: str, p: dict, emission: float = 0.0):
    m, nt, out = bu.new_material(name)
    nt.links.new(_surface(nt, p, p.get("alpha"), emission), out.inputs["Surface"])
    return m


def environment_material(name: str, p: dict, texture: dict | None = None):
    m, nt, out = bu.new_material(name)
    if texture:
        f = bu.texture_factor(nt, texture["scale_m"], texture["contrast"], texture.get("seed", 0.0))
        col = bu.math_node(nt, "MULTIPLY", f, p["reflectance"])
        b = bu.principled(nt, [1, 1, 1], 0.9)
        cmb = nt.nodes.new("ShaderNodeCombineColor")
        for k in ("Red", "Green", "Blue"):
            nt.links.new(col, cmb.inputs[k])
        nt.links.new(cmb.outputs[0], b.inputs["Base Color"])
        if "Specular IOR Level" in b.inputs:
            b.inputs["Specular IOR Level"].default_value = 0.0  # ground: diffuse only, as in RGB
    else:
        b = bu.principled(nt, [p["reflectance"]] * 3, 0.85)
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def world(spec: dict, env: dict, calib: dict):
    w, nt, bgn = bu.world("swir_world")
    atmo = spec["atmosphere"]
    div = env["swir"]["sky_divisor"]
    clouds = spec.get("clouds")
    if spec["lighting"] == "night":
        glow = env["swir"]["night_airglow"]
        base = [glow] * 3
        sky = None
        if spec.get("moon"):
            mo = spec["moon"]
            s = bu.sky_node(nt, mo["elevation_deg"], mo["azimuth_deg"], atmo)
            sep = nt.nodes.new("ShaderNodeSeparateColor")
            nt.links.new(s.outputs["Color"], sep.inputs[0])
            r = bu.math_node(nt, "MULTIPLY_ADD", sep.outputs["Red"], mo["ratio"] / div)
            r.node.inputs[2].default_value = glow
            sky = r
            cloud = float(calib["moon"]["plane_disc"][0]) * mo["ratio"] + glow
        else:
            cloud = glow * 1.2
    else:
        s = bu.sky_node(nt, spec["sun"]["elevation_deg"], spec["sun"]["azimuth_deg"], atmo)
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(s.outputs["Color"], sep.inputs[0])
        sky = bu.math_node(nt, "MULTIPLY", sep.outputs["Red"], 1.0 / div)
        cloud = float(calib["sun"]["plane_disc"][0])
        base = None
    if clouds:
        cloud = cloud * env["swir"]["cloud_brightness"] * env["swir"]["cloud_brightness_scale"]   # own draw, not RGB's
    mask = bu.cloud_mask(nt, clouds)
    if sky is None and mask is None:
        bgn.inputs["Color"].default_value = list(base) + [1.0]
        return w
    if sky is None:
        col = bu.mix_rgb(nt, mask, list(base), [cloud] * 3)
    else:
        cmb = nt.nodes.new("ShaderNodeCombineColor")
        for k in ("Red", "Green", "Blue"):
            nt.links.new(sky, cmb.inputs[k])
        col = cmb.outputs[0] if mask is None else bu.mix_rgb(nt, mask, cmb.outputs[0], [cloud] * 3)
    nt.links.new(col, bgn.inputs["Color"])
    return w


def lamps(spec: dict, calib: dict) -> list:
    out = []
    if spec["lighting"] != "night":
        E = float(calib["sun"]["sun_irradiance"][0])
        if E > 1e-6:
            out.append(bu.sun_lamp("swir_sun", spec["sun"]["elevation_deg"], spec["sun"]["azimuth_deg"], E))
    elif spec.get("moon"):
        mo = spec["moon"]
        E = float(calib["moon"]["sun_irradiance"][0]) * mo["ratio"]
        if E > 0:
            out.append(bu.sun_lamp("swir_moon", mo["elevation_deg"], mo["azimuth_deg"], E))
    return out

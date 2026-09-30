"""LWIR band (Phase 3), 8-14 um: emission-only pass, no lights, no bounces (spec 4 step 4).

Every material becomes Emission with strength = emissivity x in-band radiance of the slot's
temperature (ambient air + dT), from the Planck lookup table. Surfaces with emissivity < 1 also
reflect their surroundings; that term, (1 - e) x L_env, is added in the shader with L_env blended
from the sky (normal up) to the ground (normal down), so the pass stays emission-only.

The world carries the clear-sky radiance as a function of elevation (a 1-D lookup image), with the
same procedural clouds as the reflective bands. Values are W m^-2 sr^-1.
"""
from __future__ import annotations

import math

import bpy
import numpy as np

from render import bpy_util as bu
from thermal.planck_lut import lwir_lut

K0 = 273.15
LUT_N = 2048


def L(T_kelvin):
    return lwir_lut().radiance(T_kelvin)


def sky_profile(spec: dict, env: dict) -> tuple[np.ndarray, np.ndarray]:
    """(elevation deg, radiance) for the clear sky: dT(el) = dTz + (dTh - dTz) exp(-el / shape)."""
    Ta = spec["air_temp_c"] + K0
    s = env["sky"]
    el = ((np.arange(LUT_N) + 0.5) / LUT_N - 0.5) * 180.0
    dT = s["dT_zenith"] + (s["dT_horizon"] - s["dT_zenith"]) * np.exp(-np.clip(el, 0, None) / s["shape_deg"])
    rad = L(np.clip(Ta + dT, 200.0, 800.0))
    g = env["ground"]
    rad = np.where(el < 0, g["emissivity"] * L(Ta + g["dT"]), rad)      # below horizon: ground
    return el, rad


def environment_radiance(spec: dict, env: dict) -> dict:
    """Hemispherical sky and ground radiances used by the reflection term and the haze model."""
    Ta = spec["air_temp_c"] + K0
    el, rad = sky_profile(spec, env)
    up = el >= 0
    th = np.radians(90.0 - el[up])                 # zenith angle
    w = np.cos(th) * np.sin(th)
    L_up = float((rad[up] * w).sum() / w.sum())
    clouds = spec.get("clouds")
    L_cloud = float(L(Ta + env["clouds"]["dT"]))
    if clouds:
        L_up = (1 - clouds["coverage"]) * L_up + clouds["coverage"] * L_cloud
    g = env["ground"]
    L_ground = float(g["emissivity"] * L(Ta + g["dT"]) + (1 - g["emissivity"]) * L_up)
    return dict(L_up=L_up, L_down=L_ground, L_cloud=L_cloud, L_air=float(L(Ta)),
                L_horizon=float(np.interp(0.5, el, rad)))


def _strength(nt, emissive: float, eps: float, envr: dict, reflection: bool):
    """Emission strength socket: e L(T) + (1 - e) lerp(L_down, L_up, (n_z + 1) / 2)."""
    if not reflection or eps >= 0.999:
        return float(emissive)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], sep.inputs["Vector"])
    t = bu.math_node(nt, "MULTIPLY_ADD", sep.outputs["Z"], 0.5)
    t.node.inputs[2].default_value = 0.5
    k = (1 - eps) * (envr["L_up"] - envr["L_down"])
    refl = bu.math_node(nt, "MULTIPLY_ADD", t, k)
    refl.node.inputs[2].default_value = emissive + (1 - eps) * envr["L_down"]
    return refl


def _emission(nt, strength):
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (1, 1, 1, 1)
    if isinstance(strength, (int, float)):
        e.inputs["Strength"].default_value = float(strength)
    else:
        nt.links.new(strength, e.inputs["Strength"])
    return e.outputs[0]


def _with_alpha(nt, shader, alpha: float):
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = float(alpha)
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(shader, mix.inputs[2])
    return mix.outputs[0]


def slot_material(name: str, p: dict, spec: dict, envr: dict, reflection: bool = True, alpha: float | None = None,
                  transparent: bool = False):
    m, nt, out = bu.new_material(name)
    if transparent:            # decal quads: "as skin" (spec) -> show the skin underneath
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        nt.links.new(tr.outputs[0], out.inputs["Surface"])
        return m
    T = spec["air_temp_c"] + K0 + p["dT"]
    eps = p["emissivity"]
    sh = _emission(nt, _strength(nt, eps * float(L(T)), eps, envr, reflection))
    if alpha is not None:
        sh = _with_alpha(nt, sh, alpha)
    nt.links.new(sh, out.inputs["Surface"])
    return m


def target_material(name, slot, p, spec, envr, reflection=True, rgb_params=None):
    alpha = rgb_params["alpha"] if slot == "propeller_disc" else None
    return slot_material(name, p, spec, envr, reflection, alpha=alpha, transparent=(slot == "decal"))


def negative_material(name, slot, p, spec, envr, reflection=True, rgb_params=None):
    alpha = (rgb_params or {}).get("alpha")
    return slot_material(name, p, spec, envr, reflection, alpha=alpha)


def ground_material(name: str, p: dict, spec: dict, envr: dict, texture: dict, reflection=True):
    """Ground with a procedural +-texture_dT pattern (linearised: L(T + d) ~ L(T) + dL/dT d)."""
    m, nt, out = bu.new_material(name)
    T = spec["air_temp_c"] + K0 + p["dT"]
    eps = p["emissivity"]
    L0, dLdT = float(L(T)), float(lwir_lut().dL_dT(T))
    f = bu.texture_factor(nt, texture["scale_m"], 1.0, texture.get("seed", 0.0))     # 0..2
    dev = bu.math_node(nt, "SUBTRACT", f, 1.0)                                        # -1..1
    emis = bu.math_node(nt, "MULTIPLY_ADD", dev, eps * dLdT * p["texture_dT"])
    emis.node.inputs[2].default_value = eps * L0
    refl = (1 - eps) * envr["L_up"] if reflection else 0.0
    s = bu.math_node(nt, "ADD", emis, refl)
    nt.links.new(_emission(nt, s), out.inputs["Surface"])
    return m


def world(spec: dict, env: dict, envr: dict):
    w, nt, bgn = bu.world("lwir_world")
    el, rad = sky_profile(spec, env)
    img = bu.float_image("lwir_sky_lut", rad)
    vec, z = bu.world_direction(nt)
    asin = bu.math_node(nt, "ARCSINE", z)
    u = bu.math_node(nt, "MULTIPLY_ADD", asin, 1.0 / math.pi)
    u.node.inputs[2].default_value = 0.5
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    comb.inputs["Y"].default_value = 0.5
    nt.links.new(u, comb.inputs["X"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"
    tex.extension = "EXTEND"
    nt.links.new(comb.outputs[0], tex.inputs["Vector"])
    mask = bu.cloud_mask(nt, spec.get("clouds"))
    col = tex.outputs["Color"] if mask is None else bu.mix_rgb(nt, mask, tex.outputs["Color"], [envr["L_cloud"]] * 3)
    nt.links.new(col, bgn.inputs["Color"])
    return w

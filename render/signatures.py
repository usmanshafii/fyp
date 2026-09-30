"""Per-scene draws of the band signatures in configs/signatures.yaml (pure NumPy).

Slots that point at the same group share one draw within a band. dT entries keyed by thermal
state are resolved with the scene's state. Bands share geometry only (spec 7.2 rule 4): every
band's values are separate draws, including the SWIR finish and the temperature used for SWIR
self-emission of hot parts (`emit_dT`, drawn from the LWIR range but not the LWIR value). The
fraction of a propeller or rotor disc swept by blades (`alpha`) is geometry and is shared.
"""
from __future__ import annotations

import colorsys

import numpy as np


def _is_range(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)


def draw(rng, v):
    return float(rng.uniform(v[0], v[1])) if _is_range(v) else v


def draw_group(rng, spec: dict, state: str | None = None) -> dict:
    out = {}
    for k, v in spec.items():
        if isinstance(v, dict) and state in v:
            v = v[state]
        out[k] = draw(rng, v)
    return out


def _state_range(v, state):
    return v[state] if isinstance(v, dict) else v


def sample_target(rng, sig: dict, state: str) -> dict:
    """{band: {slot: params}} for the KAL target."""
    out = {}
    for band in ("rgb", "lwir", "swir"):
        t = sig[band]["target"]
        groups = {g: draw_group(rng, s, state) for g, s in t["groups"].items()}
        out[band] = {slot: dict(groups[g], group=g) for slot, g in t["slots"].items()}
    alpha = draw(rng, sig["rgb"]["target"]["propeller_disc_alpha"])      # blade coverage: geometry
    out["rgb"]["propeller_disc"]["alpha"] = out["swir"]["propeller_disc"]["alpha"] = alpha
    lw = sig["lwir"]["target"]
    emit = {g: draw(rng, _state_range(s["dT"], state)) for g, s in lw["groups"].items()}   # own draws
    for slot, p in out["swir"].items():
        p["emit_dT"] = emit[lw["slots"][slot]]
    return out


def sample_negative(rng, sig: dict, cls: str, state: str) -> dict:
    """{band: {slot: params}} for one hard-negative object."""
    out = {}
    for band in ("rgb", "lwir", "swir"):
        spec = sig[band]["negatives"][cls]
        d = {}
        for slot, s in spec.items():
            if "same_as" in s:
                continue
            p = draw_group(rng, s, state)
            if band == "rgb":
                if p.pop("saturated", False):
                    h, sat = rng.uniform(), rng.uniform(0.6, 1.0)
                    p["color"] = list(colorsys.hsv_to_rgb(h, sat, 1.0))
                else:
                    tint = rng.uniform(0.9, 1.1, 3)
                    p["color"] = list(np.clip(tint / tint.mean(), 0, 2))
            d[slot] = p
        for slot, s in spec.items():
            if "same_as" in s:
                d[slot] = dict(d[s["same_as"]])
        out[band] = d
    lw = sig["lwir"]["negatives"][cls]
    for slot, p in out["swir"].items():
        if "alpha" in out["rgb"].get(slot, {}):          # rotor / propeller coverage: geometry
            p["alpha"] = out["rgb"][slot]["alpha"]
        src = lw.get(slot) or lw.get(sig["swir"]["negatives"][cls].get(slot, {}).get("same_as", ""), {})
        if "dT" in src:
            p["emit_dT"] = draw(rng, _state_range(src["dT"], state))   # own draw, not the LWIR value
    return out


def sample_environment(rng, sig: dict, state: str) -> dict:
    env = {}
    r = sig["rgb"]["environment"]
    g = draw_group(rng, r["ground"])
    hue = rng.choice([[1.08, 1.0, 0.82], [0.9, 1.0, 0.78], [1.0, 1.0, 1.0], [1.1, 0.98, 0.9]])
    tint = 1.0 + g.pop("tint") * (np.array(hue) - 1.0) / 0.2
    g["color"] = list(np.clip(tint / tint.mean(), 0.5, 1.5))
    env["rgb"] = dict(ground=g, wall=draw_group(rng, r["wall"]), roof=draw_group(rng, r["roof"]))
    lw = sig["lwir"]["environment"]
    env["lwir"] = {k: draw_group(rng, lw[k], state) for k in ("sky", "clouds", "ground", "wall", "roof")}
    sw = sig["swir"]["environment"]
    env["swir"] = dict(ground=draw_group(rng, sw["ground"]), wall=draw_group(rng, sw["wall"]),
                       roof=draw_group(rng, sw["roof"]), cloud_brightness=draw(rng, sw["cloud_brightness"]),
                       cloud_brightness_scale=draw(rng, sw["cloud_brightness_scale"]))
    env["swir"]["sky_divisor"] = draw(rng, sig["swir"]["sky_divisor"])
    env["swir"]["night_airglow"] = float(np.exp(rng.uniform(*np.log(sig["swir"]["night_airglow"]))))
    return env

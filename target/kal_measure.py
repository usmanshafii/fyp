"""Measurements, invariant checks and hot-part visibility on the built KAL mesh.

Everything reads the mesh itself (vertices and faces), never the builder's intentions,
so a measurement that disagrees with the YAML is a real defect. numpy only.
Adapted from the earlier `kal_target.measure` module (validation, projected extent and
ray-cast visibility), rewritten for the photo-based geometry in kal_geometry.py.

    python target/kal_measure.py      -> invariant report + exhaust/engine visibility table
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kal_geometry as kg  # noqa: E402


# ============================================================================ basics (metres, output frame)
def triangulate(mesh: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Fan-triangulate every face; returns (T, 3) vertex indices and the source face per triangle."""
    tris, src = [], []
    for fi, f in enumerate(mesh["faces"]):
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
            src.append(fi)
    return np.array(tris, dtype=np.int64), np.array(src, dtype=np.int64)


def camera_basis(view_dir: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)):
    """Right-handed (right, up, forward) basis for a camera looking along `view_dir`."""
    f = np.asarray(view_dir, float); f /= np.linalg.norm(f)
    u = np.asarray(up, float)
    r = np.cross(f, u)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, [1.0, 0, 0] if abs(f[0]) < 0.9 else [0, 1.0, 0])
    r /= np.linalg.norm(r)
    return r, np.cross(r, f), f


def projected_extent(V: np.ndarray, view_dir, up=(0.0, 0.0, 1.0)) -> Tuple[float, float]:
    """Width, height (metres) of vertices projected along `view_dir`: W_proj for the range solver."""
    r, u, _ = camera_basis(view_dir, up)
    return float(np.ptp(V @ r)), float(np.ptp(V @ u))


VIEWS = {  # output frame: +X forward, +Y port, +Z up; look direction, image-up
    "top": ((0, 0, -1.0), (1.0, 0, 0)), "front": ((-1.0, 0, 0), (0, 0, 1.0)),
    "side": ((0, -1.0, 0), (0, 0, 1.0)), "rear": ((1.0, 0, 0), (0, 0, 1.0)),
}


# ============================================================================ measure + validate
def measure(mesh: dict) -> Dict[str, object]:
    """Dimensions read back from the mesh, in metres (output frame)."""
    g = mesh["derived"]
    S = g["span_m"]
    V = kg.to_output_frame(mesh)
    parts = np.array(mesh["face_parts"])

    def verts(pred):
        idx = sorted({i for f, p in zip(mesh["faces"], parts) if pred(p) for i in f})
        return V[idx]
    wing = verts(lambda p: p == "wing")
    fus = verts(lambda p: p == "fuselage")
    fins = verts(lambda p: p.startswith("fin_"))
    disc = verts(lambda p: p == "prop_disc")
    probe = verts(lambda p: p == "probe")
    # wing sections: group wing vertices by |y|
    ys = np.round(np.abs(wing[:, 1]), 7)
    root, tip = wing[ys == ys.min()], wing[ys == ys.max()]
    le_pts = np.array([(y, wing[ys == y][:, 0].max()) for y in np.unique(ys)])
    slope = np.polyfit(le_pts[:, 0], le_pts[:, 1], 1)[0]
    return {
        "span_wing_m": float(np.ptp(wing[:, 1])),
        "length_nose_to_hub_m": float(fus[:, 0].max() - disc[:, 0].mean()),
        "length_overall_m": float(probe[:, 0].max() - V[:, 0].min()),
        "root_chord_m": float(np.ptp(root[:, 0])),
        "tip_chord_m": float(np.ptp(tip[:, 0])),
        "le_sweep_deg": float(math.degrees(math.atan(-slope))),
        "forebody_dia_m": float(np.ptp(fus[:, 1])),
        "fin_height_m": float(np.ptp(fins[:, 2])),
        "prop_dia_m": float(np.ptp(disc[:, 1])),
        "projected_extent_m": {k: projected_extent(V, d, u) for k, (d, u) in VIEWS.items()},
        "faces": len(mesh["faces"]),
    }


def validate(mesh: dict, tol: float = 2e-3) -> List[str]:
    """Every invariant the geometry promises; returns a list of problems (empty = valid)."""
    g = mesh["derived"]
    S = g["span_m"]
    m = measure(mesh)
    out: List[str] = []

    def close(name, got, want, eps=tol):
        if abs(got - want) > eps * max(1.0, abs(want)):
            out.append(f"{name}: mesh {got:.4f}, expected {want:.4f}")
    close("wing span", m["span_wing_m"], S)
    close("nose-to-hub length", m["length_nose_to_hub_m"], (g["hub_x"] - g["nose_x"]) * S, 5e-3)
    close("root chord", m["root_chord_m"], g["te_x"] * S)
    close("tip chord", m["tip_chord_m"], g["tip_chord"] * S)
    close("LE sweep", m["le_sweep_deg"], g["sweep_deg"], 1e-3)
    close("forebody diameter", m["forebody_dia_m"], g["forebody_dia"] * S, 1e-2)
    close("fin height", m["fin_height_m"], g["fin_total_height"] * S)
    close("prop diameter", m["prop_dia_m"], g["prop_dia"] * S, 1e-2)
    bad = kg.edge_report(mesh)
    if bad:
        out.append(f"open or inconsistently wound parts: {bad}")
    V = kg.to_output_frame(mesh)
    if not np.all(np.isfinite(V)):
        out.append("non-finite vertices")
    tris, _ = triangulate(mesh)
    area = 0.5 * np.linalg.norm(np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]]), axis=1)
    if (area <= 1e-12).any():
        out.append(f"{int((area <= 1e-12).sum())} zero-area triangles")
    # mirror symmetry about y = 0 (decals excepted: text reads one way)
    cy = np.array([V[list(f), 1].mean() for f in mesh["faces"]])
    for i, slot in enumerate(kg.SLOTS):
        if slot == "decal":
            continue
        on = mesh["face_slots"] == i
        if int((on & (cy > 1e-9)).sum()) != int((on & (cy < -1e-9)).sum()):
            out.append(f"slot {slot} not mirror-symmetric")
    if not 300 <= len(mesh["faces"]) <= 5000:
        out.append(f"face count {len(mesh['faces'])} outside the 300-5000 budget")
    return out


# ============================================================================ ray casting (Moller-Trumbore)
def rays_blocked(origins, direction, tri_v, t_min=1e-7, chunk=256) -> np.ndarray:
    """Whether rays origins + t*direction (shared direction) hit any triangle at t > t_min."""
    d = np.asarray(direction, float)
    v0, e1, e2 = tri_v[:, 0], tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0]
    pvec = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > 1e-15
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    out = np.zeros(len(origins), dtype=bool)
    for s in range(0, len(origins), chunk):
        o = origins[s:s + chunk]
        tvec = o[:, None, :] - v0[None, :, :]
        u = np.einsum("rtk,tk->rt", tvec, pvec) * inv
        q = np.cross(tvec, e1[None, :, :])
        w = np.einsum("k,rtk->rt", d, q) * inv
        t = np.einsum("rtk,tk->rt", q, e2) * inv
        out[s:s + chunk] = (ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > t_min)).any(axis=1)
    return out


def hot_visibility(mesh: dict, camera_dirs: np.ndarray, slots=("exhaust_outlet",)) -> np.ndarray:
    """Visible projected area (m^2) of `slots` from distant cameras.

    camera_dirs (K, 3): unit vectors from the target toward each camera, output frame.
    A sample point counts if its face looks toward the camera and nothing else is hit on
    the way. The propeller disc is transparent (spinning blades cover ~5-10% of it);
    decals never block.
    """
    V = kg.to_output_frame(mesh)
    tris, src = triangulate(mesh)
    slot_of = mesh["face_slots"][src]
    a, b, c = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    nrm = cr / np.maximum(np.linalg.norm(cr, axis=1), 1e-30)[:, None]
    hot = np.isin(slot_of, [kg.SLOTS.index(s) for s in slots])
    occ = ~np.isin(slot_of, [kg.SLOTS.index("propeller_disc"), kg.SLOTS.index("decal")])
    occ_v = V[tris[occ]]
    bary = np.array([[1 / 3, 1 / 3, 1 / 3], [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
    pts = np.einsum("sk,tkd->tsd", bary, V[tris[hot]]).reshape(-1, 3)
    n_s = np.repeat(nrm[hot], len(bary), axis=0)
    w_s = np.repeat(area[hot] / len(bary), len(bary))
    scale = float(np.linalg.norm(np.ptp(V, axis=0)))
    pts = pts + n_s * 1e-6 * scale
    out = np.zeros(len(camera_dirs))
    for k, d in enumerate(np.asarray(camera_dirs, float)):
        d = d / np.linalg.norm(d)
        cos = n_s @ d
        facing = cos > 1e-12
        if facing.any():
            blocked = rays_blocked(pts[facing], d, occ_v, t_min=1e-6 * scale)
            out[k] = float((w_s[facing] * cos[facing])[~blocked].sum())
    return out


def direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Target-to-camera unit vector. Azimuth 0 = camera dead ahead, 90 = port, 180 = behind;
    elevation + = camera above the target."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


if __name__ == "__main__":
    cfg = kg.load_config()
    mesh = kg.build_mesh(kg.defaults(cfg))
    probs = validate(mesh)
    print("invariants:", "all hold" if not probs else "\n  " + "\n  ".join(probs))
    m = measure(mesh)
    for k, v in m.items():
        if k != "projected_extent_m":
            print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")
    print("  projected extent (w x h, m):", {k: tuple(round(x, 2) for x in v) for k, v in m["projected_extent_m"].items()})
    print("\nVisible hot area (cm^2): exhaust_outlet / engine_region")
    els = (-45, -15, 0, 15, 45)
    print("  az\\el " + "".join(f"{e:>14d}" for e in els))
    for az in (0, 30, 60, 90, 120, 150, 180):
        dirs = np.array([direction(az, e) for e in els])
        ex = hot_visibility(mesh, dirs, ("exhaust_outlet",)) * 1e4
        en = hot_visibility(mesh, dirs, ("engine_region",)) * 1e4
        print(f"  {az:4d}  " + "".join(f"{a:7.0f}/{b:<6.0f}" for a, b in zip(ex, en)))

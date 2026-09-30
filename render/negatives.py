"""Low-poly meshes for the hard-negative classes (spec 6: bird, kite, light_aircraft, helicopter,
small_uav, warm_clutter). Pure NumPy, same MeshBuilder as the target, same output frame
(metres, +X forward, +Y port, +Z up, origin at the object centre).

These are generic shapes sized from the ranges in configs/render.yaml. They exist so the detector
learns what is *not* the target at the same pixel sizes; they are not models of specific types.
Every part is closed and outward-facing (checked by kal_geometry.edge_report in the tests).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target"))
import kal_geometry as kg  # noqa: E402

CLASSES = ["target", "bird", "kite", "light_aircraft", "helicopter", "small_uav", "warm_clutter"]
NEG_SLOTS = ["body", "wing", "detail", "hot", "disc"]


class NegBuilder(kg.MeshBuilder):
    def add_face(self, idx, slot, part, uv=None):
        self.f.append(tuple(int(i) for i in idx))
        self.slot.append(NEG_SLOTS.index(slot))
        self.part.append(part)

    def prism(self, poly2d, thickness, M, slot, part):
        """Closed slab: planar polygon (local x-y) extruded along local z, then transformed by 4x4 M."""
        poly = np.asarray(poly2d, float)
        n = len(poly)
        lo = np.c_[poly, np.full(n, -thickness / 2), np.ones(n)] @ M.T
        hi = np.c_[poly, np.full(n, thickness / 2), np.ones(n)] @ M.T
        a = self.add_verts(lo[:, :3])
        b = self.add_verts(hi[:, :3])
        self.add_face([a + i for i in range(n)][::-1], slot, part)
        self.add_face([b + i for i in range(n)], slot, part)
        for i in range(n):
            j = (i + 1) % n
            self.add_face((a + i, a + j, b + j, b + i), slot, part)
        self.orient_part(part)

    def body_of_revolution(self, xs, rs, n, slot, part, z=0.0, y=0.0):
        """Closed body around the x axis with radius profile rs(xs); pointed ends where r = 0."""
        xs, rs = np.asarray(xs, float), np.asarray(rs, float)
        ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
        inner = [(x, r) for x, r in zip(xs, rs) if r > 1e-9]
        rings = [np.c_[np.full(n, x), y + r * np.cos(ang), z + r * np.sin(ang)] for x, r in inner]
        starts = [self.add_verts(r) for r in rings]
        tip0 = self.add_verts([(xs[0], y, z)])
        tip1 = self.add_verts([(xs[-1], y, z)])
        for i in range(n):
            self.add_face((tip0, starts[0] + (i + 1) % n, starts[0] + i), slot, part)
        for k in range(len(starts) - 1):
            a, b = starts[k], starts[k + 1]
            for i in range(n):
                j = (i + 1) % n
                self.add_face((a + i, a + j, b + j, b + i), slot, part)
        for i in range(n):
            self.add_face((starts[-1] + i, starts[-1] + (i + 1) % n, tip1), slot, part)
        self.orient_part(part)

    def box(self, center, size, slot, part, R=None):
        cx, cy, cz = center
        sx, sy, sz = np.asarray(size, float) / 2
        P = np.array([(x, y, z) for z in (-sz, sz) for y in (-sy, sy) for x in (-sx, sx)])
        if R is not None:
            P = P @ np.asarray(R).T
        i0 = self.add_verts(P + np.array([cx, cy, cz]))
        for q in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
            self.add_face([i0 + k for k in q], slot, part)
        self.orient_part(part)

    def disc(self, center, radius, axis, thickness, n, slot, part):
        c = np.asarray(center, float)
        ax = np.asarray(axis, float) / np.linalg.norm(axis)
        self.cylinder(c - ax * thickness / 2, c + ax * thickness / 2, radius, radius, n, slot, part)


def _M(R=None, t=(0, 0, 0)):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    M[:3, 3] = t
    return M


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _finish(mb: NegBuilder, cls: str, extra=None):
    V = np.asarray(mb.v, float)
    ctr = 0.5 * (V.min(0) + V.max(0))
    return dict(cls=cls, vertices=V - ctr, faces=mb.f, face_slots=np.array(mb.slot),
                face_parts=mb.part, slots=list(NEG_SLOTS), extra=extra or {})


# ============================================================================ classes
def bird(rng, span: float):
    mb = NegBuilder()
    L = span * rng.uniform(0.30, 0.45)
    r = span * rng.uniform(0.045, 0.07)
    xs = np.linspace(-L / 2, L / 2, 8)
    t = (xs + L / 2) / L
    rs = r * np.sqrt(np.clip(np.sin(np.pi * t ** 0.8), 0, 1))
    rs[0] = rs[-1] = 0.0
    mb.body_of_revolution(xs, rs, 8, "body", "body")
    flap = math.radians(rng.uniform(-35, 45))       # wing beat phase: dihedral from down to up
    chord_root, chord_tip = span * rng.uniform(0.18, 0.28), span * rng.uniform(0.07, 0.14)
    sweep = span * rng.uniform(0.0, 0.12)
    half = span / 2 - r
    for s, tag in ((1, "port"), (-1, "stbd")):
        poly = [(chord_root * 0.35, 0.0), (chord_root * 0.35 - sweep, half), (-chord_tip * 0.65 - sweep, half), (-chord_root * 0.65, 0.0)]
        if s < 0:
            poly = [(x, -y) for x, y in poly][::-1]
        R = _rx(s * flap)
        mb.prism(poly, span * 0.012, _M(R, (0, s * r * 0.8, r * 0.3)), "wing", f"wing_{tag}")
    tail = [(-L / 2 + L * 0.1, -span * 0.05), (-L / 2 - span * 0.12, -span * 0.08), (-L / 2 - span * 0.12, span * 0.08), (-L / 2 + L * 0.1, span * 0.05)]
    mb.prism(tail, span * 0.008, _M(), "wing", "tail")
    return _finish(mb, "bird", dict(flap_deg=math.degrees(flap)))


def kite(rng, span: float):
    mb = NegBuilder()
    h = span * rng.uniform(1.1, 1.5)
    top, bot = h * 0.35, -h * 0.65
    poly = [(0.0, top), (-span / 2, 0.0), (0.0, bot), (span / 2, 0.0)]     # in the local x(=y world)-z plane
    tilt = math.radians(rng.uniform(25, 60))       # face tilted back from vertical into the wind
    R = _ry(-tilt) @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])   # local (x, y, z) -> world (y, z, x)
    mb.prism(poly, span * 0.01, _M(R), "body", "sail")
    tl = span * rng.uniform(1.0, 2.5)
    w = span * 0.03
    tail = [(-w, bot), (w, bot), (w, bot - tl), (-w, bot - tl)]
    mb.prism(tail, span * 0.004, _M(R), "detail", "tail")
    return _finish(mb, "kite")


def light_aircraft(rng, span: float):
    mb = NegBuilder()
    L = span * rng.uniform(0.70, 0.80)
    R0 = span * rng.uniform(0.055, 0.07)
    xs = np.array([0.5, 0.47, 0.38, 0.2, 0.0, -0.2, -0.38, -0.5]) * L
    rs = R0 * np.array([0.0, 0.65, 0.95, 1.0, 0.9, 0.6, 0.3, 0.0])
    mb.body_of_revolution(xs, rs, 12, "body", "fuselage")
    cw = span * rng.uniform(0.12, 0.16)
    wx = L * 0.18
    wing = [(wx + cw / 2, -span / 2), (wx + cw / 2, span / 2), (wx - cw / 2, span / 2), (wx - cw / 2, -span / 2)]
    mb.prism(wing, cw * 0.12, _M(t=(0, 0, R0 * 0.95)), "body", "wing")
    ts, tc = span * 0.32, span * 0.09
    tx = -L * 0.45
    tail = [(tx + tc / 2, -ts / 2), (tx + tc / 2, ts / 2), (tx - tc / 2, ts / 2), (tx - tc / 2, -ts / 2)]
    mb.prism(tail, tc * 0.1, _M(t=(0, 0, R0 * 0.25)), "body", "tailplane")
    fin = [(tx + tc * 0.8, 0.0), (tx - tc * 0.2, span * 0.13), (tx - tc * 0.7, span * 0.13), (tx - tc * 0.7, 0.0)]
    mb.prism(fin, tc * 0.1, _M(_rx(math.pi / 2), (0, 0, R0 * 0.3)), "body", "fin")
    mb.box((L * 0.44, 0, -R0 * 0.35), (L * 0.08, R0 * 1.2, R0 * 0.5), "hot", "engine")
    mb.box((L * 0.36, 0, -R0 * 0.85), (L * 0.06, R0 * 0.35, R0 * 0.25), "hot", "exhaust")
    mb.disc((L * 0.52, 0, 0), span * rng.uniform(0.08, 0.10), (1, 0, 0), 0.01 * span, 20, "disc", "prop")
    return _finish(mb, "light_aircraft")


def helicopter(rng, span: float):
    """span = main rotor diameter."""
    mb = NegBuilder()
    L = span * rng.uniform(0.35, 0.45)
    R0 = span * rng.uniform(0.09, 0.12)
    xs = np.array([0.5, 0.42, 0.25, 0.0, -0.25, -0.45, -0.5]) * L
    rs = R0 * np.array([0.0, 0.7, 0.95, 1.0, 0.85, 0.45, 0.0])
    mb.body_of_revolution(xs, rs, 12, "body", "cabin")
    bl = span * rng.uniform(0.40, 0.50)
    mb.cylinder((-L * 0.4, 0, R0 * 0.3), (-L * 0.4 - bl, 0, R0 * 0.45), R0 * 0.25, R0 * 0.12, 8, "body", "boom")
    fin = [(-L * 0.4 - bl + 0.02 * span, 0.0), (-L * 0.4 - bl - 0.04 * span, span * 0.12), (-L * 0.4 - bl - 0.09 * span, span * 0.12), (-L * 0.4 - bl - 0.06 * span, 0.0)]
    mb.prism(fin, span * 0.008, _M(_rx(math.pi / 2), (0, 0, R0 * 0.4)), "body", "fin")
    mb.box((-L * 0.1, 0, R0 * 1.05), (L * 0.35, R0 * 0.9, R0 * 0.45), "hot", "engine")
    mb.cylinder((0, 0, R0 * 1.2), (0, 0, R0 * 1.6), R0 * 0.08, R0 * 0.08, 8, "body", "mast")
    mb.disc((0, 0, R0 * 1.62), span / 2, (0, 0, 1), 0.004 * span, 32, "disc", "rotor")
    mb.disc((-L * 0.4 - bl - 0.03 * span, R0 * 0.2, R0 * 0.7), span * 0.09, (0, 1, 0), 0.004 * span, 16, "disc", "tail_rotor")
    return _finish(mb, "helicopter")


def small_uav(rng, span: float):
    """Consumer-style quadcopter; span = rotor-tip to rotor-tip across the diagonal."""
    mb = NegBuilder()
    b = span * rng.uniform(0.22, 0.32)
    mb.box((0, 0, 0), (b, b * 0.7, b * 0.35), "body", "body")
    arm = span / 2 * 0.72
    rr = span * rng.uniform(0.15, 0.19)
    for k, a in enumerate((45, 135, 225, 315)):
        ang = math.radians(a)
        c, s = math.cos(ang), math.sin(ang)
        mb.cylinder((c * b * 0.3, s * b * 0.3, 0), (c * arm, s * arm, 0), span * 0.018, span * 0.018, 6, "body", f"arm{k}")
        mb.cylinder((c * arm, s * arm, -span * 0.02), (c * arm, s * arm, span * 0.04), span * 0.03, span * 0.03, 8, "hot", f"motor{k}")
        mb.disc((c * arm, s * arm, span * 0.05), rr, (0, 0, 1), 0.003 * span, 16, "disc", f"rotor{k}")
    mb.box((0, 0, -b * 0.25), (b * 0.5, b * 0.35, b * 0.15), "hot", "battery")
    return _finish(mb, "small_uav")


def warm_clutter(rng, span: float):
    """Hot rooftop unit, vent or small machine: a box or a stack."""
    mb = NegBuilder()
    if rng.uniform() < 0.6:
        mb.box((0, 0, 0), (span, span * rng.uniform(0.5, 1.0), span * rng.uniform(0.3, 0.8)), "hot", "unit")
    else:
        r = span * rng.uniform(0.15, 0.3)
        mb.cylinder((0, 0, -span / 2), (0, 0, span / 2), r, r * 0.9, 12, "hot", "stack")
    return _finish(mb, "warm_clutter")


BUILDERS = dict(bird=bird, kite=kite, light_aircraft=light_aircraft, helicopter=helicopter,
                small_uav=small_uav, warm_clutter=warm_clutter)


def build(cls: str, rng, span: float) -> dict:
    return BUILDERS[cls](rng, span)


def edge_report(mesh: dict) -> dict:
    return kg.edge_report(dict(faces=mesh["faces"], face_parts=mesh["face_parts"]))

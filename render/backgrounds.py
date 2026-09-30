"""Terrain and structures for the horizon family (FYP_Sky_and_Background.md, spec 6).

The height field, the structure list and the line-of-sight tests are pure NumPy, so placement
(render/placement.py) and the Blender scene use exactly the same surfaces: ground_z() interpolates
the rendered triangles, not a smoother analytic function.

Ground: 20 km square centred on the camera, noise-displaced low hills, a flat pad around the
camera and a distant ridge at the plane edge so the terrain silhouette always sits above the
geometric horizon (otherwise a dark sliver of below-horizon sky would show at the join).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ============================================================================ terrain
class Terrain:
    def __init__(self, rng, cfg: dict, camera_xy=(0.0, 0.0), hills: bool = True):
        self.size = float(cfg["size_m"])
        self.grid = int(cfg["grid"])
        self.cx, self.cy = map(float, camera_xy)
        self.A = float(rng.uniform(*cfg["hill_amplitude_m"])) if hills else 0.0
        self.lam = float(rng.uniform(*cfg["hill_wavelength_m"]))
        self.ridge = float(rng.uniform(*cfg["edge_ridge_m"])) if hills else 0.0
        self.flat_r = float(cfg["flat_radius_m"])
        k0 = 2 * math.pi / self.lam
        ks, ph, am = [], [], []
        for octave in range(4):
            for _ in range(6):
                a = rng.uniform(0, 2 * math.pi)
                kk = k0 * 2 ** octave * rng.uniform(0.8, 1.25)
                ks.append((kk * math.cos(a), kk * math.sin(a)))
                ph.append(rng.uniform(0, 2 * math.pi))
                am.append(0.5 ** octave)
        self.k, self.phase, self.amp = np.array(ks), np.array(ph), np.array(am)
        self.ridge_phase = rng.uniform(0, 2 * math.pi, 3)
        self._build_grid()

    def _noise(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        arg = x[..., None] * self.k[:, 0] + y[..., None] * self.k[:, 1] + self.phase
        return (np.cos(arg) * self.amp).sum(-1) / self.amp.sum()

    def analytic_height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = self.A * 0.5 * (1.0 + self._noise(x, y))
        r = np.hypot(x - self.cx, y - self.cy)
        h = h * _smoothstep(self.flat_r, 3 * self.flat_r, r)
        edge = np.maximum(np.abs(x - self.cx), np.abs(y - self.cy)) / (self.size / 2)
        az = np.arctan2(x - self.cx, y - self.cy)
        vary = 0.65 + 0.35 * np.cos(2 * az + self.ridge_phase[0]) * np.cos(5 * az + self.ridge_phase[1])
        return h + self.ridge * _smoothstep(0.72, 1.0, edge) * vary

    def _build_grid(self):
        n = self.grid
        xs = self.cx + np.linspace(-self.size / 2, self.size / 2, n)
        ys = self.cy + np.linspace(-self.size / 2, self.size / 2, n)
        X, Y = np.meshgrid(xs, ys)          # row index = y, column index = x
        self.xs, self.ys = xs, ys
        self.Z = self.analytic_height(X, Y)
        self.step = xs[1] - xs[0]

    def ground_z(self, x, y):
        """Height of the rendered surface: each grid quad is split along its (i,j)-(i+1,j+1)
        diagonal, exactly as mesh() emits its triangles."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        fx = np.clip((x - self.xs[0]) / self.step, 0, self.grid - 1 - 1e-9)
        fy = np.clip((y - self.ys[0]) / self.step, 0, self.grid - 1 - 1e-9)
        i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
        u, v = fx - i, fy - j
        z00, z10 = self.Z[j, i], self.Z[j, i + 1]
        z01, z11 = self.Z[j + 1, i], self.Z[j + 1, i + 1]
        lower = u >= v                       # triangle (00, 10, 11)
        z_lo = z00 + u * (z10 - z00) + v * (z11 - z10)
        z_up = z00 + v * (z01 - z00) + u * (z11 - z01)
        return np.where(lower, z_lo, z_up)

    def mesh(self):
        n = self.grid
        X, Y = np.meshgrid(self.xs, self.ys)
        V = np.c_[X.ravel(), Y.ravel(), self.Z.ravel()]
        idx = np.arange(n * n).reshape(n, n)
        a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
        c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        # counter-clockwise seen from above -> normals up
        F = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
        return V, F

    def los_clear(self, P0, P1, clearance: float = 0.5, n: int | None = None) -> bool:
        """True if the straight segment P0 -> P1 stays above the ground."""
        P0, P1 = np.asarray(P0, float), np.asarray(P1, float)
        L = np.linalg.norm(P1[:2] - P0[:2])
        n = n or int(min(4000, max(16, L / (self.step * 0.25))))
        t = np.linspace(0.0, 1.0, n)[1:-1]
        P = P0 + t[:, None] * (P1 - P0)
        return bool(np.all(P[:, 2] > self.ground_z(P[:, 0], P[:, 1]) + clearance))


def flat_terrain(cfg: dict, camera_xy=(0.0, 0.0)) -> Terrain:
    """Flat ground for open-sky scenes: never in frame, but it lights the target's underside."""
    return Terrain(np.random.default_rng(0), {**cfg, "grid": 2}, camera_xy, hills=False)


# ============================================================================ structures
@dataclass
class Block:
    kind: str          # building | mast
    cx: float
    cy: float
    z0: float
    w: float
    d: float
    h: float
    yaw: float         # radians

    def _local(self, P):
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        Q = np.asarray(P, float) - np.array([self.cx, self.cy, 0.0])
        return np.array([c * Q[0] - s * Q[1], s * Q[0] + c * Q[1], Q[2]])

    def blocks(self, P0, P1) -> bool:
        """Slab test: does segment P0 -> P1 pass through the box?"""
        a, b = self._local(P0), self._local(P1)
        lo = np.array([-self.w / 2, -self.d / 2, self.z0])
        hi = np.array([self.w / 2, self.d / 2, self.z0 + self.h])
        dvec = b - a
        t0, t1 = 0.0, 1.0
        for k in range(3):
            if abs(dvec[k]) < 1e-12:
                if a[k] < lo[k] or a[k] > hi[k]:
                    return False
                continue
            ta, tb = (lo[k] - a[k]) / dvec[k], (hi[k] - a[k]) / dvec[k]
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
            if t0 > t1:
                return False
        return True

    def roof_point(self, rng, margin: float = 1.0):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        lx = rng.uniform(-self.w / 2 + margin, self.w / 2 - margin) if self.w > 2 * margin else 0.0
        ly = rng.uniform(-self.d / 2 + margin, self.d / 2 - margin) if self.d > 2 * margin else 0.0
        return np.array([self.cx + c * lx - s * ly, self.cy + s * lx + c * ly, self.z0 + self.h])

    def mesh(self, segments: int = 10):
        """Vertices and faces with slot names (wall / roof)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        if self.kind == "mast":
            r = self.w / 2
            ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
            ring = np.c_[r * np.cos(ang), r * np.sin(ang)]
            V = [(self.cx + x, self.cy + y, self.z0) for x, y in ring] + \
                [(self.cx + x, self.cy + y, self.z0 + self.h) for x, y in ring]
            F, S = [], []
            for i in range(segments):
                j = (i + 1) % segments
                F.append((i, j, segments + j, segments + i)); S.append("wall")
            F.append(tuple(range(segments))[::-1]); S.append("wall")
            F.append(tuple(segments + i for i in range(segments))); S.append("roof")
            return np.array(V), F, S
        hw, hd = self.w / 2, self.d / 2
        loc = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
        ring = [(self.cx + c * x - s * y, self.cy + s * x + c * y) for x, y in loc]
        V = [(x, y, self.z0) for x, y in ring] + [(x, y, self.z0 + self.h) for x, y in ring]
        F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        S = ["wall", "roof", "wall", "wall", "wall", "wall"]
        return np.array(V), F, S


def sample_structures(rng, cfg: dict, terrain: Terrain, cam_xy, cam_az_deg: float, hfov_deg: float):
    """Buildings and masts inside the camera's azimuth wedge, standing on the terrain."""
    out = []
    half = hfov_deg / 2 + cfg["view_margin_deg"]

    def spot(dmin, dmax):
        dist = math.exp(rng.uniform(math.log(dmin), math.log(dmax)))
        az = math.radians(cam_az_deg + rng.uniform(-half, half))
        return cam_xy[0] + dist * math.sin(az), cam_xy[1] + dist * math.cos(az)

    for _ in range(int(rng.integers(cfg["buildings"][0], cfg["buildings"][1] + 1))):
        w, d, h = (rng.uniform(*cfg["building_w_m"]), rng.uniform(*cfg["building_d_m"]), rng.uniform(*cfg["building_h_m"]))
        dmin = max(cfg["building_distance_m"][0], max(w, d, h) / (0.3 * math.radians(hfov_deg)))
        if dmin >= cfg["building_distance_m"][1]:
            continue
        x, y = spot(dmin, cfg["building_distance_m"][1])
        yaw = rng.uniform(0, math.pi)
        corners = [(x + a * w / 2, y + b * d / 2) for a in (-1, 1) for b in (-1, 1)]
        z0 = float(min(terrain.ground_z(cx, cy) for cx, cy in corners)) - 0.5
        out.append(Block("building", x, y, z0, w, d, h + 0.5, yaw))
    for _ in range(int(rng.integers(cfg["masts"][0], cfg["masts"][1] + 1))):
        h, r = rng.uniform(*cfg["mast_h_m"]), rng.uniform(*cfg["mast_r_m"])
        x, y = spot(*cfg["mast_distance_m"])
        out.append(Block("mast", x, y, float(terrain.ground_z(x, y)) - 0.5, 2 * r, 2 * r, h + 0.5, 0.0))
    return out


def los_clear(P0, P1, terrain: Terrain | None, blocks, clearance: float = 0.5) -> bool:
    if terrain is not None and not terrain.los_clear(P0, P1, clearance):
        return False
    return not any(b.blocks(P0, P1) for b in blocks)


# ============================================================================ Blender builders
def build_terrain_object(terrain: Terrain, name: str = "ground"):
    import bpy
    V, F = terrain.mesh()
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F.tolist())
    me.polygons.foreach_set("use_smooth", [True] * len(F))
    me.materials.append(None)
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    obj["fyp_slots"] = ["ground"]
    return obj


def build_structures_object(blocks, name: str = "structures"):
    import bpy
    if not blocks:
        return None
    Vs, Fs, Ss, off = [], [], [], 0
    for b in blocks:
        V, F, S = b.mesh()
        Vs.append(V)
        Fs += [tuple(i + off for i in f) for f in F]
        Ss += S
        off += len(V)
    V = np.vstack(Vs)
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], Fs)
    slots = ["wall", "roof"]
    for _ in slots:
        me.materials.append(None)
    me.polygons.foreach_set("material_index", [slots.index(s) for s in Ss])
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    obj["fyp_slots"] = slots
    return obj

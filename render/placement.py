"""Camera model, projection and bin-targeted placement (spec 4 step 2, spec 5). Pure NumPy.

World frame (Blender): metres, +X east, +Y north, +Z up. Azimuths are clockwise from north.
Camera: Blender convention, looks along local -Z with local +Y up. Image coordinates are
continuous pixels with (0, 0) at the top-left corner of the top-left pixel, x right, y down;
pixel (i, j) covers [i, i+1) x [j, j+1). Boxes are (x0, y0, x1, y1) in those units.

Objects use the target's output frame: +X forward (nose), +Y port, +Z up.

Placement order (spec 4 step 2): pick the size bin and a target size inside it, pick a random
ray in the field of view, sample the orientation, set range from the projected extent
(Z = f_px * W_proj / p_target), refine on the exact projected vertex box, and verify the box lands
in the bin and inside the frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ============================================================================ basics
def focal_px(width: int, hfov_deg: float) -> float:
    return width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def focal_mm(sensor_mm: float, hfov_deg: float) -> float:
    return sensor_mm / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])


def azimuth_elevation(d) -> tuple[float, float]:
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    return math.degrees(math.atan2(d[0], d[1])) % 360.0, math.degrees(math.asin(np.clip(d[2], -1, 1)))


def camera_rotation(azimuth_deg: float, elevation_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """Camera-to-world rotation for a Blender camera looking along (azimuth, elevation)."""
    return (rot_z(-math.radians(azimuth_deg)) @ rot_x(math.pi / 2 + math.radians(elevation_deg))
            @ rot_z(math.radians(roll_deg)))


def object_rotation(heading_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Body-to-world rotation. heading: azimuth of the nose (clockwise from north); pitch: nose up
    positive; roll: starboard wing down positive."""
    return (rot_z(math.pi / 2 - math.radians(heading_deg)) @ rot_y(-math.radians(pitch_deg))
            @ rot_x(math.radians(roll_deg)))


def matrix_world(R: np.ndarray, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t, float)
    return M


# ============================================================================ camera
class Camera:
    """Pinhole camera matching Blender (sensor fit horizontal, square pixels, no shift)."""

    def __init__(self, width: int, height: int, hfov_deg: float, position, azimuth_deg: float,
                 elevation_deg: float, roll_deg: float = 0.0):
        self.W, self.H, self.hfov = int(width), int(height), float(hfov_deg)
        self.f = focal_px(width, hfov_deg)
        self.C = np.asarray(position, float)
        self.azimuth, self.elevation, self.roll = float(azimuth_deg), float(elevation_deg), float(roll_deg)
        self.R = camera_rotation(azimuth_deg, elevation_deg, roll_deg)

    @property
    def vfov(self) -> float:
        return math.degrees(2 * math.atan(self.H / 2 / self.f))

    @property
    def forward(self) -> np.ndarray:
        return -self.R[:, 2]

    def scaled(self, s: int) -> "Camera":
        return Camera(self.W * s, self.H * s, self.hfov, self.C, self.azimuth, self.elevation, self.roll)

    def to_camera(self, P) -> np.ndarray:
        return (np.atleast_2d(np.asarray(P, float)) - self.C) @ self.R

    def project(self, P):
        """World points -> (u, v, depth). depth <= 0 means behind the camera."""
        pc = self.to_camera(P)
        z = -pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.W / 2 + self.f * pc[:, 0] / z
            v = self.H / 2 - self.f * pc[:, 1] / z
        return u, v, z

    def ray(self, u: float, v: float) -> np.ndarray:
        d = self.R @ np.array([(u - self.W / 2) / self.f, -(v - self.H / 2) / self.f, -1.0])
        return d / np.linalg.norm(d)

    def pixel_elevation(self, v) -> np.ndarray:
        """Ray elevation (deg) of image rows v at the image centre column."""
        v = np.atleast_1d(np.asarray(v, float))
        return np.array([azimuth_elevation(self.ray(self.W / 2, vi))[1] for vi in v])

    def horizon_row(self) -> float:
        """Image row of the geometric horizon at the centre column (may lie outside the frame)."""
        e = math.radians(self.elevation)
        return self.H / 2 + self.f * math.tan(e)

    def bbox(self, P) -> tuple:
        u, v, z = self.project(P)
        if np.any(z <= 1e-6):
            return None
        return float(u.min()), float(v.min()), float(u.max()), float(v.max())


def box_size(b) -> float:
    """Larger side of a box (spec 1: bins use the larger side)."""
    return max(b[2] - b[0], b[3] - b[1])


def box_union(a, b):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def box_inside(b, W, H, margin=0.0) -> bool:
    return b[0] >= margin and b[1] >= margin and b[2] <= W - margin and b[3] <= H - margin


def boxes_overlap(a, b, gap=0.0) -> bool:
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def bin_index(p: float, edges) -> int:
    """Index of the size bin containing p (lower edge inclusive). The last bin is open-ended."""
    for i, (lo, hi) in enumerate(edges):
        if lo <= p < hi or (i == len(edges) - 1 and p >= lo):
            return i
    return -1


def sample_size(rng, lo: float, hi: float, mode: str = "log_uniform") -> float:
    if mode == "log_uniform":
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return float(rng.uniform(lo, hi))


# ============================================================================ placement
@dataclass
class Placement:
    position: np.ndarray            # world position of the object origin at mid-exposure
    R: np.ndarray                   # body-to-world rotation
    distance: float                 # camera to object origin, metres
    bbox: tuple                     # projected vertex box at mid-exposure (output px)
    size_px: float                  # larger side of bbox
    bin: int
    pose: dict = field(default_factory=dict)
    velocity: np.ndarray = None     # world m/s
    bbox_blur: tuple = None         # union of the boxes at shutter open and close

    def aspect(self, cam: Camera) -> dict:
        """Viewing geometry in the body frame: aspect 0 = head-on, 180 = tail-on; elevation > 0 =
        camera above the wing plane (sees the upper surface)."""
        to_cam = cam.C - self.position
        to_cam = to_cam / np.linalg.norm(to_cam)
        b = self.R.T @ to_cam
        return dict(aspect_deg=math.degrees(math.acos(np.clip(b[0], -1, 1))),
                    aspect_azimuth_deg=math.degrees(math.atan2(b[1], b[0])),
                    aspect_elevation_deg=math.degrees(math.asin(np.clip(b[2], -1, 1))))


def solve_distance(cam: Camera, verts_body: np.ndarray, R: np.ndarray, d: np.ndarray, p_target: float,
                   iters: int = 12, tol: float = 1e-4):
    """Distance along ray d at which the projected vertex box has larger side p_target."""
    Vw = verts_body @ R.T
    ext = max(np.ptp(Vw @ cam.R[:, 0]), np.ptp(Vw @ cam.R[:, 1]))
    cosang = max(1e-3, float(d @ cam.forward))
    dist = cam.f * ext / p_target / cosang
    b = None
    for _ in range(iters):
        b = cam.bbox(Vw + cam.C + d * dist)
        if b is None:
            dist *= 2.0
            continue
        p = box_size(b)
        if abs(p / p_target - 1.0) < tol:
            break
        dist *= p / p_target
    return dist, b


def place(cam: Camera, verts_body: np.ndarray, rng, bin_edges, bin_idx: int, pose_fn, valid_fn=None,
          uv_fn=None, sampling: str = "log_uniform", tries: int = 200, frame_margin: float = 1.0,
          p_range=None) -> Placement:
    """Bin-targeted placement along a random ray. Raises RuntimeError after `tries` failures.

    pose_fn(rng) -> (R, pose dict).  valid_fn(position, distance, bbox) -> bool.
    uv_fn(rng, p_target) -> (u, v) image point for the ray; default uniform over the frame.
    p_range overrides the sampling interval (default: the bin's edges).
    """
    lo, hi = p_range if p_range is not None else bin_edges[bin_idx]
    for _ in range(tries):
        p_t = sample_size(rng, lo, hi * 0.999, sampling)
        m = p_t / 2 + frame_margin + 1
        if uv_fn is not None:
            uv = uv_fn(rng, p_t)
            if uv is None:
                continue
            u, v = uv
        else:
            u, v = rng.uniform(m, cam.W - m), rng.uniform(m, cam.H - m)
        d = cam.ray(u, v)
        R, pose = pose_fn(rng)
        dist, b = solve_distance(cam, verts_body, R, d, p_t)
        if b is None or not box_inside(b, cam.W, cam.H, frame_margin):
            continue
        p = box_size(b)
        if bin_index(p, bin_edges) != bin_idx:
            continue
        pos = cam.C + d * dist
        if valid_fn is not None and not valid_fn(pos, dist, b):
            continue
        return Placement(position=pos, R=R, distance=float(dist), bbox=b, size_px=p, bin=bin_idx, pose=pose)
    raise RuntimeError("placement failed: no valid ray/pose for this bin")


def place_on_ground(cam: Camera, verts_body: np.ndarray, rng, bin_edges, bin_idx: int, ground_z_fn,
                    pose_fn, valid_fn=None, sampling: str = "log_uniform", tries: int = 200,
                    frame_margin: float = 1.0, azimuth_margin_deg: float = 0.0) -> Placement:
    """Placement for objects resting on the ground (warm clutter): choose size and distance, then
    an azimuth inside the view; the object sits at ground height, so its image row follows."""
    lo, hi = bin_edges[bin_idx]
    Vext = np.ptp(verts_body, axis=0)
    for _ in range(tries):
        p_t = sample_size(rng, lo, hi * 0.999, sampling)
        R, pose = pose_fn(rng)
        span = max(np.ptp((verts_body @ R.T)[:, :2], axis=0).max(), Vext[2])
        dist = cam.f * span / p_t
        half = cam.hfov / 2 - azimuth_margin_deg
        az = cam.azimuth + rng.uniform(-half, half)
        x = cam.C[0] + dist * math.sin(math.radians(az))
        y = cam.C[1] + dist * math.cos(math.radians(az))
        z = float(ground_z_fn(x, y))
        pos = np.array([x, y, z - float((verts_body @ R.T)[:, 2].min())])
        b = cam.bbox(verts_body @ R.T + pos)
        if b is None or not box_inside(b, cam.W, cam.H, frame_margin):
            continue
        p = box_size(b)
        if bin_index(p, bin_edges) != bin_idx:
            # one refinement step on distance keeps the bin hit rate high
            dist *= p / p_t
            x = cam.C[0] + dist * math.sin(math.radians(az))
            y = cam.C[1] + dist * math.cos(math.radians(az))
            z = float(ground_z_fn(x, y))
            pos = np.array([x, y, z - float((verts_body @ R.T)[:, 2].min())])
            b = cam.bbox(verts_body @ R.T + pos)
            if b is None or not box_inside(b, cam.W, cam.H, frame_margin):
                continue
            p = box_size(b)
            if bin_index(p, bin_edges) != bin_idx:
                continue
        if valid_fn is not None and not valid_fn(pos, float(np.linalg.norm(pos - cam.C)), b):
            continue
        return Placement(position=pos, R=R, distance=float(np.linalg.norm(pos - cam.C)), bbox=b, size_px=p,
                         bin=bin_idx, pose=pose)
    raise RuntimeError("ground placement failed")


def motion_boxes(cam: Camera, verts_body: np.ndarray, pl: Placement, exposure_s: float):
    """Box union over the shutter for linear motion at pl.velocity (for crops and metadata)."""
    if pl.velocity is None or exposure_s <= 0:
        return pl.bbox
    Vw = verts_body @ pl.R.T
    half = 0.5 * exposure_s * pl.velocity
    b0 = cam.bbox(Vw + pl.position - half)
    b1 = cam.bbox(Vw + pl.position + half)
    if b0 is None or b1 is None:
        return pl.bbox
    return box_union(box_union(b0, b1), pl.bbox)


def blur_length_px(f_px: float, speed_mps: float, exposure_s: float, distance_m: float) -> float:
    """Spec 3.1 estimate: blur length ~ f_px * V * t_exp / Z (motion perpendicular to the ray)."""
    return f_px * speed_mps * exposure_s / distance_m

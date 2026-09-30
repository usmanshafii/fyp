"""Photo acceptance check: project the model mesh onto the reference photos.

For each photo, a weak-perspective camera (rotation, scale, image offset; 6 dof) is
fitted to hand-measured keypoints, then the whole mesh is drawn over the photo.
Pass criterion: RMS keypoint error under 3% of the projected span, and no visible
outline mismatch in the overlay.

    python tools/photo_check.py          -> docs/reference/overlay_<photo>.png

Also prints the RMS for three root_chord values: a flat result means that photo
cannot constrain root chord / span (true of every single photo available so far).
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "target"))
import kal_geometry as kg  # noqa: E402

PHOTOS = os.path.join(HERE, "..", "docs", "reference", "photos")
OUT = os.path.join(HERE, "..", "docs", "reference")

# Pixel keypoints. "L"/"R" = image left/right; the fit decides which is port.
# `surface`: which wing surface faces the camera. The P1 render has a slimmer fuselage
# than the flying article, so its fuselage-width points are not used.
KEYPOINTS = {
    "P1_render_oblique_top.png": dict(surface="upper", pts={
        "nose": (375, 40), "hub": (900, 520),
        "tip_le_L": (357, 605), "tip_le_R": (1223, 283),
        "tip_te_L": (485, 672), "tip_te_R": (1240, 367),
        "break_L": (650, 600), "break_R": (1083, 385)}),
    "P6_flyby_underside_a.png": dict(surface="lower", pts={
        "nose": (1044, 176),
        "tip_le_L": (998, 245), "tip_le_R": (1087, 245),
        "tip_te_L": (999, 257), "tip_te_R": (1087, 257),
        "le_root_L": (1036, 202), "le_root_R": (1052, 202)}),
    "P8_rear_launcher.png": dict(surface="upper", camera="perspective", pts={   # close, near-axial view from behind
        "tip_te_L": (50, 115), "tip_te_R": (400, 129),
        "tip_le_L": (50, 90), "tip_le_R": (399, 109),
        "fin_top_L": (47, 59), "fin_top_R": (409, 78),
        "fin_bot_L": (38, 163), "fin_bot_R": (407, 179),
        "break_L": (125, 124),
        "le_root_L": (190, 52), "le_root_R": (238, 52)}),       # approximate: LE meets the forebody
    "P7_flyby_underside_b.png": dict(surface="lower", pts={
        "nose": (1028, 18),
        "tip_le_L": (980, 97), "tip_le_R": (1080, 108),
        "tip_te_L": (982, 103), "tip_te_R": (1078, 118),
        "le_root_L": (1022, 48), "le_root_R": (1037, 48)}),
}


def _rot(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _proj(v, P):
    """Weak perspective (6 params: rotation, scale, offset) or pinhole (8 params: rotation,
    focal length, offset, camera distance, lateral shift folded into the offset)."""
    Q = (_rot(v[:3]) @ np.atleast_2d(P).T).T
    if len(v) == 6:
        return np.c_[v[3] * Q[:, 0] + v[4], -v[3] * Q[:, 1] + v[5]]
    f, cx, cy, dist = v[3], v[4], v[5], v[6]
    z = dist - Q[:, 2]                      # camera on +z of the rotated frame, looking down -z
    return np.c_[f * Q[:, 0] / z * dist + cx, -f * Q[:, 1] / z * dist + cy]


def fit(g, spec):
    kp = kg.keypoints(g)
    obs = spec["pts"]
    n_surf = np.array([0, 0, 1.0 if spec["surface"] == "upper" else -1.0])
    best = None
    for left in ("port", "stbd"):
        right = "stbd" if left == "port" else "port"
        names = [k.replace("_L", "_" + left).replace("_R", "_" + right) for k in obs]
        M = np.array([kp[n] for n in names])
        O = np.array(list(obs.values()), float)
        span_px = np.linalg.norm(O[[i for i, k in enumerate(obs) if k == "tip_le_R"][0]] -
                                 O[[i for i, k in enumerate(obs) if k == "tip_le_L"][0]])
        for seed in range(40):
            v0 = np.r_[np.random.default_rng(seed).normal(0, 1.6, 3), span_px, O.mean(0)]
            s = least_squares(lambda v: (_proj(v, M) - O).ravel(), v0)
            facing = (_rot(s.x[:3]) @ n_surf)[2] * np.sign(s.x[3])
            if facing <= 0:
                continue
            if spec.get("camera") == "perspective":
                # refine with a pinhole camera; distance in span units, bounded to stay in front
                v1 = np.r_[s.x, 3.0]
                lb = [-np.inf] * 3 + [-np.inf, -np.inf, -np.inf, 1.0]
                ub = [np.inf] * 3 + [np.inf, np.inf, np.inf, 40.0]
                try:
                    s = least_squares(lambda v: (_proj(v, M) - O).ravel(), v1, bounds=(lb, ub))
                except ValueError:
                    continue
                if (_rot(s.x[:3]) @ n_surf)[2] * np.sign(s.x[3]) <= 0:
                    continue
            rms = math.sqrt(np.mean(np.sum(s.fun.reshape(-1, 2) ** 2, 1)))
            if best is None or rms < best["rms"]:
                best = dict(v=s.x, rms=rms, names=names, res=s.fun.reshape(-1, 2), span_px=span_px, left=left)
    return best


def overlay(mesh, spec, photo, best, out):
    V = mesh["vertices_span"]
    img = plt.imread(photo)
    h, w = img.shape[:2]
    P2 = _proj(best["v"], V)
    cam_dir = _rot(best["v"][:3]).T @ np.array([0, 0, 1.0]) * np.sign(best["v"][3])
    if len(best["v"]) == 7:   # pinhole: per-face direction toward the camera centre
        cam_pos = _rot(best["v"][:3]).T @ np.array([0, 0, best["v"][6]])
    polys, depth = [], []
    for f, s in zip(mesh["faces"], mesh["face_slots"]):
        if kg.SLOTS[s] in ("decal", "propeller_disc"):
            continue
        pts = V[list(f)]
        n = sum(np.cross(pts[k] - pts[0], pts[k + 1] - pts[0]) for k in range(1, len(f) - 1))
        d = cam_dir if len(best["v"]) == 6 else (cam_pos - pts.mean(0))
        if np.dot(n, d) <= 0:
            continue
        polys.append(P2[list(f)])
        depth.append(np.dot(pts.mean(0), cam_dir) if len(best["v"]) == 6 else -np.linalg.norm(cam_pos - pts.mean(0)))
    order = np.argsort(depth)
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.add_collection(PolyCollection([polys[i] for i in order], facecolors=(0, 0.9, 1, 0.22),
                                     edgecolors=(0, 0.9, 1, 0.55), linewidths=0.4))
    kp = kg.keypoints(mesh["derived"])
    Mp = _proj(best["v"], np.array([kp[n] for n in best["names"]]))
    O = np.array(list(spec["pts"].values()), float)
    ax.plot(O[:, 0], O[:, 1], "o", ms=7, mfc="none", mec="red", mew=1.6)
    ax.plot(Mp[:, 0], Mp[:, 1], "x", ms=7, color="yellow", mew=1.6)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    ax.text(10, h - 12, f"model (cyan), measured points (red o), model keypoints (yellow x)   "
            f"RMS {best['rms']:.1f} px = {100 * best['rms'] / best['span_px']:.1f}% of span",
            color="white", fontsize=max(8, int(w / 150)), bbox=dict(fc="black", alpha=0.6, lw=0))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def crop_overlay(out, spec, pad=0.6):
    """Flight silhouettes are small: also save a zoomed crop around the aircraft."""
    img = plt.imread(out)
    O = np.array(list(spec["pts"].values()), float)
    c = O.mean(0); r = np.ptp(O, 0).max() * (0.5 + pad)
    x0, x1 = int(max(0, c[0] - r)), int(min(img.shape[1], c[0] + r))
    y0, y1 = int(max(0, c[1] - r)), int(min(img.shape[0], c[1] + r * 1.1))
    plt.imsave(out.replace(".png", "_zoom.png"), img[y0:y1, x0:x1])


def main():
    cfg = kg.load_config()
    base = kg.defaults(cfg)
    mesh = kg.build_mesh(base)
    ok = True
    for photo, spec in KEYPOINTS.items():
        path = os.path.join(PHOTOS, photo)
        if not os.path.exists(path):
            print(f"{photo}: missing, skipped")
            continue
        best = fit(mesh["derived"], spec)
        pct = 100 * best["rms"] / best["span_px"]
        passed = pct < 3.0
        ok &= passed
        print(f"\n{photo}: RMS {best['rms']:.1f} px ({pct:.1f}% of span)  image-left = {best['left']}  "
              f"{'PASS' if passed else 'FAIL'}")
        for n, (dx, dy) in zip(best["names"], best["res"]):
            print(f"    {n:14s} {dx:+6.1f} {dy:+6.1f}")
        sens = []
        for rc in (0.90, 1.00, 1.12):
            g = kg.derive({**base, "root_chord": rc})
            sens.append(f"root {rc:.2f}: {fit(g, spec)['rms']:.1f} px")
        print("    root-chord sensitivity:", ", ".join(sens))
        out = os.path.join(OUT, "overlay_" + photo)
        overlay(mesh, spec, path, best, out)
        if "flyby" in photo:
            crop_overlay(out, spec)
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()

"""Four-view reference drawing (top, side, front, rear) projected from the actual model mesh.

Every outline in the drawing is a projected face of the same mesh build_target()
produces, so the drawing cannot drift from the model. Run after any YAML change:

    python tools/draw_reference.py            -> docs/reference/kal_views.{png,svg}
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "target"))
import kal_geometry as kg  # noqa: E402

OUT = os.path.join(HERE, "..", "docs", "reference")
COLORS = {
    "skin": "#d3d8de", "control_surface": "#b7c1cc", "fin": "#aab5c2", "fuselage": "#9ba7b4",
    "tail_cone": "#9ba7b4", "markings": "#ffffff", "engine_region": "#c0392b",
    "exhaust_outlet": "#7b1d12", "propeller": "#4a4f55", "propeller_disc": "#c9ced4",
    "antenna": "#2f3338", "nose_window": "#223",
}
INK = "#1c232b"


def project(mesh, view):
    """Return 2D polygons, depths and slot names for a view (internal frame)."""
    V = mesh["vertices_span"]
    if view == "top":      # from above, nose left, starboard up
        P, depth, n_view = V[:, [0, 1]], V[:, 2], np.array([0, 0, 1.0])
    elif view == "side":   # from port, nose left
        P, depth, n_view = V[:, [0, 2]], -V[:, 1], np.array([0, -1.0, 0])
    elif view == "rear":   # from behind, starboard on the right
        P, depth, n_view = np.c_[V[:, 1], V[:, 2]], V[:, 0], np.array([1.0, 0, 0])
    else:                  # from ahead, port on the right
        P, depth, n_view = np.c_[-V[:, 1], V[:, 2]], -V[:, 0], np.array([-1.0, 0, 0])
    polys, depths, cols = [], [], []
    for f, s, part in zip(mesh["faces"], mesh["face_slots"], mesh["face_parts"]):
        slot = kg.SLOTS[s]
        if slot == "decal":
            continue
        pts = V[list(f)]
        n = np.zeros(3)
        for k in range(1, len(f) - 1):          # polygon normal (robust for n-gons)
            n += np.cross(pts[k] - pts[0], pts[k + 1] - pts[0])
        nn = np.linalg.norm(n)
        if nn == 0 or np.dot(n, n_view) <= 0:   # back-face culling: every part is closed and outward
            continue
        shade = 0.62 + 0.38 * np.dot(n / nn, n_view)
        c = np.array(matplotlib.colors.to_rgb(COLORS[slot])) * shade
        a = 0.35 if slot == "propeller_disc" else 1.0
        polys.append(P[list(f)])
        depths.append(depth[list(f)].mean())
        cols.append((*np.clip(c, 0, 1), a))
    order = np.argsort(depths)
    return [polys[i] for i in order], [cols[i] for i in order]


def draw_view(ax, mesh, view):
    polys, cols = project(mesh, view)
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.25))


def dim(ax, p0, p1, text, off=(0, 0), rot=0, fs=7.8):
    ax.annotate("", p0, p1, arrowprops=dict(arrowstyle="<->", lw=0.8, color="#555", shrinkA=0, shrinkB=0))
    ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], text, ha="center", va="center",
            fontsize=fs, color="#222", rotation=rot,
            bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))


def main():
    cfg = kg.load_config()
    mesh = kg.build_mesh(kg.defaults(cfg))
    g = mesh["derived"]
    S = g["span_m"]
    m = lambda v: f"{v:.3f} ({v * S:.2f} m)"
    nx, te, tle, hub, zw = g["nose_x"], g["te_x"], g["tip_le_x"], g["hub_x"], g["wing_z"]
    hu, hd = g["fin_up_height"], g["fin_down_height"]

    SC, FW, FH = 3.1, 13.0, 9.6                     # inches per span unit, figure size
    X = (nx - g["probe_len"] - 0.06, hub + 0.26)
    TOPY, SIDEY, FRX, FRY = (-0.78, 0.88), (-0.24, 0.20), (-0.62, 0.62), (-0.26, 0.28)
    fig = plt.figure(figsize=(FW, FH))

    def axes_in(x0, y0, xr, yr):
        a = fig.add_axes([x0 / FW, y0 / FH, (xr[1] - xr[0]) * SC / FW, (yr[1] - yr[0]) * SC / FH])
        a.set_xlim(*xr); a.set_ylim(*yr); a.set_aspect("equal"); a.axis("off")
        return a
    side = axes_in(0.25, 0.25, X, SIDEY)
    top = axes_in(0.25, 0.25 + (SIDEY[1] - SIDEY[0]) * SC + 0.35, X, TOPY)
    xr = 0.25 + (X[1] - X[0]) * SC + 0.25
    front = axes_in(xr, FH - 0.55 - (FRY[1] - FRY[0]) * SC, FRX, FRY)
    rear = axes_in(xr, FH - 0.55 - 2 * (FRY[1] - FRY[0]) * SC - 0.45, FRX, FRY)

    # ---------------- top
    draw_view(top, mesh, "top")
    y0 = -0.62
    dim(top, (nx, y0), (0, y0), f"nose {m(-nx)}", off=(0, 0.035))
    dim(top, (0, y0), (te, y0), f"root chord {m(te)}", off=(0, 0.035))
    dim(top, (te, y0), (hub, y0), f"{m(hub - te)}", off=(0.04, -0.045))
    dim(top, (nx, y0 - 0.1), (hub, y0 - 0.1), f"length, nose tip to hub (derived) {m(g['length'])}", off=(0, 0.035))
    dim(top, (tle, 0.57), (te, 0.57), f"tip {m(g['tip_chord'])}", off=(-0.02, 0.04))
    dim(top, (X[1] - 0.07, -0.5), (X[1] - 0.07, 0.5), f"span 1.000 ({S:.2f} m)", rot=90, off=(0.035, 0))
    ang = math.degrees(math.atan(0.5 / tle))
    top.text(0.16 * tle, 0.19, f"LE sweep (derived) {g['sweep_deg']:.1f}°", fontsize=7.8, rotation=ang, rotation_mode="anchor")
    top.plot([0, 0], [-0.03, 0.03], color=INK, lw=0.8)
    top.text(0, 0.045, "apex", fontsize=7, ha="center")
    for k, xt in enumerate((nx, 0, tle, te, hub)):
        yt = 0.79 if k % 2 == 0 else 0.735
        top.plot([xt, xt], [0.66, yt - 0.01], color="#555", lw=0.6)
        top.text(xt, yt, "0" if abs(xt) < 1e-9 else f"{xt:+.3f}", ha="center", fontsize=7.2, color="#555")
    b = g["elevon_breaks"]
    top.text(te - g["elevon_c"] / 2, 0.5 * (b[0] + b[1]), "elevon", fontsize=6.5, rotation=90, ha="center", va="center")
    top.text(te - g["elevon_c"] / 2, 0.5 * (b[1] + b[2]), "elevon", fontsize=6.5, rotation=90, ha="center", va="center")
    top.text(X[0], 0.855, "TOP VIEW   span units (metres at default span)   x aft from the wing apex", fontsize=10, weight="bold")

    # ---------------- side
    draw_view(side, mesh, "side")
    xf = hub + 0.05
    side.plot([te, xf + 0.01], [zw + hu] * 2, color="#999", lw=0.5, ls=":")
    side.plot([te, xf + 0.01], [zw - hd] * 2, color="#999", lw=0.5, ls=":")
    for z0, z1, lab in ((zw, zw + hu, f"fin up {m(hu)}"), (zw, zw - hd, f"fin down {m(hd)}")):
        side.annotate("", (xf, z0), (xf, z1), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#555", shrinkA=0, shrinkB=0))
        side.text(xf + 0.02, (z0 + z1) / 2, lab, fontsize=6.8, va="center", ha="left", color="#222")
    xd = g["collar_x"] - 0.06
    dim(side, (xd, -g["R_fore"]), (xd, g["R_fore"]), "", fs=7)
    side.text(xd, -g["R_fore"] - 0.03, f"forebody {m(g['forebody_dia'])}", fontsize=7.2, ha="center")
    side.annotate(f"wing, edge-on (root section t/c {g['wing_tc']:.2f}, mid-wing)", (0.35, zw - 0.01), (0.12, zw - 0.16),
                  fontsize=7.2, color="#333", arrowprops=dict(arrowstyle="-", lw=0.6, color="#777"))
    side.annotate("engine + exhaust headers (red)", (g["eng0"] + 0.03, -0.03), (0.62, -0.20),
                  fontsize=7, color="#7b1d12", arrowprops=dict(arrowstyle="-", lw=0.6, color="#b77"))
    side.text(nx - 0.02, g["R_fore"] + 0.05, "probe + 2 antennas", fontsize=7, color="#333")
    side.text(X[0], 0.175, "SIDE VIEW   from port, same scale", fontsize=10, weight="bold")

    # ---------------- front
    draw_view(front, mesh, "front")
    R = g["prop_dia"] / 2
    dim(front, (-0.5, -0.22), (0.5, -0.22), f"span {S:.2f} m", off=(0, 0.03), fs=7.2)
    front.text(0, R + 0.02, f"prop disc {m(g['prop_dia'])}", fontsize=7, ha="center", color="#555")
    front.text(0.55, zw + hu + 0.015, "port", fontsize=7, ha="center")
    front.text(-0.55, zw + hu + 0.015, "stbd", fontsize=7, ha="center")
    front.text(FRX[0], 0.25, "FRONT VIEW   from ahead, same scale", fontsize=10, weight="bold")

    # ---------------- rear
    draw_view(rear, mesh, "rear")
    for sgn, lab in ((-1, "port"), (1, "stbd")):
        rear.text(sgn * 0.55, zw + hu + 0.015, lab, fontsize=7, ha="center")
    xf2 = 0.56
    for z0, z1, lab in ((zw, zw + hu, f"up {m(hu)}"), (zw, zw - hd, f"down {m(hd)}")):
        rear.annotate("", (xf2, z0), (xf2, z1), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#555", shrinkA=0, shrinkB=0))
        rear.text(xf2 + 0.015, (z0 + z1) / 2, lab, fontsize=6.6, va="center", ha="left", color="#222")
    ew = g["engine_width"]
    dim(rear, (-ew / 2, -0.20), (ew / 2, -0.20), f"engine {m(ew)}", off=(0, -0.035), fs=6.8)
    rear.annotate("exhaust headers + outlets", (0.02, -g["outlet_drop"]), (0.12, -0.13), fontsize=6.8, color="#7b1d12",
                  arrowprops=dict(arrowstyle="-", lw=0.6, color="#b77"))
    rear.text(FRX[0], 0.25, "REAR VIEW   from behind, same scale", fontsize=10, weight="bold")

    notes = (
        "Measured from photos (see spec §2.2):\n"
        "  centreline stations, forebody/span,\n"
        "  fin heights and split (P8), elevons\n\n"
        "Least certain (widest sampled ranges):\n"
        "  root chord / span -> sweep, length\n"
        "  absolute span (not published)\n"
        "  wing thickness / chord\n\n"
        "red = engine_region + exhaust_outlet\n"
        "white = markings band\n"
        "decals omitted in this drawing"
    )
    fig.text((xr + 0.1) / FW, 0.33, notes, fontsize=8.2, va="top", family="monospace")
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"kal_views.{ext}"), dpi=150)
    print("wrote", os.path.join(OUT, "kal_views.png"))


if __name__ == "__main__":
    main()

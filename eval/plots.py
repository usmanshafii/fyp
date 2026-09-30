"""Central plots (spec 8): detection probability vs pixel size, and vs range as a band from the span
range, with bootstrap confidence intervals; AP per pixel bin for every model."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.bins import NAMES, f_px  # noqa: E402

COLORS = {"rgb_only": "#d62728", "lwir_only": "#1f77b4", "rgb_lwir": "#7f7f7f", "teacher": "#9467bd", "student": "#2ca02c"}


def _centres(edges):
    e = np.asarray(edges, float)
    return np.sqrt(e[:-1] * e[1:])


def pd_vs_size(reports: dict, out: str, kind: str = "centre"):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, rep in reports.items():
        c = rep[f"pd_vs_size_{kind}"]
        x = _centres(c["edges"])
        y, lo, hi = (np.array(c[k], float) for k in ("pd", "lo", "hi"))
        m = ~np.isnan(y)
        col = COLORS.get(name)
        ax.plot(x[m], y[m], "o-", color=col, label=name, ms=4)
        ax.fill_between(x[m], lo[m], hi[m], color=col, alpha=0.15, lw=0)
    for e in (2, 4, 8, 16, 32, 64):
        ax.axvline(e, color="#ddd", lw=0.8, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("apparent target size, larger box side (px)")
    ax.set_ylabel("detection probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Pd vs pixel size ({'centre-distance' if kind == 'centre' else 'IoU 0.5'} match, 95% bootstrap CI)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def pd_vs_range(reports: dict, out: str, span_range=(2.0, 2.6), width=640, hfovs=(50.0, 10.0), kind="centre"):
    """Each size bin maps to a range band: Z = f_px * span / p for the span limits (spec 5).
    Horizontal bars show that band; vertical bars the bootstrap CI."""
    fig, axes = plt.subplots(1, len(hfovs), figsize=(6 * len(hfovs), 4.0), squeeze=False)
    for ax, hf in zip(axes[0], hfovs):
        f = f_px(width, hf)
        for name, rep in reports.items():
            c = rep[f"pd_vs_size_{kind}"]
            x = _centres(c["edges"])
            y, lo, hi = (np.array(c[k], float) for k in ("pd", "lo", "hi"))
            m = ~np.isnan(y)
            r_lo, r_hi = f * span_range[0] / x, f * span_range[1] / x
            mid = np.sqrt(r_lo * r_hi)
            col = COLORS.get(name)
            ax.errorbar(mid[m], y[m], xerr=[mid[m] - r_lo[m], r_hi[m] - mid[m]], yerr=[y[m] - lo[m], hi[m] - y[m]],
                        fmt="o-", color=col, ms=3, lw=1, capsize=0, label=name)
        ax.set_xscale("log")
        ax.set_xlabel(f"range (m), {hf:g} deg HFOV, span {span_range[0]:.2f}-{span_range[1]:.2f} m")
        ax.set_ylabel("detection probability")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("geometry only, no atmospheric loss in the conversion")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def ap_per_bin(reports: dict, out: str, key: str = "ap50"):
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(reports)
    w = 0.8 / max(1, len(names))
    for k, name in enumerate(names):
        v = [reports[name]["bins"][b].get(key, np.nan) for b in NAMES]
        ax.bar(np.arange(len(NAMES)) + k * w - 0.4 + w / 2, v, w, label=name, color=COLORS.get(name))
    ax.set_xticks(range(len(NAMES)))
    ax.set_xticklabels([f"{b} px" for b in NAMES])
    ax.set_ylabel(key)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{key} per apparent-size bin (target class)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def all_plots(reports: dict, out_dir: str, span_range=(2.0, 2.6)):
    os.makedirs(out_dir, exist_ok=True)
    pd_vs_size(reports, os.path.join(out_dir, "pd_vs_size_centre.png"), "centre")
    pd_vs_size(reports, os.path.join(out_dir, "pd_vs_size_iou50.png"), "iou")
    pd_vs_range(reports, os.path.join(out_dir, "pd_vs_range.png"), span_range)
    ap_per_bin(reports, os.path.join(out_dir, "ap50_per_bin.png"), "ap50")
    ap_per_bin(reports, os.path.join(out_dir, "ap_centre_per_bin.png"), "ap_centre")

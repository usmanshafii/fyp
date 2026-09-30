"""Apparent-size bins (spec 1) and the pixel-size -> range conversion (spec 5).

Bins are experimental bins on the larger side of the projected box, not DRI standards. Range is a
band, not a number: the KAL's span is not published, so a pixel size maps to the ranges at which
a 2.00 m and a 2.60 m span would produce it (target.yaml).
"""
from __future__ import annotations

import math

NAMES = ["2-4", "4-8", "8-16", "16-32", "32-64", "64+"]
EDGES = [(2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, float("inf"))]


def bin_of(size_px: float) -> str | None:
    for n, (lo, hi) in zip(NAMES, EDGES):
        if lo <= size_px < hi:
            return n
    return None


def f_px(width: int, hfov_deg: float) -> float:
    return width / (2 * math.tan(math.radians(hfov_deg) / 2))


def range_for_pixels(p_px: float, extent_m: float, width: int = 640, hfov_deg: float = 10.0) -> float:
    """Spec 5: p = N W / (2 Z tan(HFOV/2))  ->  Z = f_px W / p."""
    return f_px(width, hfov_deg) * extent_m / p_px


def range_band(p_px: float, extent_range_m=(2.00, 2.60), width: int = 640, hfov_deg: float = 10.0):
    return tuple(range_for_pixels(p_px, e, width, hfov_deg) for e in extent_range_m)

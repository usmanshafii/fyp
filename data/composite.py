"""Layer compositing with haze (spec 4; FYP_Sky_and_Background.md, "Haze comes from the Mist pass").

Each band of a scene is stored as layers at output resolution, already through the base PSF and
the supersample integration:

    sky      world radiance seen directly (Env pass), premultiplied by the sky fraction of each pixel
    ground   terrain and structures (premultiplied, film transparent), with coverage `galpha`
    gdepth   coverage-weighted ray distance of the ground (Mist pass), metres
    horizon  per-column radiance just above the horizon (path radiance at long range)
    objects  per labelled object: premultiplied radiance P and coverage A in a crop, plus its range

Haze is applied here, so visibility can be swept without re-rendering (Sky doc):

    I_out = I e^{-kd} + L_inf (1 - e^{-kd}),   k = 3.912 / V   (x a band ratio for SWIR / LWIR)

  * sky pixels are left alone: the sky model already contains the atmosphere, and applying the
    formula with the Mist value of the sky (20 km) would flatten it to the horizon colour;
  * ground: L_inf is the horizon radiance of that column (aerial perspective);
  * objects: t = e^{-k d_obj}; over sky L_inf is the sky behind the object, so the object's
    contrast against the sky is multiplied by t exactly (the Sky doc's "retains 24% of its
    contrast at 3.7 km for V = 10 km"); over ground L_inf is the horizon radiance.

Objects are composited far-to-near with the "over" operator; their crops were rendered with every
other object and the terrain as holdouts, so occlusion is already resolved.
"""
from __future__ import annotations

import math
import os

import numpy as np

K_COEF = 3.912            # Koschmieder: 2% contrast threshold


def extinction(visibility_km: float, ratio: float = 1.0) -> float:
    """Extinction coefficient per metre for meteorological visibility V."""
    return K_COEF / (visibility_km * 1000.0) * ratio


def optical_depth(dist_m, k: float, elevation_deg: float | None = None, scale_height_m: float | None = None):
    """k d for a horizontal path. With a scale height H and a slant path at elevation el > 0.5 deg the
    aerosol density falls off with height: tau = k H / sin(el) (1 - exp(-d sin(el) / H))."""
    d = np.asarray(dist_m, float)
    if scale_height_m and elevation_deg is not None and elevation_deg > 0.5:
        s = math.sin(math.radians(elevation_deg))
        return k * scale_height_m / s * (1.0 - np.exp(-d * s / scale_height_m))
    return k * d


def transmittance(dist_m, k: float, elevation_deg=None, scale_height_m=None):
    return np.exp(-optical_depth(dist_m, k, elevation_deg, scale_height_m))


def _c(a):
    return a if a.ndim == 3 else a[..., None]


def background(layer: dict, k: float):
    """Hazed background and the path-radiance field L_inf (both HxWxC)."""
    sky = _c(layer["sky"]).astype(np.float32)
    if "ground" not in layer:
        return sky.copy(), sky
    ga = layer["galpha"].astype(np.float32)[..., None]
    Lh = np.asarray(layer["horizon"], np.float32)
    Lh = Lh.reshape(1, -1, sky.shape[-1]) if Lh.ndim == 2 else Lh.reshape(1, 1, -1)
    t = np.exp(-k * layer["gdepth"].astype(np.float32))[..., None]
    L_inf = sky + ga * Lh
    return sky + _c(layer["ground"]).astype(np.float32) * t + ga * Lh * (1.0 - t), L_inf


def composite(layer: dict, objects: list, k: float, scale_height_m: float | None = None,
              skip: set | None = None) -> np.ndarray:
    """Full frame for one band. objects: dicts with crop (x0, y0, x1, y1), P, A, range_m, elevation_deg."""
    out, L_inf = background(layer, k)
    for o in sorted(objects, key=lambda o: -o["range_m"]):
        if skip and o["name"] in skip:
            continue
        x0, y0, x1, y1 = o["crop"]
        P, A = _c(o["P"]).astype(np.float32), o["A"].astype(np.float32)[..., None]
        t = float(transmittance(o["range_m"], k, o.get("elevation_deg"), scale_height_m))
        reg = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = t * P + (1.0 - t) * A * L_inf[y0:y1, x0:x1] + (1.0 - A) * reg
    return out


# ============================================================================ storage
def _pack(a: np.ndarray, half: bool):
    a = np.asarray(a, np.float32)
    if not half:
        return a, np.float32(1.0)
    s = float(np.abs(a).max()) / 1024.0
    s = s if s > 0 else 1.0
    return (a / s).astype(np.float16), np.float32(s)


def save_layers(path: str, layer: dict, objects: list, half: bool = True, background_ref: str | None = None):
    """One compressed .npz per scene and band. Float16 with a per-array scale keeps 3 significant
    digits at every magnitude (night radiances are ~1e-5 of day).

    background_ref: path of another layers file, relative to this one, whose sky / ground layers this
    file shares (render/sequence.py: every frame of a fixed-camera sequence reuses one background)."""
    arrs = {}
    if background_ref is not None:
        arrs["background_ref"] = np.array(str(background_ref).replace("\\", "/"))
    for k in ("sky", "ground", "galpha", "gdepth"):
        if k in layer:
            v, s = _pack(layer[k], half and k != "gdepth")
            arrs[k], arrs[k + "__scale"] = v, s
    if "horizon" in layer:
        arrs["horizon"] = np.asarray(layer["horizon"], np.float32)
    names = []
    for o in objects:
        n = o["name"]
        names.append(n)
        for k in ("P", "A"):
            v, s = _pack(o[k], half)
            arrs[f"obj.{n}.{k}"], arrs[f"obj.{n}.{k}__scale"] = v, s
        arrs[f"obj.{n}.crop"] = np.asarray(o["crop"], np.int32)
        arrs[f"obj.{n}.geom"] = np.asarray([o["range_m"], o.get("elevation_deg") or 0.0], np.float64)
    arrs["objects"] = np.array(names)
    arrs["psf_sigma0"] = np.float32(layer.get("psf_sigma0", 0.0))
    np.savez_compressed(path, **arrs)


_SHARED = {}             # shared background layers, per process (a sequence has one per band)


def _shared_background(path: str) -> dict:
    if path not in _SHARED:
        if len(_SHARED) >= 8:
            _SHARED.pop(next(iter(_SHARED)))
        _SHARED[path] = load_layers(path)[0]
    return _SHARED[path]


def load_layers(path: str):
    z = np.load(path, allow_pickle=False)

    def get(k):
        v = z[k]
        return v.astype(np.float32) * float(z[k + "__scale"]) if k + "__scale" in z.files else v.astype(np.float32)
    layer = {k: get(k) for k in ("sky", "ground", "galpha", "gdepth") if k in z.files}
    if "horizon" in z.files:
        layer["horizon"] = z["horizon"]
    if "background_ref" in z.files:
        ref = os.path.normpath(os.path.join(os.path.dirname(path), str(z["background_ref"])))
        layer = {**_shared_background(ref), **layer}
    layer["psf_sigma0"] = float(z["psf_sigma0"])
    objects = []
    for n in z["objects"].tolist():
        g = z[f"obj.{n}.geom"]
        objects.append(dict(name=n, P=get(f"obj.{n}.P"), A=get(f"obj.{n}.A"), crop=z[f"obj.{n}.crop"].tolist(),
                            range_m=float(g[0]), elevation_deg=float(g[1])))
    return layer, objects

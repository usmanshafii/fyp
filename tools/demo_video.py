"""Live demo video: KAL flight sequences (render/sequence.py) through the RGB and LWIR sensor
models, a detector on every frame, and telemetry, written as MP4 per clip and all clips joined.

    python tools/demo_video.py --data out/demo --out out/demo/video

Sensors: the noise models of noise/*.py with their parameters, and the LWIR fixed-pattern noise,
drawn once per clip; shot, read and NETD noise are fresh every frame, and the LWIR display
stretch (AGC) is smoothed over time like a real camera's.

Detector (demo baseline, no training): per band the local background is removed (LWIR column
offsets first), the residual is averaged over 3 x 3 px and divided by a robust noise sigma; the
two score maps are fused by their larger value, and the strongest connected blob above 6 sigma is
the detection. It stands in for the trained RGB+LWIR network until that has been trained on the
rendered dataset; the video says so on screen.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]

from data import composite as cp  # noqa: E402
from noise import lwir_fpn, rgb_noise  # noqa: E402
from render import psf  # noqa: E402
from render.sampling import load_configs  # noqa: E402

W, H = 640, 512
VW, VH = 1280, 720
TOP, PANEL_Y, FOOT_Y = 56, 56, 568
GREEN, RED, AMBER = (60, 220, 110), (240, 80, 70), (255, 196, 60)
INK, MUTED, BG = (236, 240, 242), (150, 162, 170), (11, 15, 18)
CLIPS = [("day", "DAY"), ("dusk", "DUSK"), ("night", "NIGHT")]


def font(size, bold=False, mono=False):
    names = (["consolab.ttf", "consola.ttf"] if mono else []) + (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"])
    for n in names:
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", n)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ============================================================================ sensors
class SensorSim:
    """noise/rgb_noise.py and noise/lwir_fpn.py with per-clip parameters and a fixed LWIR pattern."""

    def __init__(self, cfgs, seed):
        self.S = cfgs["sensors"]
        rng = np.random.default_rng([seed, 77])
        self.p_rgb = rgb_noise.sample_params(rng, self.S["rgb_noise"])
        self.p_lw = lwir_fpn.sample_params(rng, self.S["lwir_noise"])
        self.col = rng.standard_normal(W) * self.p_lw["col_k"]
        self.pix = rng.standard_normal((H, W)) * self.p_lw["pix_k"]
        self.agc = None

    def rgb(self, L, rng, exp_ms):
        return rgb_noise.apply(L, rng, self.S["rgb_noise"], exp_ms, params=self.p_rgb)[0]

    def lwir(self, L, rng, T_air_k):
        p, cfg = self.p_lw, self.S["lwir_noise"]
        cpk = p["counts_per_k"]
        c = lwir_fpn.to_counts(L, T_air_k, p) + (self.col[None, :] + self.pix) * cpk
        c = c + rng.standard_normal(c.shape) * p["netd_k"] * cpk
        c = np.clip(np.round(c), 0, 2 ** int(cfg["bits"]) - 1)
        lo, hi = np.percentile(c, [p["agc_lo"], p["agc_hi"]])
        self.agc = (lo, hi) if self.agc is None else (0.9 * self.agc[0] + 0.1 * lo, 0.9 * self.agc[1] + 0.1 * hi)
        lo, hi = self.agc
        return np.clip((c - lo) / max(hi - lo, 1.0), 0.0, 1.0)


def radiance(root, meta, band):
    """Haze and the rest of the PSF, as in data/dataset.py (evaluation path)."""
    layer, objects = cp.load_layers(os.path.join(root, meta["modalities"][band]["layers"]))
    w = meta["extra"]["weather"]
    img = cp.composite(layer, objects, cp.extinction(w["visibility_km"], w["band_ratio"][band]), w.get("aerosol_scale_height_m"))
    return psf.gaussian_blur(img, psf.extra_sigma(meta["extra"]["psf"][band]["nominal"], layer["psf_sigma0"]))


# ============================================================================ detector
class ContrastDetector:
    def __init__(self, thresh=8.0, grow=3.0, bg_sigma=10.0, border=16):
        self.thresh, self.grow, self.bg_sigma, self.border = thresh, grow, bg_sigma, border

    def score(self, img, band):
        Y = img.mean(-1) if img.shape[-1] == 3 else img[..., 0]
        Y = Y.astype(np.float32)
        if band == "lwir":
            Y = Y - np.median(Y, axis=0, keepdims=True)              # column offsets (FPN)
        r = ndi.uniform_filter(Y - ndi.gaussian_filter(Y, self.bg_sigma), 3)
        sigma = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-9
        return np.abs(r) / sigma

    def __call__(self, imgs):
        z = {b: self.score(im, b) for b, im in imgs.items()}
        zf = np.maximum.reduce(list(z.values()))
        e = self.border                          # the local background is unreliable at the frame edge
        zf[:e], zf[-e:], zf[:, :e], zf[:, -e:] = 0, 0, 0, 0
        lab, n = ndi.label(zf > self.grow)
        dets = []
        if n:
            peaks = ndi.maximum(zf, lab, index=np.arange(1, n + 1))
            for k, (pk, sl) in enumerate(zip(peaks, ndi.find_objects(lab))):
                if pk < self.thresh:
                    continue
                m = lab[sl] == k + 1
                ys, xs = np.nonzero(m & (zf[sl] >= max(self.grow, 0.3 * pk)))     # the core, not the faint halo
                y0, x0 = sl[0].start, sl[1].start
                dets.append(dict(box=[x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1],
                                 score=float(pk), bands={b: float(zz[sl][m].max()) for b, zz in z.items()}))
        return sorted(dets, key=lambda d: -d["score"])


# ============================================================================ drawing
def to_rgb8(img):
    a = np.clip(img, 0, 1)
    a = np.repeat(a, 3, -1) if a.shape[-1] == 1 else a
    return (a * 255 + 0.5).astype(np.uint8)


def zoom(arr8, cx, cy, win=34, scale=4):
    x0 = int(np.clip(round(cx - win / 2), 0, W - win))
    y0 = int(np.clip(round(cy - win / 2), 0, H - win))
    return Image.fromarray(arr8[y0:y0 + win, x0:x0 + win]).resize((win * scale, win * scale), Image.NEAREST), (x0, y0)


def draw_frame(clip, meta, seq, imgs8, dets, F):
    fr = Image.new("RGB", (VW, VH), BG)
    d = ImageDraw.Draw(fr)
    tgt = meta["target"]
    t = meta["sequence"]["t_s"]
    # header
    d.text((16, 12), "KAL detection demo", font=F["h1"], fill=INK)
    chip_col = {"day": (250, 204, 90), "dusk": (240, 140, 90), "night": (120, 150, 240)}[clip]
    cx0 = 16 + d.textlength("KAL detection demo", font=F["h1"]) + 16
    d.rounded_rectangle((cx0, 14, cx0 + 86, 44), 6, fill=chip_col)
    d.text((cx0 + 43, 29), clip.upper(), font=F["chip"], fill=(12, 14, 16), anchor="mm")
    d.text((cx0 + 102, 18), "synthetic RGB + LWIR sensor simulation", font=F["small"], fill=MUTED)
    d.text((VW - 16, 16), f"t = {t:5.2f} s", font=F["mono"], fill=INK, anchor="ra")
    # panels
    for i, (b, label) in enumerate((("rgb", "RGB  visible"), ("lwir", "LWIR  thermal 8-14 um, white-hot"))):
        x = i * W
        fr.paste(Image.fromarray(imgs8[b]), (x, PANEL_Y))
        d.rectangle((x + 8, PANEL_Y + 8, x + 8 + d.textlength(label, font=F["small"]) + 12, PANEL_Y + 30), fill=(0, 0, 0))
        d.text((x + 14, PANEL_Y + 10), label, font=F["small"], fill=INK)
        for k, det in enumerate(dets):
            x0, y0, x1, y1 = det["box"]
            col = GREEN if k == 0 else AMBER
            d.rectangle((x + x0 - 4, PANEL_Y + y0 - 4, x + x1 + 3, PANEL_Y + y1 + 3), outline=col, width=2)
            tag = f"KAL {det['score']:.0f}\u03c3" if k == 0 else f"? {det['score']:.0f}\u03c3"
            ty = PANEL_Y + y0 - 24 if y0 > 26 else PANEL_Y + y1 + 6
            d.rectangle((x + x0 - 4, ty, x + x0 - 4 + d.textlength(tag, font=F["tag"]) + 8, ty + 19), fill=col)
            d.text((x + x0, ty + 1), tag, font=F["tag"], fill=(8, 12, 10))
    d.line((W, PANEL_Y, W, PANEL_Y + H), fill=BG, width=2)
    # zoom insets around the detection (or where the drone is, if nothing was detected)
    if dets:
        x0, y0, x1, y1 = dets[0]["box"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    else:
        bb = tgt["bbox_px"]
        cx, cy = bb["x"] + bb["w"] / 2, bb["y"] + bb["h"] / 2
    for i, b in enumerate(("rgb", "lwir")):
        z, (zx, zy) = zoom(imgs8[b], cx, cy)
        px = 16 + i * 150
        fr.paste(z, (px, FOOT_Y + 8))
        if dets:
            bx0, by0, bx1, by1 = dets[0]["box"]
            d.rectangle((px + (bx0 - zx) * 4 - 6, FOOT_Y + 8 + (by0 - zy) * 4 - 6, px + (bx1 - zx) * 4 + 5,
                         FOOT_Y + 8 + (by1 - zy) * 4 + 5), outline=GREEN, width=2)
        d.rectangle((px, FOOT_Y + 8, px + 58, FOOT_Y + 26), fill=(0, 0, 0))
        d.text((px + 4, FOOT_Y + 9), f"{b.upper()} x4", font=F["tiny"], fill=INK)
    # telemetry
    tx = 330
    rows = [("Range", f"{tgt['range_m']:,.0f} m"), ("Altitude", f"{tgt['agl_m']:,.0f} m AGL"),
            ("Speed", f"{tgt['speed_mps'] * 3.6:.0f} km/h"), ("Heading", f"{tgt['heading_deg']:.0f}\u00b0")]
    for k, (a, v) in enumerate(rows):
        d.text((tx + k * 170, FOOT_Y + 10), a.upper(), font=F["tiny"], fill=MUTED)
        d.text((tx + k * 170, FOOT_Y + 26), v, font=F["val"], fill=INK)
    if dets:
        bs = dets[0]["bands"]
        status = f"DETECTED   score {dets[0]['score']:.0f}\u03c3   (RGB {bs['rgb']:.0f}\u03c3, LWIR {bs['lwir']:.0f}\u03c3)"
        col = GREEN
    else:
        status, col = "NO DETECTION", RED
    d.ellipse((tx, FOOT_Y + 74, tx + 14, FOOT_Y + 88), fill=col)
    d.text((tx + 22, FOOT_Y + 70), status, font=F["status"], fill=col)
    w = seq["weather"]
    d.text((tx, FOOT_Y + 102), f"Target {meta['extra']['objects'][0]['size_px']:.0f} px  \u00b7  exposure "
           f"{meta['camera']['exposure_ms']:.1f} ms  \u00b7  visibility {w['visibility_km']:.0f} km  \u00b7  air "
           f"{seq['air_temp_c']:.0f} \u00b0C  \u00b7  10\u00b0 lens, 640 x 512", font=F["small"], fill=MUTED)
    d.text((tx, FOOT_Y + 124), "Demo contrast detector (not the trained network).  KAL cruise 160-200 km/h, "
           "up to 5,000 m: Indian Masterminds, 21 Sep 2026.", font=F["tiny"], fill=MUTED)
    return fr


def title_card(lines, F):
    fr = Image.new("RGB", (VW, VH), BG)
    d = ImageDraw.Draw(fr)
    y = VH // 2 - 30 * len(lines)
    for k, (text, key, col) in enumerate(lines):
        d.text((VW // 2, y), text, font=F[key], fill=col, anchor="mm")
        y += 64 if k == 0 else 40
    return fr


# ============================================================================ main
def run_clip(root, clip, cfgs, det, out, F, fps):
    seq_path = os.path.join(root, "sequences", f"{clip}.json")
    if not os.path.exists(seq_path):
        print(f"[{clip}] no sequence at {seq_path}, skipped")
        return None
    with open(seq_path) as f:
        seq = json.load(f)
    frames_dir = os.path.join(out, f"frames_{clip}")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    fr0 = seq["frames"][0]
    title = title_card([(f"{clip.upper()}", "title", INK),
                        (f"KAL at {fr0['speed_mps'] * 3.6:.0f} km/h, {fr0['agl_m']:.0f} m altitude, "
                         f"{seq['frames'][0]['range_m'] / 1000:.1f} to {seq['frames'][-1]['range_m'] / 1000:.1f} km", "sub", MUTED),
                        ("RGB (left) and LWIR thermal (right), detector boxes in green", "sub", MUTED)], F)
    n_title = int(round(1.8 * fps))
    for k in range(n_title):
        title.save(os.path.join(frames_dir, f"{k:05d}.png"), compress_level=1)
    sim = SensorSim(cfgs, seq["seq_id"])
    hits, scores = 0, []
    for k, fid in enumerate(seq["frame_ids"]):
        with open(os.path.join(root, "meta", f"{fid}.json")) as f:
            meta = json.load(f)
        rng = np.random.default_rng([seq["seq_id"], k])
        imgs = {"rgb": sim.rgb(radiance(root, meta, "rgb"), rng, meta["camera"]["exposure_ms"]),
                "lwir": sim.lwir(radiance(root, meta, "lwir")[..., 0], rng, seq["air_temp_c"] + 273.15)[..., None]}
        dets = det(imgs)
        if dets:
            hits += 1
            scores.append(dets[0]["score"])
        img = draw_frame(clip, meta, seq, {b: to_rgb8(v) for b, v in imgs.items()}, dets[:3], F)
        img.save(os.path.join(frames_dir, f"{n_title + k:05d}.png"), compress_level=1)
    shutil.copy(os.path.join(frames_dir, f"{n_title + len(seq['frame_ids']) // 2:05d}.png"), os.path.join(out, f"still_{clip}.png"))
    mp4 = os.path.join(out, f"kal_demo_{clip}.mp4")
    subprocess.run(["ffmpeg", "-y", "-r", str(fps), "-i", os.path.join(frames_dir, "%05d.png"), "-c:v", "libx264",
                    "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", mp4],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = len(seq["frame_ids"])
    print(f"[{clip}] {hits}/{n} frames detected, median score {np.median(scores) if scores else 0:.1f} sigma -> {mp4}")
    return dict(clip=clip, mp4=mp4, frames=n, detected=hits, median_score=float(np.median(scores)) if scores else 0.0,
                range_m=[fr0["range_m"], seq["frames"][-1]["range_m"]], agl_m=fr0["agl_m"], speed_kmh=fr0["speed_mps"] * 3.6,
                size_px=[fr0["size_px"], seq["frames"][-1]["size_px"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "out", "demo"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=25.0)
    a = ap.parse_args()
    a.data = os.path.abspath(a.data)
    out = os.path.abspath(a.out or os.path.join(a.data, "video"))
    os.makedirs(out, exist_ok=True)
    cfgs = load_configs()
    F = dict(h1=font(26, bold=True), chip=font(15, bold=True), small=font(15), tiny=font(13), mono=font(20, mono=True),
             tag=font(14, bold=True), val=font(24, bold=True), status=font(20, bold=True), title=font(64, bold=True),
             sub=font(24))
    det = ContrastDetector()
    results = [r for r in (run_clip(a.data, c, cfgs, det, out, F, a.fps) for c, _ in CLIPS) if r]
    if not results:
        sys.exit("no sequences rendered yet (render/sequence.py)")
    intro = os.path.join(out, "frames_intro")
    shutil.rmtree(intro, ignore_errors=True)
    os.makedirs(intro)
    card = title_card([("KAL detection demo", "title", INK),
                       ("Synthetic multispectral simulation: Blender Cycles renders + RGB and LWIR sensor noise models", "sub", MUTED),
                       ("Day, dusk and night flights of the KAL at its published cruise speed", "sub", MUTED)], F)
    for k in range(int(round(2.5 * a.fps))):
        card.save(os.path.join(intro, f"{k:05d}.png"), compress_level=1)
    intro_mp4 = os.path.join(out, "intro.mp4")
    subprocess.run(["ffmpeg", "-y", "-r", str(a.fps), "-i", os.path.join(intro, "%05d.png"), "-c:v", "libx264",
                    "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", intro_mp4],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lst = os.path.join(out, "concat.txt")
    with open(lst, "w") as f:
        for p in [intro_mp4] + [r["mp4"] for r in results]:
            f.write(f"file '{os.path.basename(p)}'\n")
    full = os.path.join(out, "kal_detection_demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-i", lst, "-c", "copy", full], cwd=out,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # a smaller copy for web pages: night RGB is mostly sensor noise, which barely compresses
    subprocess.run(["ffmpeg", "-y", "-i", full, "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p",
                    os.path.join(out, "kal_detection_demo_web.mp4")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(dict(video=full, clips=results), f, indent=1)
    for p in glob.glob(os.path.join(out, "frames_*")):
        shutil.rmtree(p, ignore_errors=True)
    print("video:", full)


if __name__ == "__main__":
    main()

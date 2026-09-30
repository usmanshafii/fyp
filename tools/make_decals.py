"""Generate assets/decals/decal_atlas.png (+ .json UV regions) for the decal slot.

Markings only matter at 64+ px, so plain fonts are enough. Regions are sized to the
aspect ratio of the decal quads in configs/target.yaml so nothing is stretched.
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "target"))
import kal_geometry as kg  # noqa: E402

OUT = os.path.join(HERE, "..", "assets", "decals")
N = 2048
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def fit_text(draw, box, text, path):
    x0, y0, x1, y1 = box
    size = int((y1 - y0) * 0.9)
    while size > 8:
        f = font(path, size)
        l, t, r, b = draw.textbbox((0, 0), text, font=f)
        if r - l <= (x1 - x0) * 0.92 and b - t <= (y1 - y0) * 0.9:
            break
        size -= 4
    l, t, r, b = draw.textbbox((0, 0), text, font=f)
    draw.text(((x0 + x1 - (r - l)) / 2 - l, (y0 + y1 - (b - t)) / 2 - t), text, font=f, fill=(235, 235, 235, 255))


def main():
    g = kg.derive(kg.defaults(kg.load_config()))
    os.makedirs(OUT, exist_ok=True)
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    regions = {}

    def region(name, top_px, aspect):          # full-width band, height from aspect
        h = int(N / aspect)
        box = (0, top_px, N, top_px + h)
        # PIL y runs down, UV v runs up
        regions[name] = [0.0, 1 - (top_px + h) / N, 1.0, 1 - top_px / N]
        return box, top_px + h + 16

    L, H = g["wing_text_size"]
    box, y = region("wing_text", 0, L / H)
    fit_text(d, box, "KAL", SERIF)

    box, y = region("fin_text", y, 3.2)
    fit_text(d, box, "IG DEFENCE", SANS)

    fl, fw = g["flag_size"]
    h = int(N / 2 / (fl / fw))
    x0, y0, x1, y1 = 0, y, N // 2, y + h
    s = h // 3
    d.rectangle((x0, y0, x1, y0 + s), fill=(255, 153, 51, 255))
    d.rectangle((x0, y0 + s, x1, y0 + 2 * s), fill=(255, 255, 255, 255))
    d.rectangle((x0, y0 + 2 * s, x1, y1), fill=(19, 136, 8, 255))
    cx, cy, r = (x0 + x1) // 2, y0 + int(1.5 * s), int(s * 0.42)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 0, 128, 255), width=max(2, r // 8))
    regions["flag"] = [0.0, 1 - y1 / N, 0.5, 1 - y0 / N]

    img.save(os.path.join(OUT, "decal_atlas.png"))
    with open(os.path.join(OUT, "decal_atlas.json"), "w") as f:
        json.dump(regions, f, indent=2)
    print("wrote", os.path.join(OUT, "decal_atlas.png"), regions)


if __name__ == "__main__":
    main()

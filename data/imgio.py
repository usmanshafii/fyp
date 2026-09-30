"""Minimal OpenEXR and PNG I/O in pure NumPy + zlib.

Runs unchanged inside Blender's bundled Python (no OpenEXR, OpenCV or PIL there), so the
render post-processing and the data loader share one reader.

EXR support: single-part scanline files, compression NONE / ZIPS / ZIP, channel types
UINT / HALF / FLOAT. That covers everything this project writes; Blender renders are
saved with the ZIP codec (render/scene.py sets it).
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

_MAGIC = 20000630
_LINES = {0: 1, 2: 1, 3: 16}                       # compression id -> scanlines per chunk
_CNAME = {0: "NONE", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ", 5: "PXR24", 6: "B44", 7: "B44A",
          8: "DWAA", 9: "DWAB"}
_DT = {0: np.dtype("<u4"), 1: np.dtype("<f2"), 2: np.dtype("<f4")}


# ============================================================================ EXR read
def _cstr(buf, pos):
    end = buf.index(b"\x00", pos)
    return buf[pos:end].decode("latin-1"), end + 1


def _header(buf):
    magic, version = struct.unpack_from("<ii", buf, 0)
    if magic != _MAGIC:
        raise ValueError("not an OpenEXR file")
    if version & 0x200 or version & 0x800 or version & 0x1000:
        raise ValueError("tiled, deep and multi-part EXR files are not supported")
    pos, attrs = 8, {}
    while buf[pos] != 0:
        name, pos = _cstr(buf, pos)
        typ, pos = _cstr(buf, pos)
        size = struct.unpack_from("<i", buf, pos)[0]
        pos += 4
        attrs[name] = (typ, buf[pos:pos + size])
        pos += size
    return attrs, pos + 1


def _channels(raw):
    out, pos = [], 0
    while raw[pos] != 0:
        name, pos = _cstr(raw, pos)
        ptype, _plin, xs, ys = struct.unpack_from("<iB3xii", raw, pos)
        pos += 16
        if xs != 1 or ys != 1:
            raise ValueError("subsampled EXR channels are not supported")
        out.append((name, ptype))
    return out


def _unzip(data, expected):
    if len(data) == expected:                      # stored raw when compression did not help
        return data
    t = np.frombuffer(zlib.decompress(data), np.uint8).astype(np.int64)
    t[1:] -= 128
    t = (np.cumsum(t) & 0xFF).astype(np.uint8)
    half = (len(t) + 1) // 2
    out = np.empty_like(t)
    out[0::2], out[1::2] = t[:half], t[half:]
    return out.tobytes()


def read_exr(path: str, channels: list[str] | None = None) -> dict[str, np.ndarray]:
    """Return {channel name: HxW float32 array}, rows top to bottom."""
    with open(path, "rb") as f:
        buf = f.read()
    attrs, pos = _header(buf)
    comp = attrs["compression"][1][0]
    if comp not in _LINES:
        raise ValueError(f"EXR compression {_CNAME.get(comp, comp)} not supported (use ZIP, ZIPS or NONE)")
    chans = _channels(attrs["channels"][1])
    x0, y0, x1, y1 = struct.unpack("<iiii", attrs["dataWindow"][1])
    w, h = x1 - x0 + 1, y1 - y0 + 1
    lpc = _LINES[comp]
    nchunk = (h + lpc - 1) // lpc
    offsets = struct.unpack_from(f"<{nchunk}Q", buf, pos)
    want = [c for c in chans if channels is None or c[0] in channels]
    out = {name: np.empty((h, w), np.float32) for name, _ in want}
    row_bytes = sum(_DT[t].itemsize for _, t in chans) * w
    for off in offsets:
        yc, size = struct.unpack_from("<ii", buf, off)
        data = buf[off + 8: off + 8 + size]
        r0 = yc - y0
        nrows = min(lpc, h - r0)
        expected = row_bytes * nrows
        raw = data if comp == 0 else _unzip(data, expected)
        p = 0
        for r in range(nrows):
            for name, ptype in chans:
                n = _DT[ptype].itemsize * w
                if name in out:
                    out[name][r0 + r] = np.frombuffer(raw, _DT[ptype], w, p)
                p += n
    return out


def exr_channel_names(path: str) -> list[str]:
    with open(path, "rb") as f:
        buf = f.read(65536)
    attrs, _ = _header(buf)
    return [n for n, _ in _channels(attrs["channels"][1])]


# ============================================================================ EXR write
def _attr(name, typ, value: bytes) -> bytes:
    return name.encode() + b"\x00" + typ.encode() + b"\x00" + struct.pack("<i", len(value)) + value


def write_exr(path: str, channels: dict[str, np.ndarray], half: bool = False, compress: bool = True):
    """Write HxW arrays as one scanline EXR. Channel names are stored sorted (EXR rule)."""
    names = sorted(channels)
    h, w = channels[names[0]].shape
    ptype = 1 if half else 2
    dt = _DT[ptype]
    chl = b"".join(n.encode() + b"\x00" + struct.pack("<iB3xii", ptype, 0, 1, 1) for n in names) + b"\x00"
    comp = 3 if compress else 0
    lpc = _LINES[comp]
    hdr = (struct.pack("<ii", _MAGIC, 2)
           + _attr("channels", "chlist", chl)
           + _attr("compression", "compression", bytes([comp]))
           + _attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, w - 1, h - 1))
           + _attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, w - 1, h - 1))
           + _attr("lineOrder", "lineOrder", b"\x00")
           + _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
           + _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
           + _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
           + b"\x00")
    arrs = [np.ascontiguousarray(channels[n], dtype=dt) for n in names]
    chunks = []
    for r0 in range(0, h, lpc):
        rows = range(r0, min(h, r0 + lpc))
        raw = b"".join(a[r].tobytes() for r in rows for a in arrs)
        if comp:
            t = np.frombuffer(raw, np.uint8)
            t = np.concatenate([t[0::2], t[1::2]]).astype(np.int16)
            d = np.empty_like(t)
            d[0], d[1:] = t[0], (t[1:] - t[:-1] + 128 + 256) & 0xFF
            packed = zlib.compress(d.astype(np.uint8).tobytes(), 6)
            if len(packed) >= len(raw):
                packed = raw
        else:
            packed = raw
        chunks.append(struct.pack("<ii", r0, len(packed)) + packed)
    table_pos = len(hdr)
    offs, p = [], table_pos + 8 * len(chunks)
    for c in chunks:
        offs.append(p)
        p += len(c)
    with open(path, "wb") as f:
        f.write(hdr + struct.pack(f"<{len(offs)}Q", *offs) + b"".join(chunks))


# ============================================================================ PNG write
def write_png(path: str, img: np.ndarray):
    """8-bit PNG from a uint8 HxW, HxWx3 or HxWx4 array."""
    a = np.ascontiguousarray(img, np.uint8)
    h, w = a.shape[:2]
    c = 1 if a.ndim == 2 else a.shape[2]
    ctype = {1: 0, 2: 4, 3: 2, 4: 6}[c]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)

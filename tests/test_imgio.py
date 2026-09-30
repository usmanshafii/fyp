"""Pure-NumPy EXR / PNG I/O."""
import numpy as np
import pytest

from data.imgio import exr_channel_names, read_exr, write_exr, write_png


@pytest.mark.parametrize("half,compress", [(False, True), (False, False), (True, True)])
def test_exr_roundtrip(tmp_path, half, compress):
    rng = np.random.default_rng(0)
    ch = {"ViewLayer.Combined.R": rng.random((37, 53)).astype(np.float32) * 100,
          "ViewLayer.Mist.Z": np.linspace(0, 1, 37 * 53, dtype=np.float32).reshape(37, 53),
          "A": np.zeros((37, 53), np.float32)}
    p = str(tmp_path / "t.exr")
    write_exr(p, ch, half=half, compress=compress)
    assert exr_channel_names(p) == sorted(ch)
    back = read_exr(p)
    for k, v in ch.items():
        assert np.allclose(back[k], v, rtol=1e-3 if half else 0, atol=1e-3 if half else 0)


def test_png_is_readable(tmp_path):
    from PIL import Image
    img = (np.arange(24 * 30 * 3) % 251).astype(np.uint8).reshape(24, 30, 3)
    p = str(tmp_path / "t.png")
    write_png(p, img)
    assert np.array_equal(np.asarray(Image.open(p)), img)

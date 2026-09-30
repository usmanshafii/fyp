"""Detector, assignment, loss, distillation (spec 7)."""
import pytest
import torch

from models.detector import Detector
from models.distill import kd_loss
from models.loss import Assigner, DetLoss, nms

W, H = 128, 96
SMALL = dict(width=(8, 16, 16, 32, 32), depth=(1, 1, 1, 1))


def _imgs(bands, B=2):
    return {b: torch.rand(B, 3 if b == "rgb" else 1, H, W) for b in bands}


@pytest.mark.parametrize("p2", [True, False])
def test_forward_shapes(p2):
    m = Detector(["rgb", "lwir"], 7, p2=p2, **SMALL)
    raw = m(_imgs(m.bands))
    assert [r[0].shape[-1] for r in raw] == ([W // 4] if p2 else []) + [W // 8, W // 16, W // 32]
    cls, boxes, anchors, strides = m.flatten(raw)
    assert cls.shape[:2] == boxes.shape[:2] and cls.shape[-1] == 7 and anchors.shape[0] == cls.shape[1]


def test_tiny_object_always_gets_a_positive():
    m = Detector(["rgb"], 7, p2=True, **SMALL)
    cls, boxes, anchors, strides = m.flatten(m(_imgs(["rgb"], 1)))
    gt = torch.tensor([[50.2, 30.4, 52.9, 32.3]])          # 2.7 x 1.9 px: no stride-4 anchor centre inside
    ts, tb, fg, _ = Assigner(7)(cls[0].sigmoid(), boxes[0], anchors, strides, gt, torch.tensor([0]))
    assert fg.sum() >= 1 and ts[fg, 0].min() > 0
    assert strides[fg].max() <= 8                          # tiny objects stay on P2 / P3


def test_loss_backward_and_modality_dropout():
    m = Detector(["rgb", "lwir"], 7, modality_dropout=0.9, **SMALL).train()
    tg = [dict(boxes=torch.tensor([[10.0, 10, 14, 13], [60, 40, 90, 60]]), labels=torch.tensor([0, 1])),
          dict(boxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long))]
    loss, parts = DetLoss(7)(m, m(_imgs(m.bands)), tg)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in m.heads.parameters())
    mask = m.fusion.band_mask(1000, "cpu", {"rgb": True, "lwir": True})
    assert (mask.sum(1) >= 1).all()                        # never drops every band


def test_missing_band_and_predict():
    m = Detector(["rgb", "lwir"], 7, **SMALL).eval()
    out = m.predict({"rgb": None, "lwir": torch.rand(1, 1, H, W)}, conf=0.0, max_det=20)
    assert len(out) == 1 and out[0]["boxes"].shape[1] == 4 and len(out[0]["scores"]) <= 20


def test_distillation_loss():
    t = Detector(["rgb", "lwir", "swir"], 7, **SMALL).eval()
    s = Detector(["rgb", "lwir"], 7, **SMALL).train()
    imgs = _imgs(["rgb", "lwir", "swir"])
    with torch.no_grad():
        traw = t(imgs)
    loss, _ = kd_loss(s, s({b: imgs[b] for b in s.bands}), t, traw, score_thresh=0.0)
    loss.backward()
    assert torch.isfinite(loss)


def test_nms_suppresses_tiny_duplicates():
    b = torch.tensor([[10.0, 10, 13, 13], [11.0, 11, 14, 14], [40, 40, 43, 43]])   # IoU(0, 1) ~ 0.29
    keep = nms(b, torch.tensor([0.9, 0.8, 0.7]), torch.tensor([0, 0, 0]), 0.5, 2.0)
    assert keep.tolist() == [0, 2]

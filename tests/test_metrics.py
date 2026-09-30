"""Evaluation protocol (spec 8)."""
import numpy as np
import pytest

from eval import metrics as M
from eval.nwd import iou_matrix, nwd_matrix

CFG = dict(centre_match=dict(abs_px=2.0, rel=0.5), nwd_match=dict(C=12.8, thresh=0.8), iou_thresholds=[0.5, 0.75],
           score_thresh=0.3, negative_match_iou=0.1, tiny_bins=["2-4", "4-8", "8-16"],
           pd_size_edges_px=[2, 4, 8, 16, 32, 64, 256], bootstrap=50)


def rec(sid, preds, gts):
    b = np.array([p[0] for p in preds], float).reshape(-1, 4)
    return dict(scene_id=sid,
                preds=dict(boxes=b, scores=np.array([p[1] for p in preds]), labels=np.array([p[2] for p in preds])),
                gts=[dict(box=g[0], label=g[1], size_px=max(g[0][2] - g[0][0], g[0][3] - g[0][1]), size_bin=g[2],
                          range_m=1000.0, cls_name=g[3]) for g in gts],
                conditions=dict(lighting="day", weather="clear", thermal_state="day", background="open_sky"),
                camera=dict(hfov_deg=10.0))


def test_iou_unstable_for_tiny_boxes_but_centre_and_nwd_are_not():
    a, b = [[10, 10, 13, 13]], [[11, 11, 14, 14]]
    assert iou_matrix(a, b)[0, 0] == pytest.approx(4 / 14, abs=1e-6)     # ~0.29 after a 1-px diagonal shift
    assert nwd_matrix(a, b)[0, 0] > 0.85
    r = [rec("s1", [([11, 11, 14, 14], 0.9, 0)], [([10, 10, 13, 13], 0, "2-4", "target")])]
    assert M.ap(r, "iou", 0.5, cfg=CFG) == 0.0
    assert M.ap(r, "centre", 0.0, cfg=CFG) == pytest.approx(1.0)
    assert M.ap(r, "nwd", 0.0, cfg=CFG) == pytest.approx(1.0)


def test_perfect_and_per_bin_ignore():
    r = [rec("s1", [([10, 10, 13, 13], 0.9, 0), ([100, 100, 150, 140], 0.8, 0)],
             [([10, 10, 13, 13], 0, "2-4", "target"), ([100, 100, 150, 140], 0, "32-64", "target")])]
    assert M.ap(r, "iou", 0.5, cfg=CFG) == pytest.approx(1.0)
    assert M.ap(r, "iou", 0.5, "2-4", cfg=CFG) == pytest.approx(1.0)      # the 50 px hit is ignored, not an FP
    assert np.isnan(M.ap(r, "iou", 0.5, "8-16", cfg=CFG))


def test_fppi_charged_to_negative_class():
    r = [rec("s1", [([10, 10, 13, 13], 0.9, 0), ([50, 50, 56, 54], 0.8, 0), ([200, 200, 204, 204], 0.7, 0)],
             [([10, 10, 13, 13], 0, "2-4", "target"), ([50, 50, 56, 54], 1, "4-8", "bird")])]
    op = M.operating_point(r, CFG, "iou", 0.5)
    assert op["tp"] == 1 and op["fp"] == 2
    assert op["fppi_by_class"] == {"background": 1.0, "bird": 1.0}


def test_bootstrap_and_summary_run():
    rs = [rec(f"s{i}", [([10, 10, 13, 13], 0.9, 0)] if i % 2 else [], [([10, 10, 13, 13], 0, "2-4", "target")])
          for i in range(10)]
    s = M.summary(rs, CFG)
    assert s["bins"]["2-4"]["op_centre"]["recall"] == pytest.approx(0.5)
    d = M.bootstrap_diff(rs, rs, lambda r: M.ap(r, "centre", 0.0, cfg=CFG), 50)
    assert d["diff"] == 0.0 and not d["significant"]

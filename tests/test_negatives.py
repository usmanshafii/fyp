"""Hard-negative meshes: closed, outward-facing, sized from the span."""
import numpy as np
import pytest

from render import negatives as neg


@pytest.mark.parametrize("cls", list(neg.BUILDERS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_closed_and_sized(cls, seed):
    rng = np.random.default_rng(seed)
    span = {"bird": 1.0, "kite": 1.2, "light_aircraft": 10.0, "helicopter": 11.0, "small_uav": 0.6, "warm_clutter": 2.0}[cls]
    m = neg.build(cls, rng, span)
    assert neg.edge_report(m) == {}
    ext = np.ptp(m["vertices"], axis=0).max()
    assert 0.3 * span < ext < 3.5 * span
    assert np.allclose(0.5 * (m["vertices"].min(0) + m["vertices"].max(0)), 0, atol=1e-9)
    assert set(np.unique(m["face_slots"])) <= set(range(len(neg.NEG_SLOTS)))

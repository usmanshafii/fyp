"""ONNX export and CPU latency (Phase 7: edge proxy, not edge optimisation).

    python eval/latency.py --run out/runs/student --threads 4

Exports <run>/model.onnx (inputs: one [1, C, 512, 640] tensor per band in [0, 1]; outputs: raw
class logits and box parameters per level) and times it with onnxruntime on the CPU. Without
onnxruntime installed it times the PyTorch model instead and says so.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models.detector import BAND_CHANNELS  # noqa: E402


class _Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, *xs):
        outs = self.m({b: x for b, x in zip(self.m.bands, xs)})
        return tuple(t for pair in outs for t in pair)


def time_fn(fn, warm=3, n=20):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return dict(mean_ms=1000 * float(np.mean(ts)), p50_ms=1000 * float(np.median(ts)), p90_ms=1000 * float(np.percentile(ts, 90)))


def main():
    from train.engine import load_ckpt
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    ck = a.run if a.run.endswith(".pt") else os.path.join(a.run, "best.pt")
    out_dir = os.path.dirname(ck)
    model, _ = load_ckpt(ck, "cpu")
    torch.set_num_threads(a.threads)
    xs = tuple(torch.rand(1, BAND_CHANNELS[b], 512, 640) for b in model.bands)
    w = _Wrap(model).eval()
    res = dict(bands=model.bands, params_m=sum(p.numel() for p in model.parameters()) / 1e6, threads=a.threads)
    with torch.no_grad():
        res["torch_cpu"] = time_fn(lambda: w(*xs), n=a.n)
    onnx_path = os.path.join(out_dir, "model.onnx")
    try:
        torch.onnx.export(w, xs, onnx_path, input_names=list(model.bands), opset_version=17,
                          output_names=[f"{k}{i}" for i in range(len(model.strides)) for k in ("cls", "box")])
        res["onnx"] = onnx_path
    except Exception as e:                     # export needs the `onnx` package on some torch versions
        res["onnx_error"] = repr(e)
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = a.threads
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        feeds = {b: x.numpy() for b, x in zip(model.bands, xs)}
        res["onnxruntime_cpu"] = time_fn(lambda: sess.run(None, feeds), n=a.n)
    except ImportError:
        res["onnxruntime_cpu"] = "onnxruntime not installed (pip install onnxruntime); torch CPU timing above"
    except Exception as e:
        res["onnxruntime_error"] = repr(e)
    print(json.dumps(res, indent=1))
    with open(os.path.join(out_dir, "latency.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

"""Shared training loop for the baselines, the teacher and the distilled student (spec 7, Phase 6).

    AdamW, linear warm-up + cosine decay, EMA weights, gradient clipping, validation AP50 on the
    target class every `eval_every` epochs; best.pt / last.pt under <out>/<model name>/.
Device-agnostic: CUDA when available (training a real run on CPU is slow), CPU otherwise.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (ROOT, os.path.join(ROOT, "target")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.dataset import SceneDataset, collate, torch_dataset  # noqa: E402
from models.detector import build_detector  # noqa: E402
from models.loss import DetLoss  # noqa: E402
from render.sampling import load_yaml  # noqa: E402


def load_all_configs(train_cfg: str | None = None) -> dict:
    from render.sampling import load_configs
    cfgs = load_configs()
    cfgs["train"] = load_yaml(train_cfg or os.path.join(ROOT, "configs", "train.yaml"))
    return cfgs


def device_of(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class EMA:
    def __init__(self, model, decay=0.9998, tau=2000):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.n = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = self.decay * (1 - math.exp(-self.n / self.tau))
        msd = model.state_dict()
        for k, v in self.m.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def save_ckpt(path, model, cfg_model: dict, bands, epoch, metrics, train_cfg):
    torch.save(dict(state_dict=model.state_dict(), bands=list(bands), model_cfg=cfg_model, epoch=epoch,
                    metrics=metrics, detector=train_cfg["detector"], classes=train_cfg["classes"]), path)


def load_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg = dict(detector=ck["detector"], classes=ck["classes"])
    m = build_detector(cfg, ck["bands"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


def train_model(name: str, data_root: str, out_root: str, cfgs: dict, teacher_ckpt: str | None = None,
                max_iters: int | None = None, epochs: int | None = None, log=print) -> str:
    T = cfgs["train"]
    tc = T["train"]
    mcfg = T["models"][name]
    bands = list(mcfg["bands"])
    dev = device_of(tc["device"])
    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])
    teacher = None
    load_bands = list(bands)
    if mcfg.get("distill_from"):
        raise NotImplementedError(
            "spec 7.2 student training (per-batch modality subsets, response + masked feature KD, "
            "CNR-gap weights) is not wired into the engine yet; models/distill.py has the losses")
        if not teacher_ckpt:
            raise ValueError(f"model {name} distils from {mcfg['distill_from']}: pass --teacher <best.pt>")
        teacher, tck = load_ckpt(teacher_ckpt, dev)
        load_bands += [b for b in tck["bands"] if b not in load_bands]
    train_ds = SceneDataset(data_root, "train", load_bands, cfgs, train=True, seed=tc["seed"], aug=tc["augment"],
                            use_blur_box=tc.get("use_blur_box", False), cache_layers=tc.get("cache_layers", 0))
    val_ds = SceneDataset(data_root, "val", bands, cfgs, train=False, seed=tc["seed"], use_blur_box=tc.get("use_blur_box", False))
    if len(train_ds) == 0:
        raise RuntimeError(f"no training scenes under {data_root}")
    loader = torch.utils.data.DataLoader(torch_dataset(train_ds), batch_size=tc["batch"], shuffle=True,
                                         num_workers=tc["workers"], collate_fn=collate, drop_last=len(train_ds) > tc["batch"],
                                         persistent_workers=False)
    model = build_detector(T, bands).to(dev)
    crit = DetLoss(len(T["classes"]), **T["loss"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    ema = EMA(model, tc["ema_decay"])
    n_ep = epochs or tc["epochs"]
    steps = max(1, len(loader)) * n_ep
    warm = max(1, len(loader)) * tc["warmup_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)) * 0.95 + 0.05))
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(model=name, bands=bands, train=T, teacher=teacher_ckpt, data=data_root), f, indent=1)
    best, it, hist = -1.0, 0, []
    D = T["distill"]
    for ep in range(n_ep):
        train_ds.set_epoch(ep)
        model.train()
        t0 = time.time()
        for imgs, tgts in loader:
            imgs = {b: v.to(dev, non_blocking=True) for b, v in imgs.items()}
            raw = model({b: imgs[b] for b in bands})
            loss, parts = crit(model, raw, tgts)
            if teacher is not None:
                with torch.no_grad():
                    t_raw = teacher({b: imgs[b] for b in teacher.bands})
                lk, kp = kd_loss(model, raw, teacher, t_raw, D["temperature"], D["box_weight"], D["score_thresh"])
                loss = loss + D["lambda"] * lk
                parts.update(kp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            ema.update(model)
            it += 1
            if it % 10 == 0 or it == 1:
                log(f"[{name}] ep {ep} it {it} loss {float(loss):.4f} " + " ".join(f"{k} {v:.3f}" for k, v in parts.items())
                    + f" lr {sched.get_last_lr()[0]:.2e}")
            if max_iters and it >= max_iters:
                break
        log(f"[{name}] epoch {ep} done in {time.time() - t0:.0f}s")
        last = (ep + 1 == n_ep) or (max_iters and it >= max_iters)
        if len(val_ds) and ((ep + 1) % tc["eval_every"] == 0 or last):
            from eval.metrics import quick_ap50
            ap = quick_ap50(ema.m, val_ds, bands, T["eval"], dev)
            hist.append(dict(epoch=ep, iters=it, val_ap50_target=ap))
            log(f"[{name}] val AP50(target) {ap:.4f}")
            if ap > best:
                best = ap
                save_ckpt(os.path.join(out, "best.pt"), ema.m, mcfg, bands, ep, dict(val_ap50_target=ap), T)
        save_ckpt(os.path.join(out, "last.pt"), ema.m, mcfg, bands, ep, dict(history=hist), T)
        if max_iters and it >= max_iters:
            break
    if not os.path.exists(os.path.join(out, "best.pt")):
        save_ckpt(os.path.join(out, "best.pt"), ema.m, mcfg, bands, ep, dict(note="no validation scenes"), T)
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    return os.path.join(out, "best.pt")


def cli(default_model: str | None = None, needs_teacher: bool = False):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "out", "dataset"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "runs"))
    ap.add_argument("--model", default=default_model, required=default_model is None)
    ap.add_argument("--teacher", default=None, required=needs_teacher, help="teacher best.pt (student only)")
    ap.add_argument("--config", default=None, help="train.yaml override")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-iters", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-p2", action="store_true", help="drop the stride-4 head (spec 7 baseline gap run)")
    a = ap.parse_args()
    cfgs = load_all_configs(a.config)
    tc = cfgs["train"]["train"]
    for k in ("device", "batch", "workers"):
        if getattr(a, k) is not None:
            tc[k] = getattr(a, k)
    if a.no_p2:
        cfgs["train"]["detector"]["p2_head"] = False
    name = a.model
    out = a.out if not a.no_p2 else os.path.join(a.out, "no_p2")
    path = train_model(name, a.data, out, cfgs, a.teacher, a.max_iters, a.epochs)
    print("best checkpoint:", path)

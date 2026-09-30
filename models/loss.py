"""Label assignment, detection loss and NMS for tiny objects.

Assignment: task-aligned (TOOD / YOLOv8: metric = score^alpha * sim^beta, top-k per object) with
two changes for 2-16 px objects:
  * candidates are the anchors inside the box OR the k nearest anchors of each level whose stride
    is at most max(8, object size). A 3-px box usually contains no stride-4 anchor centre, so
    "inside only" would leave it without a single positive;
  * for objects under `tiny_px`, similarity is max(IoU, NWD). IoU is 0 for any non-overlapping
    prediction and jumps under 1-px shifts (a 1-px diagonal shift of a 3x3 box gives IoU ~0.3, spec 7);
    NWD (Wang et al. 2021; Xu et al. 2022) degrades smoothly with centre distance.
Every object keeps at least its nearest candidate anchor.

Candidates inside a box must also lie within `centre_radius` strides of its centre (YOLOX centre
sampling), so every positive's centre offset fits the box head's distribution range.

Loss: BCE on soft IoU/NWD-aware class targets (normalised by the sum of targets) + (1 - CIoU)
+ (1 - NWD) for tiny objects + distribution focal loss on the four box distributions, all box terms
weighted by the target score.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ============================================================================ geometry
def box_iou(a, b, eps=1e-9):
    """Pairwise IoU, a [N,4] x b [M,4] -> [N,M]."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    aa = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    ab = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (aa[:, None] + ab[None, :] - inter + eps)


def nwd_pairwise(a, b, C=12.8):
    """Normalized Wasserstein distance between boxes as 2-D Gaussians N((cx,cy), diag(w^2/4, h^2/4))."""
    ca = torch.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2, (a[:, 2] - a[:, 0]) / 2, (a[:, 3] - a[:, 1]) / 2], 1)
    cb = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, (b[:, 2] - b[:, 0]) / 2, (b[:, 3] - b[:, 1]) / 2], 1)
    w2 = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
    return torch.exp(-torch.sqrt(w2 + 1e-12) / C)


def nwd_elementwise(a, b, C=12.8):
    ca = torch.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2, (a[:, 2] - a[:, 0]) / 2, (a[:, 3] - a[:, 1]) / 2], 1)
    cb = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, (b[:, 2] - b[:, 0]) / 2, (b[:, 3] - b[:, 1]) / 2], 1)
    return torch.exp(-torch.sqrt(((ca - cb) ** 2).sum(-1) + 1e-12) / C)


def ciou(a, b, eps=1e-9):
    """Complete IoU, elementwise over [N,4] boxes."""
    lt, rb = torch.max(a[:, :2], b[:, :2]), torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    wa, ha = (a[:, 2] - a[:, 0]).clamp(min=eps), (a[:, 3] - a[:, 1]).clamp(min=eps)
    wb, hb = (b[:, 2] - b[:, 0]).clamp(min=eps), (b[:, 3] - b[:, 1]).clamp(min=eps)
    union = wa * ha + wb * hb - inter + eps
    iou = inter / union
    cw = torch.max(a[:, 2], b[:, 2]) - torch.min(a[:, 0], b[:, 0])
    ch = torch.max(a[:, 3], b[:, 3]) - torch.min(a[:, 1], b[:, 1])
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((a[:, 0] + a[:, 2] - b[:, 0] - b[:, 2]) ** 2 + (a[:, 1] + a[:, 3] - b[:, 1] - b[:, 3]) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(wb / hb) - torch.atan(wa / ha)) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def giou(a, b, eps=1e-9):
    lt, rb = torch.max(a[:, :2], b[:, :2]), torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    aa = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    ab = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    union = aa + ab - inter + eps
    c = (torch.max(a[:, 2:], b[:, 2:]) - torch.min(a[:, :2], b[:, :2])).clamp(min=0).prod(-1) + eps
    return inter / union - (c - union) / c


# ============================================================================ assignment
class Assigner:
    def __init__(self, num_classes, topk=10, alpha=0.5, beta=6.0, near_k=4, tiny_px=16.0, nwd_C=12.8,
                 centre_radius=2.5):
        self.nc, self.topk, self.alpha, self.beta = num_classes, topk, alpha, beta
        self.near_k, self.tiny_px, self.C, self.radius = near_k, tiny_px, nwd_C, centre_radius

    @torch.no_grad()
    def __call__(self, scores, boxes, anchors, strides, gt_boxes, gt_labels):
        """scores [A,nc] (sigmoid), boxes [A,4] -> target_scores [A,nc], target_boxes [A,4], fg [A], gt_idx [A]."""
        A = scores.shape[0]
        dev = scores.device
        G = gt_boxes.shape[0]
        if G == 0:
            return torch.zeros_like(scores), torch.zeros_like(boxes), torch.zeros(A, dtype=torch.bool, device=dev), \
                torch.zeros(A, dtype=torch.long, device=dev)
        size = (gt_boxes[:, 2:] - gt_boxes[:, :2]).max(1).values                       # [G]
        ctr = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        ax, ay = anchors[:, 0], anchors[:, 1]
        inside = (ax[None] > gt_boxes[:, :1]) & (ax[None] < gt_boxes[:, 2:3]) & \
                 (ay[None] > gt_boxes[:, 1:2]) & (ay[None] < gt_boxes[:, 3:4])
        r = self.radius * strides[None]                                                  # centre sampling
        inside &= ((ax[None] - ctr[:, :1]).abs() <= r) & ((ay[None] - ctr[:, 1:2]).abs() <= r)
        dist = ((anchors[None] - ctr[:, None]) ** 2).sum(-1)                              # [G,A]
        level_ok = strides[None] <= torch.clamp(size, min=8.0)[:, None]
        near = torch.zeros_like(inside)
        for s in torch.unique(strides):
            idx = (strides == s).nonzero().squeeze(1)
            k = min(self.near_k, len(idx))
            nn_ = dist[:, idx].topk(k, dim=1, largest=False).indices
            near[torch.arange(G, device=dev)[:, None], idx[nn_]] = True
        cand = (inside | near) & level_ok
        iou = box_iou(gt_boxes, boxes)
        tiny = (size < self.tiny_px)[:, None]
        sim = torch.where(tiny, torch.max(iou, nwd_pairwise(gt_boxes, boxes, self.C)), iou)
        sc = scores[:, gt_labels].T                                                       # [G,A]
        metric = sc.pow(self.alpha) * sim.pow(self.beta) * cand
        k = min(self.topk, A)
        top = metric.topk(k, dim=1).indices
        pos = torch.zeros_like(cand)
        pos[torch.arange(G, device=dev)[:, None], top] = True
        pos &= cand & (metric > 0)
        nearest = torch.where(cand, dist, torch.full_like(dist, float("inf"))).argmin(1)  # always one positive
        pos[torch.arange(G, device=dev), nearest] = True
        # an anchor claimed by several objects keeps the most similar one
        multi = pos.sum(0) > 1
        if multi.any():
            best = (sim * pos).argmax(0)
            pos[:, multi] = False
            pos[best[multi], multi.nonzero().squeeze(1)] = True
        fg = pos.any(0)
        gt_idx = pos.float().argmax(0)
        align = metric * pos
        norm = align / (align.amax(1, keepdim=True) + 1e-9) * (sim * pos).amax(1, keepdim=True)
        tscore = norm.amax(0)
        # forced nearest anchors of tiny objects may have metric 0 early on: floor their target
        tscore = torch.where(fg, torch.clamp(tscore, min=0.05), tscore)
        target_scores = torch.zeros_like(scores)
        target_scores[fg, gt_labels[gt_idx[fg]]] = tscore[fg]
        return target_scores, gt_boxes[gt_idx], fg, gt_idx


def encode(gt_boxes, anchors, strides):
    """Boxes -> the head's parameters: centre offset from the grid point and log2 size, in strides."""
    c = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
    wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=1e-3)
    return torch.cat([(c - anchors) / strides[:, None], torch.log2(wh / strides[:, None])], 1)


def dfl_loss(logits, target, values):
    """Distribution focal loss: cross-entropy of each distribution to the two bins around the target.
    logits [N, 4, K], target [N, 4], values [4, K] (uniform bins) -> [N]."""
    K = logits.shape[-1]
    v0, dv = values[:, 0], values[:, 1] - values[:, 0]
    t = ((target - v0) / dv).clamp(0, K - 1 - 1e-4)
    lo = t.floor().long()
    wr = t - lo
    logp = logits.log_softmax(-1)
    lp_lo = logp.gather(-1, lo.unsqueeze(-1)).squeeze(-1)
    lp_hi = logp.gather(-1, (lo + 1).clamp(max=K - 1).unsqueeze(-1)).squeeze(-1)
    return -((1 - wr) * lp_lo + wr * lp_hi).mean(-1)


def assign_batch(assigner, flat, targets):
    """Run the assigner image by image on one model's (detached) predictions."""
    scores = flat.cls.detach().sigmoid()
    tss, tbs, fgs = [], [], []
    for b in range(scores.shape[0]):
        gtb = targets[b]["boxes"].to(scores.device).float()
        gtl = targets[b]["labels"].to(scores.device).long()
        ts, tb, fg, _ = assigner(scores[b], flat.boxes[b].detach(), flat.anchors, flat.strides, gtb, gtl)
        tss.append(ts)
        tbs.append(tb)
        fgs.append(fg)
    return torch.stack(tss), torch.stack(tbs), torch.stack(fgs)


class DetLoss:
    def __init__(self, num_classes, w_cls=0.5, w_box=7.5, w_nwd=2.0, w_dfl=1.5, tiny_px=16.0, nwd_C=12.8, **assigner):
        self.assign = Assigner(num_classes, tiny_px=tiny_px, nwd_C=nwd_C, **assigner)
        self.w = dict(cls=w_cls, box=w_box, nwd=w_nwd, dfl=w_dfl)
        self.tiny_px, self.C = tiny_px, nwd_C

    def __call__(self, model, raw, targets, flat=None):
        f = flat if flat is not None else model.flatten(raw)
        ts, tb, fg = assign_batch(self.assign, f, targets)
        norm = ts.sum().clamp(min=1.0)
        l_cls = F.binary_cross_entropy_with_logits(f.cls, ts, reduction="sum") / norm
        if fg.any():
            w = ts.sum(-1)[fg]
            pb, gb = f.boxes[fg], tb[fg]
            l_box = ((1.0 - ciou(pb, gb)) * w).sum() / norm
            tiny = (gb[:, 2:] - gb[:, :2]).max(1).values < self.tiny_px
            l_nwd = ((1.0 - nwd_elementwise(pb, gb, self.C)) * w * tiny).sum() / norm
            B, A = fg.shape
            anc = f.anchors.unsqueeze(0).expand(B, A, 2)[fg]
            stv = f.strides.unsqueeze(0).expand(B, A)[fg]
            l_dfl = (dfl_loss(f.box_logits[fg], encode(gb, anc, stv), model.bin_values) * w).sum() / norm
        else:
            l_box = l_nwd = l_dfl = f.cls.sum() * 0.0
        total = self.w["cls"] * l_cls + self.w["box"] * l_box + self.w["nwd"] * l_nwd + self.w["dfl"] * l_dfl
        return total, dict(cls=float(l_cls), box=float(l_box), nwd=float(l_nwd), dfl=float(l_dfl))


# ============================================================================ NMS
def nms(boxes, scores, labels, iou_thr=0.5, centre_px=2.0):
    """Class-wise greedy NMS. For tiny boxes IoU misses 1-2 px duplicates, so a box is also
    suppressed when its centre is within max(centre_px, 0.5 * smaller size) of a kept box."""
    if len(scores) == 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)
    order = scores.argsort(descending=True)
    boxes, labels = boxes[order], labels[order]
    iou = box_iou(boxes, boxes)
    ctr = (boxes[:, :2] + boxes[:, 2:]) / 2
    d = torch.cdist(ctr, ctr)
    sz = (boxes[:, 2:] - boxes[:, :2]).max(1).values
    tol = torch.clamp(0.5 * torch.min(sz[:, None], sz[None, :]), min=centre_px)
    same = labels[:, None] == labels[None, :]
    sup = same & ((iou > iou_thr) | (d < tol))
    keep = torch.ones(len(order), dtype=torch.bool, device=scores.device)
    for i in range(len(order)):
        if keep[i]:
            s = sup[i].clone()
            s[: i + 1] = False
            keep &= ~s
    return order[keep]

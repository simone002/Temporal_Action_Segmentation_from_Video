"""
LightningModule per temporal action segmentation.
Loss: CrossEntropy + Dice (escluso background)
Metriche: acc, acc_fg, mIoU epoch-level, F1@{10,25,50}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import defaultdict


# ── Dice Loss (background escluso) ───────────────────────────────────────────

def dice_loss(logits: torch.Tensor, targets: torch.Tensor,
              num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss multiclasse senza background (classe 0).
    logits:  (N, C)
    targets: (N,)
    """
    probs      = torch.softmax(logits, dim=-1)           # (N, C)
    targets_oh = F.one_hot(targets, num_classes).float() # (N, C)

    intersection = (probs * targets_oh).sum(0)           # (C,)
    probs_sum    = probs.sum(0)                          # (C,)
    targets_sum  = targets_oh.sum(0)                     # (C,)

    dice_per_cls = (2 * intersection + eps) / (probs_sum + targets_sum + eps)
    dice_per_cls = dice_per_cls[1:]  # ← escludi background (classe 0)
    return 1 - dice_per_cls.mean()


# ── Metriche frame-level ─────────────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor,
                    num_classes: int, ignore_index: int = 0):
    acc  = (preds == targets).float().mean()
    mask = targets != ignore_index

    if mask.sum().item() == 0:
        z = torch.tensor(0.0, device=preds.device)
        return acc, z, z, z, z, z

    p_fg = preds[mask]
    t_fg = targets[mask]
    acc_fg = (p_fg == t_fg).float().mean()

    iou_list, dice_list, prec_list, rec_list = [], [], [], []
    for c in t_fg.unique():
        pred_c = (p_fg == c)
        tgt_c  = (t_fg == c)
        tp = (pred_c & tgt_c).sum().float()
        fp = (pred_c & ~tgt_c).sum().float()
        fn = (~pred_c & tgt_c).sum().float()

        iou_list.append(tp / (tp + fp + fn + 1e-6))
        dice_list.append(2 * tp / (2 * tp + fp + fn + 1e-6))
        prec_list.append(tp / (tp + fp + 1e-6))
        rec_list.append(tp / (tp + fn + 1e-6))

    return (
        acc, acc_fg,
        torch.stack(iou_list).mean(),
        torch.stack(dice_list).mean(),
        torch.stack(prec_list).mean(),
        torch.stack(rec_list).mean(),
    )


# ── F1@k (segment-level) ─────────────────────────────────────────────────────

def f1_at_k(pred_seq: torch.Tensor, target_seq: torch.Tensor,
             overlap_thresh: float, bg_class: int = 0) -> float:
    def get_segments(seq):
        segments, seq, i = [], seq.tolist(), 0
        while i < len(seq):
            label, j = seq[i], i
            while j < len(seq) and seq[j] == label:
                j += 1
            if label != bg_class:
                segments.append((label, i, j - 1))
            i = j
        return segments

    pred_segs   = get_segments(pred_seq)
    target_segs = get_segments(target_seq)
    tp, used    = 0, [False] * len(target_segs)

    for p_label, p_start, p_end in pred_segs:
        best_iou, best_idx = 0, -1
        for i, (t_label, t_start, t_end) in enumerate(target_segs):
            if used[i] or t_label != p_label:
                continue
            intersection = max(0, min(p_end, t_end) - max(p_start, t_start) + 1)
            union        = max(p_end, t_end) - min(p_start, t_start) + 1
            iou          = intersection / union
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= overlap_thresh and best_idx >= 0:
            tp += 1
            used[best_idx] = True

    fp        = len(pred_segs) - tp
    fn        = len(target_segs) - tp
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)


# ── LightningModule ──────────────────────────────────────────────────────────

class TemporalSegmentationModule(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        num_classes: int       = 106,
        lr: float              = 1e-3,
        weight_decay: float    = 1e-4,
        label_smoothing: float = 0.05,
        bg_weight: float       = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model       = model

        weights    = torch.ones(num_classes)
        weights[0] = bg_weight
        self.criterion = nn.CrossEntropyLoss(
            weight         = weights,
            label_smoothing= label_smoothing,
        )

        # Accumulatori mIoU epoch-level su TUTTE le classi foreground
        self._tp: dict = defaultdict(float)
        self._fp: dict = defaultdict(float)
        self._fn: dict = defaultdict(float)

        # Accumulatori F1@k
        self._f1_preds:   list = []
        self._f1_targets: list = []

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y    = batch
        logits  = self(x)
        B, T, C = logits.shape
        N       = B * T

        logits_flat  = logits.reshape(N, C)
        targets_flat = y.reshape(N)

        # ── Loss ────────────────────────────────────────────────────────────
        ce   = self.criterion(logits_flat, targets_flat)
        dl   = dice_loss(logits_flat, targets_flat, C)
        loss = ce + 0.5 * dl 

        # ── Metriche frame-level ─────────────────────────────────────────────
        preds = logits_flat.argmax(dim=-1)
        acc, acc_fg, iou, dice, precision, recall = compute_metrics(
            preds, targets_flat, num_classes=C
        )

        on_step = (stage == "train")
        self.log(f"{stage}/loss",      loss,      prog_bar=True,  on_epoch=True, on_step=on_step)
        self.log(f"{stage}/loss_ce",   ce,        prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}/loss_dice", dl,        prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}/acc",       acc,       prog_bar=True,  on_epoch=True, on_step=False)
        self.log(f"{stage}/acc_fg",    acc_fg,    prog_bar=True,  on_epoch=True, on_step=False)
        self.log(f"{stage}/iou",       iou,       prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}/dice_m",    dice,      prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}/precision", precision, prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}/recall",    recall,    prog_bar=False, on_epoch=True, on_step=False)

        # ── Accumula per mIoU e F1@k (solo validation) ──────────────────────
        if stage == "val":
            mask = targets_flat != 0
            if mask.sum().item() > 0:
                p_fg = preds[mask]
                t_fg = targets_flat[mask]
                # Itera su TUTTE le classi foreground, non solo quelle presenti
                for c in range(1, self.hparams.num_classes):
                    pred_c = (p_fg == c)
                    tgt_c  = (t_fg == c)
                    self._tp[c] += (pred_c & tgt_c).sum().item()
                    self._fp[c] += (pred_c & ~tgt_c).sum().item()
                    self._fn[c] += (~pred_c & tgt_c).sum().item()

            for b in range(B):
                self._f1_preds.append(logits[b].argmax(-1).cpu())
                self._f1_targets.append(y[b].cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        # ── mIoU epoch-level su tutte le classi foreground ──────────────────
        ious = []
        for c in range(1, self.hparams.num_classes):
            tp = self._tp[c]
            fp = self._fp[c]
            fn = self._fn[c]
            denom = tp + fp + fn
            if denom > 0:
                ious.append(tp / denom)

        if ious:
            miou = sum(ious) / len(ious)
            self.log("val/mIoU_epoch", miou, prog_bar=True)

        self._tp.clear()
        self._fp.clear()
        self._fn.clear()

        # ── F1@{10, 25, 50} ─────────────────────────────────────────────────
        if self._f1_preds:
            for k in [0.10, 0.25, 0.50]:
                scores = [
                    f1_at_k(p, t, overlap_thresh=k)
                    for p, t in zip(self._f1_preds, self._f1_targets)
                ]
                self.log(f"val/F1@{int(k*100)}", sum(scores) / len(scores), prog_bar=False)

        self._f1_preds.clear()
        self._f1_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr           = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }
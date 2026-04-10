"""
run_experiments.py  ──  DLCV 2026 Assignment 2
================================================
All 7 experiments with:
  • EarlyStopping  (patience | overfit-gap | accuracy target)
  • Rich per-epoch metrics: grad norm, LR, train-val gap
  • W&B real-time logging — one run per variant, grouped by experiment
  • Post-training: class accuracy, confusion matrix, gradient flow
  • All plots auto-saved to results/exp{N}_*/

Setup:
    pip install wandb
    wandb login          # paste your API key once

Usage:
    python run_experiments.py              # all 7 experiments
    python run_experiments.py --exp 1 2   # subset
    python run_experiments.py --no-wandb  # disable W&B
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR,
)
from tqdm.auto import tqdm

from src.ViT import vit_tiny, VisionTransformer, vit_small
from src.resnet import resnet18_cifar
from src.data_setup import get_dataloaders

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
#  W&B  (gracefully disabled when not installed or --no-wandb passed)
# ═══════════════════════════════════════════════════════════════════════════
try:
    import wandb as _wandb_module
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb_module    = None
    _WANDB_AVAILABLE = False
    print("[wandb] not installed — `pip install wandb` to enable logging.")

# ── Edit these two lines to match your account ───────────────────────────
WANDB_PROJECT = "dlcv2026-assignment2"
WANDB_ENTITY  = None   # your W&B username / team, or None to use default
# ─────────────────────────────────────────────────────────────────────────
    
# Controlled at runtime via --no-wandb flag (default: enabled)
_USE_WANDB = True


def _wb_init(run_name: str, config: dict,
             group: str = None, tags=None):
    """Open a W&B run.  Returns run object, or None when W&B is off."""
    if not (_USE_WANDB and _WANDB_AVAILABLE):
        return None
    return _wandb_module.init(
        project = WANDB_PROJECT,
        entity  = WANDB_ENTITY,
        name    = run_name,
        group   = group,
        tags    = tags or [],
        config  = config,
        reinit  = True,
    )


def _wb_log(metrics: dict, step: int = None):
    """Log scalar metrics.  No-op when W&B is off."""
    if _USE_WANDB and _WANDB_AVAILABLE and _wandb_module.run is not None:
        _wandb_module.log(metrics, step=step)


def _wb_log_img(key: str, path):
    """Log a saved PNG.  No-op when W&B is off."""
    if (_USE_WANDB and _WANDB_AVAILABLE
            and _wandb_module.run is not None
            and Path(path).exists()):
        _wandb_module.log({key: _wandb_module.Image(str(path))})


def _wb_log_table(key: str, columns, data):
    """Log a W&B Table (renders as interactive bar chart in UI)."""
    if _USE_WANDB and _WANDB_AVAILABLE and _wandb_module.run is not None:
        _wandb_module.log({key: _wandb_module.Table(
            columns=columns, data=data)})


def _wb_finish():
    """Close the current W&B run."""
    if _USE_WANDB and _WANDB_AVAILABLE and _wandb_module.run is not None:
        _wandb_module.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  Globals
# ═══════════════════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results")
MODELS_DIR  = Path("saved_models")
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

print(f"\n{'='*60}")
print(f"  DLCV 2026 Assignment 2")
print(f"  Device  : {DEVICE}" +
      (f" — {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else ""))
print(f"  W&B     : {'enabled → ' + WANDB_PROJECT if _WANDB_AVAILABLE else 'disabled'}")
print(f"{'='*60}\n")

# ═══════════════════════════════════════════════════════════════════════════
#  Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════
VIT_CFG = dict(
    epochs=150, lr=3e-4, weight_decay=0.05, batch_size=256,
    warmup_epochs=15, grad_clip=1.0, label_smoothing=0.1,
)
CNN_CFG = dict(
    epochs=150, lr=0.1, weight_decay=5e-4, batch_size=512,
    warmup_epochs=5, grad_clip=0.0, label_smoothing=0.0,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Early Stopping
# ═══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Fires on whichever condition triggers first:
      1. Patience   : val_acc stagnates for `patience` epochs.
      2. Overfit gap: train_acc − val_acc > `gap_threshold`
                      for `gap_patience` consecutive epochs.
      3. Target acc : val_acc >= `target_acc`.
    """
    def __init__(self, patience=20, min_delta=5e-4,
                 gap_threshold=0.25, gap_patience=15,
                 target_acc=None):
        self.patience      = patience
        self.min_delta     = min_delta
        self.gap_threshold = gap_threshold
        self.gap_patience  = gap_patience
        self.target_acc    = target_acc
        self._best         = 0.0
        self._no_improve   = 0
        self._gap_streak   = 0
        self.triggered     = False
        self.reason        = ""
        self.stopped_epoch = None

    def step(self, val_acc: float, train_acc: float, epoch: int) -> bool:
        if self.target_acc and val_acc >= self.target_acc:
            return self._fire(
                f"target_acc {self.target_acc:.3f} reached ({val_acc:.4f})",
                epoch)
        if val_acc > self._best + self.min_delta:
            self._best, self._no_improve = val_acc, 0
        else:
            self._no_improve += 1
        if self._no_improve >= self.patience:
            return self._fire(
                f"no improvement for {self.patience} epochs "
                f"(best={self._best:.4f})", epoch)
        gap = train_acc - val_acc
        self._gap_streak = self._gap_streak + 1 if gap > self.gap_threshold else 0
        if self._gap_streak >= self.gap_patience:
            return self._fire(
                f"overfit gap {gap:.3f} > {self.gap_threshold} "
                f"for {self.gap_patience} epochs", epoch)
        return False

    def _fire(self, reason, epoch):
        self.triggered, self.reason, self.stopped_epoch = True, reason, epoch
        tqdm.write(f"\n  ⏹  EarlyStopping @ epoch {epoch+1}: {reason}\n")
        return True


# ═══════════════════════════════════════════════════════════════════════════
#  Optimizer / Scheduler
# ═══════════════════════════════════════════════════════════════════════════

def make_vit_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 or "bias" in name else decay).append(p)
    return AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999))


def make_cnn_optimizer(model, lr, wd):
    return SGD(model.parameters(), lr=lr, momentum=0.9,
               weight_decay=wd, nesterov=True)


def make_scheduler(optimizer, warmup, total):
    if warmup and warmup > 0:
        w = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                     total_iters=warmup)
        c = CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=1e-6)
        return SequentialLR(optimizer, [w, c], milestones=[warmup])
    return CosineAnnealingLR(optimizer, T_max=total, eta_min=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
#  Unified Trainer
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Universal trainer for ViT and ResNet.

    Per-epoch tracking
    ------------------
    train_loss, train_acc, val_loss, val_acc  — standard
    grad_norms    : mean pre-clip gradient L2 norm
    lrs           : learning-rate at epoch start
    train_val_gap : train_acc − val_acc  (overfitting indicator)

    W&B logging
    -----------
    Every epoch: all scalars above, streamed in real-time.
    End of run : summary metrics + all analysis plots as W&B Images.
    """

    def __init__(self, model, train_loader, val_loader, optimizer,
                 scheduler=None, early_stopping=None, label_smoothing=0.1,
                 grad_clip=1.0, use_amp=True, save_path=None,
                 wb_run=None, wb_prefix=""):
        self.model        = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.es           = early_stopping
        self.grad_clip    = grad_clip
        self.save_path    = save_path
        self.wb_run       = wb_run
        self.wb_prefix    = (wb_prefix.rstrip("/") + "/") if wb_prefix else ""
        self.criterion    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.use_amp      = use_amp and (DEVICE == "cuda")
        self.scaler       = torch.amp.GradScaler(enabled=self.use_amp)
        self.best_val_acc = 0.0
        self.grad_flow    = {}   # set at end of each epoch via capture_grad_flow

        self.history = dict(
            train_loss=[], train_acc=[], val_loss=[], val_acc=[],
            grad_norms=[], lrs=[], train_val_gap=[],
        )

    # ── forward ─────────────────────────────────────────────────────────

    def _fwd(self, imgs, labels):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE, enabled=self.use_amp):
            out = self.model(imgs)
            if isinstance(out, tuple):
                out = out[0]
            loss = self.criterion(out, labels)
        return loss, out.detach(), labels

    # ── train one epoch ─────────────────────────────────────────────────

    def train_one_epoch(self):
        self.model.train()
        total_loss = correct = total = 0
        total_gnorm = n_batches = 0

        for imgs, labels in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, out, labels = self._fwd(imgs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # clip_grad_norm_ returns total norm BEFORE clipping — true signal
            gnorm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip if self.grad_clip else float("inf"),
            )
            total_gnorm += gnorm.item()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
            n_batches  += 1

        return total_loss / total, correct / total, total_gnorm / n_batches

    # ── evaluate ────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader=None):
        self.model.eval()
        loader = loader or self.val_loader
        total_loss = correct = total = 0
        for imgs, labels in loader:
            loss, out, labels = self._fwd(imgs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        return total_loss / total, correct / total

    # ── gradient flow snapshot ──────────────────────────────────────────

    def capture_grad_flow(self):
        """Snapshot mean/max |grad| per named parameter (after last backward)."""
        self.grad_flow = {
            name: (p.grad.detach().abs().mean().item(),
                   p.grad.detach().abs().max().item())
            for name, p in self.model.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    # ── main fit loop ────────────────────────────────────────────────────

    def fit(self, epochs, desc="Training"):
        p = self.wb_prefix   # shorthand

        for epoch in tqdm(range(epochs), desc=desc):
            cur_lr = self.optimizer.param_groups[0]["lr"]

            t_loss, t_acc, gnorm = self.train_one_epoch()
            self.capture_grad_flow()
            v_loss, v_acc = self.evaluate()

            if self.scheduler:
                self.scheduler.step()

            # ── history ──────────────────────────────────────────────────
            self.history["train_loss"].append(t_loss)
            self.history["train_acc"].append(t_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)
            self.history["grad_norms"].append(gnorm)
            self.history["lrs"].append(cur_lr)
            self.history["train_val_gap"].append(t_acc - v_acc)

            # ── checkpoint best ───────────────────────────────────────────
            if v_acc > self.best_val_acc:
                self.best_val_acc = v_acc
                if self.save_path:
                    torch.save(self.model.state_dict(), self.save_path)

            # ── W&B per-epoch scalars ─────────────────────────────────────
            _wb_log({
                f"{p}train/loss":      t_loss,
                f"{p}train/acc":       t_acc,
                f"{p}val/loss":        v_loss,
                f"{p}val/acc":         v_acc,
                f"{p}train/grad_norm": gnorm,
                f"{p}train/lr":        cur_lr,
                f"{p}train/val_gap":   t_acc - v_acc,
                f"{p}val/best_acc":    self.best_val_acc,
            }, step=epoch)

            # ── console ───────────────────────────────────────────────────
            if (epoch + 1) % 10 == 0 or epoch == 0:
                tqdm.write(
                    f"  [{epoch+1:>3}/{epochs}]  "
                    f"TL={t_loss:.4f} TA={t_acc:.4f} | "
                    f"VL={v_loss:.4f} VA={v_acc:.4f} | "
                    f"gap={t_acc-v_acc:+.3f} gN={gnorm:.3f} "
                    f"lr={cur_lr:.2e}  best={self.best_val_acc:.4f}"
                )

            # ── early stopping ────────────────────────────────────────────
            if self.es and self.es.step(v_acc, t_acc, epoch):
                if self.es.triggered:
                    _wb_log({f"{p}early_stop/epoch":  epoch,
                             f"{p}early_stop/reason": self.es.reason})
                break

        # Reload best weights
        if self.save_path and Path(self.save_path).exists():
            self.model.load_state_dict(
                torch.load(self.save_path, map_location=DEVICE,
                           weights_only=True))

        # W&B run-level summary
        _wb_log({
            f"{p}summary/best_val_acc":   self.best_val_acc,
            f"{p}summary/epochs_trained": len(self.history["val_acc"]),
        })

        return self.history

    # ── detailed evaluation ──────────────────────────────────────────────

    @torch.no_grad()
    def detailed_evaluate(self, loader, num_classes=10):
        """Returns (class_acc ndarray, confusion matrix ndarray)."""
        self.model.eval()
        conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = self.model(imgs)
            if isinstance(out, tuple):
                out = out[0]
            for t, pred in zip(labels.cpu(), out.argmax(1).cpu()):
                conf[t, pred] += 1
        class_acc = conf.diag().float() / conf.sum(1).float().clamp(min=1)
        return class_acc.numpy(), conf.numpy()


# ═══════════════════════════════════════════════════════════════════════════
#  Shared run helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_es(**kw):
    return EarlyStopping(**kw)


def _run_vit(model, train_loader, val_loader, cfg, save_path,
             desc="ViT", es_kwargs=None, wb_run=None, wb_prefix=""):
    opt   = make_vit_optimizer(model, cfg["lr"], cfg["weight_decay"])
    sched = make_scheduler(opt, cfg["warmup_epochs"], cfg["epochs"])
    es    = _build_es(**(es_kwargs or {})) if es_kwargs is not None else None
    tr    = Trainer(model, train_loader, val_loader, opt, sched,
                    early_stopping=es,
                    label_smoothing=cfg["label_smoothing"],
                    grad_clip=cfg["grad_clip"],
                    use_amp=True,
                    save_path=save_path,
                    wb_run=wb_run,
                    wb_prefix=wb_prefix)
    h = tr.fit(cfg["epochs"], desc=desc)
    return h, tr.best_val_acc, tr


def _run_cnn(model, train_loader, val_loader, cfg, save_path,
             desc="CNN", es_kwargs=None, wb_run=None, wb_prefix=""):
    opt   = make_cnn_optimizer(model, cfg["lr"], cfg["weight_decay"])
    sched = make_scheduler(opt, cfg["warmup_epochs"], cfg["epochs"])
    es    = _build_es(**(es_kwargs or {})) if es_kwargs is not None else None
    tr    = Trainer(model, train_loader, val_loader, opt, sched,
                    early_stopping=es,
                    label_smoothing=cfg["label_smoothing"],
                    grad_clip=cfg["grad_clip"],
                    use_amp=True,
                    save_path=save_path,
                    wb_run=wb_run,
                    wb_prefix=wb_prefix)
    h = tr.fit(cfg["epochs"], desc=desc)
    return h, tr.best_val_acc, tr


@torch.no_grad()
def _test_acc(model, loader):
    model.eval().to(DEVICE)
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        if isinstance(out, tuple):
            out = out[0]
        correct += (out.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def _load_weights(model, path):
    model.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True))
    return model


def _save_json(data, path):
    def _cvt(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_cvt)
    print(f"  Saved JSON : {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting utilities
# ═══════════════════════════════════════════════════════════════════════════

def _savefig(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot : {path}")


def plot_loss_curves(histories, labels, title, out_dir, fname):
    n      = len(histories)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for hist, lbl, col in zip(histories, labels, colors):
        ep = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(ep, hist["train_loss"], color=col,
                     label=f"{lbl} train")
        axes[0].plot(ep, hist["val_loss"], color=col, linestyle="--",
                     label=f"{lbl} val")
        axes[1].plot(ep, hist["train_acc"], color=col,
                     label=f"{lbl} train")
        axes[1].plot(ep, hist["val_acc"], color=col, linestyle="--",
                     label=f"{lbl} val")
    for ax, yl in zip(axes, ["Loss", "Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl)
        ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    axes[0].set_title(f"{title} — Loss", fontsize=13)
    axes[1].set_title(f"{title} — Accuracy", fontsize=13)
    _savefig(out_dir / fname)


def plot_training_dynamics(histories, labels, out_dir, fname):
    """
    Three-panel: gradient norm (log) | learning rate | train-val gap
    These three together reveal training stability, schedule correctness,
    and the onset/severity of overfitting.
    """
    n      = len(histories)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [("grad_norms", "Gradient Norm (pre-clip)", True),
              ("lrs",        "Learning Rate",            False),
              ("train_val_gap", "Train − Val Gap",       False)]

    for hist, lbl, col in zip(histories, labels, colors):
        ep = range(1, len(hist["train_loss"]) + 1)
        for ax, (key, _, _log) in zip(axes, panels):
            if hist.get(key):
                ax.plot(ep, hist[key], color=col, label=lbl, lw=1.5)

    for ax, (_, title, use_log) in zip(axes, panels):
        ax.set_xlabel("Epoch"); ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if use_log and all(v > 0 for h in histories for v in h.get("grad_norms", [1])):
            ax.set_yscale("log")

    axes[2].axhline(0.0,  color="black", linestyle=":", alpha=0.5)
    axes[2].axhline(0.25, color="red",   linestyle="--", alpha=0.4,
                    label="gap threshold (0.25)")
    axes[2].legend(fontsize=8)
    _savefig(out_dir / fname)


def plot_gradient_flow(grad_flow: dict, out_dir, fname="gradient_flow.png",
                       top_n=30):
    """
    Horizontal bar chart — mean |grad| per named layer.
    Dead layers → near-zero bars.  Exploding layers → huge bars.
    """
    if not grad_flow:
        return
    names = list(grad_flow.keys())
    means = [grad_flow[n][0] for n in names]
    order = np.argsort(means)[-top_n:]
    names = [names[i] for i in order]
    means = [means[i] for i in order]

    cmap   = plt.cm.viridis
    colors = [cmap(i / max(len(names) - 1, 1)) for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.3)))
    ax.barh(range(len(names)), means, color=colors, edgecolor="none")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [n.replace("encoder.blocks.", "blk.")
          .replace("module.", "")[:50] for n in names],
        fontsize=7)
    ax.set_xlabel("Mean |∇|", fontsize=11)
    ax.set_title("Gradient Flow — Final Epoch Snapshot", fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, axis="x", alpha=0.3)
    _savefig(out_dir / fname)


def plot_class_accuracy(class_acc, class_names, out_dir,
                        fname="class_accuracy.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = plt.cm.RdYlGn(class_acc)
    bars    = ax.bar(class_names, class_acc, color=colors,
                     edgecolor="black", alpha=0.9)
    for bar, acc in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=13)
    ax.axhline(np.mean(class_acc), color="navy", linestyle="--", alpha=0.6,
               label=f"mean={np.mean(class_acc):.3f}")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    _savefig(out_dir / fname)


def plot_confusion_matrix(conf, class_names, out_dir,
                           fname="confusion_matrix.png"):
    norm = conf.astype(float) / conf.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix", fontsize=13)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if norm[i, j] > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Recall")
    _savefig(out_dir / fname)


def plot_accuracy_vs_data(vit_accs, cnn_accs, fractions, out_dir):
    pct = [f * 100 for f in fractions]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pct, vit_accs, "o-", label="ViT-Tiny",  color="royalblue", lw=2.5, ms=8)
    ax.plot(pct, cnn_accs, "s-", label="ResNet-18", color="tomato",    lw=2.5, ms=8)
    for x, vy, cy in zip(pct, vit_accs, cnn_accs):
        ax.annotate(f"{vy:.3f}", (x, vy), textcoords="offset points",
                    xytext=(0, 7),  ha="center", fontsize=8, color="royalblue")
        ax.annotate(f"{cy:.3f}", (x, cy), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="tomato")
    ax.set_xlabel("Training Data (%)"); ax.set_ylabel("Val Accuracy")
    ax.set_title("Data Efficiency: ViT-Tiny vs ResNet-18", fontsize=14)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    _savefig(out_dir / "accuracy_vs_data.png")


def plot_bar(keys, vals, xlabel, ylabel, title, out_dir, fname,
             color="steelblue"):
    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.6), 5))
    bars = ax.bar([str(k) for k in keys], vals,
                  color=color, edgecolor="black", alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13); ax.grid(True, axis="y", alpha=0.3)
    _savefig(out_dir / fname)


def plot_entropy_per_layer(entropy_matrix, out_dir):
    L, heads = entropy_matrix.shape
    layers   = list(range(1, L + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mean_e = entropy_matrix.mean(axis=1)
    axes[0].plot(layers, mean_e, "o-", color="darkorchid", lw=2.5, ms=7)
    axes[0].fill_between(layers, entropy_matrix.min(axis=1),
                         entropy_matrix.max(axis=1),
                         alpha=0.2, color="darkorchid", label="head range")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Entropy (nats)")
    axes[0].set_title("Attention Entropy vs Layer", fontsize=13)
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    im = axes[1].imshow(entropy_matrix.T, aspect="auto", cmap="plasma",
                        interpolation="nearest")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Head")
    axes[1].set_title("Entropy Heatmap (head × layer)", fontsize=13)
    axes[1].set_xticks(range(L)); axes[1].set_xticklabels(layers)
    axes[1].set_yticks(range(heads)); axes[1].set_yticklabels(range(1, heads+1))
    plt.colorbar(im, ax=axes[1], label="Entropy (nats)")
    _savefig(out_dir / "entropy_analysis.png")


def plot_attention_maps(attn_maps_list, orig_imgs, out_dir, num_images=4):
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
    L           = len(attn_maps_list)
    num_patches = attn_maps_list[0].shape[-1] - 1
    grid        = int(num_patches ** 0.5)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    for img_i in range(min(num_images, orig_imgs.shape[0])):
        cols = min(5, L); rows = (L + cols) // cols + 1
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = np.array(axes).flatten()
        orig = (orig_imgs[img_i].cpu() * std + mean).permute(1,2,0).numpy().clip(0,1)
        axes[0].imshow(orig); axes[0].set_title("Input"); axes[0].axis("off")
        for li, attn in enumerate(attn_maps_list):
            ca = attn[img_i].mean(0)[0, 1:].cpu().numpy().reshape(grid, grid)
            ca = (ca - ca.min()) / (ca.max() - ca.min() + 1e-8)
            axes[li+1].imshow(ca, cmap="inferno", interpolation="bilinear")
            axes[li+1].set_title(f"L{li+1}", fontsize=8)
            axes[li+1].axis("off")
        for j in range(L+1, len(axes)):
            axes[j].axis("off")
        plt.suptitle(f"CLS→Patch Attention (img {img_i})", fontsize=11)
        _savefig(out_dir / f"attn_img{img_i}.png")


def plot_layer_accuracy(layer_accs, out_dir):
    layers = list(range(1, len(layer_accs) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layers, layer_accs, "o-", color="teal", lw=2.5, ms=8)
    for l, acc in zip(layers, layer_accs):
        ax.annotate(f"{acc:.3f}", (l, acc),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    ax.axhline(max(layer_accs), color="teal", linestyle=":", alpha=0.5,
               label=f"best={max(layer_accs):.4f}")
    ax.set_xlabel("Layer Index"); ax.set_ylabel("Linear Probe Accuracy")
    ax.set_title("Layer-wise Representation Quality", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(layers)
    _savefig(out_dir / "layer_accuracy.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Post-training analysis  (called after every experiment variant)
# ═══════════════════════════════════════════════════════════════════════════

def run_post_training_analysis(trainer: Trainer, test_loader,
                               out_dir: Path, tag: str = "",
                               wb_group: str = ""):
    """
    Generates & saves:
      class_accuracy.png, confusion_matrix.png,
      gradient_flow.png, training_dynamics.png

    All four plots are also uploaded to the *currently active* W&B run
    (the one that was open when the trainer ran) so they appear in that
    run's Media tab.
    """
    pfx  = f"{tag}_" if tag else ""
    wkey = f"{wb_group}/{tag}/" if wb_group else f"{tag}/" if tag else ""

    paths = {
        "class_accuracy":    out_dir / f"{pfx}class_accuracy.png",
        "confusion_matrix":  out_dir / f"{pfx}confusion_matrix.png",
        "gradient_flow":     out_dir / f"{pfx}gradient_flow.png",
        "training_dynamics": out_dir / f"{pfx}training_dynamics.png",
    }

    class_acc, conf = trainer.detailed_evaluate(test_loader)

    plot_class_accuracy(class_acc, CIFAR10_CLASSES, out_dir,
                        f"{pfx}class_accuracy.png")
    plot_confusion_matrix(conf, CIFAR10_CLASSES, out_dir,
                          f"{pfx}confusion_matrix.png")
    plot_gradient_flow(trainer.grad_flow, out_dir,
                       f"{pfx}gradient_flow.png")
    plot_training_dynamics([trainer.history], [tag or "model"],
                           out_dir, f"{pfx}training_dynamics.png")

    # Upload plots to the currently open W&B run
    for name, path in paths.items():
        _wb_log_img(f"{wkey}{name}", path)

    # Per-class accuracy as a W&B Table → automatic bar chart in UI
    _wb_log_table(
        f"{wkey}class_acc_table",
        columns=["class", "accuracy"],
        data=[[c, float(a)] for c, a in zip(CIFAR10_CLASSES, class_acc)],
    )

    return {
        "class_accuracies": class_acc.tolist(),
        "mean_class_acc":   float(class_acc.mean()),
        "worst_class":      CIFAR10_CLASSES[int(class_acc.argmin())],
        "best_class":       CIFAR10_CLASSES[int(class_acc.argmax())],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Experiment 1: Data Efficiency
# ═══════════════════════════════════════════════════════════════════════════

class Exp1_DataEfficiency:
    """ViT-Tiny vs ResNet-18 across 5%, 10%, 25%, 50%, 100% of CIFAR-10."""
    NAME      = "exp1_data_efficiency"
    FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
    EPOCHS    = 100

    def __init__(self):
        self.out_dir = RESULTS_DIR / self.NAME
        self.out_dir.mkdir(exist_ok=True)

    def run(self):
        print("\n" + "="*60)
        print("  Experiment 1: Data Efficiency  (ViT vs CNN)")
        print("="*60)

        cfg_v = {**VIT_CFG, "epochs": self.EPOCHS, "warmup_epochs": 5}
        cfg_c = {**CNN_CFG, "epochs": self.EPOCHS, "warmup_epochs": 3}
        es_small = dict(patience=15, gap_threshold=0.20, gap_patience=10)
        es_full  = dict(patience=20, gap_threshold=0.25, gap_patience=15,
                        target_acc=0.89)

        results = {"fractions": self.FRACTIONS,
                   "vit_val_accs": [], "cnn_val_accs": []}
        vit_histories, cnn_histories = [], []
        last_vit_tr = last_cnn_tr = None

        _, test_loader, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=256)

        for frac in self.FRACTIONS:
            print(f"\n  ── Fraction {frac*100:.0f}% ──")
            es_kw = es_full if frac == 1.0 else es_small
            frac_tag = f"{frac*100:.0f}pct"

            tr_v, val_ldr, _ = get_dataloaders(
                data_fraction=frac, img_size=32, batch_size=cfg_v["batch_size"])
            tr_c, _, _ = get_dataloaders(
                data_fraction=frac, img_size=32, batch_size=cfg_c["batch_size"])

            # ── ViT ──────────────────────────────────────────────────────
            wb = _wb_init(
                run_name=f"exp1-vit-{frac_tag}",
                config={**cfg_v, "model": "vit_small", "data_fraction": frac},
                group="exp1_data_efficiency", tags=["exp1", "vit"])
            vit = vit_small(num_classes=10)
            h_v, best_v, tr_vit = _run_vit(
                vit, tr_v, val_ldr, cfg_v,
                MODELS_DIR / f"exp1_vit_{frac:.2f}.pt",
                desc=f"ViT {frac_tag}", es_kwargs=es_kw,
                wb_run=wb, wb_prefix="exp1/vit")
            if frac == 1.0:
                run_post_training_analysis(
                    tr_vit, test_loader, self.out_dir,
                    tag="vit_full", wb_group="exp1")
                last_vit_tr = tr_vit
            _wb_finish()
            vit_histories.append(h_v)
            results["vit_val_accs"].append(best_v)

            # ── CNN ──────────────────────────────────────────────────────
            wb = _wb_init(
                run_name=f"exp1-cnn-{frac_tag}",
                config={**cfg_c, "model": "resnet18", "data_fraction": frac},
                group="exp1_data_efficiency", tags=["exp1", "cnn"])
            cnn = resnet18_cifar(num_classes=10, base_channels=64, dropout=0.1)
            h_c, best_c, tr_cnn = _run_cnn(
                cnn, tr_c, val_ldr, cfg_c,
                MODELS_DIR / f"exp1_cnn_{frac:.2f}.pt",
                desc=f"CNN {frac_tag}", es_kwargs=es_kw,
                wb_run=wb, wb_prefix="exp1/cnn")
            if frac == 1.0:
                run_post_training_analysis(
                    tr_cnn, test_loader, self.out_dir,
                    tag="cnn_full", wb_group="exp1")
                last_cnn_tr = tr_cnn
            _wb_finish()
            cnn_histories.append(h_c)
            results["cnn_val_accs"].append(best_c)

            print(f"  → ViT={best_v:.4f}  CNN={best_c:.4f}")

        # Test on best full-data models
        vit_f = _load_weights(vit_small(num_classes=10),
                               MODELS_DIR / "exp1_vit_1.00.pt")
        cnn_f = _load_weights(
            resnet18_cifar(num_classes=10, base_channels=64, dropout=0.1),
            MODELS_DIR / "exp1_cnn_1.00.pt")
        results["vit_test_acc_full"] = _test_acc(vit_f, test_loader)
        results["cnn_test_acc_full"] = _test_acc(cnn_f, test_loader)

        _save_json(results, self.out_dir / "results.json")
        plot_accuracy_vs_data(results["vit_val_accs"], results["cnn_val_accs"],
                              self.FRACTIONS, self.out_dir)
        plot_loss_curves([vit_histories[-1], cnn_histories[-1]],
                         ["ViT (100%)", "CNN (100%)"],
                         "Full Data", self.out_dir, "loss_curves_full.png")
        plot_training_dynamics([vit_histories[-1], cnn_histories[-1]],
                               ["ViT (100%)", "CNN (100%)"],
                               self.out_dir, "dynamics_full.png")
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  Experiment 2: Patch Size
# ═══════════════════════════════════════════════════════════════════════════

class Exp2_PatchSize:
    """ViT-Tiny with patch sizes 4 × 4, 8 × 8, 16 × 16."""
    NAME        = "exp2_patch_size"
    PATCH_SIZES = [4, 8, 16]

    def __init__(self):
        self.out_dir = RESULTS_DIR / self.NAME
        self.out_dir.mkdir(exist_ok=True)

    def run(self):
        print("\n" + "="*60)
        print("  Experiment 2: Patch Size")
        print("="*60)
        train_loader, val_loader, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=VIT_CFG["batch_size"])
        _, test_loader, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=256)

        results = {"patch_sizes": self.PATCH_SIZES, "num_patches": [],
                   "test_accs": [], "best_val_accs": [], "training_times_s": []}
        histories, labels = [], []
        es_kw = dict(patience=20, gap_threshold=0.25, target_acc=0.89)

        for ps in self.PATCH_SIZES:
            print(f"\n  ── patch_size={ps} ──")
            model   = vit_small(num_classes=10, patch_size=ps)
            num_p   = model.patch_embed.num_patches
            nparams = sum(p.numel() for p in model.parameters())
            print(f"  num_patches={num_p}  params={nparams/1e6:.2f}M")

            wb = _wb_init(
                run_name=f"exp2-patch{ps}",
                config={**VIT_CFG, "patch_size": ps,
                        "num_patches": num_p, "model_params": nparams},
                group="exp2_patch_size", tags=["exp2", f"patch{ps}"])

            t0 = time.time()
            h, best, tr = _run_vit(
                model, train_loader, val_loader, VIT_CFG,
                MODELS_DIR / f"exp2_patch{ps}.pt",
                desc=f"patch={ps}", es_kwargs=es_kw,
                wb_run=wb, wb_prefix=f"exp2/patch{ps}")
            elapsed = time.time() - t0

            test_acc = _test_acc(model, test_loader)
            run_post_training_analysis(
                tr, test_loader, self.out_dir,
                tag=f"patch{ps}", wb_group="exp2")
            _wb_finish()

            results["num_patches"].append(num_p)
            results["best_val_accs"].append(best)
            results["test_accs"].append(test_acc)
            results["training_times_s"].append(elapsed)
            histories.append(h); labels.append(f"p={ps}")
            print(f"  test={test_acc:.4f}  time={elapsed/60:.1f}min")

        _save_json(results, self.out_dir / "results.json")
        plot_loss_curves(histories, labels, "Patch Size",
                         self.out_dir, "loss_curves.png")
        plot_training_dynamics(histories, labels,
                               self.out_dir, "training_dynamics.png")
        plot_bar(self.PATCH_SIZES, results["test_accs"],
                 "Patch Size", "Test Accuracy", "Accuracy vs Patch Size",
                 self.out_dir, "accuracy_bar.png")
        plot_bar(self.PATCH_SIZES, results["training_times_s"],
                 "Patch Size", "Time (s)", "Training Time vs Patch Size",
                 self.out_dir, "time_bar.png", "orange")
        # Tokens vs accuracy
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(results["num_patches"], results["test_accs"],
                   s=120, zorder=5, color="royalblue", edgecolors="black")
        for ps, np_, acc in zip(self.PATCH_SIZES, results["num_patches"],
                                 results["test_accs"]):
            ax.annotate(f"p={ps} ({np_} tok)", (np_, acc),
                        textcoords="offset points", xytext=(6, 4), fontsize=10)
        ax.set_xlabel("Tokens"); ax.set_ylabel("Test Accuracy")
        ax.set_title("Tokens vs Accuracy"); ax.grid(True, alpha=0.3)
        _savefig(self.out_dir / "tokens_vs_accuracy.png")
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  Experiment 3: CLS vs Mean Pooling
# ═══════════════════════════════════════════════════════════════════════════

class Exp3_CLSvsMean:
    NAME     = "exp3_cls_vs_mean"
    POOLINGS = ["cls", "mean"]

    def __init__(self):
        self.out_dir = RESULTS_DIR / self.NAME
        self.out_dir.mkdir(exist_ok=True)

    def run(self):
        print("\n" + "="*60)
        print("  Experiment 3: CLS Token vs Mean Pooling")
        print("="*60)
        train_loader, val_loader, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=VIT_CFG["batch_size"])
        _, test_loader, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=256)

        results = {}
        histories, labels = [], []
        es_kw = dict(patience=20, gap_threshold=0.25, target_acc=0.89)

        configs = [
            ("cls",  VIT_CFG, "cls"),
            ("mean", VIT_CFG, "mean"),
            ("mean_lr1e4", {**VIT_CFG, "lr": 1e-4}, "mean"),  # ablation
        ]

        for name, cfg, pooling in configs:
            print(f"\n  ── {name} ──")
            model = vit_small(num_classes=10, pooling=pooling)
            wb = _wb_init(
                run_name=f"exp3-{name}",
                config={**cfg, "pooling": pooling},
                group="exp3_cls_vs_mean", tags=["exp3", name])
            h, best, tr = _run_vit(
                model, train_loader, val_loader, cfg,
                MODELS_DIR / f"exp3_{name}.pt",
                desc=name, es_kwargs=es_kw,
                wb_run=wb, wb_prefix=f"exp3/{name}")
            test_acc = _test_acc(model, test_loader)
            ana = run_post_training_analysis(
                tr, test_loader, self.out_dir,
                tag=name, wb_group="exp3")
            _wb_finish()
            results[name] = {"best_val": best, "test_acc": test_acc, **ana}
            histories.append(h); labels.append(name)
            print(f"  test={test_acc:.4f}")

        _save_json(results, self.out_dir / "results.json")
        plot_loss_curves(histories, labels, "CLS vs Mean Pool",
                         self.out_dir, "loss_curves.png")
        plot_training_dynamics(histories, labels,
                               self.out_dir, "training_dynamics.png")
        plot_bar(list(results.keys()),
                 [results[k]["test_acc"] for k in results],
                 "Pooling", "Test Accuracy", "Pooling Comparison",
                 self.out_dir, "accuracy_bar.png")
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  Experiment 4: Positional Encoding
# ═══════════════════════════════════════════════════════════════════════════

class Exp4_PosEncoding:
    NAME     = "exp4_pos_encoding"
    PE_TYPES = ["learnable", "sinusoidal", "none"]

    def __init__(self):
        self.out_dir = RESULTS_DIR / self.NAME
        self.out_dir.mkdir(exist_ok=True)

    def run(self):
        print("\n" + "="*60)
        print("  Experiment 4: Positional Encoding Ablation")
        print("="*60)
        train_loader, val_loader, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=VIT_CFG["batch_size"])
        _, test_loader, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=256)

        results = {}
        histories, labels = [], []
        es_kw = dict(patience=20, gap_threshold=0.25, target_acc=0.89)

        for pe in self.PE_TYPES:
            print(f"\n  ── pos_encoding={pe} ──")
            model = vit_small(num_classes=10, pos_encoding=pe)
            wb = _wb_init(
                run_name=f"exp4-{pe}",
                config={**VIT_CFG, "pos_encoding": pe},
                group="exp4_pos_encoding", tags=["exp4", pe])
            h, best, tr = _run_vit(
                model, train_loader, val_loader, VIT_CFG,
                MODELS_DIR / f"exp4_{pe}.pt",
                desc=f"pos={pe}", es_kwargs=es_kw,
                wb_run=wb, wb_prefix=f"exp4/{pe}")
            test_acc  = _test_acc(model, test_loader)
            stability = float(np.std(h["val_loss"][-20:]))
            gn_std    = float(np.std(h["grad_norms"][-30:]) if h["grad_norms"] else 0)
            ana = run_post_training_analysis(
                tr, test_loader, self.out_dir,
                tag=pe, wb_group="exp4")
            _wb_finish()
            results[pe] = {"best_val": best, "test_acc": test_acc,
                            "stability": stability, "grad_norm_std": gn_std,
                            **ana}
            histories.append(h); labels.append(pe)
            print(f"  test={test_acc:.4f}  stability={stability:.5f}")

        _save_json(results, self.out_dir / "results.json")
        plot_loss_curves(histories, labels, "Pos Encoding",
                         self.out_dir, "loss_curves.png")
        plot_training_dynamics(histories, labels,
                               self.out_dir, "training_dynamics.png")
        plot_bar(self.PE_TYPES,
                 [results[pe]["test_acc"] for pe in self.PE_TYPES],
                 "Pos Encoding", "Test Accuracy", "Accuracy vs PE",
                 self.out_dir, "accuracy_bar.png")
        plot_bar(self.PE_TYPES,
                 [results[pe]["stability"] for pe in self.PE_TYPES],
                 "Pos Encoding", "Val-Loss Std (last 20 ep)",
                 "Training Stability", self.out_dir, "stability_bar.png",
                 color="salmon")
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  Experiment 5: Attention Maps & Entropy
# ═══════════════════════════════════════════════════════════════════════════

class Exp5_AttentionAnalysis:
    NAME = "exp5_attention"

    def __init__(self):
        self.out_dir = RESULTS_DIR / self.NAME
        self.out_dir.mkdir(exist_ok=True)

    @staticmethod
    def _entropy(attn):
        eps = 1e-9
        H   = -(attn * (attn + eps).log()).sum(dim=-1)
        return H.mean(dim=(0, 2)).cpu().numpy()

    def run(self):
        print("\n" + "="*60)
        print("  Experiment 5: Attention Maps & Entropy")
        print("="*60)

        ckpt = MODELS_DIR / "exp5_attn_model.pt"
        base = MODELS_DIR / "exp4_learnable.pt"
        attn_model = vit_small(num_classes=10, return_attention=True)

        wb = _wb_init(run_name="exp5-attention",
                      config={**VIT_CFG, "return_attention": True},
                      group="exp5_attention", tags=["exp5"])

        if not ckpt.exists():
            if base.exists():
                print("  Loading from exp4_learnable.pt…")
                attn_model.load_state_dict(
                    torch.load(base, map_location=DEVICE, weights_only=True))
                torch.save(attn_model.state_dict(), ckpt)
            else:
                print("  Training from scratch…")
                tr_l, vl, _ = get_dataloaders(
                    data_fraction=1.0, img_size=32,
                    batch_size=VIT_CFG["batch_size"])
                _run_vit(attn_model, tr_l, vl, VIT_CFG, save_path=ckpt,
                         desc="Exp5", es_kwargs=dict(patience=20, target_acc=0.89),
                         wb_run=wb, wb_prefix="exp5/train")
        else:
            print("  Loading saved model…")

        attn_model.load_state_dict(
            torch.load(ckpt, map_location=DEVICE, weights_only=True))
        attn_model.to(DEVICE).eval()

        _, vis_loader, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=32)

        L = entropy_accum = sample_maps = sample_imgs = None
        n_batches = 0

        for imgs, _ in tqdm(vis_loader, desc="Entropy computation"):
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                result = attn_model(imgs)
            if not isinstance(result, tuple) or len(result) < 2:
                print("  [ERROR] No attention maps."); _wb_finish(); return {}
            attn_maps = result[1]
            if L is None:
                L             = len(attn_maps)
                entropy_accum = np.zeros((L, attn_maps[0].shape[1]))
                sample_maps   = [a.cpu() for a in attn_maps]
                sample_imgs   = imgs.cpu()
            for i, a in enumerate(attn_maps):
                entropy_accum[i] += self._entropy(a)
            n_batches += 1
            if n_batches >= 30:
                break

        entropy_matrix = entropy_accum / n_batches

        # Log entropy scalars to W&B
        for li, row in enumerate(entropy_matrix):
            _wb_log({f"exp5/entropy/layer_{li+1}/mean": float(row.mean()),
                     **{f"exp5/entropy/layer_{li+1}/head_{hi+1}": float(v)
                        for hi, v in enumerate(row)}})

        results = {
            "num_layers": L, "num_heads": int(entropy_matrix.shape[1]),
            "mean_entropy_per_layer": entropy_matrix.mean(axis=1).tolist(),
            "entropy_matrix": entropy_matrix.tolist(),
        }
        _save_json(results, self.out_dir / "results.json")
        plot_entropy_per_layer(entropy_matrix, self.out_dir)
        plot_attention_maps(sample_maps, sample_imgs,
                            self.out_dir / "maps", num_images=4)

        # Upload entropy plot + sample attention maps
        _wb_log_img("exp5/entropy_analysis", self.out_dir / "entropy_analysis.png")
        for img_i in range(4):
            p = self.out_dir / "maps" / f"attn_img{img_i}.png"
            _wb_log_img(f"exp5/attn_map/img{img_i}", p)

        _wb_finish()

        for li, e in enumerate(entropy_matrix.mean(axis=1)):
            print(f"  Layer {li+1}: {e:.4f} nats")
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  Experiment 6: Overlapping Patches
# ═══════════════════════════════════════════════════════════════════════════

class Exp6_OverlappingPatches:
    NAME = "exp6_overlap"
    CONFIGS = [
        ("non_overlap", dict(patch_size=4, patch_stride=4)),
        ("overlap_s3",  dict(patch_size=4, patch_stride=3)),
        ("overlap_s2",  dict(patch_size=4, patch_stride=2)),
    ]

    def __init__(self):
        self.out_dir = RESULTS_DIR / self.NAME
        self.out_dir.mkdir(exist_ok=True)

    @staticmethod
    def _peak_gpu_mb(model):
        if DEVICE != "cuda":
            return 0.0
        model.to(DEVICE).eval()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            _ = model(torch.randn(256, 3, 32, 32, device=DEVICE))
        return torch.cuda.max_memory_allocated() / 1e6

    def run(self):
        print("\n" + "="*60)
        print("  Experiment 6: Overlapping Patches")
        print("="*60)
        train_loader, val_loader, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=VIT_CFG["batch_size"])
        _, test_loader, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=256)

        results = {}
        histories, labels = [], []
        es_kw = dict(patience=20, gap_threshold=0.25, target_acc=0.89)

        for name, kwargs in self.CONFIGS:
            print(f"\n  ── {name} ──")
            model = VisionTransformer(
                img_size=32, num_classes=10, embed_dim=192, depth=9,
                num_heads=12, mlp_ratio=2.0, dropout=0.1, attn_dropout=0.0,
                **kwargs)
            num_p = model.patch_embed.num_patches
            mem   = self._peak_gpu_mb(model)
            print(f"  num_patches={num_p}  mem={mem:.0f}MB")

            wb = _wb_init(
                run_name=f"exp6-{name}",
                config={**VIT_CFG, **kwargs, "num_patches": num_p,
                        "gpu_mem_mb": mem},
                group="exp6_overlap", tags=["exp6", name])

            t0 = time.time()
            h, best, tr = _run_vit(
                model, train_loader, val_loader, VIT_CFG,
                MODELS_DIR / f"exp6_{name}.pt",
                desc=name, es_kwargs=es_kw,
                wb_run=wb, wb_prefix=f"exp6/{name}")
            elapsed = time.time() - t0

            test_acc = _test_acc(model, test_loader)
            ana = run_post_training_analysis(
                tr, test_loader, self.out_dir,
                tag=name, wb_group="exp6")
            _wb_finish()

            results[name] = {
                "patch_size": kwargs["patch_size"],
                "stride": kwargs["patch_stride"],
                "num_patches": num_p, "best_val": best,
                "test_acc": test_acc, "training_time_s": elapsed,
                "gpu_memory_mb": mem, **ana,
            }
            histories.append(h); labels.append(name)
            print(f"  test={test_acc:.4f}  time={elapsed/60:.1f}min")

        _save_json(results, self.out_dir / "results.json")
        plot_loss_curves(histories, labels, "Overlapping Patches",
                         self.out_dir, "loss_curves.png")
        plot_training_dynamics(histories, labels,
                               self.out_dir, "training_dynamics.png")
        names = list(results.keys())
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        pal = ["steelblue", "seagreen", "darkorange"]
        for ax, (vals, yl) in zip(axes, [
            ([results[n]["test_acc"]        for n in names], "Test Accuracy"),
            ([results[n]["training_time_s"] for n in names], "Training Time (s)"),
            ([results[n]["gpu_memory_mb"]   for n in names], "Peak GPU Mem (MB)"),
        ]):
            bars = ax.bar(names, vals, color=pal[:len(names)],
                          edgecolor="black", alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=10)
            ax.set_ylabel(yl); ax.set_title(yl)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=15)
        plt.suptitle("Overlapping Patches Comparison", fontsize=14)
        _savefig(self.out_dir / "comparison_summary.png")
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  Experiment 7: Linear Probing
# ═══════════════════════════════════════════════════════════════════════════

class Exp7_LinearProbing:
    NAME         = "exp7_linear_probe"
    PROBE_LR     = 1e-3
    PROBE_EPOCHS = 30
    PROBE_BATCH  = 512

    def __init__(self):
        self.out_dir = RESULTS_DIR / self.NAME
        self.out_dir.mkdir(exist_ok=True)

    @torch.no_grad()
    def _extract(self, model, loader):
        model.eval().to(DEVICE)
        buckets = None; labels = []
        for imgs, lbls in tqdm(loader, desc="  Extracting", leave=False):
            out = model(imgs.to(DEVICE))
            if not isinstance(out, tuple) or len(out) < 2:
                raise RuntimeError("Need return_all_layers=True")
            feats = out[-1]
            if buckets is None:
                buckets = [[] for _ in feats]
            for li, f in enumerate(feats):
                buckets[li].append(f.cpu())
            labels.append(lbls)
        return [torch.cat(b) for b in buckets], torch.cat(labels)

    def _train_probe(self, tr_f, tr_l, te_f, te_l, layer_idx, wb_run):
        probe = nn.Linear(tr_f.shape[1], 10).to(DEVICE)
        opt   = torch.optim.Adam(probe.parameters(), lr=self.PROBE_LR,
                                  weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, T_max=self.PROBE_EPOCHS, eta_min=1e-5)
        crit  = nn.CrossEntropyLoss()
        ds    = torch.utils.data.TensorDataset(tr_f, tr_l)
        ldr   = torch.utils.data.DataLoader(
            ds, batch_size=self.PROBE_BATCH, shuffle=True, num_workers=0)
        best = 0.0
        for ep in range(self.PROBE_EPOCHS):
            probe.train()
            for f, l in ldr:
                opt.zero_grad()
                crit(probe(f.to(DEVICE)), l.to(DEVICE)).backward()
                opt.step()
            sched.step()
            probe.eval()
            with torch.no_grad():
                acc = (probe(te_f.to(DEVICE)).argmax(1)
                       == te_l.to(DEVICE)).float().mean().item()
            best = max(best, acc)
            _wb_log({f"exp7/probe/layer_{layer_idx+1}/acc": acc}, step=ep)
        return best

    def run(self):
        print("\n" + "="*60)
        print("  Experiment 7: Layer-wise Linear Probing")
        print("="*60)

        wb = _wb_init(run_name="exp7-linear-probe",
                      config={"probe_lr": self.PROBE_LR,
                               "probe_epochs": self.PROBE_EPOCHS},
                      group="exp7_linear_probe", tags=["exp7"])

        probe_model = vit_small(num_classes=10, return_all_layers=True)
        for cand in [MODELS_DIR / "exp4_learnable.pt",
                     MODELS_DIR / "exp3_cls.pt",
                     MODELS_DIR / "exp2_patch4.pt"]:
            if cand.exists():
                print(f"  Loading {cand.name}…")
                probe_model.load_state_dict(
                    torch.load(cand, map_location=DEVICE, weights_only=True))
                break
        else:
            print("  Training backbone from scratch…")
            tr_l, vl, _ = get_dataloaders(
                data_fraction=1.0, img_size=32,
                batch_size=VIT_CFG["batch_size"])
            ck = MODELS_DIR / "exp7_backbone.pt"
            _run_vit(probe_model, tr_l, vl, VIT_CFG, save_path=ck,
                     desc="Exp7 backbone",
                     es_kwargs=dict(patience=20, target_acc=0.89),
                     wb_run=wb, wb_prefix="exp7/backbone")
            probe_model.load_state_dict(
                torch.load(ck, map_location=DEVICE, weights_only=True))

        probe_model.eval().to(DEVICE)
        tr_ldr, _, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=256)
        _, te_ldr, _ = get_dataloaders(
            data_fraction=1.0, img_size=32, batch_size=256)

        print("  Extracting features…")
        tr_feats, tr_labels = self._extract(probe_model, tr_ldr)
        te_feats, te_labels = self._extract(probe_model, te_ldr)

        L = len(tr_feats)
        layer_accs = []
        for li in range(L):
            acc = self._train_probe(
                tr_feats[li], tr_labels, te_feats[li], te_labels,
                layer_idx=li, wb_run=wb)
            layer_accs.append(acc)
            _wb_log({"exp7/probe/layer_acc": acc,
                     "exp7/probe/layer_idx": li + 1})
            print(f"  Layer {li+1:>2}/{L}  acc={acc:.4f}")

        results = {"layer_accuracies": layer_accs,
                   "best_layer": int(np.argmax(layer_accs)) + 1,
                   "best_accuracy": float(max(layer_accs))}
        _save_json(results, self.out_dir / "results.json")
        plot_layer_accuracy(layer_accs, self.out_dir)
        _wb_log_img("exp7/layer_accuracy", self.out_dir / "layer_accuracy.png")

        # W&B table: layer index vs accuracy (renders as line chart)
        _wb_log_table("exp7/probe_results",
                      columns=["layer", "accuracy"],
                      data=[[i+1, a] for i, a in enumerate(layer_accs)])
        _wb_finish()

        print(f"\n  Best: layer {results['best_layer']}"
              f"  acc={results['best_accuracy']:.4f}")
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

_EXP_MAP = {
    # 1: Exp1_DataEfficiency,
    2: Exp2_PatchSize,
    3: Exp3_CLSvsMean,
    4: Exp4_PosEncoding,
    5: Exp5_AttentionAnalysis,
    6: Exp6_OverlappingPatches,
    7: Exp7_LinearProbing,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=int, nargs="+",
                   default=list(range(1, 8)),
                   help="Experiment IDs (default: all)")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable W&B logging for this run")
    return p.parse_args()


def main():
    global _USE_WANDB
    args = parse_args()
    if args.no_wandb:
        _USE_WANDB = False
        print("  W&B logging disabled via --no-wandb\n")

    all_res  = {}
    total_t0 = time.time()

    for eid in sorted(args.exp):
        if eid not in _EXP_MAP:
            print(f"[WARN] Unknown experiment {eid} — skipping.")
            continue
        t0  = time.time()
        res = _EXP_MAP[eid]().run()
        all_res[f"exp{eid}"] = res
        print(f"\n  Exp {eid} done in {(time.time()-t0)/60:.1f} min\n")

    total = time.time() - total_t0
    print("\n" + "="*60)
    print(f"  All done in {total/60:.1f} min")
    print("="*60)

    summary = {}
    for k, v in all_res.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    summary[f"{k}/{kk}"] = round(float(vv), 5)
    _save_json(summary, RESULTS_DIR / "summary.json")


if __name__ == "__main__":
    main()
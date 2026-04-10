"""
exp1_data_efficiency.py — Experiment 1: Data Efficiency of ViT vs CNN
======================================================================
Objective : Compare data efficiency of ViT and CNN on CIFAR-10.
Task      : Train ViT and ResNet-18 on 5%, 10%, 25%, 50%, 100% of data.
Metrics   : Test accuracy for each data fraction.
Plot      : Accuracy vs % of training data (ViT vs CNN).
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from src.ViT import vit_small
from src.resnet import resnet18_cifar
from src.data_setup import get_dataloaders

#  Device 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results/exp1_data_efficiency")
MODELS_DIR  = Path("saved_models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

#  Hyperparameters ─
VIT_CFG = dict(
    epochs=100, lr=3e-4, weight_decay=0.05, batch_size=256,
    warmup_epochs=5, grad_clip=1.0, label_smoothing=0.1,
)
CNN_CFG = dict(
    epochs=100, lr=0.1, weight_decay=5e-4, batch_size=512,
    warmup_epochs=3, grad_clip=0.0, label_smoothing=0.0,
)

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]


#  Optimizer / Scheduler ─

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


#  Training utilities 

def train_one_epoch(model, loader, optimizer, criterion, grad_clip, scaler, use_amp):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=DEVICE, enabled=use_amp):
            out = model(imgs)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.detach().argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, use_amp):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE, enabled=use_amp):
            out = model(imgs)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def train(model, train_loader, val_loader, cfg, save_path, desc, is_vit):
    """Train model and return (history, best_val_acc)."""
    use_amp  = DEVICE == "cuda"
    scaler   = torch.amp.GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    if is_vit:
        optimizer = make_vit_optimizer(model, cfg["lr"], cfg["weight_decay"])
    else:
        optimizer = make_cnn_optimizer(model, cfg["lr"], cfg["weight_decay"])

    scheduler    = make_scheduler(optimizer, cfg["warmup_epochs"], cfg["epochs"])
    best_val_acc = 0.0
    history      = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    model.to(DEVICE)
    for epoch in tqdm(range(cfg["epochs"]), desc=desc):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            cfg["grad_clip"], scaler, use_amp)
        v_loss, v_acc = evaluate(model, val_loader, criterion, use_amp)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            tqdm.write(
                f"  [{epoch+1:>3}/{cfg['epochs']}]  "
                f"TL={t_loss:.4f} TA={t_acc:.4f} | "
                f"VL={v_loss:.4f} VA={v_acc:.4f} | best={best_val_acc:.4f}"
            )

    # Reload best weights
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    return history, best_val_acc


@torch.no_grad()
def test_accuracy(model, loader):
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


#  Plotting 

def plot_accuracy_vs_data(vit_accs, cnn_accs, fractions):
    pct = [f * 100 for f in fractions]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pct, vit_accs, "o-", label="ViT-Small",  color="royalblue", lw=2.5, ms=8)
    ax.plot(pct, cnn_accs, "s-", label="ResNet-18", color="tomato",    lw=2.5, ms=8)
    for x, vy, cy in zip(pct, vit_accs, cnn_accs):
        ax.annotate(f"{vy:.3f}", (x, vy), textcoords="offset points",
                    xytext=(0, 7),  ha="center", fontsize=8, color="royalblue")
        ax.annotate(f"{cy:.3f}", (x, cy), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="tomato")
    ax.set_xlabel("Training Data (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Data Efficiency: ViT-Small vs ResNet-18 on CIFAR-10", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / "accuracy_vs_data.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_loss_curves(vit_history, cnn_history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in zip(axes, ["loss", "acc"], ["Loss", "Accuracy"]):
        ep = range(1, len(vit_history[f"train_{key}"]) + 1)
        ax.plot(ep, vit_history[f"train_{key}"], color="royalblue", label="ViT train")
        ax.plot(ep, vit_history[f"val_{key}"],   color="royalblue", linestyle="--", label="ViT val")
        ep = range(1, len(cnn_history[f"train_{key}"]) + 1)
        ax.plot(ep, cnn_history[f"train_{key}"], color="tomato", label="CNN train")
        ax.plot(ep, cnn_history[f"val_{key}"],   color="tomato", linestyle="--", label="CNN val")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.set_title(f"Full Data (100%) — {title}", fontsize=13)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / "loss_curves_full.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_json(data, path):
    def cvt(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=cvt)
    print(f"  Saved: {path}")


#  Main experiment ─

def run():
    print("\n" + "="*60)
    print("  Experiment 1: Data Efficiency  (ViT vs CNN)")
    print("="*60)

    _, test_loader, _ = get_dataloaders(data_fraction=1.0, img_size=32, batch_size=256)

    results = {"fractions": FRACTIONS, "vit_test_accs": [], "cnn_test_accs": []}
    vit_hist_full = cnn_hist_full = None

    for frac in FRACTIONS:
        print(f"\n Fraction {frac*100:.0f}% ")
        frac_tag = f"{frac*100:.0f}pct"

        tr_v, val_ldr, _ = get_dataloaders(
            data_fraction=frac, img_size=32, batch_size=VIT_CFG["batch_size"])
        tr_c, _, _ = get_dataloaders(
            data_fraction=frac, img_size=32, batch_size=CNN_CFG["batch_size"])

        # Train ViT
        vit = vit_small(num_classes=10)
        h_v, _ = train(vit, tr_v, val_ldr, VIT_CFG,
                        MODELS_DIR / f"exp1_vit_{frac_tag}.pt",
                        desc=f"ViT {frac_tag}", is_vit=True)
        vit_acc = test_accuracy(vit, test_loader)
        results["vit_test_accs"].append(vit_acc)
        if frac == 1.0:
            vit_hist_full = h_v

        # Train CNN
        cnn = resnet18_cifar(num_classes=10, base_channels=64, dropout=0.1)
        h_c, _ = train(cnn, tr_c, val_ldr, CNN_CFG,
                        MODELS_DIR / f"exp1_cnn_{frac_tag}.pt",
                        desc=f"CNN {frac_tag}", is_vit=False)
        cnn_acc = test_accuracy(cnn, test_loader)
        results["cnn_test_accs"].append(cnn_acc)
        if frac == 1.0:
            cnn_hist_full = h_c

        print(f"  → ViT test acc={vit_acc:.4f}  |  CNN test acc={cnn_acc:.4f}")

    # Plots
    plot_accuracy_vs_data(results["vit_test_accs"], results["cnn_test_accs"], FRACTIONS)
    if vit_hist_full and cnn_hist_full:
        plot_loss_curves(vit_hist_full, cnn_hist_full)

    save_json(results, RESULTS_DIR / "results.json")

    print("\n  Summary:")
    for frac, va, ca in zip(FRACTIONS, results["vit_test_accs"], results["cnn_test_accs"]):
        print(f"    {frac*100:>5.0f}%  ViT={va:.4f}  CNN={ca:.4f}")
    return results


if __name__ == "__main__":
    run()
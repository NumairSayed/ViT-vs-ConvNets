
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from src.ViT import vit_small
from src.data_setup import get_dataloaders

#  Device 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results/exp3_cls_vs_mean")
MODELS_DIR  = Path("saved_models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

#  Hyperparameters ─
VIT_CFG = dict(
    epochs=150, lr=3e-4, weight_decay=0.05, batch_size=256,
    warmup_epochs=15, grad_clip=1.0, label_smoothing=0.1,
)

VARIANTS = [
    ("cls",  "cls",  VIT_CFG),   # Standard CLS token
    ("mean", "mean", VIT_CFG),   # Mean pooling (no CLS token)
]


#  Optimizer / Scheduler ─

def make_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 or "bias" in name else decay).append(p)
    return AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999))


def make_scheduler(optimizer, warmup, total):
    if warmup and warmup > 0:
        w = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup)
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


def train(model, train_loader, val_loader, cfg, save_path, desc):
    """Train model and return (history, best_val_acc)."""
    use_amp   = DEVICE == "cuda"
    scaler    = torch.amp.GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = make_optimizer(model, cfg["lr"], cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, cfg["warmup_epochs"], cfg["epochs"])

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

def plot_loss_curves(histories, labels):
    """Training loss curves for all variants."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["royalblue", "tomato"]
    for hist, lbl, col in zip(histories, labels, colors):
        ep = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(ep, hist["train_loss"], color=col, label=f"{lbl} train")
        axes[0].plot(ep, hist["val_loss"],   color=col, linestyle="--", label=f"{lbl} val")
        axes[1].plot(ep, hist["train_acc"],  color=col, label=f"{lbl} train")
        axes[1].plot(ep, hist["val_acc"],    color=col, linestyle="--", label=f"{lbl} val")
    for ax, yl in zip(axes, ["Loss", "Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    axes[0].set_title("CLS vs Mean Pooling — Loss", fontsize=13)
    axes[1].set_title("CLS vs Mean Pooling — Accuracy", fontsize=13)
    plt.tight_layout()
    path = RESULTS_DIR / "loss_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_bar(names, test_accs):
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["royalblue", "tomato"]
    bars   = ax.bar(names, test_accs, color=colors[:len(names)],
                    edgecolor="black", alpha=0.85)
    for bar, v in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=12)
    ax.set_xlabel("Pooling Method", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("CLS Token vs Mean Pooling — Test Accuracy", fontsize=13)
    ax.set_ylim(0, min(1.0, max(test_accs) + 0.05))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / "accuracy_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_json(data, path):
    def cvt(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=cvt)
    print(f"  Saved: {path}")


#  Main experiment ─

def run():
    print("\n" + "="*60)
    print("  Experiment 3: CLS Token vs Mean Pooling")
    print("="*60)

    train_loader, val_loader, _ = get_dataloaders(
        data_fraction=1.0, img_size=32, batch_size=VIT_CFG["batch_size"])
    _, test_loader, _ = get_dataloaders(
        data_fraction=1.0, img_size=32, batch_size=256)

    results    = {}
    histories, hist_labels = [], []

    for name, pooling, cfg in VARIANTS:
        print(f"\n variant={name}  pooling={pooling} ")
        model = vit_small(num_classes=10, pooling=pooling)

        h, best = train(model, train_loader, val_loader, cfg,
                        MODELS_DIR / f"exp3_{name}.pt",
                        desc=name)
        test_acc = test_accuracy(model, test_loader)

        results[name] = {"pooling": pooling, "best_val_acc": best,
                         "test_acc": test_acc}
        histories.append(h); hist_labels.append(name)
        print(f"  test_acc={test_acc:.4f}")

    # Plots
    names     = [v[0] for v in VARIANTS]
    test_accs = [results[n]["test_acc"] for n in names]
    plot_loss_curves(histories, hist_labels)
    plot_accuracy_bar(names, test_accs)

    save_json(results, RESULTS_DIR / "results.json")

    print("\n  Summary:")
    for n in names:
        print(f"    {n:10s}  test_acc={results[n]['test_acc']:.4f}")
    return results


if __name__ == "__main__":
    run()

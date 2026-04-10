"""
exp7_linear_probe.py — Experiment 7: Layer-wise Representation Quality
========================================================================
Objective : Understand how feature representations evolve across layers.
Task      :
  (a) Extract CLS token (or pooled) features from each transformer layer.
  (b) Train a linear classifier on features from each layer separately.
Metrics   : Classification accuracy per layer.
Plot      : Layer index vs accuracy.

Prerequisite: Uses exp4_learnable.pt if available, otherwise trains from scratch.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from src.ViT import vit_small
from src.data_setup import get_dataloaders

#  Device 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results/exp7_linear_probe")
MODELS_DIR  = Path("saved_models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

#  Hyperparameters ─
BACKBONE_CFG = dict(
    epochs=150, lr=3e-4, weight_decay=0.05, batch_size=256,
    warmup_epochs=15, grad_clip=1.0, label_smoothing=0.1,
)
PROBE_LR     = 1e-3
PROBE_EPOCHS = 30
PROBE_BATCH  = 512


#  Feature extraction 

@torch.no_grad()
def extract_layer_features(model, loader):
    """
    Extract CLS token (or pooled) features from every transformer layer.

    Returns:
        layer_feats : list of L tensors, each shape (N, D)
        all_labels  : tensor of shape (N,)

    Requires model built with return_all_layers=True so that forward()
    returns (logits, [feat_layer_0, feat_layer_1, ..., feat_layer_L-1]).
    """
    model.eval().to(DEVICE)
    buckets = None
    labels  = []

    for imgs, lbls in tqdm(loader, desc="  Extracting features", leave=False):
        out = model(imgs.to(DEVICE))
        if not isinstance(out, tuple) or len(out) < 2:
            raise RuntimeError(
                "Model must be built with return_all_layers=True and return "
                "(logits, [layer_features]) from forward().")
        layer_feats = out[-1]  # list of L tensors (B, D)

        if buckets is None:
            buckets = [[] for _ in layer_feats]
        for li, f in enumerate(layer_feats):
            buckets[li].append(f.cpu())
        labels.append(lbls)

    layer_feats_all = [torch.cat(b) for b in buckets]
    all_labels      = torch.cat(labels)
    return layer_feats_all, all_labels


#  Linear probe training ─

def train_linear_probe(tr_feats, tr_labels, te_feats, te_labels, layer_idx):
    """
    Train a single linear layer on pre-extracted features.
    Returns best test accuracy achieved over PROBE_EPOCHS epochs.
    """
    dim   = tr_feats.shape[1]
    probe = nn.Linear(dim, 10).to(DEVICE)
    opt   = torch.optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=PROBE_EPOCHS, eta_min=1e-5)
    crit  = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(tr_feats, tr_labels)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=PROBE_BATCH, shuffle=True, num_workers=0)

    best_acc = 0.0
    for ep in range(PROBE_EPOCHS):
        # Train
        probe.train()
        for f, l in loader:
            opt.zero_grad()
            crit(probe(f.to(DEVICE)), l.to(DEVICE)).backward()
            opt.step()
        sched.step()

        # Evaluate on test features
        probe.eval()
        with torch.no_grad():
            preds = probe(te_feats.to(DEVICE)).argmax(1)
            acc   = (preds == te_labels.to(DEVICE)).float().mean().item()
        best_acc = max(best_acc, acc)

    return best_acc


#  Plotting 

def plot_layer_accuracy(layer_accs):
    layers = list(range(1, len(layer_accs) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layers, layer_accs, "o-", color="teal", lw=2.5, ms=8)
    for l, acc in zip(layers, layer_accs):
        ax.annotate(f"{acc:.3f}", (l, acc),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    best_l   = int(np.argmax(layer_accs)) + 1
    best_acc = max(layer_accs)
    ax.axhline(best_acc, color="teal", linestyle=":", alpha=0.5,
               label=f"best: layer {best_l} = {best_acc:.4f}")
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Linear Probe Accuracy", fontsize=12)
    ax.set_title("Layer-wise Representation Quality (Linear Probing)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    plt.tight_layout()
    path = RESULTS_DIR / "layer_accuracy.png"
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


#  Backbone trainer (used only when no checkpoint is available) 

def _train_backbone(model, save_path):
    """Minimal backbone training loop for Exp 7."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    tr_l, vl, _ = get_dataloaders(
        data_fraction=1.0, img_size=32, batch_size=BACKBONE_CFG["batch_size"])

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 or "bias" in name else decay).append(p)
    optimizer = AdamW(
        [{"params": decay, "weight_decay": BACKBONE_CFG["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=BACKBONE_CFG["lr"], betas=(0.9, 0.999))
    w = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                 total_iters=BACKBONE_CFG["warmup_epochs"])
    c = CosineAnnealingLR(optimizer,
                          T_max=BACKBONE_CFG["epochs"] - BACKBONE_CFG["warmup_epochs"],
                          eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [w, c],
                              milestones=[BACKBONE_CFG["warmup_epochs"]])
    criterion = nn.CrossEntropyLoss(label_smoothing=BACKBONE_CFG["label_smoothing"])
    use_amp   = DEVICE == "cuda"
    scaler    = torch.amp.GradScaler(enabled=use_amp)

    model.to(DEVICE)
    best_acc = 0.0

    for epoch in tqdm(range(BACKBONE_CFG["epochs"]), desc="Backbone training"):
        model.train()
        for imgs, labels in tr_l:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE, enabled=use_amp):
                out  = model(imgs)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), BACKBONE_CFG["grad_clip"])
            scaler.step(optimizer); scaler.update()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval(); correct = total = 0
            with torch.no_grad():
                for imgs, labels in vl:
                    out = model(imgs.to(DEVICE))
                    logits = out[0] if isinstance(out, tuple) else out
                    correct += (logits.argmax(1) == labels.to(DEVICE)).sum().item()
                    total   += labels.size(0)
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)
            tqdm.write(f"  [{epoch+1}]  val_acc={acc:.4f}  best={best_acc:.4f}")

    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))


#  Main experiment ─

def run():
    print("\n" + "="*60)
    print("  Experiment 7: Layer-wise Linear Probing")
    print("="*60)

    # Build backbone that returns features from all layers
    probe_model = vit_small(num_classes=10, return_all_layers=True)

    # Load pre-trained weights (prefer earlier experiment checkpoints)
    ckpt = None
    for cand in [MODELS_DIR / "exp4_learnable.pt",
                 MODELS_DIR / "exp3_cls.pt",
                 MODELS_DIR / "exp2_patch4.pt"]:
        if cand.exists():
            ckpt = cand
            break

    if ckpt is not None:
        print(f"  Loading backbone from {ckpt.name} …")
        probe_model.load_state_dict(
            torch.load(ckpt, map_location=DEVICE, weights_only=True))
    else:
        scratch_ckpt = MODELS_DIR / "exp7_backbone.pt"
        print("  No existing checkpoint found — training backbone from scratch …")
        _train_backbone(probe_model, scratch_ckpt)

    probe_model.eval().to(DEVICE)

    # Data loaders (use train set to extract train features, val set for test)
    tr_ldr, te_ldr, _ = get_dataloaders(
        data_fraction=1.0, img_size=32, batch_size=256)
    _, te_ldr, _       = get_dataloaders(
        data_fraction=1.0, img_size=32, batch_size=256)

    print("  Extracting training features …")
    tr_feats, tr_labels = extract_layer_features(probe_model, tr_ldr)
    print("  Extracting test features …")
    te_feats, te_labels = extract_layer_features(probe_model, te_ldr)

    L = len(tr_feats)
    print(f"  Number of layers: {L}  |  Feature dim: {tr_feats[0].shape[1]}")

    # Train one linear probe per layer
    layer_accs = []
    for li in range(L):
        acc = train_linear_probe(
            tr_feats[li], tr_labels,
            te_feats[li], te_labels,
            layer_idx=li)
        layer_accs.append(acc)
        print(f"  Layer {li+1:>2}/{L}  acc={acc:.4f}")

    # Plot & save
    plot_layer_accuracy(layer_accs)

    results = {
        "layer_accuracies": layer_accs,
        "best_layer":       int(np.argmax(layer_accs)) + 1,
        "best_accuracy":    float(max(layer_accs)),
    }
    save_json(results, RESULTS_DIR / "results.json")

    print(f"\n  Best layer: {results['best_layer']}  "
          f"accuracy: {results['best_accuracy']:.4f}")
    return results


if __name__ == "__main__":
    run()


import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from src.ViT import vit_small
from src.data_setup import get_dataloaders

#  Device 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results/exp5_attention")
MODELS_DIR  = Path("saved_models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "maps").mkdir(exist_ok=True)

#  Hyperparameters (used only if training from scratch) 
VIT_CFG = dict(
    epochs=150, lr=3e-4, weight_decay=0.05, batch_size=256,
    warmup_epochs=15, grad_clip=1.0, label_smoothing=0.1,
)


#  Entropy computation ─

def attention_entropy(attn):
    """
    Compute Shannon entropy per head averaged over batch and query positions.
    attn: (B, heads, seq, seq)
    Returns: ndarray of shape (heads,) — mean entropy in nats.
    """
    eps = 1e-9
    H   = -(attn * (attn + eps).log()).sum(dim=-1)  # (B, heads, seq)
    return H.mean(dim=(0, 2)).cpu().numpy()          # (heads,)


#  Plotting 

def plot_entropy_per_layer(entropy_matrix):
    """
    entropy_matrix: ndarray of shape (L, heads)
      Row i = entropy per head averaged over all batches for layer i.
    """
    L, heads = entropy_matrix.shape
    layers   = list(range(1, L + 1))
    mean_e   = entropy_matrix.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean entropy per layer with head range ribbon
    axes[0].plot(layers, mean_e, "o-", color="darkorchid", lw=2.5, ms=7,
                 label="mean across heads")
    axes[0].fill_between(layers,
                         entropy_matrix.min(axis=1),
                         entropy_matrix.max(axis=1),
                         alpha=0.2, color="darkorchid", label="head min/max")
    axes[0].set_xlabel("Layer", fontsize=12)
    axes[0].set_ylabel("Entropy (nats)", fontsize=12)
    axes[0].set_title("Attention Entropy vs Layer Depth", fontsize=13)
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Right: heatmap (head × layer)
    im = axes[1].imshow(entropy_matrix.T, aspect="auto", cmap="plasma",
                        interpolation="nearest")
    axes[1].set_xlabel("Layer", fontsize=12); axes[1].set_ylabel("Head", fontsize=12)
    axes[1].set_title("Entropy Heatmap (Head × Layer)", fontsize=13)
    axes[1].set_xticks(range(L));      axes[1].set_xticklabels(layers)
    axes[1].set_yticks(range(heads));  axes[1].set_yticklabels(range(1, heads + 1))
    plt.colorbar(im, ax=axes[1], label="Entropy (nats)")

    plt.tight_layout()
    path = RESULTS_DIR / "entropy_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_attention_maps(attn_maps_list, orig_imgs, num_images=4):
    """
    Visualise CLS→patch attention for each layer on a few sample images.
    attn_maps_list : list of length L, each tensor (B, heads, seq, seq)
    orig_imgs      : tensor (B, 3, H, W) normalised images
    """
    L = len(attn_maps_list)
    num_patches = attn_maps_list[0].shape[-1] - 1  # subtract CLS
    grid = int(num_patches ** 0.5)

    # ImageNet-style denorm for CIFAR-10
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    for img_i in range(min(num_images, orig_imgs.shape[0])):
        cols = min(5, L + 1)
        rows = (L + 1 + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.array(axes).flatten()

        # Original image
        orig = (orig_imgs[img_i].cpu() * std + mean).permute(1, 2, 0).numpy().clip(0, 1)
        axes[0].imshow(orig); axes[0].set_title("Input", fontsize=9); axes[0].axis("off")

        # One panel per layer: mean-head CLS→patch attention
        for li, attn in enumerate(attn_maps_list):
            # attn: (B, heads, seq, seq); take CLS row [0], mean over heads
            cls_attn = attn[img_i].mean(0)[0, 1:].cpu().numpy()  # (num_patches,)
            cls_attn = cls_attn.reshape(grid, grid)
            cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
            axes[li + 1].imshow(cls_attn, cmap="inferno", interpolation="bilinear")
            axes[li + 1].set_title(f"Layer {li+1}", fontsize=8)
            axes[li + 1].axis("off")

        for j in range(L + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(f"CLS → Patch Attention (sample {img_i})", fontsize=11)
        plt.tight_layout()
        path = RESULTS_DIR / "maps" / f"attn_img{img_i}.png"
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
    print("  Experiment 5: Attention Maps & Entropy Analysis")
    print("="*60)

    ckpt = MODELS_DIR / "exp5_attn_model.pt"

    # Build model with return_attention=True so forward() returns (logits, [attn_per_layer])
    attn_model = vit_small(num_classes=10, return_attention=True)

    # Load from exp4 learnable checkpoint if available, else train from scratch
    if not ckpt.exists():
        fallback = MODELS_DIR / "exp4_learnable.pt"
        if fallback.exists():
            print(f"  Loading weights from {fallback.name} …")
            attn_model.load_state_dict(
                torch.load(fallback, map_location=DEVICE, weights_only=True))
            torch.save(attn_model.state_dict(), ckpt)
        else:
            print("  No pre-trained checkpoint found — training from scratch …")
            import torch.nn as nn
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            tr_l, vl, _ = get_dataloaders(
                data_fraction=1.0, img_size=32, batch_size=VIT_CFG["batch_size"])

            decay, no_decay = [], []
            for name, p in attn_model.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim == 1 or "bias" in name else decay).append(p)
            optimizer = AdamW(
                [{"params": decay, "weight_decay": VIT_CFG["weight_decay"]},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=VIT_CFG["lr"], betas=(0.9, 0.999))
            w = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                         total_iters=VIT_CFG["warmup_epochs"])
            c = CosineAnnealingLR(optimizer, T_max=VIT_CFG["epochs"] - VIT_CFG["warmup_epochs"],
                                  eta_min=1e-6)
            scheduler = SequentialLR(optimizer, [w, c], milestones=[VIT_CFG["warmup_epochs"]])
            criterion = nn.CrossEntropyLoss(label_smoothing=VIT_CFG["label_smoothing"])
            use_amp   = DEVICE == "cuda"
            scaler    = torch.amp.GradScaler(enabled=use_amp)
            attn_model.to(DEVICE)
            best_acc  = 0.0

            for epoch in tqdm(range(VIT_CFG["epochs"]), desc="Exp5 training"):
                attn_model.train()
                for imgs, labels in tr_l:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=DEVICE, enabled=use_amp):
                        out = attn_model(imgs)
                        logits = out[0] if isinstance(out, tuple) else out
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(attn_model.parameters(), VIT_CFG["grad_clip"])
                    scaler.step(optimizer); scaler.update()
                scheduler.step()
                # Quick val check every 10 epochs
                if (epoch + 1) % 10 == 0:
                    attn_model.eval(); correct = total = 0
                    with torch.no_grad():
                        for imgs, labels in vl:
                            out = attn_model(imgs.to(DEVICE))
                            logits = out[0] if isinstance(out, tuple) else out
                            correct += (logits.argmax(1) == labels.to(DEVICE)).sum().item()
                            total   += labels.size(0)
                    acc = correct / total
                    if acc > best_acc:
                        best_acc = acc
                        torch.save(attn_model.state_dict(), ckpt)
                    tqdm.write(f"  [{epoch+1}]  val_acc={acc:.4f}  best={best_acc:.4f}")

    # Load best weights
    attn_model.load_state_dict(
        torch.load(ckpt, map_location=DEVICE, weights_only=True))
    attn_model.to(DEVICE).eval()
    print(f"  Model loaded from {ckpt}")

    # Use validation loader for entropy / attention map computation
    _, vis_loader, _ = get_dataloaders(data_fraction=1.0, img_size=32, batch_size=32)

    L              = None
    entropy_accum  = None
    sample_maps    = None
    sample_imgs    = None
    n_batches      = 0

    print("  Computing attention entropy …")
    for imgs, _ in tqdm(vis_loader, desc="  Entropy"):
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            result = attn_model(imgs)

        if not isinstance(result, tuple) or len(result) < 2:
            print("  [ERROR] Model did not return attention maps. "
                  "Ensure return_attention=True is implemented.")
            return {}

        attn_maps = result[1]  # list of length L, each (B, heads, seq, seq)

        if L is None:
            L             = len(attn_maps)
            entropy_accum = np.zeros((L, attn_maps[0].shape[1]))
            sample_maps   = [a.cpu() for a in attn_maps]
            sample_imgs   = imgs.cpu()

        for i, a in enumerate(attn_maps):
            entropy_accum[i] += attention_entropy(a)

        n_batches += 1
        if n_batches >= 30:  # 30 batches of 32 = 960 samples
            break

    entropy_matrix = entropy_accum / n_batches  # (L, heads)

    # Plots
    plot_entropy_per_layer(entropy_matrix)
    plot_attention_maps(sample_maps, sample_imgs, num_images=4)

    # Numerical summary
    results = {
        "num_layers":              L,
        "num_heads":               int(entropy_matrix.shape[1]),
        "mean_entropy_per_layer":  entropy_matrix.mean(axis=1).tolist(),
        "entropy_matrix":          entropy_matrix.tolist(),  # (L, heads)
    }
    save_json(results, RESULTS_DIR / "results.json")

    print("\n  Entropy per layer (mean over heads):")
    for li, e in enumerate(entropy_matrix.mean(axis=1)):
        print(f"    Layer {li+1:>2}:  {e:.4f} nats")

    return results


if __name__ == "__main__":
    run()

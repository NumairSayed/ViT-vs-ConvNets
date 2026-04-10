
ORGANIZATION
main.py                  — Entry point; runs any/all experiments
exp1_data_efficiency.py  — Exp 1: ViT vs CNN data efficiency
exp2_patch_size.py       — Exp 2: Effect of patch size
exp3_cls_vs_mean.py      — Exp 3: CLS token vs mean pooling
exp4_pos_encoding.py     — Exp 4: Positional encoding ablation
exp5_attention.py        — Exp 5: Attention maps & entropy
exp6_overlap.py          — Exp 6: Overlapping vs non-overlapping patches
exp7_linear_probe.py     — Exp 7: Layer-wise linear probing

src/
  ViT.py                 — VisionTransformer, vit_small, etc.
  resnet.py              — resnet18_cifar (used in Exp 1)
  data_setup.py          — get_dataloaders() for CIFAR-10

results/                 — Auto-created; one sub-folder per experiment
saved_models/            — Auto-created; best checkpoints saved here


HOW TO RUN
Run ALL 7 experiments sequentially:
    python main.py

Run specific experiments (e.g., 1, 3, and 5 only):
    python main.py --exp 1 3 5

Run a single experiment file directly:
    python exp1_data_efficiency.py
    python exp2_patch_size.py
    python exp3_cls_vs_mean.py
    python exp4_pos_encoding.py
    python exp5_attention.py
    python exp6_overlap.py
    python exp7_linear_probe.py


NOTES ON EXPERIMENT DEPENDENCIES
- Exp 5 (attention) uses exp4_learnable.pt if present, otherwise trains fresh.
- Exp 7 (linear probe) uses exp4_learnable.pt → exp3_cls.pt → exp2_patch4.pt
  (in priority order) if present; otherwise trains a backbone from scratch.
- Running experiments in order (1→7) is therefore recommended to save time,
  but each experiment is self-contained and can be run independently.


OUTPUTS
After each experiment, results are written to results/exp{N}_<name>/:
  results.json    — Numerical results (accuracies, times, etc.)
  loss_curves.png — Training/validation loss and accuracy curves
  accuracy_bar.png / comparison_summary.png — Bar charts of key metrics
  entropy_analysis.png (Exp 5) — Entropy vs layer + heatmap
  maps/attn_img*.png    (Exp 5) — Per-layer attention visualisations
  layer_accuracy.png    (Exp 7) — Layer index vs probe accuracy

A cross-experiment summary is written to results/summary.json.


HARDWARE

A CUDA-capable GPU is strongly recommended.
On CPU, reduce `epochs` in the VIT_CFG / CNN_CFG dicts inside each file.
Mixed-precision (AMP) is used automatically when CUDA is available.
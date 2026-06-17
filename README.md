# KG-MixNet

**Knowledge-Guided Geometric Mixup for Zero-Shot Compound Deficiency Recognition in Coffee Leaves**

This repository contains the official implementation used to produce the results
in the paper. KG-MixNet recognizes **unseen compound** nutrient deficiencies in
coffee leaves while training only on **single-deficiency** (and healthy) images,
formulated as generalized zero-shot learning (GZSL).

The model combines:
- a **DINOv3 ViT-L/16** visual encoder (only the last two blocks + final norm are fine-tuned);
- a lightweight **residual bottleneck adapter** (FRA, 1024 → 512 → 1024);
- **MPNet** semantic prototypes built from a hierarchical **symptom knowledge graph**;
- a **Multimodal Geometric Mixup (MMG)** branch that synthesizes visual–semantic
  pairs on the unit hypersphere (linear interpolation + spherical re-projection —
  a hypersphere-aware approximation, not exact geodesic interpolation);
- a **dual objective**: seen-class cross-entropy + cosine alignment + KL regularization.

## Repository structure

```
KG-MixNet/
├── models/
│   ├── network.py          # KG_MixNet (main model; adapter & freeze flags)
│   └── baselines.py         # adapted in-house baselines (Table 2)
├── utils/
│   ├── taxonomy.py          # symptom knowledge graph (class descriptions)
│   └── custom_dataset.py    # GZSL dataset (source single-label / target multi-label)
├── scripts/
│   ├── build_kg.py          # encode KG descriptions -> class_embeddings.pt (MPNet)
│   ├── generate_simple_emb.py  # plain class-name prototypes (Table 4 ablation)
│   ├── data_loader.py       # CoLeaf-DB loading + augmentation
│   ├── class_embeddings.pt        # precomputed KG prototypes (9 x 768)
│   └── class_embeddings_simple.pt # precomputed simple-name prototypes (9 x 768)
├── train.py                 # 5-fold training (main results, Table 1)
├── eval_non_oracle.py       # practical non-oracle inference analysis (Table 6)
├── requirements.txt
└── LICENSE
```

## Installation

```bash
python -m venv venv && source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.x, CUDA 12.x, on a single GPU.

## Data and backbone

1. **CoLeaf-DB dataset** (Tuesta-Monteza et al., 2023, *Data in Brief*,
   DOI: [10.1016/j.dib.2023.109226](https://doi.org/10.1016/j.dib.2023.109226)).
   Arrange it so the root contains nine single-deficiency / healthy folders plus a
   compound folder:

   ```
   CoLeaf_raw_images/
   ├── boron-B/  calcium-Ca/  iron-Fe/  magnesium-Mg/  manganese-Mn/
   ├── nitrogen-N/  phosphorus-P/  potassium-K/  healthy/
   └── more-deficiencies/        # compound images, multi-label parsed from filenames
   ```

   Compound filenames encode their components, e.g. `B_Ca (1).jpg` → `['B','Ca']`.
   Point the code at your copy via `--data_root` or the `COLEAF_DATA_ROOT` env var.

2. **DINOv3 ViT-L/16** backbone (`dinov3-vitl16-pretrain-lvd1689m`). Download the
   weights and pass the folder via `--dino_path` or the `DINOV3_PATH` env var.

> Note: image counts in the paper follow a file-level audit of the local copy
> (901 seen + 104 unseen compound across 21 combinations). Counts may differ
> slightly across dataset mirrors.

## Knowledge-graph prototypes

The precomputed prototypes are included under `scripts/`. To regenerate them:

```bash
python scripts/build_kg.py            # -> class_embeddings.pt   (KG descriptions)
python scripts/generate_simple_emb.py # -> class_embeddings_simple.pt (plain names)
```

## Training (reproducing the paper)

```bash
# Main 5-fold result (Table 1, oracle-cardinality protocol)
python train.py --data_root /path/to/CoLeaf_raw_images --dino_path /path/to/dinov3-vitl16

# Fixed single-checkpoint protocol used for Tables 2-5 (fold-1)
python train.py --folds 1
```

Ablations (Table 3) and design-choice studies (Table 4) reuse the **same** loop;
each configuration corresponds to a switch the authors toggled:

| Configuration | Command |
|---|---|
| Full model | `python train.py` |
| w/o residual adapter (FRA) | `python train.py --no_adapter` |
| w/o dual-loss (align + KL off) | `python train.py --lambda_align 0 --lambda_kl 0` |
| Frozen backbone | `python train.py --freeze_backbone` |
| Simple class names instead of KG | `python train.py --embeddings simple` |

Seen accuracy is evaluated on each fold's held-out 20% split (no leakage); unseen
accuracy uses the fixed compound set under the **oracle-cardinality** protocol
(prediction size = ground-truth component count). The best checkpoint per fold is
selected by H-Mean.

## Pretrained checkpoints

Trained weights are released as **GitHub Release assets** (each checkpoint
bundles the DINOv3 backbone, ≈1.2 GB, so it cannot live in Git). Download
`best_fold_1.pth` … `best_fold_5.pth` from the
[Releases page](https://github.com/1931840268/KG-MixNet/releases) and put them
in `checkpoints/` (see `checkpoints/README.md`). Re-running `train.py` with the
fixed seed reproduces equivalent checkpoints from scratch.

## Non-oracle evaluation (Table 6)

The oracle-cardinality numbers are an **upper bound**. For a deployment-oriented
estimate, calibrate an adaptive decision rule (no retraining) and grid-search it:

```bash
python eval_non_oracle.py --checkpoint checkpoints/best_fold_1.pth \
                          --data_root /path/to/CoLeaf_raw_images \
                          --dino_path /path/to/dinov3-vitl16
```

This reports unseen sample-wise F1, seen strict accuracy, and seen false-positive
rate. These numbers must not be mixed with the oracle-cardinality tables.

## Default hyperparameters

AdamW (head lr 1e-4, backbone lr 1e-6, weight decay 1e-4), cosine annealing
(T_max = 40, eta_min = 1e-6), batch size 32, 40 epochs, 5-fold `KFold(shuffle=True,
random_state=42)`. Loss weights: CE 1.0, align 1.0, KL 0.5. MMG: perturb p=0.3
(σ=0.05), dual-mix p=0.5 (λ ~ Beta(1,1)), identity p=0.2.

## Honest notes on scope

- The comparison baselines in `models/baselines.py` are **adapted in-house**
  re-implementations on a shared backbone and common protocol, not official
  reproductions; the comparison is an internal controlled study.
- The study uses a single, small, imbalanced public benchmark (CoLeaf-DB); the
  paper reports statistical uncertainty (incl. a Wilson interval for the small
  unseen set) and does not claim state-of-the-art or field-ready performance.

## Citation

```bibtex
@article{wang2026kgmixnet,
  title   = {KG-MixNet: Knowledge-Guided Geometric Mixup for Zero-Shot Compound
             Deficiency Recognition in Coffee Leaves},
  author  = {Wang, Yuqi},
  year    = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`).

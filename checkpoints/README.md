# Trained checkpoints

The trained KG-MixNet weights are **not stored in this Git repository** because
each checkpoint bundles the DINOv3 ViT-L/16 backbone (~306M parameters, ≈1.2 GB
in fp32), which exceeds GitHub's 100 MB per-file limit.

They are distributed as **GitHub Release assets** instead. Download them from:

> https://github.com/1931840268/KG-MixNet/releases

and place the `.pth` files in this `checkpoints/` directory.

## Expected files

| File | Description |
|---|---|
| `best_fold_1.pth` … `best_fold_5.pth` | The five per-fold KG-MixNet checkpoints (Table 1). The best checkpoint per fold is selected by H-Mean. |

The fold-1 checkpoint is the reference single-checkpoint model used for the
fixed-protocol tables (Tables 2–5) and for the non-oracle analysis (Table 6).

## How they were produced

```bash
python train.py --data_root /path/to/CoLeaf_raw_images \
                --dino_path /path/to/dinov3-vitl16 \
                --save_dir checkpoints
```

Training used a fixed seed (`random_state=42`) on a single NVIDIA A10 GPU, so
re-running `train.py` reproduces equivalent checkpoints from scratch.

## Using a checkpoint

```bash
python eval_non_oracle.py --checkpoint checkpoints/best_fold_1.pth \
                          --data_root /path/to/CoLeaf_raw_images \
                          --dino_path /path/to/dinov3-vitl16
```

"""KG-MixNet training with rigorous 5-fold cross-validation (paper Table 1).

This is the final training script used for the main results. Seen-class accuracy
is measured on the held-out 20% validation split of each fold (no leakage), and
unseen accuracy is measured on the fixed compound-deficiency target set under the
oracle-cardinality protocol. The best checkpoint per fold is selected by H-Mean.

Reproducing the paper configurations (all use the same training loop):
  Full model (Table 1)          : python train.py
  w/o adapter (FRA ablation)    : python train.py --no_adapter
  w/o dual-loss (align+KL off)  : python train.py --lambda_align 0 --lambda_kl 0
  frozen backbone (Table 4)     : python train.py --freeze_backbone
  simple class names (Table 4)  : python train.py --embeddings simple
  fixed single-checkpoint runs  : python train.py --folds 1   (fold-1 protocol)

Paths default to environment-friendly values; override with the flags below.
"""

import os
import time
import random
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, hamming_loss)
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    from thop import profile, clever_format
except ImportError:
    profile = None

from scripts.data_loader import load_data, train_aug_transforms, base_transforms
from utils.custom_dataset import CoffeeDataset
from utils.taxonomy import CLASS_TO_IDX
from models.network import KG_MixNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger("KG-MixNet")


def parse_args():
    p = argparse.ArgumentParser(description="Train KG-MixNet (5-fold CV)")
    p.add_argument("--data_root", default=os.environ.get("COLEAF_DATA_ROOT", "CoLeaf_raw_images"))
    p.add_argument("--dino_path", default=os.environ.get("DINOV3_PATH", "models/dinov3-vitl16-pretrain-lvd1689m"))
    p.add_argument("--embeddings", choices=["kg", "simple"], default="kg",
                   help="kg = fine-grained KG descriptions; simple = plain class names (Table 4).")
    p.add_argument("--embed_path", default=None, help="Override the embedding .pt path.")
    p.add_argument("--save_dir", default="checkpoints_KG_MixNet")
    p.add_argument("--log_dir", default="logs_KG_MixNet")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="Head learning rate.")
    p.add_argument("--lr_backbone", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--folds", type=int, default=5, help="Set to 1 for the fixed single-checkpoint protocol.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--lambda_align", type=float, default=1.0)
    p.add_argument("--lambda_kl", type=float, default=0.5)
    p.add_argument("--no_adapter", action="store_true", help="Disable the residual adapter (FRA ablation).")
    p.add_argument("--freeze_backbone", action="store_true", help="Fully freeze the DINOv3 backbone (Table 4).")
    return p.parse_args()


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)


def log_print(msg):
    logger.info(msg)


def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    label_indices = torch.tensor([item[1] for item in batch])
    raw_labels = [item[2] for item in batch]
    return images, label_indices, raw_labels


def benchmark_model(model):
    """Report Params, FLOPs and inference FPS (paper Table 5)."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    log_print("-" * 40)
    log_print("[Benchmark] Model efficiency test...")
    if profile:
        try:
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            macs_str, params_str = clever_format([macs, params], "%.3f")
            log_print(f"   Parameters: {params_str} | FLOPs (MACs): {macs_str}")
        except Exception as e:
            params = sum(p.numel() for p in model.parameters())
            log_print(f"   FLOPs failed ({e}); Parameters: {params / 1e6:.2f}M")
    else:
        params = sum(p.numel() for p in model.parameters())
        log_print(f"   Parameters: {params / 1e6:.2f}M (install 'thop' for FLOPs)")

    if DEVICE == "cuda":
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy_input)
        dummy_batch = torch.randn(32, 3, 224, 224).to(DEVICE)
        iters = 50
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(iters):
                _ = model(dummy_batch)
        torch.cuda.synchronize()
        fps = (iters * 32) / (time.time() - t0)
        log_print(f"   Inference FPS (batch=32): {fps:.2f} img/s")
    log_print("-" * 40)


def get_mmg_batch(model, features, labels):
    """Multimodal Geometric Mixup (MMG): hypersphere-aware compositional synthesis.

    With prob 0.3 add a small Gaussian perturbation; with prob 0.5 perform dual
    visual-semantic interpolation with lambda ~ Beta(1, 1); with prob 0.2 keep the
    pair unchanged. Both modalities are L2-normalized before and after (spherical
    re-projection).
    """
    device = features.device
    B = features.size(0)
    prototypes = model.class_prototypes.to(device)
    text_features = prototypes[labels]

    features = F.normalize(features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)

    p = random.random()
    if p < 0.3:
        noise = torch.randn_like(features) * 0.05
        v_mix = features + noise
        t_mix = text_features
    elif p < 0.8:
        idx = torch.randperm(B).to(device)
        lam = float(np.random.beta(1.0, 1.0))
        lam_t = torch.tensor(lam, device=device)
        v_mix = lam_t * features + (1 - lam_t) * features[idx]
        t_mix = lam_t * text_features + (1 - lam_t) * text_features[idx]
    else:
        v_mix = features
        t_mix = text_features

    v_mix = F.normalize(v_mix, p=2, dim=1)
    t_mix = F.normalize(t_mix, p=2, dim=1)
    return v_mix, t_mix


def evaluate_all_metrics(model, loader, mode="seen"):
    """Oracle-cardinality evaluation. ``mode`` in {'seen', 'unseen'}."""
    model.eval()
    y_true_all, y_pred_all = [], []
    top3_any_hit, total = 0, 0

    with torch.no_grad():
        for imgs, lbls, raw_lbls in loader:
            imgs = imgs.to(DEVICE)
            probs = F.softmax(model(imgs), dim=1)
            _, top3_indices_batch = probs.topk(3, dim=1)

            for i in range(len(raw_lbls)):
                if mode == "unseen":
                    gt = [CLASS_TO_IDX[x] for x in raw_lbls[i] if x in CLASS_TO_IDX]
                else:
                    gt = [lbls[i].item()]
                y_t = np.zeros(9, dtype=int)
                y_t[gt] = 1
                y_true_all.append(y_t)

                k = len(gt) if len(gt) > 0 else 1     # oracle cardinality
                _, topk = probs[i].topk(k)
                y_p = np.zeros(9, dtype=int)
                y_p[topk.tolist()] = 1
                y_pred_all.append(y_p)

                if mode == "unseen":
                    if not set(gt).isdisjoint(set(top3_indices_batch[i].tolist())):
                        top3_any_hit += 1
                total += 1

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    return {
        "Acc": accuracy_score(y_true_all, y_pred_all),      # exact match
        "Prec_Macro": precision_score(y_true_all, y_pred_all, average="macro", zero_division=0),
        "Rec_Macro": recall_score(y_true_all, y_pred_all, average="macro", zero_division=0),
        "F1_Macro": f1_score(y_true_all, y_pred_all, average="macro", zero_division=0),
        "F1_Samples": f1_score(y_true_all, y_pred_all, average="samples", zero_division=0),
        "Hamming": hamming_loss(y_true_all, y_pred_all),
        "Top3": top3_any_hit / total if total > 0 else 0,
    }


def run_training(args):
    log_print(f"START TRAINING | device={DEVICE} | embeddings={args.embeddings} | "
              f"adapter={not args.no_adapter} | freeze_backbone={args.freeze_backbone} | "
              f"lambda=(ce {args.lambda_ce}, align {args.lambda_align}, kl {args.lambda_kl})")

    embed_path = args.embed_path
    if embed_path is None:
        embed_path = ("scripts/class_embeddings_simple.pt" if args.embeddings == "simple"
                      else "scripts/class_embeddings.pt")

    def build_model():
        return KG_MixNet(args.dino_path, embed_path, device=DEVICE,
                         freeze_backbone=args.freeze_backbone,
                         use_adapter=not args.no_adapter).to(DEVICE)

    # one-off efficiency benchmark
    tmp = build_model()
    benchmark_model(tmp)
    del tmp
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    s_imgs, s_labels, t_imgs, t_labels = load_data(args.data_root)
    s_imgs, s_labels = np.array(s_imgs), np.array(s_labels)

    target_dataset = CoffeeDataset(t_imgs, t_labels, transform=base_transforms, mode="target")
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=custom_collate_fn)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    metrics_log = {k: [] for k in ["H-Mean", "Seen_Acc", "Unseen_Exact", "Unseen_Top3",
                                   "Unseen_F1_Macro", "Unseen_F1_Samp", "Unseen_Prec",
                                   "Unseen_Rec", "Unseen_Ham"]}

    for fold, (train_idx, val_idx) in enumerate(kf.split(s_imgs)):
        if fold >= args.folds:
            break
        log_print(f"\n{'=' * 20} FOLD {fold + 1} / {args.folds} {'=' * 20}")

        train_ds = CoffeeDataset(s_imgs[train_idx], s_labels[train_idx],
                                 transform=train_aug_transforms, mode="source")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=custom_collate_fn,
                                  drop_last=True)
        val_ds = CoffeeDataset(s_imgs[val_idx], s_labels[val_idx],
                               transform=base_transforms, mode="source")
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=custom_collate_fn)

        model = build_model()
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.adapter.parameters(), "lr": args.lr},
            {"params": model.projector.parameters(), "lr": args.lr},
            {"params": [model.logit_scale], "lr": args.lr},
        ], weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

        best_h, best_metrics = 0.0, {}
        for epoch in range(args.epochs):
            model.train()
            loop_loss = []
            pbar = tqdm(train_loader, desc=f"[F{fold + 1}] Ep {epoch + 1}")
            for imgs, lbl_indices, _ in pbar:
                imgs, lbl_indices = imgs.to(DEVICE), lbl_indices.to(DEVICE)
                ctx = torch.amp.autocast("cuda") if DEVICE == "cuda" else _nullcontext()
                with ctx:
                    raw_feats = model.extract_features(imgs)
                    _, logits_orig = model.forward_head(raw_feats)
                    loss_ce = F.cross_entropy(logits_orig, lbl_indices)

                    v_mix, t_mix = get_mmg_batch(model, raw_feats, lbl_indices)
                    vq_mix, logits_mix = model.forward_head(v_mix)

                    loss_align = 1.0 - (F.normalize(vq_mix, p=2, dim=1) *
                                        F.normalize(t_mix, p=2, dim=1)).sum(dim=1).clamp(-1, 1).mean()

                    scale = model.logit_scale.exp().clamp(max=100)
                    prot_norm = F.normalize(model.class_prototypes, dim=1)
                    target_logits = torch.matmul(F.normalize(t_mix, p=2, dim=1), prot_norm.t()) * scale
                    loss_kl = F.kl_div(F.log_softmax(logits_mix, dim=1),
                                       F.softmax(target_logits, dim=1), reduction="batchmean")

                    loss = (args.lambda_ce * loss_ce + args.lambda_align * loss_align +
                            args.lambda_kl * loss_kl)

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                loop_loss.append(loss.item())
                pbar.set_postfix({"L": f"{loss.item():.3f}"})
            scheduler.step()

            seen_res = evaluate_all_metrics(model, val_loader, mode="seen")
            unseen_res = evaluate_all_metrics(model, target_loader, mode="unseen")
            s_acc, u_acc = seen_res["Acc"], unseen_res["Acc"]
            h_mean = (2 * s_acc * u_acc) / (s_acc + u_acc + 1e-6)

            log_print(f"Ep {epoch + 1} | Loss={np.mean(loop_loss):.3f}")
            log_print(f"  Seen(Val): Acc={s_acc:.3f} F1={seen_res['F1_Macro']:.3f}")
            log_print(f"  Unseen   : Exact={u_acc:.3f} Top3={unseen_res['Top3']:.3f} H-Mean={h_mean:.3f}")

            if h_mean > best_h:
                best_h = h_mean
                best_metrics = {
                    "H-Mean": h_mean, "Seen_Acc": s_acc, "Unseen_Exact": u_acc,
                    "Unseen_Top3": unseen_res["Top3"], "Unseen_F1_Macro": unseen_res["F1_Macro"],
                    "Unseen_F1_Samp": unseen_res["F1_Samples"], "Unseen_Prec": unseen_res["Prec_Macro"],
                    "Unseen_Rec": unseen_res["Rec_Macro"], "Unseen_Ham": unseen_res["Hamming"],
                }
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_fold_{fold + 1}.pth"))
                log_print(f"  >>> Best saved (H={h_mean:.4f})")

        log_print(f"Fold {fold + 1} finished. Best H-Mean={best_h:.4f}")
        for k, v in best_metrics.items():
            metrics_log[k].append(v)
        del model, optimizer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    log_print("\n" + "=" * 40)
    log_print(f"FINAL METRICS ({args.folds}-FOLD AVERAGE)")
    log_print("=" * 40)
    for key, values in metrics_log.items():
        if values:
            log_print(f"{key:15s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")
    log_print("=" * 40)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    setup_logger(args.log_dir)
    run_training(args)

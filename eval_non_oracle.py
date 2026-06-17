"""Practical (non-oracle) inference analysis for KG-MixNet (paper Table 6).

The main tables use the oracle-cardinality protocol (the number of true
components is known). This script removes that assumption and evaluates a
deployment-oriented adaptive decision rule whose thresholds are calibrated on
MMG-generated virtual compound samples and then grid-searched. It reports the
unseen sample-wise F1, the seen strict accuracy, and the seen false-positive
rate trade-off. These numbers must NOT be mixed with the oracle-cardinality
tables; they are a supplemental deployment-oriented study.
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from models.network import KG_MixNet
from utils.custom_dataset import CoffeeDataset
from scripts.data_loader import load_data, base_transforms
from utils.taxonomy import CLASS_TO_IDX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Non-oracle adaptive-threshold evaluation")
    p.add_argument("--data_root", default=os.environ.get("COLEAF_DATA_ROOT", "CoLeaf_raw_images"))
    p.add_argument("--dino_path", default=os.environ.get("DINOV3_PATH", "models/dinov3-vitl16-pretrain-lvd1689m"))
    p.add_argument("--embed_path", default="scripts/class_embeddings.pt")
    p.add_argument("--checkpoint", required=True, help="Path to a trained KG-MixNet .pth checkpoint.")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    raw_labels = [item[2] for item in batch]
    return images, labels, raw_labels


def load_checkpoint(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_final_features(model, images):
    raw_feats = model.extract_features(images)
    final_feats, _ = model.forward_head(raw_feats)
    return final_feats


def calibrate_threshold_mmg(model, seen_loader, device, num_virtual_samples=3000):
    """Generate virtual compound samples via MMG and find the best low threshold."""
    model.eval()
    print("\n[*] Step 1: Calibrating tau_low via MMG-generated virtual compounds...")
    prototypes = F.normalize(model.class_prototypes.to(device), p=2, dim=1)

    feats_bank, labels_bank = [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(seen_loader, desc="Extracting seen"):
            images = images.to(device)
            feats_bank.append(get_final_features(model, images).cpu())
            labels_bank.append(torch.tensor(labels))
    feats_bank = torch.cat(feats_bank, dim=0)
    labels_bank = torch.cat(labels_bank, dim=0)

    num_classes = prototypes.size(0)
    virtual_feats, virtual_targets = [], []
    for _ in range(num_virtual_samples):
        idx1, idx2 = np.random.choice(len(feats_bank), 2, replace=False)
        lam = np.random.uniform(0.2, 0.8)
        v_mix = F.normalize(lam * feats_bank[idx1] + (1 - lam) * feats_bank[idx2], p=2, dim=0)
        target = torch.zeros(num_classes)
        target[labels_bank[idx1].item()] = 1
        target[labels_bank[idx2].item()] = 1
        virtual_feats.append(v_mix)
        virtual_targets.append(target)

    virtual_feats = torch.stack(virtual_feats).to(device)
    virtual_targets = torch.stack(virtual_targets).cpu().numpy()
    scores = torch.matmul(virtual_feats, prototypes.t()).cpu().numpy()

    best_tau, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.60, 0.01):
        preds_t = (scores > t).astype(int)
        f1 = f1_score(virtual_targets, preds_t, average="samples", zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, t
    print(f"    > Optimal tau_low = {best_tau:.3f} (virtual F1 = {best_f1:.4f})")
    return best_tau


def run_inference_fast(sims, tau_low, energy_thr, gap_thr, tau_sec):
    """Adaptive decision rule over similarity scores."""
    batch_preds = []
    for i in range(len(sims)):
        scores = sims[i]
        sorted_indices = np.argsort(scores)[::-1]
        top1_idx, top2_idx = sorted_indices[0], sorted_indices[1]
        max_score = scores[top1_idx]
        gap = max_score - scores[top2_idx]
        pred_row = np.zeros(9)

        if max_score < energy_thr:                 # weak signal -> recall-first
            pred_row = (scores > tau_low).astype(int)
            if pred_row.sum() == 0:
                pred_row[top1_idx] = 1
        else:                                       # strong signal
            if gap < gap_thr:                       # ambiguous -> single label
                pred_row[top1_idx] = 1
            else:                                   # dominant + optional secondary
                pred_row[top1_idx] = 1
                if scores[top2_idx] > tau_sec:
                    pred_row[top2_idx] = 1
        batch_preds.append(pred_row)
    return np.vstack(batch_preds)


def _precompute_scores(model, loader, prototypes, mode):
    sims_all, gt_all = [], []
    with torch.no_grad():
        for images, labels, raw_labels in tqdm(loader, desc=f"{mode} scores"):
            images = images.to(DEVICE)
            feats = get_final_features(model, images)
            sims_all.append(torch.matmul(feats, prototypes.t()).cpu().numpy())
            batch_gt = np.zeros((len(images), 9))
            if mode == "seen":
                for k, l in enumerate(labels):
                    val = l.item() if isinstance(l, torch.Tensor) else l
                    batch_gt[k, val] = 1
            else:
                for k, raw_l in enumerate(raw_labels):
                    current = [raw_l] if isinstance(raw_l, str) else raw_l
                    for c in current:
                        if c in CLASS_TO_IDX:
                            batch_gt[k, CLASS_TO_IDX[c]] = 1
            gt_all.append(batch_gt)
    return np.vstack(sims_all), np.vstack(gt_all)


def main():
    args = parse_args()
    print("\n>>> Loading data and model...")
    s_imgs, s_labels, t_imgs, t_labels = load_data(args.data_root)

    seen_ds = CoffeeDataset(s_imgs, s_labels, transform=base_transforms, mode="source")
    unseen_ds = CoffeeDataset(t_imgs, t_labels, transform=base_transforms, mode="target")
    seen_loader = DataLoader(seen_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=custom_collate_fn)
    unseen_loader = DataLoader(unseen_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=0, collate_fn=custom_collate_fn)

    model = KG_MixNet(args.dino_path, args.embed_path, device=DEVICE).to(DEVICE)
    model = load_checkpoint(model, args.checkpoint)
    prototypes = F.normalize(model.class_prototypes.to(DEVICE), p=2, dim=1)

    tau_low = calibrate_threshold_mmg(model, seen_loader, DEVICE)

    print("\n>>> Pre-computing scores...")
    seen_sims, seen_gt = _precompute_scores(model, seen_loader, prototypes, "seen")
    unseen_sims, unseen_gt = _precompute_scores(model, unseen_loader, prototypes, "unseen")

    energy_range = np.arange(0.20, 0.32, 0.02)
    gap_range = np.arange(0.03, 0.13, 0.01)
    sec_range = np.arange(0.20, 0.36, 0.02)
    print(f"\n>>> Grid search ({len(energy_range) * len(gap_range) * len(sec_range)} combos)...")

    results = []
    for e in energy_range:
        for g in gap_range:
            for s in sec_range:
                unseen_preds = run_inference_fast(unseen_sims, tau_low, e, g, s)
                f1 = f1_score(unseen_gt, unseen_preds, average="samples", zero_division=0)
                if f1 > 0.59:
                    seen_preds = run_inference_fast(seen_sims, tau_low, e, g, s)
                    fpr = (seen_preds.sum(axis=1) > 1).sum() / len(seen_sims)
                    strict_acc = np.all(seen_preds == seen_gt, axis=1).mean()
                    results.append({"e": e, "g": g, "s": s, "f1": f1,
                                    "strict": strict_acc, "fpr": fpr})

    gold = [r for r in results if r["f1"] >= 0.60 and r["fpr"] <= 0.08]
    silver = [r for r in results if r["f1"] >= 0.59 and r["fpr"] <= 0.10]
    if gold:
        gold.sort(key=lambda x: x["strict"], reverse=True)
        final_choice, shortlist, tag = gold[0], gold[:5], "GOLD"
    elif silver:
        silver.sort(key=lambda x: x["f1"], reverse=True)
        final_choice, shortlist, tag = silver[0], silver[:5], "SILVER"
    elif results:
        results.sort(key=lambda x: x["f1"] + x["strict"] - x["fpr"], reverse=True)
        final_choice, shortlist, tag = results[0], results[:5], "BEST-BALANCED"
    else:
        print("No configuration met the criteria.")
        return

    print(f"\n>>> {tag} configurations:")
    print(f"{'Energy':<8}{'Gap':<8}{'Sec':<8}{'Strict':<10}{'FPR':<10}{'UnseenF1':<10}")
    for r in shortlist:
        mark = "  <-- chosen" if r is final_choice else ""
        print(f"{r['e']:<8.2f}{r['g']:<8.2f}{r['s']:<8.2f}{r['strict']:<10.4f}{r['fpr']:<10.4f}{r['f1']:<10.4f}{mark}")

    print("\n" + "=" * 50)
    print("FINAL RECOMMENDED (non-oracle) CONFIGURATION")
    print(f"  Energy / Gap / Sec     : {final_choice['e']:.2f} / {final_choice['g']:.2f} / {final_choice['s']:.2f}")
    print(f"  Unseen sample-wise F1  : {final_choice['f1']:.4f}")
    print(f"  Seen strict accuracy   : {final_choice['strict']:.4f}")
    print(f"  Seen false-positive rate: {final_choice['fpr']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

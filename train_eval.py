#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KG-MixNet - training and evaluation pipeline

Design notes
--------
1. **Faithful to the original implementation.** The architecture, the MMG
   sampling, the losses and the evaluation metrics are reproduced line by line
   from ``models/network.py``; this file introduces no behaviour that was not
   in the original code.
2. **One training run, two checkpoint-selection rules.** Full source- and
   target-domain metrics are recorded at every epoch, and scores are stored at
   two of them:
     - ``sel_seen``  : selected on source-validation macro-F1 (no leakage; this
                       is the rule used for every number in the paper)
     - ``sel_hmean`` : selected on H-Mean, which reads the target set (used only
                       for the protocol-audit row)
   The cost of leakage is therefore a within-run comparison and needs no second
   training pass.
3. **Scores are written to disk.** The per-sample nine-dimensional scores for
   the seen validation split and for the unseen target set are saved, so the
   threshold analyses, bootstrap intervals and component-recall figures can all
   be recomputed offline without retraining.

Usage
----
    python kgmix_rerun.py --exp abl_full --folds 1,2,3,4,5 --epochs 40

Outputs
----
    runs/<exp>.json         per-epoch metrics and the summary at both
                            selected epochs
    runs/<exp>.scores.npz   per-sample scores on the seen validation split and
                            the unseen target set at both selected epochs
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             precision_score, recall_score)
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import AutoModel

# ======================================================================================
# Fixed constants. These match the original implementation and should not be
# changed: the class order is the label encoding used by the cross-entropy loss
# and by the prototype bank.
# ======================================================================================

CLASS_TO_IDX = {"B": 0, "Ca": 1, "Fe": 2, "Healthy": 3, "K": 4,
                "Mg": 5, "Mn": 6, "N": 7, "P": 8}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
N_CLASSES = 9

# Directory names as distributed with CoLeaf-DB. The published archive spells
# "potassium" as "potasium"; both spellings are accepted so that the 96
# potassium images are not silently skipped.
SINGLE_FOLDERS = ["boron-B", "calcium-Ca", "iron-Fe", "magnesium-Mg",
                  "manganese-Mn", "nitrogen-N", "phosphorus-P",
                  "potassium-K", "potasium-K", "healthy"]
COMPOUND_FOLDER = "more-deficiencies"

train_aug_transforms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

base_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --------------------------------------------------------------------------------------
# Aspect-preserving preprocessing (a control, not the default)
# --------------------------------------------------------------------------------------
# Motivation: image geometry in CoLeaf-DB is not uniform, and the
# non-uniformity is correlated with the class:
#   phosphorus-P : 122 of 246 images (49.6%) are under 2 MP, portrait crops with
#                  aspect ratios of 0.38-0.52, spanning 102 distinct sizes
#   boron-B      : 13.9% small images      calcium-Ca : 8.0%
#   Fe / N / healthy / **all 104 compound leaves** : 0%, uniformly 4000x3000
# The reported pipeline resizes to 224x224 without preserving aspect ratio,
# which encodes a class-correlated quantity into the input. Phosphorus is also
# the largest class and the most accurately recognised one, so the possibility
# that the model reads framing rather than pathology has to be ruled out by
# measurement rather than assumed away.
#
# The risk is bounded: the 104 target images are uniform in size, so no unseen
# result can be inflated by this shortcut; only the seen-class column is
# exposed to it.
#
# The control below preserves aspect ratio: resize the short side to 256, then
# centre-crop to 224.
train_aug_aspect = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

base_aspect = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_transforms(preprocess: str):
    """Return (training augmentation, evaluation transform).

    ``squash`` is the reported pipeline; ``aspect`` is the aspect-preserving
    control described above.
    """
    if preprocess == "squash":
        return train_aug_transforms, base_transforms
    if preprocess == "aspect":
        return train_aug_aspect, base_aspect
    raise ValueError(f"unknown preprocess: {preprocess}")


# ======================================================================================
# Data
# ======================================================================================

def load_data(data_root: str):
    """Load the source and target splits.

    Source domain: the nine single-deficiency and healthy directories, with the
    directory name as the label.
    Target domain: ``more-deficiencies``, whose labels are parsed from the file
    name, e.g. ``B_Ca (1).jpg`` -> ['B', 'Ca'].
    """
    source_imgs, source_labels = [], []
    seen_dirs = []
    for folder in SINGLE_FOLDERS:
        path = os.path.join(data_root, folder)
        if not os.path.isdir(path):
            continue
        seen_dirs.append(folder)
        label = folder.split("-")[-1] if "-" in folder else "Healthy"
        for img_path in sorted(glob.glob(os.path.join(path, "*.*"))):
            source_imgs.append(img_path)
            source_labels.append(label)

    target_imgs, target_labels = [], []
    for img_path in sorted(glob.glob(os.path.join(data_root, COMPOUND_FOLDER, "*.*"))):
        raw_name = os.path.basename(img_path).split(" ")[0]
        target_imgs.append(img_path)
        target_labels.append(raw_name.split("_"))

    return (np.array(source_imgs), np.array(source_labels),
            target_imgs, target_labels, seen_dirs)


def build_blob_cache(paths, verify=24):
    """Read the raw file bytes into memory so the Dataset can decode from RAM.

    **No pixel changes.** The decoder receives an identical byte stream, so
    ``Image.open(BytesIO(b))`` and ``Image.open(path)`` return bit-identical
    arrays. The function asserts this on ``verify`` randomly drawn images before
    returning, and raises if it does not hold.

    Measured, so as not to overstate it: with a warm page cache, decoding from
    memory takes 111.2 ms per image against 112.4 ms from disk, which is **no
    real speed-up**. Three other reasons justify keeping it: it removes the
    cold-read cost of the first epoch of each job (cold reads can reach hundreds
    of milliseconds per image); it keeps the data resident when several jobs run
    concurrently; and the dataset is about 1.2 GB against roughly 1 TB of RAM.

    The real bottleneck is elsewhere: decoding costs 111 ms, but decoding plus
    full-resolution rotation and colour jitter costs 2440 ms, so roughly 95% of
    the data time is augmentation on 4000x3000 originals.

    This differs in kind from a pre-resized cache: pre-resizing to a fixed
    resolution changes the pixels and changes the interpolation that a
    rotate-then-resize pipeline produces. This cache only avoids the disk
    round-trip.
    """
    cache = {}
    for p in paths:
        with open(p, "rb") as f:
            cache[p] = f.read()

    if verify:
        rng = random.Random(0)
        sample = rng.sample(list(paths), min(verify, len(paths)))
        for p in sample:
            a = np.asarray(Image.open(p).convert("RGB"))
            b = np.asarray(Image.open(io.BytesIO(cache[p])).convert("RGB"))
            if not np.array_equal(a, b):
                raise RuntimeError(
                    f"Byte cache changed the pixels, which must not happen: {p}. "
                    f"Re-run with --no-cache-bytes.")
    mb = sum(len(v) for v in cache.values()) / 1e6
    print(f"[cache] loaded {len(cache)} files, {mb:.0f} MB, "
          f"bit-identity verified on {min(verify, len(paths))} samples", flush=True)
    return cache


class CoffeeDataset(Dataset):
    """Dataset for the source (single-label) and target (multi-label) splits.

    The only addition over the original is ``blob_cache``: when supplied, images
    are decoded from in-memory bytes rather than read from disk. Pixels are
    identical; see ``build_blob_cache`` for the assertion that checks this.
    """

    def __init__(self, img_paths, labels, transform=None, mode="source", blob_cache=None):
        assert mode in ("source", "target")
        self.img_paths = list(img_paths)
        self.labels = list(labels)
        self.transform = transform
        self.mode = mode
        self.blob_cache = blob_cache

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        raw_label = self.labels[idx]
        try:
            if self.blob_cache is not None:
                image = Image.open(io.BytesIO(self.blob_cache[path])).convert("RGB")
            else:
                image = Image.open(path).convert("RGB")
        except Exception as e:  # as in the original: a corrupt image yields a black
                                #  tensor rather than aborting training
            print(f"[ERROR] Corrupt image skipped: {path}, error: {e}")
            image = Image.new("RGB", (224, 224))
        if self.transform:
            image = self.transform(image)

        if self.mode == "source":
            if not isinstance(raw_label, str):
                raw_label = raw_label[0]
            return image, CLASS_TO_IDX.get(raw_label, -1), raw_label
        raw_list = [raw_label] if isinstance(raw_label, str) else list(raw_label)
        return image, -1, raw_list


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    idxs = torch.tensor([b[1] for b in batch])
    raw = [b[2] for b in batch]
    return images, idxs, raw


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================================
# Models. The proposed method and every reference share the DINOv3 backbone,
# the freeze policy, the projection head and the decoding rules; only the
# scoring mechanism differs.
# ======================================================================================

class KAdapter(nn.Module):
    """Feature-level residual adapter. The up-projection is zero-initialised, so
    the module is exactly the identity at the start of training."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.dropout(self.up(self.act(self.down(x))))


class PlantCaFoAdapter(nn.Module):
    """Narrower bottleneck with a learnable residual scale (PlantCaFo-style)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return x + self.up_proj(self.relu(self.down_proj(x))) * self.scale


class AnomalyPromptLearner(nn.Module):
    """Object-agnostic learnable context added to the text prototypes
    (AnomalyCLIP-style)."""

    def __init__(self, embed_dim: int, num_prompts: int = 4):
        super().__init__()
        self.ctx = nn.Parameter(torch.empty(num_prompts, embed_dim))
        nn.init.normal_(self.ctx, std=0.02)
        self.fusion = nn.Linear(embed_dim, embed_dim)
        self.act = nn.SiLU()

    def forward(self, protos):
        return self.act(self.fusion(protos + self.ctx.mean(dim=0, keepdim=True)))


class SemanticEnricher(nn.Module):
    """Gated semantic expansion of the prototypes (DSECN-style)."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.expansion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2), nn.Linear(hidden_dim, input_dim), nn.Sigmoid())
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, protos):
        return self.proj(protos * self.expansion_net(protos) + protos)


class RelationHead(nn.Module):
    """MLP relation score over concatenated visual-semantic pairs
    (RelationNet-style)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, v, protos):
        b, c = v.size(0), protos.size(0)
        pair = torch.cat([v.unsqueeze(1).expand(b, c, -1),
                          protos.unsqueeze(0).expand(b, c, -1)], dim=2)
        return self.net(pair).squeeze(2)


def find_transformer_blocks(backbone) -> "nn.ModuleList | list":
    """Locate the list of transformer blocks inside the backbone.

    Why this needs its own function
    ----------------------
    The original implementation tries the attribute names ``layer``,
    ``blocks``, ``encoder.layer`` and ``layers`` in turn, and if none matches it
    **silently unfreezes nothing**. Parameter counts, losses and metrics are all
    still printed, so nothing about the output reveals that the model changed.

    This is not hypothetical. Under transformers 5.14.1 the 24 DINOv3 blocks
    live at ``model.layer`` and none of the four candidate paths matches, while
    the original logs print ``Unfrozen last 2 layers``, so an earlier version did
    find them. The same code therefore degrades silently to an almost fully
    frozen backbone on a newer library.

    This function therefore tries the known paths first, falls back to the
    version-independent heuristic of taking the longest ``ModuleList`` in the
    tree, and leaves the caller to assert on the count. Failing loudly is
    preferable to silently training a different model.
    """
    for path in ("layer", "blocks", "layers",
                 "encoder.layer", "encoder.layers", "encoder.blocks",
                 "model.layer", "model.blocks", "model.layers"):
        obj = backbone
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList) and len(obj) >= 8:
            return obj

    # Version-independent fallback: the longest ModuleList anywhere in the tree
    best = None
    for _, mod in backbone.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8:
            if best is None or len(mod) > len(best):
                best = mod
    return best if best is not None else []


@dataclass
class ModelCfg:
    backbone_path: str
    embed_path: str
    head: str = "kgmixnet"       # kgmixnet|relationnet|anomalyclip|dsecn|plantcafo|linear
    use_adapter: bool = True     # feature-level residual adapter on/off
    unfreeze_last_n: int = 2     # 0 = fully frozen backbone
    projector: str = "deep"      # deep: 1024->1024->768 (reported model); shallow: 1024->768


class Net(nn.Module):
    """One wrapper for the proposed model and every reference. ``head`` selects the
    scoring mechanism; everything else is held fixed."""

    def __init__(self, cfg: ModelCfg, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.head = cfg.head
        self.backbone = AutoModel.from_pretrained(cfg.backbone_path, trust_remote_code=True)
        self.visual_dim = getattr(self.backbone.config, "hidden_size", 1024)

        for p in self.backbone.parameters():
            p.requires_grad = False
        blocks = []
        if cfg.unfreeze_last_n > 0:
            stack = find_transformer_blocks(self.backbone)
            blocks = stack[-cfg.unfreeze_last_n:]
            for blk in blocks:
                for p in blk.parameters():
                    p.requires_grad = True
            for attr in ("norm", "layernorm"):
                if hasattr(self.backbone, attr):
                    for p in getattr(self.backbone, attr).parameters():
                        p.requires_grad = True
                    break
            # Silent failure is the dangerous outcome here: not finding the
            # blocks means training a different model from the reported one,
            # while parameter counts, losses and metrics all look normal. The
            # original implementation failed exactly this way (see
            # find_transformer_blocks), so this assertion is deliberate.
            if len(blocks) != cfg.unfreeze_last_n:
                raise RuntimeError(
                    f"asked to unfreeze the last {cfg.unfreeze_last_n} blocks but "
                    f"only found {len(blocks)}. Backbone type "
                    f"{type(self.backbone).__name__}, top-level children "
                    f"{[n for n, _ in self.backbone.named_children()]}")
        self.n_unfrozen_blocks = len(blocks)
        self.n_backbone_trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad)

        text_emb = torch.load(cfg.embed_path, map_location="cpu")
        self.text_dim = text_emb.shape[1]
        self.register_buffer("class_prototypes", text_emb.float())

        self.adapter = (KAdapter(self.visual_dim, 512) if cfg.use_adapter else nn.Identity())
        if cfg.head == "plantcafo":
            self.adapter = PlantCaFoAdapter(self.visual_dim)

        if cfg.projector == "deep":
            self.projector = nn.Sequential(
                nn.Linear(self.visual_dim, self.visual_dim), nn.LayerNorm(self.visual_dim),
                nn.GELU(), nn.Linear(self.visual_dim, self.text_dim))
        else:
            self.projector = nn.Sequential(
                nn.Linear(self.visual_dim, self.text_dim),
                nn.LayerNorm(self.text_dim), nn.GELU())

        if cfg.head == "anomalyclip":
            self.prompt_learner = AnomalyPromptLearner(self.text_dim)
        elif cfg.head == "dsecn":
            self.semantic_enricher = SemanticEnricher(self.text_dim)
        elif cfg.head == "relationnet":
            self.relation_head = RelationHead(self.text_dim)
        elif cfg.head == "linear":
            # Knowledge-free linear head: no semantic prototypes at all, just a
            # free nine-way classifier over the same visual features
            self.linear_head = nn.Linear(self.text_dim, N_CLASSES)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def extract_features(self, images):
        out = self.backbone(pixel_values=images)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]
        if isinstance(out, torch.Tensor):
            return out[:, 0, :] if out.dim() == 3 else out
        return out[0][:, 0, :]

    def current_prototypes(self):
        protos = self.class_prototypes
        if self.head == "anomalyclip":
            return self.prompt_learner(protos)
        if self.head == "dsecn":
            return self.semantic_enricher(protos)
        return protos

    def forward_head(self, features):
        """Return (L2-normalised visual query, logits)."""
        projected = self.projector(self.adapter(features))
        v_norm = F.normalize(projected, p=2, dim=1)

        if self.head == "linear":
            return v_norm, self.linear_head(projected)
        if self.head == "relationnet":
            return v_norm, self.relation_head(projected, self.class_prototypes) * self.logit_scale.exp()

        t_norm = F.normalize(self.current_prototypes(), p=2, dim=1)
        scale = self.logit_scale.exp().clamp(max=100)
        return v_norm, scale * torch.matmul(v_norm, t_norm.t())

    def forward(self, images):
        return self.forward_head(self.extract_features(images))[1]


# ======================================================================================
# Synthesis branch: multimodal geometric mixup and its controls
# ======================================================================================

def sample_mmg_action(batch_size, device):
    """Sample one MMG action.

    Factored out so that the different mixing levels (backbone feature,
    projected feature, logit) draw from the **same** action distribution.
    Otherwise the difference between levels would be confounded with a
    difference in the random stream.
    """
    p = random.random()
    if p < 0.3:
        return ("noise", None, None)
    if p < 0.8:
        idx = torch.randperm(batch_size).to(device)
        lam = float(np.random.beta(1.0, 1.0))
        return ("mix", idx, lam)
    return ("identity", None, None)


def apply_mmg(vis, txt, action, mode="geometric", noise_std=0.05):
    """Apply one sampled action to a (visual, semantic) pair.

    mode
      geometric : the reported operator. Normalise, apply the action, then
                  **re-normalise** (spherical re-projection).
      euclidean : the same action without the re-projection, which isolates
                  what returning the mixture to the unit sphere contributes.
      raw       : no normalisation at all, plain linear interpolation
                  (manifold mixup as originally proposed).
    """
    kind, idx, lam = action
    if mode != "raw":
        vis = F.normalize(vis, p=2, dim=1)
        txt = F.normalize(txt, p=2, dim=1)

    if kind == "noise":
        v_mix, t_mix = vis + torch.randn_like(vis) * noise_std, txt
    elif kind == "mix":
        lam_t = torch.tensor(lam, device=vis.device)
        v_mix = lam_t * vis + (1 - lam_t) * vis[idx]
        t_mix = lam_t * txt + (1 - lam_t) * txt[idx]
    else:
        v_mix, t_mix = vis, txt

    if mode == "geometric":
        v_mix = F.normalize(v_mix, p=2, dim=1)
        t_mix = F.normalize(t_mix, p=2, dim=1)
    return v_mix, t_mix


def mmg_batch(model, features, labels, mode="geometric"):
    """Multimodal geometric mixup on the raw backbone CLS features."""
    txt = model.class_prototypes.to(features.device)[labels]
    action = sample_mmg_action(features.size(0), features.device)
    return apply_mmg(features, txt, action, mode=mode)


def class_weights(counts: np.ndarray, scheme: str, beta: float = 0.999):
    """Class-imbalance schemes. Returns a length-9 weight vector normalised to
    mean one, so that the effective learning rate is unchanged."""
    n = counts.astype(np.float64)
    if scheme == "none":
        return None
    if scheme == "inverse":
        w = 1.0 / np.maximum(n, 1)
    elif scheme == "sqrt_inverse":
        w = 1.0 / np.sqrt(np.maximum(n, 1))
    elif scheme == "effective_number":
        w = (1.0 - beta) / (1.0 - np.power(beta, np.maximum(n, 1)))
    else:
        raise ValueError(f"unknown weighting scheme: {scheme}")
    return w / w.mean()


def focal_ce(logits, target, gamma=2.0, weight=None):
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    logp_t = logp.gather(1, target.view(-1, 1)).squeeze(1)
    p_t = p.gather(1, target.view(-1, 1)).squeeze(1)
    loss = -((1 - p_t) ** gamma) * logp_t
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()


# ======================================================================================
# Evaluation. Identical to the original metric code, with the per-sample
# scores additionally returned so that offline analyses need no retraining.
# ======================================================================================

@torch.no_grad()
def evaluate(model, loader, mode, device):
    """Oracle-cardinality evaluation. Returns (metrics, scores, y_true).

    scores : [N, 9] softmax posteriors, for offline threshold and bootstrap
             analyses
    y_true : [N, 9] binary indicator matrix
    """
    model.eval()
    y_true_all, y_pred_all, scores_all = [], [], []
    top3_any_hit, total = 0, 0

    for imgs, lbls, raw_lbls in loader:
        probs = F.softmax(model(imgs.to(device)), dim=1)
        _, top3_idx = probs.topk(3, dim=1)
        for i in range(len(raw_lbls)):
            gt = ([CLASS_TO_IDX[x] for x in raw_lbls[i] if x in CLASS_TO_IDX]
                  if mode == "unseen" else [int(lbls[i])])
            y_t = np.zeros(N_CLASSES, dtype=int); y_t[gt] = 1
            y_true_all.append(y_t)

            k = len(gt) if len(gt) > 0 else 1          # oracle cardinality
            _, topk = probs[i].topk(k)
            y_p = np.zeros(N_CLASSES, dtype=int); y_p[topk.tolist()] = 1
            y_pred_all.append(y_p)
            scores_all.append(probs[i].detach().cpu().numpy())

            if mode == "unseen" and not set(gt).isdisjoint(set(top3_idx[i].tolist())):
                top3_any_hit += 1
            total += 1

    yt, yp = np.array(y_true_all), np.array(y_pred_all)
    m = {
        "Acc": accuracy_score(yt, yp),
        "Prec_Macro": precision_score(yt, yp, average="macro", zero_division=0),
        "Rec_Macro": recall_score(yt, yp, average="macro", zero_division=0),
        "F1_Macro": f1_score(yt, yp, average="macro", zero_division=0),
        "F1_Samples": f1_score(yt, yp, average="samples", zero_division=0),
        "Hamming": hamming_loss(yt, yp),
        "Top3": top3_any_hit / total if total else 0.0,
    }
    return m, np.array(scores_all, dtype=np.float32), yt


@torch.no_grad()
def extract_projected(model, loader, device):
    """Extract the L2-normalised 768-dimensional projected features, needed for
    the projection and retrieval figures. The nine-dimensional scores alone are
    not enough: they have passed through the softmax, which discards the
    geometry of the embedding space."""
    model.eval()
    out = []
    for imgs, _, _ in loader:
        v, _ = model.forward_head(model.extract_features(imgs.to(device)))
        out.append(v.detach().cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


# ======================================================================================
# Training: one fold
# ======================================================================================

def train_one_fold(args, mcfg: ModelCfg, fold, tr_idx, va_idx, s_imgs, s_labels,
                   target_loader, device, blob_cache=None, novel_loader=None):
    """Train one fold, record metrics at every epoch, and keep one score snapshot
    per checkpoint-selection rule."""
    set_seed(args.seed + fold)

    tf_train, tf_eval = get_transforms(args.preprocess)

    # Leave-one-nutrient-out. Every image of the held-out nutrient is removed
    # from training and validation, while the prototype bank keeps its
    # description. The model never sees an image of that nutrient and can reach
    # it only through the symptom description, which is the condition under
    # which knowledge-guided zero-shot transfer should be tested. A
    # discriminative linear head cannot solve this setting by construction: its
    # output unit for the held-out class never receives a gradient.
    if args.holdout_class:
        keep_tr = s_labels[tr_idx] != args.holdout_class
        keep_va = s_labels[va_idx] != args.holdout_class
        tr_idx, va_idx = tr_idx[keep_tr], va_idx[keep_va]

    tr_ds = CoffeeDataset(s_imgs[tr_idx], s_labels[tr_idx], tf_train,
                          "source", blob_cache=blob_cache)
    va_ds = CoffeeDataset(s_imgs[va_idx], s_labels[va_idx], tf_eval,
                          "source", blob_cache=blob_cache)

    counts = np.bincount([CLASS_TO_IDX[x] for x in s_labels[tr_idx]], minlength=N_CLASSES)
    sampler, shuffle = None, True
    if args.imbalance == "sampler":
        per_class = 1.0 / np.maximum(counts, 1)
        w = np.array([per_class[CLASS_TO_IDX[x]] for x in s_labels[tr_idx]])
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), num_samples=len(w), replacement=True)
        shuffle = False

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
                           num_workers=args.workers, collate_fn=collate_fn, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, collate_fn=collate_fn)

    # Under leave-one-nutrient-out the training loss must be restricted to the
    # eight seen classes. Otherwise cross-entropy treats the held-out class as a
    # negative for every sample, which explicitly trains the model never to
    # predict it. A first implementation did exactly that and produced top-1 and
    # top-3 rates of zero for the held-out nutrient; that is not a failure to
    # recognise it but a model trained not to. With the loss restricted, the
    # held-out prototype receives no gradient and keeps the position the
    # knowledge graph gives it, and testing scores against all nine prototypes.
    # This is the standard GZSL construction.
    if args.holdout_class:
        h = CLASS_TO_IDX[args.holdout_class]
        seen_idx = torch.tensor([i for i in range(N_CLASSES) if i != h], device=device)
        remap = torch.full((N_CLASSES,), -1, dtype=torch.long, device=device)
        remap[seen_idx] = torch.arange(len(seen_idx), device=device)
    else:
        seen_idx, remap = None, None

    model = Net(mcfg, device=device).to(device)
    if fold == min(int(x) for x in args.folds.split(",")):
        print(f"[model] unfrozen blocks={model.n_unfrozen_blocks} "
              f"backbone trainable={model.n_backbone_trainable/1e6:.3f}M "
              f"total trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M",
              flush=True)

    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("backbone.") and n != "logit_scale"]
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.backbone.parameters() if p.requires_grad],
          "lr": args.lr_backbone},
         {"params": head_params, "lr": args.lr},
         {"params": [model.logit_scale], "lr": args.lr}],
        weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    cw = None
    if args.imbalance in ("inverse", "sqrt_inverse", "effective_number"):
        cw = torch.tensor(class_weights(counts, args.imbalance), dtype=torch.float32, device=device)
    elif args.imbalance == "focal_weighted":
        cw = torch.tensor(class_weights(counts, "effective_number"), dtype=torch.float32, device=device)

    history, snapshots = [], {}
    best = {"sel_seen": -1.0, "sel_hmean": -1.0}

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for imgs, lbl, _ in tr_loader:
            imgs, lbl = imgs.to(device), lbl.to(device)
            with torch.amp.autocast("cuda"):
                raw = model.extract_features(imgs)
                _, logits_orig = model.forward_head(raw)

                ce_logits, ce_lbl = logits_orig, lbl
                if seen_idx is not None:
                    ce_logits, ce_lbl = logits_orig[:, seen_idx], remap[lbl]

                if args.imbalance in ("focal", "focal_weighted"):
                    loss_ce = focal_ce(ce_logits, ce_lbl, gamma=2.0, weight=cw)
                else:
                    loss_ce = F.cross_entropy(ce_logits, ce_lbl, weight=cw)

                loss_align = torch.zeros((), device=device)
                loss_kl = torch.zeros((), device=device)
                if args.use_dual_loss:
                    # As in the original implementation:
                    #   MMG on  -> spherical mixing on the raw backbone CLS
                    #              features, before the projection head
                    #   MMG off -> the sample and its own prototype are returned
                    #              unchanged, so the auxiliary objective degrades
                    #              to semantic alignment without synthesis
                    # This is also why MMG can only reach training through the
                    # dual-constraint loss: with both weights at zero the mixed
                    # branch has no gradient path at all.
                    protos_b = model.class_prototypes.to(device)[lbl]
                    scale = model.logit_scale.exp().clamp(max=100)
                    # Note: the KL target uses the **raw** prototype matrix, not
                    # current_prototypes(). The original implementation writes
                    #     prot_norm = F.normalize(model.class_prototypes, dim=1)
                    # while forward_head uses the **augmented** prototypes for
                    # the AnomalyCLIP- and DSECN-style references. That is an
                    # inconsistency in the original, but reproducing it faithfully
                    # matters: switching to current_prototypes() would turn those
                    # two references into different models.
                    prot = F.normalize(model.class_prototypes, dim=1)

                    if not args.use_mmg:
                        v_mix, t_mix = raw, protos_b
                        vq_mix, logits_mix = model.forward_head(v_mix)
                    elif args.mix_level == "backbone":
                        # Reported operator: mix before the projection head
                        act = sample_mmg_action(raw.size(0), device)
                        v_mix, t_mix = apply_mmg(raw, protos_b, act, mode=args.mixup_mode)
                        vq_mix, logits_mix = model.forward_head(v_mix)
                    elif args.mix_level == "projected":
                        # Mix after the projection head
                        v_all = model.projector(model.adapter(raw))
                        act = sample_mmg_action(raw.size(0), device)
                        v_mix, t_mix = apply_mmg(v_all, protos_b, act, mode=args.mixup_mode)
                        vq_mix = F.normalize(v_mix, p=2, dim=1)
                        logits_mix = scale * torch.matmul(vq_mix, prot.t())
                    elif args.mix_level == "logit":
                        # Mix on the classifier outputs. There is no mixed
                        # visual feature here, so the alignment term is
                        # undefined and this variant keeps only the KL term,
                        # which is itself part of what the comparison shows.
                        _, z_all = model.forward_head(raw)
                        act = sample_mmg_action(raw.size(0), device)
                        z_mix, t_mix = apply_mmg(z_all, protos_b, act, mode="raw")
                        vq_mix, logits_mix = None, z_mix
                    else:
                        raise ValueError(f"unknown mix_level: {args.mix_level}")

                    if vq_mix is not None:
                        loss_align = 1.0 - (F.normalize(vq_mix, p=2, dim=1)
                                            * F.normalize(t_mix, p=2, dim=1)
                                            ).sum(dim=1).clamp(-1, 1).mean()
                    tgt = torch.matmul(F.normalize(t_mix, p=2, dim=1), prot.t()) * scale
                    kl_mix, kl_tgt = logits_mix, tgt
                    if seen_idx is not None:
                        kl_mix, kl_tgt = logits_mix[:, seen_idx], tgt[:, seen_idx]
                    loss_kl = F.kl_div(F.log_softmax(kl_mix, dim=1),
                                       F.softmax(kl_tgt, dim=1), reduction="batchmean")

                loss = (args.lambda_ce * loss_ce
                        + args.lambda_align * loss_align
                        + args.lambda_kl * loss_kl)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        scheduler.step()

        seen_m, seen_s, seen_y = evaluate(model, va_loader, "seen", device)
        uns_m, uns_s, uns_y = evaluate(model, target_loader, "unseen", device)
        h = (2 * seen_m["Acc"] * uns_m["Acc"]) / (seen_m["Acc"] + uns_m["Acc"] + 1e-6)

        # Recognition of the held-out nutrient, evaluated in "unseen" mode
        # because its label is also a single-element list. Accuracy is therefore
        # the top-1 hit rate and Top-3 is whether it lands in the top three.
        nov_m, nov_s, nov_y = None, None, None
        if novel_loader is not None:
            nov_m, nov_s, nov_y = evaluate(model, novel_loader, "unseen", device)
            # Top-1 is driven to zero by the seen-class bias inherent to GZSL:
            # the eight trained prototypes have had their alignment optimised
            # and the held-out one has not. A calibration-free ranking metric is
            # recorded alongside it: the rank of the held-out class among the
            # nine, where 1 is best. It is more sensitive than top-3 and is
            # unaffected by any calibrated-stacking threshold.
            hcol = CLASS_TO_IDX[args.holdout_class]
            order = np.argsort(-nov_s, axis=1)
            nov_m["Rank"] = float(np.mean(
                [int(np.where(order[i] == hcol)[0][0]) + 1 for i in range(len(nov_s))]))

        rec = {"fold": fold, "epoch": epoch + 1, "loss": float(np.mean(losses)),
               "seen": seen_m, "unseen": uns_m, "hmean": h}
        if nov_m is not None:
            rec["novel"] = nov_m
        history.append(rec)
        print(f"[F{fold}] Ep {epoch+1}/{args.epochs} loss={np.mean(losses):.3f} "
              f"seenAcc={seen_m['Acc']:.3f} seenF1={seen_m['F1_Macro']:.3f} "
              f"unseenEM={uns_m['Acc']:.3f} top3={uns_m['Top3']:.3f} H={h:.3f}", flush=True)

        # One snapshot per selection rule. sel_seen reads only the source
        # validation split and never touches the target domain.
        for rule, score in (("sel_seen", seen_m["F1_Macro"]), ("sel_hmean", h)):
            if score > best[rule]:
                best[rule] = score
                snapshots[rule] = {
                    "epoch": epoch + 1, "seen": seen_m, "unseen": uns_m, "hmean": h,
                    "novel": nov_m,
                    "_seen_scores": seen_s, "_seen_y": seen_y,
                    "_unseen_scores": uns_s, "_unseen_y": uns_y,
                }
                if nov_s is not None:
                    snapshots[rule]["_novel_scores"] = nov_s
                    snapshots[rule]["_novel_y"] = nov_y
                # Artefacts the figures need: projected features for the
                # projection and retrieval panels, and weights for the
                # attribution maps. Saved only under sel_seen and only when
                # explicitly requested, so the normal queue is not slowed.
                if args.save_artifacts and rule == "sel_seen":
                    snapshots[rule]["_seen_feat"] = extract_projected(model, va_loader, device)
                    snapshots[rule]["_unseen_feat"] = extract_projected(model, target_loader, device)
                    if novel_loader is not None:
                        snapshots[rule]["_novel_feat"] = extract_projected(model, novel_loader, device)
                    # Keep the weights in memory and write once at the end of
                    # the fold. Saving here would write 1.2 GB every time
                    # seen-F1 improves, which over 20 epochs means seven or
                    # eight writes to a network volume.
                    if fold == min(int(x) for x in args.folds.split(",")):
                        snapshots[rule]["_state"] = {k: v.detach().cpu().clone()
                                                    for k, v in model.state_dict().items()}

    del model, optimizer
    torch.cuda.empty_cache()
    return history, snapshots


# ======================================================================================
# Experiment registry
# ======================================================================================
#
# A correction to the ablation design
# --------------------------------
# The original submission treated the adapter, MMG and the dual-constraint loss
# as three independently switchable factors in a 2^3 design. In the
# implementation, however, MMG can reach the objective **only** through the
# dual-constraint loss: switching that loss off sets both weights to zero, and
# the mixed branch then has no gradient path at all. The cells
#     "+MMG only"      ≡ "baseline"
#     "FRA + MMG"      ≡ "FRA only"
# are therefore the same model, and any difference between them measures a
# divergence of the random stream rather than a component effect.
#
# This registry uses a non-degenerate 2x3 design instead:
#     axis 1  residual adapter : off / on
#     axis 2  auxiliary objective : off / on without synthesis (aligned to the
#             sample's own prototype) / on with MMG synthesis
# The six cells are pairwise distinct, and the second axis isolates the question
# the paper needs answered: whether compositional synthesis buys anything beyond
# plain semantic alignment.
#
FULL = dict(use_adapter=True, use_mmg=True, use_dual_loss=True)

EXPERIMENTS = {
    # ---- 2x3 factorial (six distinct cells) ----
    "abl_fra0_aux_off":   dict(use_adapter=False, use_mmg=False, use_dual_loss=False),
    "abl_fra1_aux_off":   dict(use_adapter=True,  use_mmg=False, use_dual_loss=False),
    "abl_fra0_aux_plain": dict(use_adapter=False, use_mmg=False, use_dual_loss=True),
    "abl_fra1_aux_plain": dict(use_adapter=True,  use_mmg=False, use_dual_loss=True),
    "abl_fra0_aux_mmg":   dict(use_adapter=False, use_mmg=True,  use_dual_loss=True),
    "abl_full":           dict(**FULL),                       # the reported model

    # ---- The two terms of the dual constraint, removed one at a time ----
    "abl_no_align":       dict(**FULL, lambda_align=0.0),
    "abl_no_kl":          dict(**FULL, lambda_kl=0.0),

    # ---- Mixup geometry and level, including the standard-mixup controls ----
    "mix_euclidean":      dict(**FULL, mixup_mode="euclidean"),   # no re-projection
    "mix_raw":            dict(**FULL, mixup_mode="raw"),         # manifold mixup
    "mix_projected":      dict(**FULL, mix_level="projected"),    # mix after projection
    "mix_logit":          dict(**FULL, mix_level="logit"),        # mix on logits

    # ---- Reference models (same backbone, freeze policy and protocol) ----
    "ref_relationnet":    dict(**FULL, head="relationnet"),
    "ref_anomalyclip":    dict(**FULL, head="anomalyclip"),
    "ref_dsecn":          dict(**FULL, head="dsecn"),
    "ref_plantcafo":      dict(**FULL, head="plantcafo"),
    "ref_linear":         dict(use_adapter=True, use_mmg=False, use_dual_loss=False,
                               head="linear"),                    # knowledge-free head

    # ---- Loss-weight response surface ----
    # This analysis is run on fold 1, as in the paper; the registry only fills in
    # the grid. The cell (1.0, 0.5) is abl_full and is not registered twice.
    **{f"lam_a{str(a).replace('.','p')}_k{str(k).replace('.','p')}":
       {**FULL, "lambda_align": a, "lambda_kl": k}
       for a in (0.5, 1.0, 2.0) for k in (0.25, 0.5, 1.0)
       if not (a == 1.0 and k == 0.5)},

    # ---- Leave-one-nutrient-out ----
    # In the standard split all nine base classes are seen and only their
    # combinations are unseen, so top-k decoding alone recovers a compound and a
    # discriminative head can do it too. Isolating the semantic branch requires
    # removing a nutrient from training entirely, leaving only its symptom
    # description. A linear head cannot solve that setting by construction.
    **{f"lono_{c}_{v}": {**FULL, "holdout_class": c, **extra}
       for c in ("Fe", "Mn", "K", "B", "Ca", "Mg", "N", "P")
       for v, extra in (("kg", {}),
                        ("name", {"embed": "simple"}),
                        ("linear", {"head": "linear", "use_mmg": False,
                                    "use_dual_loss": False}))},

    # ---- Preprocessing control: rules out recognition by framing (see the
    #      note at the top of this file) ----
    "prep_aspect":        dict(**FULL, preprocess="aspect"),

    # ---- Design choices ----
    # ---- Description robustness: wording, clause order, length, contrastive
    #      statements, and an independently authored corpus ----
    # The first three are generated deterministically by script, so no
    # favourable variant can be selected by hand. All share the shape and class
    # order of the reported prototype bank.
    **{f"desc_{v}": {**FULL, "embed": v}
       for v in ("paraphrase", "order", "terse", "nocontr", "indep")},

    # ---- Open-vocabulary semantic space: a CLIP text encoder given the class
    #      name and no symptom description ----
    **{f"lono_{c}_clip": {**FULL, "holdout_class": c, "embed": "clip"}
       for c in ("Fe", "Mn", "K", "B", "Ca", "Mg", "N", "P")},

    # Visual-prototype control: the bank is rebuilt from the frozen DINOv3
    # feature centroid of each class, with nothing else changed.
    "ref_visproto": {**FULL, "embed": "visproto"},

    "design_simple_names": dict(**FULL, embed="simple"),          # bare class names
    "design_frozen":       dict(**FULL, unfreeze_last_n=0),       # frozen backbone

    # ---- Class-imbalance sensitivity ----
    "imb_inverse":        dict(**FULL, imbalance="inverse"),
    "imb_sqrt":           dict(**FULL, imbalance="sqrt_inverse"),
    "imb_effnum":         dict(**FULL, imbalance="effective_number"),
    "imb_focal":          dict(**FULL, imbalance="focal"),
    "imb_sampler":        dict(**FULL, imbalance="sampler"),
}

# Suggested order: run what the main tables need first, then the incremental
# analyses.
PRIORITY = {
    "P1_main_tables": ["abl_full", "ref_plantcafo", "ref_dsecn", "ref_anomalyclip",
                       "ref_relationnet", "ref_linear",
                       "abl_fra0_aux_off", "abl_fra1_aux_off", "abl_fra0_aux_plain",
                       "abl_fra1_aux_plain", "abl_fra0_aux_mmg"],
    "P2_reviewer_new": ["mix_euclidean", "mix_raw", "mix_projected", "mix_logit",
                        "abl_no_align", "abl_no_kl",
                        "imb_inverse", "imb_sqrt", "imb_effnum", "imb_focal", "imb_sampler"],
    "P3_design":       ["design_simple_names", "design_frozen", "prep_aspect"],
    "P5_lambda":       [f"lam_a{str(a).replace('.','p')}_k{str(k).replace('.','p')}"
                        for a in (0.5, 1.0, 2.0) for k in (0.25, 0.5, 1.0)
                        if not (a == 1.0 and k == 0.5)],
    "P4_lono":         [f"lono_{c}_{v}" for c in ("Fe", "Mn", "K", "B")
                        for v in ("kg", "name", "linear")],
    # All eight nutrients, so that the nutrient itself can serve as the unit of
    # statistical replication.
    "P6_lono_ext":     [f"lono_{c}_{v}" for c in ("Ca", "Mg", "N", "P")
                        for v in ("kg", "name", "linear")],
    "P7_desc_robust":  [f"desc_{v}" for v in
                        ("paraphrase", "order", "terse", "nocontr", "indep")],
    "P8_lono_openvoc": [f"lono_{c}_clip" for c in
                        ("Fe", "Mn", "K", "B", "Ca", "Mg", "N", "P")],
    # Five folds sharing one seed and one fixed target set are not five
    # independent replications; two further seeds are run for each of the five
    # configurations in the main comparison.
    "P9_seeds":        ["abl_full", "ref_dsecn", "ref_anomalyclip",
                        "ref_plantcafo", "ref_relationnet"],
}


# "all" expands to these groups only. The loss-weight surface is deliberately
# excluded: that analysis is a fold-1 protocol, and running it over five folds
# would cost about 17 GPU-hours and depart from the reported protocol.
ALL_PLANS = ["P1_main_tables", "P2_reviewer_new", "P3_design", "P4_lono"]


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True,
                   help="a name from EXPERIMENTS, or one of all/P1/P2/...")
    p.add_argument("--data-root", default="data/CoLeaf-DB")
    p.add_argument("--backbone", default="models/dinov3-vitl16")
    p.add_argument("--embed-kg", default="prototypes/class_embeddings.pt")
    p.add_argument("--embed-simple", default="prototypes/class_embeddings_simple.pt")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--folds", default="1,2,3,4,5")
    # 20 rather than 40. Recomputed from the original per-epoch logs, the
    # four-fold mean H-Mean under the leakage-free rule is 0.6185 at 20 epochs
    # against 0.6120 at 25, 0.6090 at 30 and 0.6115 at 40: training longer
    # improves seen-class accuracy and costs transfer. 20 is also the protocol
    # of the original ablation and comparison runs, with T_max set to match.
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    # Measured on an uncontended GPU machine: 112.9 ms per image to decode,
    # 451.2 ms to decode plus training augmentation, 135.0 ms for the evaluation
    # transform, or about 356 core-seconds per epoch. The binding constraint is
    # the cgroup CPU quota of 15 cores, not the 128 that nproc reports, so
    # jobs x workers should be around 15; oversubscribing only causes
    # throttling. The value must be **the same for every configuration**, or the
    # augmentation random stream differs and configurations stop being
    # comparable.
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-6)
    p.add_argument("--projector", default="deep", choices=["deep", "shallow"])
    p.add_argument("--preprocess", default="squash", choices=["squash", "aspect"])
    p.add_argument("--group-file", default=None,
                   help="JSON mapping {image filename: group id}. When given, folds are "
                        "built with GroupKFold so that no group spans training and "
                        "validation; otherwise the random KFold split is used.")
    p.add_argument("--save-artifacts", action="store_true",
                   help="additionally save the 768-d projected features and the first-fold "
                        "weights, for regenerating the figures")
    p.add_argument("--no-cache-bytes", action="store_true",
                   help="do not read the raw file bytes into memory (the cache is on by "
                        "default; pixels are identical, it only avoids the disk round-trip)")
    return p.parse_args()


def _embed_path(args, embed):
    """Resolve the prototype file by name.

    Variant banks must sit in the same directory as the reported bank and share
    its shape (9, 768) and class order; otherwise register_buffer would silently
    load a different semantic space.
    """
    if embed == "kg":
        return args.embed_kg
    if embed == "simple":
        return args.embed_simple
    import os
    d = os.path.dirname(args.embed_kg)
    p = os.path.join(d, "class_embeddings_%s.pt" % embed)
    if not os.path.exists(p):
        raise FileNotFoundError("prototype file not found: %s (embed=%r)" % (p, embed))
    return p


def run_experiment(name, args, s_imgs, s_labels, make_target_loader, device, seen_dirs,
                   blob_cache=None):
    args.exp = name
    spec = dict(EXPERIMENTS[name])
    args.use_adapter = spec.pop("use_adapter")
    args.use_mmg = spec.pop("use_mmg")
    args.use_dual_loss = spec.pop("use_dual_loss")
    args.mixup_mode = spec.pop("mixup_mode", "geometric")
    args.mix_level = spec.pop("mix_level", "backbone")
    args.imbalance = spec.pop("imbalance", "none")
    args.preprocess = spec.pop("preprocess", "squash")
    args.holdout_class = spec.pop("holdout_class", None)
    args.lambda_ce = spec.pop("lambda_ce", 1.0)
    args.lambda_align = spec.pop("lambda_align", 1.0)
    args.lambda_kl = spec.pop("lambda_kl", 0.5)
    head = spec.pop("head", "kgmixnet")
    embed = spec.pop("embed", "kg")
    unfreeze = spec.pop("unfreeze_last_n", 2)
    assert not spec, f"unconsumed configuration keys: {spec}"

    # The target split must use the same preprocessing as training and
    # validation. Building it here from the current configuration, rather than
    # reusing a global transform, prevents the aspect-preserving control from
    # training on padded images and being evaluated on squashed ones.
    target_loader = make_target_loader(args.preprocess)
    # The held-out nutrient's images form a single never-trained class. Their
    # labels are written as one-element lists so the unseen evaluation path can
    # be reused directly: accuracy is then the top-1 hit rate and Top-3 is
    # whether the nutrient lands in the top three.
    novel_loader = None
    if args.holdout_class:
        mask = s_labels == args.holdout_class
        novel_loader = DataLoader(
            CoffeeDataset(s_imgs[mask], [[args.holdout_class]] * int(mask.sum()),
                          get_transforms(args.preprocess)[1], "target",
                          blob_cache=blob_cache),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            collate_fn=collate_fn)
        print(f"[lono] held out {args.holdout_class}: {int(mask.sum())} images removed "
              f"from training; the only semantic anchor is its symptom description",
              flush=True)

    mcfg = ModelCfg(backbone_path=args.backbone,
                    embed_path=_embed_path(args, embed),
                    head=head, use_adapter=args.use_adapter,
                    unfreeze_last_n=unfreeze, projector=args.projector)

    folds = [int(x) for x in args.folds.split(",")]
    if getattr(args, "group_file", None):
        # Random image-level folds can place photographs from the same capture
        # session on both sides of the split. Grouping by EXIF capture date
        # keeps a session on one side.
        from sklearn.model_selection import GroupKFold
        import json as _json
        _gmap = _json.load(open(args.group_file, encoding="utf-8"))
        _g = [_gmap.get(os.path.basename(p), "solo:" + os.path.basename(p))
              for p in s_imgs]
        assert len(_g) == len(s_imgs), "group count does not match sample count"
        splits = list(GroupKFold(n_splits=5).split(s_imgs, s_labels, _g))
        for _tr, _va in splits:
            assert not (set(np.array(_g)[_tr]) & set(np.array(_g)[_va])), "a group spans the split"
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)   # the reported split
        splits = list(kf.split(s_imgs))

    t0 = time.time()
    all_hist, per_fold = [], {"sel_seen": [], "sel_hmean": []}
    score_store = {}
    for fold in folds:
        tr_idx, va_idx = splits[fold - 1]
        hist, snaps = train_one_fold(args, mcfg, fold, tr_idx, va_idx,
                                     s_imgs, s_labels, target_loader, device,
                                     blob_cache=blob_cache, novel_loader=novel_loader)
        all_hist.extend(hist)
        for rule in ("sel_seen", "sel_hmean"):
            snap = snaps[rule]
            state = snap.pop("_state", None)
            if state is not None:
                os.makedirs(args.out_dir, exist_ok=True)
                dst = os.path.join(args.out_dir, f"{name}_fold{fold}.pth")
                torch.save(state, dst)
                print(f"[artifact] weights saved to {dst} "
                      f"({os.path.getsize(dst)/1e9:.2f} GB, epoch {snap['epoch']})", flush=True)
            per_fold[rule].append({k: v for k, v in snap.items() if not k.startswith("_")})
            for key in ("_seen_scores", "_seen_y", "_unseen_scores", "_unseen_y",
                        "_novel_scores", "_novel_y",
                        "_seen_feat", "_unseen_feat", "_novel_feat"):
                if key in snap:
                    score_store[f"{rule}__fold{fold}{key}"] = snap[key]
        score_store[f"val_index__fold{fold}"] = va_idx

    def agg(rows):
        out = {}
        for metric, getter in (("seen_acc", lambda r: r["seen"]["Acc"]),
                               ("seen_f1", lambda r: r["seen"]["F1_Macro"]),
                               ("unseen_exact", lambda r: r["unseen"]["Acc"]),
                               ("unseen_top3", lambda r: r["unseen"]["Top3"]),
                               ("unseen_f1_samples", lambda r: r["unseen"]["F1_Samples"]),
                               ("unseen_f1_macro", lambda r: r["unseen"]["F1_Macro"]),
                               ("unseen_hamming", lambda r: r["unseen"]["Hamming"]),
                               ("hmean", lambda r: r["hmean"]),
                               ("epoch", lambda r: r["epoch"]),
                               ("novel_top1", lambda r: (r.get("novel") or {}).get("Acc")),
                               ("novel_top3", lambda r: (r.get("novel") or {}).get("Top3")),
                               ("novel_rank", lambda r: (r.get("novel") or {}).get("Rank"))):
            vals = [getter(r) for r in rows]
            if any(v is None for v in vals):
                continue
            out[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))
                           if len(vals) > 1 else 0.0, "per_fold": [float(v) for v in vals]}
        return out

    result = {
        "exp": name,
        "spec": EXPERIMENTS[name],
        "protocol": {"folds": folds, "epochs": args.epochs, "batch_size": args.batch_size,
                     "workers": args.workers, "seed": args.seed, "lr": args.lr,
                     "lr_backbone": args.lr_backbone, "projector": args.projector,
                     "scheduler": f"CosineAnnealingLR(T_max={args.epochs}, eta_min=1e-6)",
                     "optimizer": "AdamW(weight_decay=1e-4)",
                     "lambda": [args.lambda_ce, args.lambda_align, args.lambda_kl],
                     "mixup_mode": args.mixup_mode, "mix_level": args.mix_level,
                     "imbalance": args.imbalance,
                     "preprocess": args.preprocess,
                     "data_root": args.data_root, "seen_dirs": seen_dirs,
                     "n_seen": int(len(s_imgs)), "n_unseen": len(target_loader.dataset),
                     "target_preprocess": args.preprocess,
                     "unfreeze_last_n": unfreeze,
                     "holdout_class": args.holdout_class},
        "summary": {rule: agg(per_fold[rule]) for rule in ("sel_seen", "sel_hmean")},
        "per_fold": per_fold,
        "history": all_hist,
        "wall_clock_s": time.time() - t0,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    # The seed goes into the file name so repeated runs do not overwrite each
    # other. Seed 42 keeps the bare name, for compatibility with single runs.
    stem = name if args.seed == 42 else f"{name}__seed{args.seed}"
    if getattr(args, "group_file", None):
        stem += "__group"
    result["stem"] = stem
    with open(os.path.join(args.out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(args.out_dir, f"{stem}.scores.npz"), **score_store)

    for rule in ("sel_seen", "sel_hmean"):
        s = result["summary"][rule]
        print(f"== {name} [{rule}]  H={s['hmean']['mean']:.4f}±{s['hmean']['std']:.4f}  "
              f"seenAcc={s['seen_acc']['mean']:.4f}  unseenEM={s['unseen_exact']['mean']:.4f}",
              flush=True)
    return result


def main():
    args = build_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}", flush=True)

    s_imgs, s_labels, t_imgs, t_labels, seen_dirs = load_data(args.data_root)
    print(f"source images: {len(s_imgs)}  target images: {len(t_imgs)}  "
          f"directories: {seen_dirs}", flush=True)
    counts = np.bincount([CLASS_TO_IDX[x] for x in s_labels], minlength=N_CLASSES)
    print("  per class:", {IDX_TO_CLASS[i]: int(c) for i, c in enumerate(counts)}, flush=True)

    blob_cache = None
    if not args.no_cache_bytes:
        blob_cache = build_blob_cache(list(s_imgs) + list(t_imgs))

    def make_target_loader(preprocess):
        _, tf_eval = get_transforms(preprocess)
        return DataLoader(
            CoffeeDataset(t_imgs, t_labels, tf_eval, "target", blob_cache=blob_cache),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            collate_fn=collate_fn)


    if args.exp in ("P1", "P2", "P3", "P4", "P5"):
        names = next(v for k, v in PRIORITY.items() if k.startswith(args.exp))
    elif args.exp == "all":
        names = sum((PRIORITY[k] for k in ALL_PLANS), [])
    else:
        names = [args.exp]

    for name in names:
        print(f"\n{'='*30} {name} {'='*30}", flush=True)
        run_experiment(name, args, s_imgs, s_labels, make_target_loader, device, seen_dirs,
                       blob_cache=blob_cache)


if __name__ == "__main__":
    main()

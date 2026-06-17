"""Adapted in-house baselines used for the internal comparison (paper Table 2).

IMPORTANT (honest disclosure, matching the paper): these are *adapted in-house*
re-implementations aligned to the shared DINOv3 backbone and the common
evaluation protocol. They are NOT official reproductions of the original
published systems, and the comparison is an internal controlled study rather
than a cross-paper leaderboard. The CLIP zero-shot baseline is evaluated
separately (see ``scripts/eval_clip.py``).

Supported ``model_type`` values for :class:`CompareModel`:
  * ``'anomaly_clip'`` -- AnomalyCLIP-inspired object-agnostic prompt learning.
  * ``'dsecn'``        -- DSECN-inspired semantic enrichment.
  * ``'plant_cafo'``   -- PlantCaFo-inspired efficient few-shot adapter.

:class:`RelationNetModel` implements the RelationNet-style metric-matching head.
All baselines share the same backbone-handling (unfreeze last two blocks) as the
main model for a fair comparison.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# --------------------------------------------------------------------------- #
# Shared / baseline-specific components
# --------------------------------------------------------------------------- #
class K_Adapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.down = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.dropout(self.up(self.act(self.down(x))))


class AnomalyPromptLearner(nn.Module):
    """AnomalyCLIP (ICLR 2024) object-agnostic prompt learning (simulated)."""

    def __init__(self, embed_dim, num_prompts=4):
        super().__init__()
        self.ctx = nn.Parameter(torch.empty(num_prompts, embed_dim))
        nn.init.normal_(self.ctx, std=0.02)
        self.fusion = nn.Linear(embed_dim, embed_dim)
        self.act = nn.SiLU()

    def forward(self, text_prototypes):
        global_ctx = self.ctx.mean(dim=0, keepdim=True)
        enhanced_text = text_prototypes + global_ctx
        return self.act(self.fusion(enhanced_text))


class DSECN_SemanticEnricher(nn.Module):
    """DSECN (CVPR 2024) diverse-semantic expansion module (simulated)."""

    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.expansion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, text_prototypes):
        semantic_gate = self.expansion_net(text_prototypes)
        enriched_text = self.proj(text_prototypes * semantic_gate + text_prototypes)
        return enriched_text


class PlantCaFo_Adapter(nn.Module):
    """PlantCaFo (2025) efficient few-shot adapter (simulated)."""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        shortcut = x
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return shortcut + x * self.scale


def _unfreeze_last_two(backbone):
    for param in backbone.parameters():
        param.requires_grad = False
    layers_to_unfreeze = []
    if hasattr(backbone, "layer"):
        layers_to_unfreeze = backbone.layer[-2:]
    elif hasattr(backbone, "blocks"):
        layers_to_unfreeze = backbone.blocks[-2:]
    for block in layers_to_unfreeze:
        for param in block.parameters():
            param.requires_grad = True
    if hasattr(backbone, "norm"):
        for param in backbone.norm.parameters():
            param.requires_grad = True


# --------------------------------------------------------------------------- #
# Prompt / semantic baselines (AnomalyCLIP, DSECN, PlantCaFo)
# --------------------------------------------------------------------------- #
class CompareModel(nn.Module):
    def __init__(self, dino_path, class_embeddings_path, device="cuda",
                 model_type="anomaly_clip"):
        super().__init__()
        self.device = device
        self.model_type = model_type
        print(f">>>> [Comparison] Initializing '{model_type.upper()}' <<<<")

        self.backbone = AutoModel.from_pretrained(dino_path, trust_remote_code=True)
        self.visual_dim = getattr(self.backbone.config, "hidden_size", 1024)
        _unfreeze_last_two(self.backbone)

        text_emb = torch.load(class_embeddings_path, map_location=device)
        self.num_classes = text_emb.shape[0]
        self.text_dim = text_emb.shape[1]
        self.register_buffer("class_prototypes", text_emb)

        self.adapter = K_Adapter(self.visual_dim, hidden_dim=512)
        if model_type == "plant_cafo":
            self.adapter = PlantCaFo_Adapter(self.visual_dim)

        self.projector = nn.Sequential(
            nn.Linear(self.visual_dim, self.text_dim),
            nn.LayerNorm(self.text_dim),
            nn.GELU(),
        )

        if model_type == "anomaly_clip":
            self.prompt_learner = AnomalyPromptLearner(self.text_dim)
        elif model_type == "dsecn":
            self.semantic_enricher = DSECN_SemanticEnricher(self.text_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def extract_features(self, images):
        outputs = self.backbone(pixel_values=images)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0, :]
        return outputs[0][:, 0, :]

    def forward_head(self, features):
        adapted = self.adapter(features)
        visual_query = self.projector(adapted)

        current_protos = self.class_prototypes
        if self.model_type == "anomaly_clip":
            current_protos = self.prompt_learner(current_protos)
        elif self.model_type == "dsecn":
            current_protos = self.semantic_enricher(current_protos)

        visual_norm = F.normalize(visual_query, p=2, dim=1)
        text_norm = F.normalize(current_protos, p=2, dim=1)
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * torch.matmul(visual_norm, text_norm.t())
        return visual_norm, logits

    def forward(self, images):
        feats = self.extract_features(images)
        _, logits = self.forward_head(feats)
        return logits


# --------------------------------------------------------------------------- #
# RelationNet-style metric matching baseline
# --------------------------------------------------------------------------- #
class RelationNetModel(nn.Module):
    def __init__(self, dino_path, class_embeddings_path, device="cuda"):
        super().__init__()
        self.device = device
        print(">>>> [Comparison] Simulating RelationNet architecture <<<<")

        self.backbone = AutoModel.from_pretrained(dino_path, trust_remote_code=True)
        self.visual_dim = getattr(self.backbone.config, "hidden_size", 1024)
        _unfreeze_last_two(self.backbone)

        text_emb = torch.load(class_embeddings_path, map_location=device)
        self.num_classes = text_emb.shape[0]
        self.text_dim = text_emb.shape[1]
        self.register_buffer("class_prototypes", text_emb)

        self.adapter = K_Adapter(self.visual_dim, hidden_dim=512)
        self.projector = nn.Sequential(
            nn.Linear(self.visual_dim, self.visual_dim),
            nn.LayerNorm(self.visual_dim),
            nn.GELU(),
            nn.Linear(self.visual_dim, self.text_dim),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(self.text_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def extract_features(self, images):
        outputs = self.backbone(pixel_values=images)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0, :]
        return outputs[0][:, 0, :]

    def forward_head(self, features):
        adapted = self.adapter(features)
        projected = self.projector(adapted)
        b = projected.size(0)
        v_expand = projected.unsqueeze(1).repeat(1, self.num_classes, 1)
        t_expand = self.class_prototypes.unsqueeze(0).repeat(b, 1, 1)
        cat_feat = torch.cat((v_expand, t_expand), dim=2)
        logits = self.relation_head(cat_feat).squeeze(2)
        logits = logits * self.logit_scale.exp()
        return projected, logits

    def forward(self, images):
        feats = self.extract_features(images)
        _, logits = self.forward_head(feats)
        return logits

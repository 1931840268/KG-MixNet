"""KG-MixNet main network.

This is the model used for all main results in the paper. It consists of:
  * a DINOv3 ViT-L/16 backbone (only the last two transformer blocks and the
    final norm are unfrozen by default),
  * a lightweight residual bottleneck adapter (K_Adapter, 1024 -> 512 -> 1024),
  * a projection head to the 768-dim shared visual-semantic space,
  * scaled cosine classification against fixed MPNet text prototypes.

The two ablation switches used in the paper are exposed as constructor flags so
that the exact configurations can be reproduced without editing the source:
  * ``use_adapter=False``     -> removes the feature-level residual adapter (FRA).
  * ``freeze_backbone=True``  -> fully frozen backbone (linear-probing variant).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class K_Adapter(nn.Module):
    """Residual bottleneck adapter (feature-level residual adapter, FRA)."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.down = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.1)
        # zero-init the up-projection so the adapter starts as identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(self.act(self.down(x))))


class KG_MixNet(nn.Module):
    def __init__(self, dino_path: str, class_embeddings_path: str,
                 device: str = "cuda", freeze_backbone: bool = False,
                 use_adapter: bool = True):
        super().__init__()
        self.device = device

        self.backbone = AutoModel.from_pretrained(dino_path, trust_remote_code=True)

        if hasattr(self.backbone.config, "hidden_size"):
            self.visual_dim = self.backbone.config.hidden_size
        else:
            self.visual_dim = 1024

        # --- backbone freezing / selective unfreezing ---
        for param in self.backbone.parameters():
            param.requires_grad = False

        if freeze_backbone:
            print(">>>> [KG_MixNet] FROZEN BACKBONE (linear-probing mode) <<<<")
        else:
            print(">>>> [KG_MixNet] Unfreezing last 2 transformer blocks <<<<")
            layers_to_unfreeze = []
            if hasattr(self.backbone, "layer"):
                layers_to_unfreeze = self.backbone.layer[-2:]
            elif hasattr(self.backbone, "blocks"):
                layers_to_unfreeze = self.backbone.blocks[-2:]
            elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
                layers_to_unfreeze = self.backbone.encoder.layer[-2:]
            elif hasattr(self.backbone, "layers"):
                layers_to_unfreeze = self.backbone.layers[-2:]

            for block in layers_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True

            if hasattr(self.backbone, "norm"):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
            elif hasattr(self.backbone, "layernorm"):
                for param in self.backbone.layernorm.parameters():
                    param.requires_grad = True

        # --- semantic prototypes (fixed buffer) ---
        text_emb = torch.load(class_embeddings_path, map_location=device)
        self.num_classes = text_emb.shape[0]
        self.text_dim = text_emb.shape[1]
        self.register_buffer("class_prototypes", text_emb)

        # --- feature-level residual adapter (innovation 1, ablatable) ---
        if use_adapter:
            self.adapter = K_Adapter(self.visual_dim, hidden_dim=512)
        else:
            self.adapter = nn.Identity()

        # --- projection head into the shared 768-dim space ---
        self.projector = nn.Sequential(
            nn.Linear(self.visual_dim, self.visual_dim),
            nn.LayerNorm(self.visual_dim),
            nn.GELU(),
            nn.Linear(self.visual_dim, self.text_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def extract_features(self, images):
        outputs = self.backbone(pixel_values=images)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0, :]
        elif isinstance(outputs, torch.Tensor):
            if outputs.dim() == 3:
                return outputs[:, 0, :]
            return outputs
        elif isinstance(outputs, tuple):
            return outputs[0][:, 0, :]
        else:
            return outputs.last_hidden_state[:, 0, :]

    def forward_head(self, features):
        adapted = self.adapter(features)
        projected = self.projector(adapted)
        visual_query_norm = F.normalize(projected, p=2, dim=1)
        text_protos_norm = F.normalize(self.class_prototypes, p=2, dim=1)

        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * torch.matmul(visual_query_norm, text_protos_norm.t())
        return visual_query_norm, logits

    def forward(self, images):
        feats = self.extract_features(images)
        _, logits = self.forward_head(feats)
        return logits

from __future__ import annotations

import torch
import torch.nn as nn

from networks.SliceStudent_MultiModal import SliceStudent_MultiModal


class SliceStudent_MultiModal_Mixed(nn.Module):
    """
    Simple mixed baseline: image encoder + clinical MLP + feature concatenation.
    """

    def __init__(
        self,
        n_slices: int = 32,
        lora_rank: int = 4,
        ckpt_path: str | None = None,
        embed_dim: int = 768,
        structured_dim: int = 7,
        structured_hidden_dim: int = 128,
        frozen: bool = True,
        backbone_name: str = "dinov3",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if structured_dim <= 0:
            raise ValueError("structured_dim must be positive")

        self.dropout = float(dropout)
        self.image_backbone = SliceStudent_MultiModal(
            n_slices=n_slices,
            lora_rank=lora_rank,
            ckpt_path=ckpt_path,
            embed_dim=embed_dim,
            num_classes=2,
            frozen=frozen,
            backbone_name=backbone_name,
            reg_head=False,
        )

        self.image_dim = self.image_backbone.embed_dim
        self.structured_hidden_dim = int(structured_hidden_dim)

        self.structured_encoder = nn.Sequential(
            nn.LayerNorm(structured_dim),
            nn.Linear(structured_dim, self.structured_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.structured_hidden_dim, self.structured_hidden_dim),
            nn.GELU(),
        )

        fusion_dim = self.image_dim + self.structured_hidden_dim
        self.out_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(self.dropout),
            nn.Linear(fusion_dim, self.image_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.image_dim, 1),
        )

    def forward(
        self,
        multi_modal_volume: torch.Tensor,
        structured_features: torch.Tensor,
        slice_weights: torch.Tensor | None = None,
    ):
        image_features = self.image_backbone.extract_averaged_features(
            multi_modal_volume,
            slice_weights=slice_weights,
        )
        structured_repr = self.structured_encoder(structured_features)
        fused = torch.cat([image_features, structured_repr], dim=1)
        return self.out_head(fused)

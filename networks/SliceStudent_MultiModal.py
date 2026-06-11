"""
Multi-modality version of SliceStudent.

Input: (B, 4, D, H, W) - 4 modalities per patient
Process: Each modality → encoder → feature → average across modalities → classifier

This ensures each modality goes through the encoder independently,
then features are averaged before the classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.SliceStudent import SliceStudent


class SliceStudent_MultiModal(nn.Module):
    """
    Multi-modality wrapper for SliceStudent.

    Input shape: (B, 4, D, H, W)
    - B: batch size
    - 4: number of modalities (T1, T1CE, T2, FLAIR)
    - D, H, W: depth, height, width

    Forward process:
    1. Split into 4 separate modalities
    2. Pass each modality through encoder independently
    3. Get 4 feature vectors (each B x 768)
    4. Average features across modalities
    5. Pass averaged feature to classification head
    """

    def __init__(self,
                 n_slices=32,
                 lora_rank=4,
                 ckpt_path=None,
                 embed_dim=768,
                 num_classes=3,
                 frozen=True,
                 backbone_name='dinov3',
                 reg_head=False):
        super().__init__()

        self.n_slices = n_slices
        self.use_reg_head = reg_head
        self.num_modalities = 4
        self.backbone_name = backbone_name

        # Build single-modality encoder
        # Use SliceStudent_comparisons for BrainMVP/bm_mae, SliceStudent for others
        TEACHERS = ["brainmvp", "bm_mae", "brainiac"]
        enc = backbone_name.lower()

        # Adjust embed_dim based on encoder type
        if enc == "brainmvp":
            actual_embed_dim = 512  # BrainMVP outputs 512-dim features
        else:
            actual_embed_dim = embed_dim  # Others use 768

        if enc in TEACHERS:
            from networks.SliceStudent_comparisons import SliceStudent_comparisons
            self.encoder = SliceStudent_comparisons(
                ckpt_path=ckpt_path,
                n_slices=n_slices,
                lora_rank=lora_rank,
                num_classes=num_classes,
                teacher_name=enc
            )
        else:
            self.encoder = SliceStudent(
                n_slices=n_slices,
                lora_rank=lora_rank,
                ckpt_path=ckpt_path,
                embed_dim=actual_embed_dim,
                num_classes=num_classes,
                frozen=frozen,
                backbone_name=backbone_name,
                reg_head=False  # We'll add our own head
            )

        # Remove the original classifier head from encoder
        # We'll add a new one that takes averaged features
        self.encoder.cls_head = nn.Identity()  # Disable original head

        # Store actual embedding dimension
        self.embed_dim = actual_embed_dim

        # Our classification head (takes averaged features)
        # Added dropout to prevent overfitting
        # Use self.embed_dim to support different feature dimensions
        if enc == "brainiac":
            self.cls_head = nn.Sequential(
                nn.Linear(self.embed_dim, num_classes)  # ★★ BrainIAC: simple linear head
            )
        else:
            self.cls_head = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(0.3),  # Dropout after norm
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Dropout(0.3),  # Dropout after activation
                nn.Linear(self.embed_dim, num_classes)
            )

        # Regression head (optional)
        # Added dropout to prevent overfitting
        if enc == "brainiac":
            self.reg_head = nn.Sequential(
                nn.Linear(self.embed_dim, 1)  # ★ BrainIAC: simple linear head
            )
        else:
            self.reg_head = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(0.3),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(self.embed_dim, 1)
            )

    def forward(self, multi_modal_volume, slice_weights: torch.Tensor | None = None):
        """
        Args:
            multi_modal_volume: (B, 4, D, H, W)

        Returns:
            logits: (B, num_classes) or (B, 1) if regression
        """
        averaged_features = self.extract_averaged_features(
            multi_modal_volume,
            slice_weights=slice_weights,
        )

        # Classification
        if self.use_reg_head:
            return self.reg_head(averaged_features)
        else:
            return self.cls_head(averaged_features)

    def extract_modality_features(
        self,
        multi_modal_volume,
        slice_weights: torch.Tensor | None = None,
    ):
        """
        Return one feature vector per modality before fusion.

        Args:
            multi_modal_volume: (B, 4, D, H, W)

        Returns:
            modality_features: (B, 4, embed_dim)
        """
        _, num_mods, _, _, _ = multi_modal_volume.shape

        assert num_mods == self.num_modalities, \
            f"Expected {self.num_modalities} modalities, got {num_mods}"

        modality_features = []
        for i in range(self.num_modalities):
            single_modality = multi_modal_volume[:, i:i + 1, :, :, :]
            feat = self._extract_features(single_modality, slice_weights=slice_weights)
            modality_features.append(feat)

        return torch.stack(modality_features, dim=1)

    def extract_modality_cls_tokens(self, multi_modal_volume):
        """
        Return per-slice CLS tokens for each modality.

        Args:
            multi_modal_volume: (B, 4, D, H, W)

        Returns:
            modality_cls_tokens: (B, 4, n_slices, embed_dim)
        """
        _, num_mods, _, _, _ = multi_modal_volume.shape

        assert num_mods == self.num_modalities, \
            f"Expected {self.num_modalities} modalities, got {num_mods}"

        modality_cls_tokens = []
        for i in range(self.num_modalities):
            single_modality = multi_modal_volume[:, i:i + 1, :, :, :]
            cls_tokens = self._extract_cls_tokens(single_modality)
            modality_cls_tokens.append(cls_tokens)

        return torch.stack(modality_cls_tokens, dim=1)

    def extract_averaged_features(
        self,
        multi_modal_volume,
        slice_weights: torch.Tensor | None = None,
    ):
        """
        Return the modality-averaged patient representation used by the
        standard image-only head.
        """
        modality_features = self.extract_modality_features(
            multi_modal_volume,
            slice_weights=slice_weights,
        )
        return modality_features.mean(dim=1)

    def _pool_cls_tokens(
        self,
        cls_tokens: torch.Tensor,
        slice_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if slice_weights is None:
            return cls_tokens.mean(dim=1)

        weights = slice_weights.to(device=cls_tokens.device, dtype=cls_tokens.dtype)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0).expand(cls_tokens.shape[0], -1)
        if weights.shape != cls_tokens.shape[:2]:
            raise ValueError(
                f"slice_weights must have shape {(cls_tokens.shape[0], cls_tokens.shape[1])}, "
                f"got {tuple(weights.shape)}"
            )

        weights = weights.clamp_min(0.0)
        denom = weights.sum(dim=1, keepdim=True)
        fallback = torch.full_like(weights, 1.0 / weights.shape[1])
        normalized = torch.where(denom > 0, weights / denom.clamp_min(1e-6), fallback)
        return (normalized.unsqueeze(-1) * cls_tokens).sum(dim=1)

    def _extract_features(self, volume, slice_weights: torch.Tensor | None = None):
        """
        Extract features from encoder without classification.

        Args:
            volume: (B, 1, D, H, W)

        Returns:
            features: (B, 768)
        """
        B, C, D, H, W = volume.shape

        # Check if using comparison encoder (BrainMVP/bm_mae)
        from networks.SliceStudent_comparisons import SliceStudent_comparisons

        if isinstance(self.encoder, SliceStudent_comparisons):
            if slice_weights is not None:
                raise RuntimeError(
                    "Seg-guided slice selection is only supported for SliceStudent-based "
                    "2D ViT backbones such as BrainDINO and DINOv3."
                )
            # SliceStudent_comparisons processes 3D volume directly
            # It returns cls_head output, we need features before that

            # Temporarily save and replace cls_head
            original_cls_head = self.encoder.cls_head
            original_use_reg_head = self.encoder.use_reg_head

            # Replace with identity to get raw features
            self.encoder.cls_head = nn.Identity()
            self.encoder.use_reg_head = False

            # Get features (with teacher frozen, so use no_grad)
            with torch.no_grad():
                volume_feat = self.encoder(volume)  # (B, embed_dim)

            # Restore original head
            self.encoder.cls_head = original_cls_head
            self.encoder.use_reg_head = original_use_reg_head

            return volume_feat

        else:
            # Standard SliceStudent: sample slices and use 2D ViT
            idx = torch.linspace(0, D - 1, self.n_slices).long().to(volume.device)
            slices = volume[:, :, idx, :, :]
            slices = slices.permute(0, 2, 1, 3, 4)
            slices = slices.reshape(B * self.n_slices, 1, H, W)
            slices = slices.repeat(1, 3, 1, 1)  # DINO expects 3 channels

            # 2D ViT features
            out = self.encoder.student.forward_features(slices)

            # CLS token
            cls_tokens = out["x_norm_clstoken"]  # (B*N, 768)

            # Reshape to (B, N, 768)
            cls_tokens = cls_tokens.reshape(B, self.n_slices, -1)

            # Mean pooling across slices, or seg-guided selected-slice mean.
            volume_feat = self._pool_cls_tokens(cls_tokens, slice_weights=slice_weights)

            return volume_feat

    def _extract_cls_tokens(self, volume):
        """
        Extract per-slice CLS tokens from the frozen 2D ViT backbone.

        Args:
            volume: (B, 1, D, H, W)

        Returns:
            cls_tokens: (B, n_slices, embed_dim)
        """
        B, _, D, H, W = volume.shape

        from networks.SliceStudent_comparisons import SliceStudent_comparisons

        if isinstance(self.encoder, SliceStudent_comparisons):
            raise RuntimeError(
                "Raw CLS-token extraction is only supported for SliceStudent-based "
                "2D ViT backbones such as BrainDINO/meddinov3."
            )

        idx = torch.linspace(0, D - 1, self.n_slices).long().to(volume.device)
        slices = volume[:, :, idx, :, :]
        slices = slices.permute(0, 2, 1, 3, 4)
        slices = slices.reshape(B * self.n_slices, 1, H, W)
        slices = slices.repeat(1, 3, 1, 1)

        out = self.encoder.student.forward_features(slices)
        cls_tokens = out["x_norm_clstoken"]
        return cls_tokens.reshape(B, self.n_slices, -1)

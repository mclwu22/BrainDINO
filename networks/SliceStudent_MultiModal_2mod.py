"""
2-Modality version of SliceStudent.

Input: (B, 2, D, H, W) - 2 modalities per patient (T1c and FLAIR)
Process: Each modality → encoder → feature → average across modalities → classifier

Based on SliceStudent_MultiModal.py but adapted for 2 modalities instead of 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.SliceStudent import SliceStudent


class SliceStudent_MultiModal_2mod(nn.Module):
    """
    2-modality wrapper for SliceStudent.

    Input shape: (B, 2, D, H, W)
    - B: batch size
    - 2: number of modalities (T1c, FLAIR)
    - D, H, W: depth, height, width

    Forward process:
    1. Split into 2 separate modalities
    2. Pass each modality through encoder independently
    3. Get 2 feature vectors (each B x 768)
    4. Average features across modalities
    5. Pass averaged feature to classification head
    """

    def __init__(self,
                 n_slices=128,
                 lora_rank=8,
                 ckpt_path=None,
                 embed_dim=768,
                 num_classes=2,
                 frozen=True,
                 backbone_name='dinov3'):
        super().__init__()

        self.n_slices = n_slices
        self.num_modalities = 2  # T1c and FLAIR
        self.backbone_name = backbone_name

        # Build single-modality encoder
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
        self.encoder.cls_head = nn.Identity()

        # Store actual embedding dimension
        self.embed_dim = actual_embed_dim

        # Our classification head (takes averaged features)
        # Added dropout to prevent overfitting
        if enc == "brainiac":
            self.cls_head = nn.Sequential(
                nn.Linear(self.embed_dim, num_classes)  # ★★ BrainIAC: simple linear head
            )
        else:
            self.cls_head = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(0.3),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(self.embed_dim, num_classes)
            )

    def forward(self, multi_modal_volume):
        """
        Args:
            multi_modal_volume: (B, 2, D, H, W)

        Returns:
            logits: (B, num_classes)
        """
        B, num_mods, D, H, W = multi_modal_volume.shape

        assert num_mods == self.num_modalities, \
            f"Expected {self.num_modalities} modalities, got {num_mods}"

        # Extract features for each modality independently
        modality_features = []

        for i in range(self.num_modalities):
            # Get single modality: (B, 1, D, H, W)
            single_modality = multi_modal_volume[:, i:i+1, :, :, :]

            # Pass through encoder to get features
            feat = self._extract_features(single_modality)  # (B, embed_dim)

            modality_features.append(feat)

        # Stack features: (B, 2, embed_dim)
        modality_features = torch.stack(modality_features, dim=1)

        # Average across modalities: (B, embed_dim)
        averaged_features = modality_features.mean(dim=1)

        # Classification
        return self.cls_head(averaged_features)

    def _extract_features(self, volume):
        """
        Extract features from encoder without classification.

        Args:
            volume: (B, 1, D, H, W)

        Returns:
            features: (B, embed_dim)
        """
        B, C, D, H, W = volume.shape

        # Check if using comparison encoder (BrainMVP/bm_mae)
        from networks.SliceStudent_comparisons import SliceStudent_comparisons

        if isinstance(self.encoder, SliceStudent_comparisons):
            # SliceStudent_comparisons processes 3D volume directly

            # Temporarily save and replace cls_head
            original_cls_head = self.encoder.cls_head
            original_use_reg_head = getattr(self.encoder, 'use_reg_head', False)

            # Replace with identity to get raw features
            self.encoder.cls_head = nn.Identity()
            if hasattr(self.encoder, 'use_reg_head'):
                self.encoder.use_reg_head = False

            # Get features (with teacher frozen, so use no_grad)
            with torch.no_grad():
                volume_feat = self.encoder(volume)  # (B, embed_dim)

            # Restore original head
            self.encoder.cls_head = original_cls_head
            if hasattr(self.encoder, 'use_reg_head'):
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
            cls_tokens = out["x_norm_clstoken"]  # (B*N, embed_dim)

            # Reshape to (B, N, embed_dim)
            cls_tokens = cls_tokens.reshape(B, self.n_slices, -1)

            # Mean pooling across slices
            volume_feat = cls_tokens.mean(dim=1)  # (B, embed_dim)

            return volume_feat

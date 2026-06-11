import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import your dinov3 backbone and LoRA
from networks.dinov3.dinov3.models.vision_transformer import vit_base
from networks.dinov3.dinov3.models.stage2.lora import LoRA


def add_lora_to_vit(model, r=4):
    import math
    for blk in model.blocks:
        old_qkv = blk.attn.qkv
        dim = old_qkv.in_features

        w_a_q = nn.Linear(dim, r, bias=False)
        w_b_q = nn.Linear(r, dim, bias=False)
        w_a_v = nn.Linear(dim, r, bias=False)
        w_b_v = nn.Linear(r, dim, bias=False)

        nn.init.kaiming_uniform_(w_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(w_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(w_b_q.weight)
        nn.init.zeros_(w_b_v.weight)

        lora_qkv = LoRA(old_qkv, w_a_q, w_b_q, w_a_v, w_b_v)
        lora_qkv.qkv.weight.data.copy_(old_qkv.weight.data)

        if old_qkv.bias is not None:
            lora_qkv.qkv.bias.data.copy_(old_qkv.bias.data)

        blk.attn.qkv = lora_qkv

    return model

# ====== Transformer encoder for slice-level modeling ======
class SliceTransformer(nn.Module):
    def __init__(self, embed_dim=768, depth=2, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x: (B, N_slices, 768)
        return self.encoder(x)  # (B, N_slices, 768)


# ====== Attention Pooling ======
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.att = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (B, N, 768)
        w = self.att(x)  # (B, N, 1)
        w = torch.softmax(w, dim=1)  # attention weights
        return (w * x).sum(dim=1)  # (B, 768)



# ====== 4. universal classifier head (supports K classes) ======
class SliceStudent(nn.Module):
    """
    Downstream version (CE Loss, K-class classification)
    """
    def __init__(self, 
                 n_slices=32, 
                 lora_rank=4, 
                 ckpt_path=None,
                 embed_dim=768,
                 num_classes=3,
                 frozen = True,
                 backbone_name='dinov3',
                 reg_head=False,
                 drop_path_rate=0.2):       # ★★ add universal K classes
        super().__init__()
        self.n_slices = n_slices
        self.use_reg_head = reg_head
        # === build backbone ===
        self.student = vit_base(
            drop_path_rate=drop_path_rate,
            layerscale_init=1e-5,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True
        )

        self.load_backbone_checkpoint(
            ckpt_path=ckpt_path,
            backbone_name=backbone_name
        )

        # === slice-level Transformer ===
        self.slice_transformer = SliceTransformer(embed_dim=embed_dim, depth=2, num_heads=8)
        self.att_pool = AttentionPooling(embed_dim=embed_dim)

        # === universal classifier head ===
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes)  # ★★ replace 1 → K classes
        )
        if frozen:
            for p in self.student.parameters():
                p.requires_grad = False

        self.reg_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)   # ★ 关键：1 维
        )
        # self.reg_head = nn.Linear(embed_dim, 1)

    def forward(self, volume):
        B, C, D, H, W = volume.shape

        # sample N slices
        idx = torch.linspace(0, D - 1, self.n_slices).long().to(volume.device)
        slices = volume[:, :, idx, :, :]       
        slices = slices.permute(0, 2, 1, 3, 4)
        slices = slices.reshape(B * self.n_slices, 1, H, W)
        slices = slices.repeat(1, 3, 1, 1)       # DINO expects 3ch

        # # 2D ViT features
        # out = self.student.forward_features(slices)
        # patch_tokens = out["x_norm_patchtokens"]  # (BN, 14*14, 768)

        # token_number = patch_tokens.shape[1]
        # patch_tokens = patch_tokens.reshape(B, self.n_slices, token_number, 768)
        # tokens = patch_tokens.reshape(B, self.n_slices * token_number, 768)

        # # transformer + pooling
        # tokens = self.slice_transformer(tokens)
        # vol_feat = self.att_pool(tokens)

        # # logits (B, K)
        # return self.cls_head(vol_feat)
        # ===== 2. ViT forward (per slice) =====
        out = self.student.forward_features(slices)

        # === (key change) grab CLS token only ===
        cls_tokens = out["x_norm_clstoken"]             # (B*N, 768)

        # reshape to (B, N, 768)
        cls_tokens = cls_tokens.reshape(B, self.n_slices, -1)

        # ===== 3. Mean pooling across slices =====
        volume_feat = cls_tokens.mean(dim=1)            # (B, 768)

        # ===== 4. classifier =====
        if self.use_reg_head is True:
            return self.reg_head(volume_feat)
        else:
            return self.cls_head(volume_feat)
    def load_backbone_checkpoint(self, ckpt_path: str, backbone_name: str):
        """
        Load pretrained checkpoint into self.student backbone.

        Supports:
        - dinov3
        - meddinov3
        - brainiac

        Assumptions:
        - self.student is already constructed
        - ckpt contains DINO-style keys (teacher / model / pure state_dict)
        """

        if ckpt_path is None:
            return

        print(f"[SliceStudent] Loading checkpoint: {ckpt_path}")
        print(f"[SliceStudent] Backbone type: {backbone_name}")

        # --------------------------------------------------
        # Choose safe loading strategy
        # --------------------------------------------------
        if backbone_name == "dinov3":
            chkpt = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=True
            )
        elif backbone_name in ["meddinov3", "brainiac"]:
            chkpt = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False
            )
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        # --------------------------------------------------
        # Extract state_dict
        # --------------------------------------------------
        if isinstance(chkpt, dict):
            if "teacher" in chkpt:
                state_dict = chkpt["teacher"]
            elif "model" in chkpt:
                state_dict = chkpt["model"]
            else:
                state_dict = chkpt
        else:
            raise RuntimeError("Checkpoint format not understood")

        # --------------------------------------------------
        # Clean keys: remove heads / ibot / prefix
        # --------------------------------------------------
        cleaned_state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if (
                "ibot" not in k
                and "dino_head" not in k
                and not k.startswith("head.")
            )
        }

        # --------------------------------------------------
        # Load into backbone
        # --------------------------------------------------
        msg = self.student.load_state_dict(cleaned_state_dict, strict=False)
        print("[SliceStudent] load_state_dict msg:", msg)


class SliceStudent_attn_pooling(SliceStudent):
    """
    Stage-2 downstream version:
    - Reuse everything from SliceStudent except forward()
    - Replace mean pooling with Stage-2 slice-transformer + attention pooling
    """
    def __init__(self, 
                 n_slices=32, 
                 lora_rank=4, 
                 ckpt_path=None,
                 embed_dim=768,
                 num_classes=3,
                 frozen=True):
        
        super().__init__(
            n_slices=n_slices,
            lora_rank=lora_rank,
            ckpt_path=ckpt_path,
            embed_dim=embed_dim,
            num_classes=num_classes,
            frozen=frozen
        )

        # --- Stage-2 extra blocks (override parent) ---
        self.slice_attn1 = SliceAttention(embed_dim, 8)
        self.slice_ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )
        self.slice_attn2 = SliceAttention(embed_dim, 8)
        self.slice_ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        # attention pooling replaces mean pooling
        self.attn_pool = nn.Linear(embed_dim, 1)

        # retrainable head
        self.cls_head = nn.Linear(embed_dim, num_classes)

    # ==========================================================
    # override forward ONLY
    # ==========================================================
    def forward(self, volume):
        B, C, D, H, W = volume.shape

        # sample slices (same as parent)
        idx = torch.linspace(0, D - 1, self.n_slices).long().to(volume.device)
        slices = volume[:, :, idx, :, :]       
        slices = slices.permute(0, 2, 1, 3, 4)
        slices = slices.reshape(B * self.n_slices, 1, H, W)
        slices = slices.repeat(1, 3, 1, 1)

        # ViT forward
        out = self.student.forward_features(slices)
        s = out["x_norm_clstoken"].reshape(B, self.n_slices, -1)

        # Stage-2 transformer
        s = s + self.slice_attn1(s)
        s = s + self.slice_ffn1(s)
        s = s + self.slice_attn2(s)
        s = s + self.slice_ffn2(s)

        # attention pooling
        w = torch.softmax(self.attn_pool(s), dim=1)  # (B,N,1)
        g = (w * s).sum(dim=1)                       # (B,768)

        # classifier
        return self.cls_head(g)



from networks.dinov3.dinov3.models.vision_transformer import vit_base
from networks.dinov3.dinov3.models.stage2.lora import LoRA


# ---------------------------------------------------------
# LoRA 插入
# ---------------------------------------------------------
def add_lora_to_vit(model, r=4):
    import math
    for blk in model.blocks:
        old_qkv = blk.attn.qkv
        dim = old_qkv.in_features

        w_a_q = nn.Linear(dim, r, bias=False)
        w_b_q = nn.Linear(r, dim, bias=False)
        w_a_v = nn.Linear(dim, r, bias=False)
        w_b_v = nn.Linear(r, dim, bias=False)

        nn.init.kaiming_uniform_(w_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(w_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(w_b_q.weight)
        nn.init.zeros_(w_b_v.weight)

        lora_qkv = LoRA(old_qkv, w_a_q, w_b_q, w_a_v, w_b_v)

        lora_qkv.qkv.weight.data.copy_(old_qkv.weight.data)
        if old_qkv.bias is not None:
            lora_qkv.qkv.bias.data.copy_(old_qkv.bias.data)

        blk.attn.qkv = lora_qkv

    return model


# ---------------------------------------------------------
# RoPE
# ---------------------------------------------------------
def apply_rope_1d(q, k):
    B, H, N, Dh = q.shape
    device = q.device

    pos = torch.arange(N, device=device).unsqueeze(1)
    theta = 1.0 / (10000 ** (torch.arange(0, Dh, 2, device=device) / Dh))
    angle = pos * theta

    sin = angle.sin()[None, None, :, :]
    cos = angle.cos()[None, None, :, :]

    def rotate(x):
        xe, xo = x[..., 0::2], x[..., 1::2]
        return torch.cat([xe * cos - xo * sin, xe * sin + xo * cos], dim=-1)

    return rotate(q), rotate(k)


# ---------------------------------------------------------
# SliceAttention (RPB + RoPE)
# ---------------------------------------------------------
class SliceAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8, max_slices=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.rpb = nn.Parameter(torch.zeros(2 * max_slices - 1, num_heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope_1d(q, k)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        idx = torch.arange(N, device=x.device)
        offset = idx[None, :] - idx[:, None]
        bias_index = offset + (self.rpb.shape[0] // 2)
        rpb = self.rpb[bias_index].permute(2, 0, 1)
        attn = attn + rpb.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(out)




class SliceStudent_stage2_cls_head(nn.Module):
    def __init__(self,
                 n_slices=32,
                 lora_rank=8,
                 ckpt_path=None,
                 embed_dim=768,
                 num_classes=3):
        super().__init__()
        self.n_slices = n_slices

        # -------------------------
        # student backbone
        # -------------------------
        self.student = vit_base(
            drop_path_rate=0.2,
            layerscale_init=1e-5,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )
        self.student = add_lora_to_vit(self.student, r=lora_rank)

        # -------------------------
        # slice transformer structure (Stage-2)
        # -------------------------
        self.slice_attn1 = SliceAttention(embed_dim, 8)
        self.slice_ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        self.slice_attn2 = SliceAttention(embed_dim, 8)
        self.slice_ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        # -------------------------
        # Stage-2 pooling + head
        # -------------------------
        self.attn_pool = nn.Linear(embed_dim, 1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # -------------------------
        # load Stage-2 checkpoint
        # -------------------------
        if ckpt_path is not None:
            self.load_stage2_ckpt(ckpt_path)

        # -------------------------
        # final classifier head (only part that is trainable)
        # -------------------------
        self.cls_head = nn.Linear(embed_dim, num_classes)

        # -------------------------
        # freeze encoder (most important!)
        # -------------------------
        self.freeze_encoder()

    def freeze_encoder(self):

        # 1) freeze EVERYTHING first
        for p in self.parameters():
            p.requires_grad = False

        # # ------------------------------------------------------
        # # 2) UNFREEZE attn_pool (trainable)
        # # ------------------------------------------------------
        # for name, p in self.attn_pool.named_parameters():
        #     p.requires_grad = True

        # # ------------------------------------------------------
        # # 3) UNFREEZE mlp_head (trainable)
        # # ------------------------------------------------------
        # for name, p in self.mlp_head.named_parameters():
        #     p.requires_grad = True
            # unfreeze proj_head
        # ------------------------------------------------------
        # 4) UNFREEZE cls_head (trainable)
        # ------------------------------------------------------
        for name, p in self.cls_head.named_parameters():
            p.requires_grad = True

        # ------------------------------------------------------
        # print summary
        # ------------------------------------------------------
        print("\n[Stage2-Downstream] Encoder frozen except cls_head")
        print("[Trainable parameters]:")
        for name, p in self.named_parameters():
            if p.requires_grad:
                print("  🔸", name)

    # ------------------------------------------------------
    # load stage-2 checkpoint (filtered)
    # ------------------------------------------------------
    def load_stage2_ckpt(self, ckpt_path):
        print(f"[Stage2-Downstream] Loading Stage-2 ckpt: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")

        allow = [
            "student.",
            "slice_attn1.",
            "slice_ffn1.",
            "slice_attn2.",
            "slice_ffn2.",
            "attn_pool.",
            "mlp_head.",
        ]

        filtered = {k: v for k, v in sd.items()
                    if any(k.startswith(a) for a in allow)}

        missing, unexpected = self.load_state_dict(filtered, strict=False)

        print(f"  Missing: {len(missing)}")
        print(f"  Unexpected: {len(unexpected)}")

    # ------------------------------------------------------
    # forward path
    # ------------------------------------------------------
    def forward(self, volume):
        B, C, D, H, W = volume.shape

        idx = torch.linspace(0, D - 1, self.n_slices).long().to(volume.device)
        slices = volume[:, :, idx, :, :]
        slices = slices.permute(0, 2, 1, 3, 4)
        slices = slices.reshape(B * self.n_slices, 1, H, W).repeat(1, 3, 1, 1)

        out = self.student.forward_features(slices)
        s = out["x_norm_clstoken"].reshape(B, self.n_slices, -1)

        s = s + self.slice_attn1(s)
        s = s + self.slice_ffn1(s)
        s = s + self.slice_attn2(s)
        s = s + self.slice_ffn2(s)

        w = torch.softmax(self.attn_pool(s), dim=1)
        g = (w * s).sum(dim=1)

        feat = self.mlp_head(g)
        logit = self.cls_head(feat)
        return logit
# =========================================================
#   SliceStudent_stage2 — FINAL DOWNSTREAM VERSION
# =========================================================
class SliceStudent_stage2_projection_head(nn.Module):
    def __init__(self,
                 n_slices=32,
                 lora_rank=8,
                 ckpt_path=None,
                 embed_dim=768,
                 num_classes=3):
        super().__init__()
        self.n_slices = n_slices

        # -------------------------
        # student backbone
        # -------------------------
        self.student = vit_base(
            drop_path_rate=0.2,
            layerscale_init=1e-5,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )
        self.student = add_lora_to_vit(self.student, r=lora_rank)

        # -------------------------
        # slice transformer structure (Stage-2)
        # -------------------------
        self.slice_attn1 = SliceAttention(embed_dim, 8)
        self.slice_ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        self.slice_attn2 = SliceAttention(embed_dim, 8)
        self.slice_ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        # -------------------------
        # Stage-2 pooling + head
        # -------------------------
        self.attn_pool = nn.Linear(embed_dim, 1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.proj_head = nn.Linear(embed_dim, embed_dim)
        # -------------------------
        # load Stage-2 checkpoint
        # -------------------------
        if ckpt_path is not None:
            self.load_stage2_ckpt(ckpt_path)

        # -------------------------
        # final classifier head (only part that is trainable)
        # -------------------------
        self.cls_head = nn.Linear(embed_dim, num_classes)

        # -------------------------
        # freeze encoder (most important!)
        # -------------------------
        self.freeze_encoder()

    def freeze_encoder(self):

        # 1) freeze EVERYTHING first
        for p in self.parameters():
            p.requires_grad = False

        # # ------------------------------------------------------
        # # 2) UNFREEZE attn_pool (trainable)
        # # ------------------------------------------------------
        # for name, p in self.attn_pool.named_parameters():
        #     p.requires_grad = True

        # # ------------------------------------------------------
        # # 3) UNFREEZE mlp_head (trainable)
        # # ------------------------------------------------------
        # for name, p in self.mlp_head.named_parameters():
        #     p.requires_grad = True
            # unfreeze proj_head
        for name, p in self.proj_head.named_parameters():
            p.requires_grad = True
        # ------------------------------------------------------
        # 4) UNFREEZE cls_head (trainable)
        # ------------------------------------------------------
        for name, p in self.cls_head.named_parameters():
            p.requires_grad = True

        # ------------------------------------------------------
        # print summary
        # ------------------------------------------------------
        print("\n[Stage2-Downstream] Encoder frozen except attn_pool + mlp_head + cls_head")
        print("[Trainable parameters]:")
        for name, p in self.named_parameters():
            if p.requires_grad:
                print("  🔸", name)

    # ------------------------------------------------------
    # load stage-2 checkpoint (filtered)
    # ------------------------------------------------------
    def load_stage2_ckpt(self, ckpt_path):
        print(f"[Stage2-Downstream] Loading Stage-2 ckpt: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")

        allow = [
            "student.",
            "slice_attn1.",
            "slice_ffn1.",
            "slice_attn2.",
            "slice_ffn2.",
            "attn_pool.",
            "mlp_head.",
        ]

        filtered = {k: v for k, v in sd.items()
                    if any(k.startswith(a) for a in allow)}

        missing, unexpected = self.load_state_dict(filtered, strict=False)

        print(f"  Missing: {len(missing)}")
        print(f"  Unexpected: {len(unexpected)}")

    # ------------------------------------------------------
    # forward path
    # ------------------------------------------------------
    def forward(self, volume):
        B, C, D, H, W = volume.shape

        idx = torch.linspace(0, D - 1, self.n_slices).long().to(volume.device)
        slices = volume[:, :, idx, :, :]
        slices = slices.permute(0, 2, 1, 3, 4)
        slices = slices.reshape(B * self.n_slices, 1, H, W).repeat(1, 3, 1, 1)

        out = self.student.forward_features(slices)
        s = out["x_norm_clstoken"].reshape(B, self.n_slices, -1)

        s = s + self.slice_attn1(s)
        s = s + self.slice_ffn1(s)
        s = s + self.slice_attn2(s)
        s = s + self.slice_ffn2(s)

        w = torch.softmax(self.attn_pool(s), dim=1)
        g = (w * s).sum(dim=1)

        feat = self.mlp_head(g)
        feat = self.proj_head(feat)  # trainable
        logit = self.cls_head(feat)
        return logit



class SliceStudent_stage2_mlp(nn.Module):
    def __init__(self,
                 n_slices=32,
                 lora_rank=8,
                 ckpt_path=None,
                 embed_dim=768,
                 num_classes=3):
        super().__init__()
        self.n_slices = n_slices

        # -------------------------
        # student backbone
        # -------------------------
        self.student = vit_base(
            drop_path_rate=0.2,
            layerscale_init=1e-5,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )
        self.student = add_lora_to_vit(self.student, r=lora_rank)

        # -------------------------
        # slice transformer structure (Stage-2)
        # -------------------------
        self.slice_attn1 = SliceAttention(embed_dim, 8)
        self.slice_ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        self.slice_attn2 = SliceAttention(embed_dim, 8)
        self.slice_ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        # -------------------------
        # Stage-2 pooling + head
        # -------------------------
        self.attn_pool = nn.Linear(embed_dim, 1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # -------------------------
        # load Stage-2 checkpoint
        # -------------------------
        if ckpt_path is not None:
            self.load_stage2_ckpt(ckpt_path)

        # -------------------------
        # final classifier head (only part that is trainable)
        # -------------------------
        self.cls_head = nn.Linear(embed_dim, num_classes)

        # -------------------------
        # freeze encoder (most important!)
        # -------------------------
        self.freeze_encoder()

    def freeze_encoder(self):

        # 1) freeze EVERYTHING first
        for p in self.parameters():
            p.requires_grad = False

        # # ------------------------------------------------------
        # # 2) UNFREEZE attn_pool (trainable)
        # # ------------------------------------------------------
        # for name, p in self.attn_pool.named_parameters():
        #     p.requires_grad = True

        # ------------------------------------------------------
        # 3) UNFREEZE mlp_head (trainable)
        # ------------------------------------------------------
        for name, p in self.mlp_head.named_parameters():
            p.requires_grad = True
        # ------------------------------------------------------
        # 4) UNFREEZE cls_head (trainable)
        # ------------------------------------------------------
        for name, p in self.cls_head.named_parameters():
            p.requires_grad = True

        # ------------------------------------------------------
        # print summary
        # ------------------------------------------------------
        print("\n[Stage2-Downstream] Encoder frozen except cls_head")
        print("[Trainable parameters]:")
        for name, p in self.named_parameters():
            if p.requires_grad:
                print("  🔸", name)

    # ------------------------------------------------------
    # load stage-2 checkpoint (filtered)
    # ------------------------------------------------------
    def load_stage2_ckpt(self, ckpt_path):
        print(f"[Stage2-Downstream] Loading Stage-2 ckpt: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")

        allow = [
            "student.",
            "slice_attn1.",
            "slice_ffn1.",
            "slice_attn2.",
            "slice_ffn2.",
            "attn_pool."
        ]

        filtered = {k: v for k, v in sd.items()
                    if any(k.startswith(a) for a in allow)}

        missing, unexpected = self.load_state_dict(filtered, strict=False)

        print(f"  Missing: {len(missing)}")
        print(f"  Unexpected: {len(unexpected)}")

    # ------------------------------------------------------
    # forward path
    # ------------------------------------------------------
    def forward(self, volume):
        B, C, D, H, W = volume.shape

        idx = torch.linspace(0, D - 1, self.n_slices).long().to(volume.device)
        slices = volume[:, :, idx, :, :]
        slices = slices.permute(0, 2, 1, 3, 4)
        slices = slices.reshape(B * self.n_slices, 1, H, W).repeat(1, 3, 1, 1)

        out = self.student.forward_features(slices)
        s = out["x_norm_clstoken"].reshape(B, self.n_slices, -1)

        s = s + self.slice_attn1(s)
        s = s + self.slice_ffn1(s)
        s = s + self.slice_attn2(s)
        s = s + self.slice_ffn2(s)

        w = torch.softmax(self.attn_pool(s), dim=1)
        g = (w * s).sum(dim=1)

        feat = self.mlp_head(g)
        logit = self.cls_head(feat)
        return logit
# =========================================================
#   SliceStudent_stage2 — FINAL DOWNSTREAM VERSION
# =========================================================
class SliceStudent_stage2_mean(nn.Module):
    def __init__(self,
                 n_slices=32,
                 lora_rank=8,
                 ckpt_path=None,
                 embed_dim=768,
                 num_classes=3):
        super().__init__()
        self.n_slices = n_slices

        # -------------------------
        # backbone
        # -------------------------
        self.student = vit_base(
            drop_path_rate=0.2,
            layerscale_init=1e-5,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )

        # -------------------------
        # stage-2 slice transformer
        # -------------------------
        self.slice_attn1 = SliceAttention(embed_dim, 8)
        self.slice_ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        self.slice_attn2 = SliceAttention(embed_dim, 8)
        self.slice_ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        # -------------------------
        # ⚠️ 删除 attn_pool & mlp_head
        # -------------------------
        # self.attn_pool = nn.Linear(embed_dim, 1)
        # self.mlp_head = ...

        # final classifier
        self.cls_head = nn.Linear(embed_dim, num_classes)

        # -------------------------
        # freeze encoder
        # -------------------------
        self.freeze_encoder()

        # load ckpt AFTER freezing
        if ckpt_path is not None:
            self.load_stage2_ckpt(ckpt_path)

    # ------------------------------------------------------
    def freeze_encoder(self):
        for p in self.parameters():
            p.requires_grad = False

        # Only classifier is trainable
        for name, p in self.cls_head.named_parameters():
            p.requires_grad = True

    # ------------------------------------------------------
    # STILL using ckpt filtering from your original version
    # ------------------------------------------------------
    def load_stage2_ckpt(self, ckpt_path):
        print(f"[Load] {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")

        allow = [
            "student.",
            "slice_attn1.",
            "slice_ffn1.",
            "slice_attn2.",
            "slice_ffn2.",
        ]

        filtered = {k: v for k, v in sd.items()
                    if any(k.startswith(a) for a in allow)}

        self.load_state_dict(filtered, strict=False)

    # ------------------------------------------------------
    # FORWARD = Stage-1 logic (mean pooling)
    # ------------------------------------------------------
    def forward(self, volume):
        B, C, D, H, W = volume.shape

        idx = torch.linspace(0, D - 1, self.n_slices).long().to(volume.device)
        slices = volume[:, :, idx, :, :]
        slices = slices.permute(0, 2, 1, 3, 4)
        slices = slices.reshape(B * self.n_slices, 1, H, W).repeat(1, 3, 1, 1)

        # slice-level ViT
        out = self.student.forward_features(slices)
        s = out["x_norm_clstoken"].reshape(B, self.n_slices, -1)

        # stage-2 slice transformer
        s = s + self.slice_attn1(s)
        s = s + self.slice_ffn1(s)
        s = s + self.slice_attn2(s)
        s = s + self.slice_ffn2(s)

        # --------------------------------------------
        # ⭐⭐⭐ Replace attn_pool → simple mean pooling
        # --------------------------------------------
        g = s.mean(dim=1)    # (B, 768)

        # classifier
        return self.cls_head(g)


class SliceStudent_stage2_attn_pooling(nn.Module):
    def __init__(self,
                 n_slices=32,
                 lora_rank=8,
                 ckpt_path=None,
                 embed_dim=768,
                 num_classes=3):
        super().__init__()
        self.n_slices = n_slices

        # -------------------------
        # backbone
        # -------------------------
        self.student = vit_base(
            drop_path_rate=0.2,
            layerscale_init=1e-5,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )
        self.att_pool = nn.Linear(embed_dim, 1) 
        # -------------------------
        # stage-2 slice transformer
        # -------------------------
        self.slice_attn1 = SliceAttention(embed_dim, 8)
        self.slice_ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        self.slice_attn2 = SliceAttention(embed_dim, 8)
        self.slice_ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        # -------------------------
        # ⚠️ 删除 attn_pool & mlp_head
        # -------------------------
        # self.attn_pool = nn.Linear(embed_dim, 1)
        # self.mlp_head = ...

        # final classifier
        self.cls_head = nn.Linear(embed_dim, num_classes)

        # -------------------------
        # freeze encoder
        # -------------------------
        self.freeze_encoder()

        # load ckpt AFTER freezing
        if ckpt_path is not None:
            self.load_stage2_ckpt(ckpt_path)

    # ------------------------------------------------------
    def freeze_encoder(self):
        for p in self.parameters():
            p.requires_grad = False

        # Only classifier is trainable
        for name, p in self.cls_head.named_parameters():
            p.requires_grad = True
        for p in self.att_pool.parameters():
            p.requires_grad = True


    # ------------------------------------------------------
    # STILL using ckpt filtering from your original version
    # ------------------------------------------------------
    def load_stage2_ckpt(self, ckpt_path):
        print(f"[Load] {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")

        allow = [
            "student.",
            "slice_attn1.",
            "slice_ffn1.",
            "slice_attn2.",
            "slice_ffn2.",
        ]

        filtered = {k: v for k, v in sd.items()
                    if any(k.startswith(a) for a in allow)}

        self.load_state_dict(filtered, strict=False)

    # ------------------------------------------------------
    # FORWARD = Stage-1 logic (mean pooling)
    # ------------------------------------------------------
    def forward(self, volume):
        B, C, D, H, W = volume.shape

        idx = torch.linspace(0, D - 1, self.n_slices).long().to(volume.device)
        slices = volume[:, :, idx, :, :]
        slices = slices.permute(0, 2, 1, 3, 4)
        slices = slices.reshape(B * self.n_slices, 1, H, W).repeat(1, 3, 1, 1)

        # slice-level ViT
        out = self.student.forward_features(slices)
        s = out["x_norm_clstoken"].reshape(B, self.n_slices, -1)

        # stage-2 slice transformer
        s = s + self.slice_attn1(s)
        s = s + self.slice_ffn1(s)
        s = s + self.slice_attn2(s)
        s = s + self.slice_ffn2(s)

        # --------------------------------------------
        # ⭐⭐⭐ Replace attn_pool → simple mean pooling
        # --------------------------------------------
        w = torch.softmax(self.att_pool(s), dim=1)
        g = (w * s).sum(dim=1)  
        # classifier
        return self.cls_head(g)


class SliceStudent_stage1concate2_mean(nn.Module):
    def __init__(self,
                 n_slices=32,
                 lora_rank=8,
                 ckpt_path=None,
                 embed_dim=768,
                 num_classes=3):
        super().__init__()
        self.n_slices = n_slices

        # -------------------------
        # backbone
        # -------------------------
        self.student = vit_base(
            drop_path_rate=0.2,
            layerscale_init=1e-5,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )

        # -------------------------
        # stage-2 slice transformer
        # -------------------------
        self.slice_attn1 = SliceAttention(embed_dim, 8)
        self.slice_ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        self.slice_attn2 = SliceAttention(embed_dim, 8)
        self.slice_ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3072), nn.GELU(),
            nn.Linear(3072, embed_dim)
        )

        # -------------------------
        # ⚠️ 删除 attn_pool & mlp_head
        # -------------------------
        # self.attn_pool = nn.Linear(embed_dim, 1)
        # self.mlp_head = ...

        # final classifier
        self.cls_head = nn.Linear(embed_dim*2, num_classes)

        # -------------------------
        # freeze encoder
        # -------------------------
        self.freeze_encoder()

        # load ckpt AFTER freezing
        if ckpt_path is not None:
            self.load_stage2_ckpt(ckpt_path)

    # ------------------------------------------------------
    def freeze_encoder(self):
        for p in self.parameters():
            p.requires_grad = False

        # Only classifier is trainable
        for name, p in self.cls_head.named_parameters():
            p.requires_grad = True

    # ------------------------------------------------------
    # STILL using ckpt filtering from your original version
    # ------------------------------------------------------
    def load_stage2_ckpt(self, ckpt_path):
        print(f"[Load] {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")

        allow = [
            "student.",
            "slice_attn1.",
            "slice_ffn1.",
            "slice_attn2.",
            "slice_ffn2.",
        ]

        filtered = {k: v for k, v in sd.items()
                    if any(k.startswith(a) for a in allow)}

        self.load_state_dict(filtered, strict=False)

    # ------------------------------------------------------
    # FORWARD = Stage-1 logic (mean pooling)
    # ------------------------------------------------------
    def forward(self, volume):
        B, C, D, H, W = volume.shape

        idx = torch.linspace(0, D - 1, self.n_slices).long().to(volume.device)
        slices = volume[:, :, idx, :, :]
        slices = slices.permute(0, 2, 1, 3, 4)
        slices = slices.reshape(B * self.n_slices, 1, H, W).repeat(1, 3, 1, 1)

        # slice-level ViT
        out = self.student.forward_features(slices)
        s_1 = out["x_norm_clstoken"].reshape(B, self.n_slices, -1)

        # stage-2 slice transformer
        s_2 = s_1 + self.slice_attn1(s_1)
        s_2 = s_2 + self.slice_ffn1(s_2)
        s_2 = s_2 + self.slice_attn2(s_2)
        s_2 = s_2 + self.slice_ffn2(s_2)

        # --------------------------------------------
        # ⭐⭐⭐ Replace attn_pool → simple mean pooling
        # --------------------------------------------
        s_concat = torch.cat([s_1, s_2], dim=-1)  
        g = s_concat.mean(dim=1)    # (B, 768*2)

        # classifier
        return self.cls_head(g)

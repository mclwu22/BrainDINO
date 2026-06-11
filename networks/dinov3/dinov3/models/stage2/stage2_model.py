
import torch
import torch.nn as nn
from dinov3.models.vision_transformer import vit_base
import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov3.models.vision_transformer import vit_base
import math
import sys
import os
import torch
import logging

def load_teacher_unified(teacher_name, dune_base="/scr2/yz_wu/foundation/dune/models/mf", device="cuda"):
    """
    Unified teacher loader for Stage-2 pretraining.

    Args:
        teacher_name (str): one of {"bm_mae", "brainmvp", "sam_med3d", "brainiac"}.
        dune_base (str): path to the DUNE teacher module folder (contains builder.py, config.py).
        device (str): device to load the model on.

    Returns:
        torch.nn.Module: teacher model (already .eval() and .cuda()).
    """
    logger = logging.getLogger(__name__)

    print("🔍 [DEBUG] Starting to load teacher:", teacher_name)
    print(f"🔍 [DEBUG] DUNE base path: {dune_base}")
    print(f"🔍 [DEBUG] Device: {device}")

    try:
        builder_path = os.path.join(dune_base)
        if builder_path not in sys.path:
            sys.path.append(builder_path)
            print(f"📁 [DEBUG] Added {builder_path} to sys.path")

        # 导入 builder 模块
        print("🧩 [DEBUG] Importing build_teachers from dinov3.models.stage2.builder ...")
        from dinov3.models.stage2.builder import build_teachers

        print(f"🧠 [DEBUG] Building teacher '{teacher_name}' ...")
        teachers = build_teachers([teacher_name])
        print(f"✅ [DEBUG] build_teachers() returned: {list(teachers.keys())}")

        teacher = teachers[teacher_name]

        print("🚀 [DEBUG] Moving teacher to device ...")
        teacher = teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        print(f"✅ [SUCCESS] Teacher '{teacher_name}' loaded and moved to {device}")
        logger.info(f"✅ Teacher '{teacher_name}' loaded and moved to {device}")
        return teacher

    except Exception as e:
        print(f"❌ [ERROR] Failed to load teacher '{teacher_name}': {e}")
        logger.error(f"❌ Failed to load teacher '{teacher_name}': {e}")
        raise


from dinov3.models.vision_transformer import vit_base
from dinov3.models.stage2.lora import LoRA  # 复用他们的 LoRA 定义

def add_lora_to_vit(model, r=4):
    """
    在加载完预训练权重的 DINOv3 ViT 模型每个 block 的 qkv 层上插入 LoRA 模块，
    并保留原 qkv 权重。
    """
    from dinov3.models.stage2.lora import LoRA
    import torch.nn as nn, math
    device = next(model.parameters()).device
    for i, block in enumerate(model.blocks):
        old_qkv = block.attn.qkv
        dim = old_qkv.in_features

        # 构造 LoRA 权重
        w_a_q = nn.Linear(dim, r, bias=False, device=device)
        w_b_q = nn.Linear(r, dim, bias=False, device=device)
        w_a_v = nn.Linear(dim, r, bias=False, device=device)
        w_b_v = nn.Linear(r, dim, bias=False, device=device)
        nn.init.kaiming_uniform_(w_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(w_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(w_b_q.weight)
        nn.init.zeros_(w_b_v.weight)

        # 用原来的 qkv 构建 LoRA 模块
        lora_qkv = LoRA(old_qkv, w_a_q, w_b_q, w_a_v, w_b_v)

        # ✅ 保留 Stage1 的 qkv 权重
        lora_qkv.qkv.weight.data.copy_(old_qkv.weight.data)
        if old_qkv.bias is not None:
            lora_qkv.qkv.bias.data.copy_(old_qkv.bias.data)

        # 替换
        block.attn.qkv = lora_qkv

    print(f"✅ Inserted LoRA (rank={r}) after loading Stage-1 weights.")
    return model

def apply_rope_1d(q, k):
    """
    q, k: [B, H, N, Dh]
    """
    B, H, N, Dh = q.shape
    device = q.device

    # positions: 0 ... N-1  → shape [N, 1]
    pos = torch.arange(N, device=device).unsqueeze(1)

    # rotary frequencies → shape [Dh/2]
    theta = 1.0 / (10000 ** (torch.arange(0, Dh, 2, device=device) / Dh))

    # angle matrix → [N, Dh/2]
    angles = pos * theta

    sin = angles.sin()[None, None, :, :]    # [1,1,N,Dh/2]
    cos = angles.cos()[None, None, :, :]

    # split even/odd
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]

    # rotate Q/K
    q_rot = torch.cat([q_even * cos - q_odd * sin,
                       q_even * sin + q_odd * cos], dim=-1)
    k_rot = torch.cat([k_even * cos - k_odd * sin,
                       k_even * sin + k_odd * cos], dim=-1)

    return q_rot, k_rot


class Stage2Model(nn.Module):
    def __init__(self, teacher, device="cuda", freeze_student=True,use_lora=False, lora_rank=4,depth_token_mode="depth_token"):
        super().__init__()
        # -----------------------------
        # ✅ 初始化 student 和 teacher
        # -----------------------------
        # self.student = vit_base(
        #     patch_size=16,
        #     num_register_tokens=0,
        #     drop_path_rate=0.0,
        #     layerscale_init=1.0e-5,
        #     qkv_bias=True,
        #     proj_bias=True,
        #     ffn_bias=True,
        # ).to(device)

        self.student = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-05, n_storage_tokens=4, 
                    qkv_bias = False, mask_k_bias= True)
        
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.depth_token_mode = depth_token_mode
        # ✅ 如果需要，插入 LoRA 模块
        # if use_lora:
        #     print(f"🔹 Inserting LoRA (rank={lora_rank}) into student ViT...")
        #     self.student = add_lora_to_vit(self.student, r=lora_rank)

        self.teacher = teacher.to(device)
        self.proj = nn.Linear(768, 768, bias=False)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=768,
        #     nhead=8,
        #     dim_feedforward=3072,
        #     dropout=0.1,
        #     batch_first=True
        # )
        # self.slice_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ---- replace with RPB version ----
        self.slice_attn1 = SliceAttention(dim=768, num_heads=8, max_slices=128)
        self.slice_ffn1 = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )

        self.slice_attn2 = SliceAttention(dim=768, num_heads=8, max_slices=128)
        self.slice_ffn2 = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )
        self.depth_token = nn.Parameter(torch.randn(1, 1, 768))
        self.attn_pool = nn.Linear(768, 1)   # 每个 slice 一个权重
        self.mlp_head = nn.Sequential(       # 可选的小 MLP 聚合头
            nn.LayerNorm(768),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 768)
        )

        if freeze_student:
            self.freeze_student()
            
            # 如果用了 LoRA，就重新打开 LoRA 参数的 requires_grad
            if use_lora:
                for name, param in self.student.named_parameters():
                    if "linear_a" in name or "linear_b" in name:
                        param.requires_grad = True


    # -----------------------------
    # ✅ 单独的权重加载函数
    # -----------------------------
    # def load_student_checkpoint(self, ckpt_path):
    #     print(f"🔹 Loading DINOv3 checkpoint from: {ckpt_path}")
    #     ckpt = torch.load(ckpt_path, map_location="cpu")
    #     state_dict = ckpt.get("teacher", ckpt)

    #     cleaned = {
    #         k.replace("backbone.", ""): v
    #         for k, v in state_dict.items()
    #         if not any(s in k for s in ["ibot", "dino_head"])
    #     }

    #     missing, unexpected = self.student.load_state_dict(cleaned, strict=False)
    #     print(f"✅ Loaded student weights. Missing={missing}, Unexpected={unexpected}")
    #     print(f"✅ Loaded student weights. Missing={len(missing)}, Unexpected={len(unexpected)}")
    def load_student_checkpoint(self, ckpt_path):
        print(f"🔹 Loading DINOv3 checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("teacher", ckpt)

        # 清理前缀
        cleaned = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if not any(s in k for s in ["ibot", "dino_head"])
        }

        # ---------------------------
        # 🔍 Debug: 比较模型与checkpoint key
        # ---------------------------
        model_keys = list(self.student.state_dict().keys())
        ckpt_keys = list(cleaned.keys())

        common = sorted(set(model_keys) & set(ckpt_keys))
        only_model = sorted(set(model_keys) - set(ckpt_keys))
        only_ckpt = sorted(set(ckpt_keys) - set(model_keys))

        print(f"\n🧩 [DEBUG] Model vs Checkpoint key analysis")
        print(f"  • Total model keys: {len(model_keys)}")
        print(f"  • Total ckpt keys:  {len(ckpt_keys)}")
        print(f"  ✅ Common keys: {len(common)}")
        print(f"  ❌ Only in model (missing): {len(only_model)}")
        print(f"  ⚠️ Only in ckpt (unexpected): {len(only_ckpt)}")

        print("\n  --- Example Common keys ---")
        print("  " + "\n  ".join(common[:8]))
        print("\n  --- Example Only-in-Model keys ---")
        print("  " + "\n  ".join(only_model[:8]))
        print("\n  --- Example Only-in-Checkpoint keys ---")
        print("  " + "\n  ".join(only_ckpt[:8]))

        # 识别可能命名不同但本质相同的key（如 LoRA 的qkv层改名）
        possible_matches = [
            k for k in only_model if k.replace(".qkv.qkv", ".qkv") in only_ckpt
        ]
        if len(possible_matches) > 0:
            print("\n  🔧 Possible renamed keys (qkv vs qkv.qkv):")
            print("  " + "\n  ".join(possible_matches[:8]))

        # 保存对比结果文件
        debug_path = "student_ckpt_compare.txt"
        with open(debug_path, "w") as f:
            f.write(f"=== Common ({len(common)}) ===\n")
            f.writelines("\n".join(common))
            f.write(f"\n\n=== Only-in-Model ({len(only_model)}) ===\n")
            f.writelines("\n".join(only_model))
            f.write(f"\n\n=== Only-in-Checkpoint ({len(only_ckpt)}) ===\n")
            f.writelines("\n".join(only_ckpt))
        print(f"  💾 Saved full comparison to {debug_path}")

        # ---------------------------
        # ✅ 实际加载
        # ---------------------------
        missing, unexpected = self.student.load_state_dict(cleaned, strict=False)
        print(f"\n✅ Loaded student weights.")
        print(f"  Missing ({len(missing)}): {missing[:6]}")
        print(f"  Unexpected ({len(unexpected)}): {unexpected[:6]}")
    
        if getattr(self, "use_lora", False):
            print(f"\n🔹 Adding LoRA (rank={self.lora_rank}) after Stage1 load...")
            self.student = add_lora_to_vit(self.student, r=self.lora_rank)

    # -----------------------------
    # ✅ 冻结 student 权重
    # -----------------------------
    def freeze_student(self):
        for p in self.student.parameters():
            p.requires_grad = False
        self.student.eval()

    # -----------------------------
    # ✅ 前向计算
    # -----------------------------
    def forward(self, slices, volume,):
        """
        slices: [B, N, 1, H, W]
        volume: [B, 1, D, H, W]
        """
        if self.depth_token_mode == "cls_token":
            B, N, C, H, W = slices.shape
            slices = slices.view(B * N, C, H, W)
            if slices.shape[1] == 1:
                slices = slices.repeat(1, 3, 1, 1)  # -> [B*N, 3, H, W]

            # 1️⃣ 每个 slice 的 CLS token
            s = self.student.forward_features(slices)["x_norm_clstoken"]  # [B*N, 768]
            s = s.view(B, N, -1)                                          # [B, N, 768]

            # 2️⃣ Transformer：学习 slice 间依赖（深度关系）
            # s = self.slice_transformer(s)                                 # [B, N, 768]
            # 2️⃣ 两层 Transformer block（RoPE + RPB 已在 SliceAttention 内）
            # ---- Block 1 ----
            s = s + self.slice_attn1(s)     # [B, N, 768]
            s = s + self.slice_ffn1(s)

            # ---- Block 2 ----
            s = s + self.slice_attn2(s)
            s = s + self.slice_ffn2(s)

            # 3️⃣ Attention Pooling：聚合成 volume-level 表征
            weights = torch.softmax(self.attn_pool(s), dim=1)             # [B, N, 1]
            s = (weights * s).sum(dim=1)                                  # [B, 768]
            s = self.mlp_head(s)                                          # [B, 768]

            # 4️⃣ Teacher 输出
            with torch.no_grad():
                t = self.teacher(volume)                                  # [B, 513,768]
                t_cls = t[:, 0, :]  # CLS token

            # 5️⃣ Cosine loss
            s = F.normalize(self.proj(s), dim=-1)
            t_cls = F.normalize(t_cls, dim=-1)

            loss = 2 - 2 * (s * t_cls).sum(dim=-1).mean()
            # 返回 attention 权重方便可视化
            return loss, s, t, weights
        else:
            B, N, C, H, W = slices.shape
            slices = slices.view(B * N, C, H, W)
            if slices.shape[1] == 1:
                slices = slices.repeat(1, 3, 1, 1)  # -> [B*N, 3, H, W]

            # 1️⃣ 每个 slice 的 CLS token
            s = self.student.forward_features(slices)["x_norm_clstoken"]  # [B*N, 768]
            s = s.view(B, N, -1)                                          # [B, N, 768]

            # ✅ 插入 depth token
            depth_token = self.depth_token.expand(B, -1, -1)              # [B, 1, 768]
            s = torch.cat([depth_token, s], dim=1)                        # [B, N+1, 768]

            # 2️⃣ Transformer：学习 slice 间依赖（深度关系）
            # s = self.slice_transformer(s)                                 # [B, N, 768]
            # 2️⃣ 两层 Transformer block（RoPE + RPB 已在 SliceAttention 内）
            # ---- Block 1 ----
            s = s + self.slice_attn1(s)     # [B, N, 768]
            s = s + self.slice_ffn1(s)

            # ---- Block 2 ----
            s = s + self.slice_attn2(s)
            s = s + self.slice_ffn2(s)
            # # 3️⃣ Attention Pooling：聚合成 volume-level 表征
            # weights = torch.softmax(self.attn_pool(s), dim=1)             # [B, N, 1]
            # s = (weights * s).sum(dim=1)                                  # [B, 768]
            # s = self.mlp_head(s)                                          # [B, 768]

            # ✅ depth token 输出即 volume-level 表征
            s = s[:, 0, :]                                                # [B, 768]
            s = self.mlp_head(s)                                          # [B, 768]        

            # 4️⃣ Teacher 输出
            with torch.no_grad():
                t = self.teacher(volume)                                  # [B, 513,768]
                t_cls = t[:, 0, :]  # CLS token

            # 5️⃣ Cosine loss
            s = F.normalize(self.proj(s), dim=-1)
            t_cls = F.normalize(t_cls, dim=-1)

            loss = 2 - 2 * (s * t_cls).sum(dim=-1).mean()
            # 返回 attention 权重方便可视化
            return loss, s, t_cls, None


class SliceAttention(nn.Module):
    """
    Multi-head Attention + Relative Positional Bias (RPB) for slice fusion.
    """
    def __init__(self, dim=768, num_heads=8, max_slices=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        # qkv
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # ---- Relative Positional Bias table ----
        # offsets: [-(D-1) ... +(D-1)]
        self.max_slices = max_slices
        self.rpb = nn.Parameter(
            torch.zeros(2 * max_slices - 1, num_heads)
        )
        nn.init.trunc_normal_(self.rpb, std=0.02)

    def forward(self, x):
        """
        x: [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x)                           # [B, N, 3C]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope_1d(q, k) 

        attn = (q @ k.transpose(-2, -1)) * self.scale     # [B, H, N, N]

        # ========= relative positional bias =========
        # offset[i,j] = j - i ∈ [-(N-1)...+(N-1)]
        idx = torch.arange(N, device=x.device)
        offset = idx[None, :] - idx[:, None]              # [N, N]
        bias_index = offset + (self.max_slices - 1)       # shift to 0...(2D-2)
        # lookup → [N, N, num_heads]
        rpb = self.rpb[bias_index]                        # [N, N, H]
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)   # add to [B,H,N,N]

        # softmax
        attn = attn.softmax(dim=-1)

        out = (attn @ v)                                  # [B,H,N,head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

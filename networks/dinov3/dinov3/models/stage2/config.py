# from .dinov2.models.vision_transformer import vit_large as dinov2_vitlarge
# from .vit_master import mast3r

# MRI teacher imports - these will be imported conditionally in builder.py
# to avoid dependency issues if MRI packages are not installed

root_path = "/path/to/Med_DINOv3/Transfer/mf"
TEACHER_CFG = {
    # "dino2reg_vitlarge_14": {
    #     "loader": dinov2_vitlarge,
    #     "ckpt_path": "/scr2/yz_wu/foundation/Teachers/dinov2/vitlarge14_reg/dinov2_vitl14_reg4_pretrain.pth",
    #     "ckpt_key": "",
    #     "num_features": 1024,
    #     "image_size": 518,
    #     "patch_size": 14,
    #     "num_register_tokens": 4,
    #     "init_values": 1,
    #     "token_types": ["cls", "patch"],
    # },
    # "mast3r_vitlarge_16": {
    #     "loader": mast3r,
    #     "ckpt_path": "/scr2/yz_wu/foundation/Teachers/mast3r/vitlarge16/model.pth",
    #     "ckpt_key": None,
    #     "code_dir": "/scr2/yz_wu/foundation/Teachers/mast3r/vitlarge16/code",  # Path to the MAST3R code, downloaded by scripts/teachers/mast3r.sh
    #     "num_features": 1024,
    #     "image_size": 512,  # shortest side
    #     "patch_size": 16,
    #     "token_types": ["patch"],  # ignore the cls token
    # },
    # "multihmr_vitlarge_14_672": {
    #     "loader": dinov2_vitlarge,
    #     "ckpt_path": "/scr2/yz_wu/foundation/Teachers/multihmr/vitlarge14_672/model.pth",
    #     "ckpt_key": "model_state_dict",
    #     "num_features": 1024,
    #     "image_size": 518,  # model has 37 ** 2 + 1 positional embeddings, but fine-tuned at resolution 672
    #     "patch_size": 14,
    #     "num_register_tokens": 0,
    #     "init_values": 1,
    #     "token_types": ["patch"],  # ignore the cls token
    # },
    
    # =====================================================
    # MRI 3D Teachers Configuration
    # =====================================================
    
    "bm_mae": {
        "loader": "bm_mae_loader",  # Custom loader function
        "ckpt_path": f"{root_path}/bmmae.pth",
        "ckpt_key": "",
        "num_features": 768,
        "input_size": [128, 128, 128],
        "patch_size": [16, 16, 16],
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 1536,
        "dropout_rate": 0.0,
        "qkv_bias": True,
        "cls_token": True,
        "modalities": ["T1"],
        "teacher_type": "mri_3d",
        "token_types": ["cls", "patch"],
    },
    
    "brainiac": {
        "loader": "brainiac_loader",
        "ckpt_path": "/path/to/Med_DINOv3/pretrained_weights/BrainIAC.ckpt",
        "ckpt_key": "state_dict",
        "num_features": 768,  # ViT-B hidden size (SimCLR-ViT-B)
        "input_size": [96, 96, 96],  # BrainIAC uses 96x96x96 input
        "patch_size": [16, 16, 16],
        "hidden_size": 768,
        "mlp_dim": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "teacher_type": "mri_3d",
        "token_types": ["cls"],  # Uses CLS token from ViT
    },
    
    "brainmvp": {
        "loader": "brainmvp_loader", 
        "ckpt_path": f"{root_path}/BrainMVP_uniformer.pt",
        "ckpt_key": "",
        "num_features": 512,  # Last layer features
        "input_size": [128, 128, 128],
        "img_size": 128,
        "in_chans": 1,
        "num_classes": 0,
        "teacher_type": "mri_3d",
        "token_types": ["patch"],  # Treat as patch tokens
    },
    
    "sam_med3d": {
        "loader": "sam_med3d_loader",
        "ckpt_path": "/scr2/yz_wu/foundation/Teachers/SAM-Med3D/ckpt/sam_med3d_turbo.pth",
        "ckpt_key": "",
        "num_features": 384,
        "input_size": [128, 128, 128],
        "patch_size": 16,
        "encoder_embed_dim": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "encoder_global_attn_indexes": [2, 5, 8, 11],
        "prompt_embed_dim": 384,
        "image_size": 128,
        "vit_patch_size": 16,
        "teacher_type": "mri_3d",
        "token_types": ["patch"],  # Treat as patch tokens
    }
}

import os
import sys, os
project_root = "/scr2/yz_wu/foundation/DINOV3/dinov3"
if project_root not in sys.path:
    sys.path.append(project_root)
import logging
from collections import OrderedDict
from typing import List, Dict

import torch

from dinov3.models.stage2.config import TEACHER_CFG


logger = logging.getLogger()
root_path = "/path/to/Med_DINOv3/Transfer/Teachers"

def build_teachers(teacher_names: List[str]) -> Dict[str, torch.nn.Module]:
    teachers = OrderedDict()

    for tname in teacher_names:
        logger.info("Loading teacher '{}'".format(tname))
        teachers[tname] = _build_teacher(tname)

    return teachers


def _build_teacher(name):
    if name not in TEACHER_CFG.keys():
        raise ValueError(
            "Unsupported teacher name: {} (supported ones: {})".format(
                name, TEACHER_CFG.keys()
            )
        )

    config = TEACHER_CFG[name]
    ckpt_path = config["ckpt_path"]
    ckpt_key = config["ckpt_key"]

    if not os.path.exists(ckpt_path):
        raise ValueError("Invalid teacher model path/directory: {}".format(ckpt_path))

    # Check if it's an MRI 3D teacher
    if config.get("teacher_type") == "mri_3d":
        return _build_mri_teacher(name, config)

    if name.startswith("mast3r"):
        code_dir = TEACHER_CFG[name]["code_dir"]
        model = TEACHER_CFG[name]["loader"](code_dir, ckpt_path)

    else:
        # Teacher models which are loaded from the checkpoint files
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt_key != "" and ckpt_key in state_dict.keys():
            state_dict = state_dict[ckpt_key]

        # dinov2 models require some modifications to the state_dict
        state_dict = _update_state_dict_for_dinov2_models(name, state_dict)

        model_args = {
            "img_size": TEACHER_CFG[name]["image_size"],
            "patch_size": TEACHER_CFG[name]["patch_size"],
        }
        for key in ["init_values", "num_register_tokens"]:
            if key in TEACHER_CFG[name]:
                model_args[key] = TEACHER_CFG[name][key]
        model = TEACHER_CFG[name]["loader"](**model_args)
        model.load_state_dict(state_dict, strict=True)

    model = model.cuda()
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def _update_state_dict_for_dinov2_models(tname, state_dict):

    if tname.startswith("multihmr"):
        state_dict = {
            k.replace("backbone.encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("backbone.encoder.")
        }

    # Add the "blocks.0" prefix to the transformer block keys
    state_dict = {k.replace("blocks.", "blocks.0."): v for k, v in state_dict.items()}

    return state_dict


def _build_mri_teacher(name, config):
    """Build MRI 3D teacher models."""
    logger.info(f"Building MRI teacher: {name}")
    
    if name == "bm_mae":
        return _load_bm_mae(config)
    elif name == "brainiac":
        return _load_brainiac(config)
    elif name == "brainmvp":
        return _load_brainmvp(config)
    elif name == "sam_med3d":
        return _load_sam_med3d(config)
    else:
        raise ValueError(f"Unknown MRI teacher: {name}")


def _load_bm_mae(config):
    """Load BM-MAE teacher model."""
    try:
        # Add BM-MAE path to sys.path
        import sys
        bm_mae_path = f"{root_path}/BM-MAE"
        if bm_mae_path not in sys.path:
            sys.path.append(bm_mae_path)
            
        from bmmae.model import ViTEncoder
        from bmmae.tokenizers import MRITokenizer
        
        # Create tokenizer
        tokenizer = MRITokenizer(
            patch_size=config["patch_size"],
            img_size=config["input_size"],
            hidden_size=config["hidden_size"],
            in_channels=1,
            num_heads=config["num_heads"],
            proj_type="conv",
            pos_embed_type="sincos"
        )
        
        # Create encoder
        modalities = tuple(config["modalities"])
        tokenizers = {config["modalities"][0]: tokenizer}
        
        model = ViTEncoder(
            modalities=modalities,
            tokenizers=tokenizers,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            mlp_dim=config["mlp_dim"],
            dropout_rate=config["dropout_rate"],
            qkv_bias=config["qkv_bias"],
            cls_token=config["cls_token"],
        )
        
        # Load checkpoint
        ckpt = torch.load(config["ckpt_path"], map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt, strict=False)
        
        # Move model to GPU
        model = model.cuda()
        model.eval()
        
        logger.info(f"BM-MAE teacher loaded successfully from {config['ckpt_path']}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load BM-MAE teacher: {e}")
        raise


def _load_brainiac(config):
    """Load BrainIAC teacher model (SimCLR-ViT-B)."""
    try:
        # BrainIAC uses MONAI ViT
        from monai.networks.nets import ViT
        import torch

        # Create ViT backbone (same architecture as BrainIAC official code)
        model = ViT(
            in_channels=1,
            img_size=tuple(config["input_size"]),  # (96, 96, 96)
            patch_size=tuple(config["patch_size"]),  # (16, 16, 16)
            hidden_size=config["hidden_size"],  # 768
            mlp_dim=config["mlp_dim"],  # 3072
            num_layers=config["num_layers"],  # 12
            num_heads=config["num_heads"],  # 12
            proj_type="conv",  # Convolutional patch projection (renamed from pos_embed in MONAI >= 1.3)
            classification=False,  # We only want features, not classification
            dropout_rate=0.0,
        )

        # Load checkpoint
        checkpoint = torch.load(config["ckpt_path"], map_location="cpu", weights_only=False)
        state_dict = checkpoint.get(config["ckpt_key"], checkpoint)

        # Extract only backbone weights (remove "backbone." prefix)
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                new_key = key[9:]  # Remove "backbone." prefix (len = 9)
                backbone_state_dict[new_key] = value

        # Load the backbone weights
        missing_keys, unexpected_keys = model.load_state_dict(backbone_state_dict, strict=False)

        if missing_keys:
            logger.warning(f"BrainIAC missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"BrainIAC unexpected keys: {unexpected_keys[:5]}...")

        logger.info(f"BrainIAC teacher (SimCLR-ViT-B) loaded successfully from {config['ckpt_path']}")
        logger.info(f"  Loaded {len(backbone_state_dict)} backbone weights")

        return model

    except Exception as e:
        logger.error(f"Failed to load BrainIAC teacher: {e}")
        import traceback
        traceback.print_exc()
        raise


def _load_brainmvp(config):
    """Load BrainMVP teacher model with proper import isolation."""
    import sys
    import os
    import importlib.util
    
    brainmvp_path = f"{root_path}/BrainMVP"

    models_path = os.path.join(brainmvp_path, "models")
    utils_path  = os.path.join(brainmvp_path, "mvp_utils")

    # --- Insert paths cleanly ---
    sys.path.insert(0, models_path)
    sys.path.insert(0, utils_path)
    sys.path.insert(0, brainmvp_path)
    # --- Clean import ---
    from Uniformer import uniformer_small

    # --- Build model ---
    model = uniformer_small(
        img_size=config["img_size"],
        in_chans=config["in_chans"],
        num_classes=config["num_classes"],
    )

    # --- Load checkpoint ---
    ckpt = torch.load(config["ckpt_path"], map_location="cpu")
    ckpt = {k: v for k, v in ckpt.items() if "head" not in k}
    model.load_state_dict(ckpt, strict=False)

    # --- GPU ---
    model = model.cuda().eval()
    return model


def _load_sam_med3d(config):
    """Load SAM-Med3D teacher model."""
    try:
        # Add SAM-Med3D path to sys.path
        import sys
        sam_med3d_path = f"{root_path}/SAM-Med3D/MedIM"
        if sam_med3d_path not in sys.path:
            sys.path.append(sam_med3d_path)
            
        from medim import create_model
        
        # Create SAM-Med3D model using factory
        model = create_model(
            "SAM-Med3D",
            pretrained=True,
            checkpoint_path=config["ckpt_path"]
        )
        
        # Move model to GPU
        model = model.cuda()
        model.eval()
        
        logger.info(f"SAM-Med3D teacher loaded successfully from {config['ckpt_path']}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load SAM-Med3D teacher: {e}")
        raise


def _test_teachers():
    """
    Load all teachers and test if they can be loaded successfully.
    """

    logging.basicConfig(level=logging.INFO)

    for tname in TEACHER_CFG.keys():
        logger.info("Testing teacher '{}'".format(tname))
        _ = _build_teacher(tname)
        logger.info(" - Teacher '{}' loaded successfully".format(tname))


if __name__ == "__main__":
    _test_teachers()

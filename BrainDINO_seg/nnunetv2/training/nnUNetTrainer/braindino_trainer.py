import os
import random

import torch
from torch import nn

from dinov3.models.primus import Primus_Multiscale_conv
from dinov3.models.vision_transformer import vit_base
from nnunetv2.paths import nnUNet_results
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.nnUNetTrainer.dinov3_base_trainer import dinov3Trainer


class BrainDINO_Primus_Multiscale_HighRes_Trainer(dinov3Trainer):
    """
    BrainDINO segmentation trainer used for the released brain-tumor experiments.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.initial_lr = 3e-4
        self.vit_lr = 3e-5
        self.weight_decay = 5e-2
        self.vit_weight_decay = 5e-2
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.warmup_epochs = 0
        self.num_epochs = 300
        self.current_epoch = 0
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(*args, **kwargs) -> nn.Module:
        """
        Supports both nnU-Net trainer call signatures:
        - old: (arch_class_name, arch_init_kwargs, arch_init_kwargs_req_import, ...)
        - new: (patch_size, arch_class_name, arch_init_kwargs, arch_init_kwargs_req_import, ...)
        """
        if len(args) < 5:
            raise TypeError(
                f"Unexpected build_network_architecture args: {len(args)}. "
                "Expected old (5/6) or new (6/7 with patch_size) signature."
            )

        if isinstance(args[0], str):
            num_output_channels = args[4]
        else:
            num_output_channels = args[5]

        ckpt_path = os.environ.get("BRAINDINO_PRETRAINED_CKPT")
        if not ckpt_path:
            raise RuntimeError(
                "BRAINDINO_PRETRAINED_CKPT is not set. "
                "Point it to the converted BrainDINO teacher checkpoint before training."
            )

        model = vit_base(
            drop_path_rate=0.2,
            layerscale_init=1.0e-05,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )

        chkpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = chkpt["teacher"]
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if "ibot" not in k and "dino_head" not in k
        }
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"Loaded BrainDINO checkpoint from: {ckpt_path}")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        return Primus_Multiscale_conv(
            embed_dim=768,
            patch_embed_size=16,
            num_classes=num_output_channels,
            dino_encoder=model,
            interaction_indices=[2, 5, 8, 11],
        )

    def configure_optimizers(self):
        vit_params = []
        other_params = []

        for name, param in self.network.named_parameters():
            if "dino_encoder" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": self.initial_lr, "weight_decay": self.weight_decay},
                {"params": vit_params, "lr": self.vit_lr, "weight_decay": self.vit_weight_decay},
            ],
            betas=(0.9, 0.98),
        )

        total_iters = self.num_epochs

        def lr_lambda(current_iter):
            if current_iter < self.warmup_epochs:
                return (1e-6 + (self.initial_lr - 1e-6) * (current_iter / self.warmup_epochs)) / self.initial_lr

            progress = (current_iter - self.warmup_epochs) / (total_iters - self.warmup_epochs)
            return (1 - progress) ** 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, lr_scheduler


class BrainDINO_Primus_Multiscale_HighRes_Trainer_DataRatio(
    BrainDINO_Primus_Multiscale_HighRes_Trainer
):
    """
    BrainDINO trainer with deterministic training-set subsampling.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        self.train_data_ratio = float(os.environ.get("TRAIN_DATA_RATIO", "1.0"))
        if not (0.0 < self.train_data_ratio <= 1.0):
            raise ValueError(f"TRAIN_DATA_RATIO must be between 0.0 and 1.0, got {self.train_data_ratio}")

        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        if self.train_data_ratio < 1.0:
            ratio_suffix = f"_ratio{self.train_data_ratio}"
            self.output_folder_base = os.path.join(
                nnUNet_results,
                self.plans_manager.dataset_name,
                self.__class__.__name__ + "__" + self.plans_manager.plans_name + "__" + configuration + ratio_suffix,
            )
            self.output_folder = os.path.join(self.output_folder_base, f"fold_{fold}")

            self.print_to_log_file(f"\n{'=' * 80}")
            self.print_to_log_file("DATA RATIO MODE ENABLED")
            self.print_to_log_file(f"Training data ratio: {self.train_data_ratio:.2%}")
            self.print_to_log_file(f"Output folder: {self.output_folder_base}")
            self.print_to_log_file("Random seed for sampling: 42 (fixed for reproducibility)")
            self.print_to_log_file(f"{'=' * 80}\n")

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        original_tr_size = len(tr_keys)

        if self.train_data_ratio < 1.0:
            random.seed(42)
            num_samples = max(1, int(len(tr_keys) * self.train_data_ratio))
            tr_keys = sorted(random.sample(tr_keys, num_samples))

            self.print_to_log_file("\nTraining data subsampling:")
            self.print_to_log_file(f"  Original training samples: {original_tr_size}")
            self.print_to_log_file(f"  Selected training samples: {len(tr_keys)}")
            self.print_to_log_file(f"  Actual ratio: {len(tr_keys) / original_tr_size:.2%}")
            self.print_to_log_file(f"  Validation samples: {len(val_keys)} (unchanged)")
            if len(tr_keys) > 5:
                self.print_to_log_file(f"  Selected case IDs: {tr_keys[:5]}...")
            else:
                self.print_to_log_file(f"  Selected case IDs: {tr_keys}")
            self.print_to_log_file("")

        dataset_tr = nnUNetDataset(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )

        dataset_val = nnUNetDataset(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )

        return dataset_tr, dataset_val

from trainers.base_trainer import BaseTrainer
from sklearn import metrics
from torch.utils.data import DataLoader
from contextlib import nullcontext
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import torch.nn as nn

# ★ add your dataloader
from datasets_3D.Classification.abide_classification import ABIDEClassificationSet
from datasets_3D.Classification.adni_classification import ADNIClassificationSet
from datasets_3D.Classification.brats_2023_classification import BraTS_2023_ClassificationSet
from datasets_3D.Classification.Survival_upenn_classification import Survival_upenn_ClassificationSet
from datasets_3D.Regression.brain_age_regression import BrainAgeRegressionSet
from datasets_3D.Regression.atlas_regression import ATLASRegressionSet
from networks.SliceStudent import SliceStudent,SliceStudent_attn_pooling
from networks.SliceStudent_comparisons import SliceStudent_comparisons

DATASET_REGISTRY = {
    "Combine_IXI_LONG_PIXAR": {
        "dataset": BrainAgeRegressionSet,
        "num_classes": 1, #just default value, ignore
        "data_root": "/path/to/Med_DINOv3/Datasets/Finetune_regression/Combine_IXI_LONG_PIXAR/"

    },
    "ATLAS": {
        "dataset": ATLASRegressionSet,
        "num_classes": 1, #just default value, ignore (regression task)
        "data_root": "/path/to/Med_DINOv3/Datasets/Finetune_regression/ATLAS/"
    }
}
BACKBONE_REGISTRY = {
    "meddinov3": {
        "feat_dim": 768,
        "ckpt": "/path/to/Med_DINOv3/Models/Meddinov3/high_res/57999/model.pth",
    },
    "dinov3": {
        "feat_dim": 768,
        "ckpt": "/path/to/Med_DINOv3/pretrained_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    },
    "brainmvp": {
        "feat_dim": 768,
        "ckpt": "/path/to/Med_DINOv3/Transfer/mf/BrainMVP_uniformer.pt",
    },
    "bm_mae": {
        "feat_dim": 768,
        "ckpt": "/path/to/Med_DINOv3/Transfer/mf/bmmae.pth",
    },
    "brainiac": {
        "feat_dim": 768,
        "ckpt": "/path/to/Med_DINOv3/pretrained_weights/BrainIAC.ckpt",
    },
    "scratch": {
        "feat_dim": 768,
        "ckpt": None,      # ★ 无 checkpoint
    }
}


class dinov3_volume2d_trainer_general_regression(BaseTrainer):
    """
    A trainer for 2.5D models:
    3D volume → slice sampler → 2D encoder → aggregation → classifier.
    Compatible with BaseTrainer.
    """

    def __init__(self, config):
        super().__init__(config, init_dataloader=False)  # ★ 禁用 BaseTrainer dataloader
        self.target_transform = str(getattr(config, "regression_target_transform", "none")).lower()
        self.use_log1p_target = (self.target_transform == "log1p")

        # -----------------------------------------
        # 🔥 Build dataloaders here
        # -----------------------------------------
        info = DATASET_REGISTRY[config.dataset_name]

        DatasetClass = info["dataset"]
        data_root    = info["data_root"]
        num_classes  = info["num_classes"]

        enc_name = str(config.encoder_name).lower()
        if enc_name not in BACKBONE_REGISTRY:
            raise KeyError(f"Unknown encoder_name '{config.encoder_name}'. Available: {sorted(BACKBONE_REGISTRY.keys())}")
        backbone_info = BACKBONE_REGISTRY[enc_name]
        print(f"[Trainer] Using backbone: {enc_name}")
        self.ckpt_path = backbone_info["ckpt"]
        if enc_name == "dinov3" and bool(getattr(config, "disable_pretrain", False)):
            self.ckpt_path = None
            print("[Trainer] dinov3 no-pretrain mode enabled: ckpt_path=None")
        teacher = config.teacher

        self.train_dataset = DatasetClass(
            config,
            data_root,
            flag="train",
            train_ratio=config.train_ratio,
            teacher=teacher
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.eval_dataset = DatasetClass(
            config,
            data_root,
            flag="valid",
            teacher=teacher
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=config.val_batch,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.start_epoch = 0

        # -----------------------------------------
        # ★ Optional: split training data for in-training validation
        # When enabled, 5% of training data is used for early stopping,
        # and the real held-out val set is saved for final evaluation.
        # Toggle via config.use_train_val_split (default: False)
        # -----------------------------------------
        self.holdout_eval_dataloader = None  # default: not used
        use_split = getattr(config, 'use_train_val_split', False)
        use_predefined_split = bool(getattr(config, 'use_predefined_atlas_split', False))
        split_ratio = getattr(config, 'train_val_split_ratio', 0.05)

        if use_predefined_split:
            # Predefined split mode:
            # - train loader already uses config.atlas_train_csv
            # - eval loader already uses config.atlas_valid_csv
            # - build holdout loader from config.atlas_test_csv
            holdout_dataset = DatasetClass(
                config,
                data_root,
                flag="test",
                teacher=teacher
            )
            self.holdout_eval_dataloader = DataLoader(
                holdout_dataset,
                batch_size=config.val_batch,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )
            print(
                f"[Trainer] ★ Predefined ATLAS split enabled: "
                f"{len(self.train_dataloader.dataset)} train, "
                f"{len(self.eval_dataloader.dataset)} val, "
                f"{len(self.holdout_eval_dataloader.dataset)} holdout-val"
            )

        elif use_split:
            from torch.utils.data import random_split
            full_train = self.train_dataset
            n_total = len(full_train)
            n_val = max(1, int(n_total * split_ratio))
            n_train = n_total - n_val

            train_subset, val_subset = random_split(
                full_train, [n_train, n_val],
                generator=torch.Generator().manual_seed(0)
            )

            self.train_dataloader = DataLoader(
                train_subset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
            )

            # Save the real held-out val for final evaluation
            self.holdout_eval_dataloader = self.eval_dataloader

            # Use train-val split for in-training early stopping
            self.eval_dataloader = DataLoader(
                val_subset,
                batch_size=config.val_batch,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )

            print(f"[Trainer] ★ Train-val split enabled: {n_train} train, {n_val} in-train-val, "
                  f"{len(self.holdout_eval_dataloader.dataset)} holdout-val")

        TEACHERS = ["brainmvp", "bm_mae", "brainiac"]
        enc = enc_name
        frozen = bool(getattr(config, "frozen", True))
        drop_path_rate = float(getattr(config, "drop_path_rate", 0.2))
        teacher_dropout_prob = float(getattr(config, "teacher_dropout_prob", 0.2))

        if enc in TEACHERS:
            self.model = SliceStudent_comparisons(
                ckpt_path=self.ckpt_path,
                n_slices=config.n_slices,
                lora_rank=config.lora_rank,
                num_classes=num_classes,
                teacher_name=enc,
                reg_head=True,
                dropout_prob=teacher_dropout_prob,
            )
        else:
            self.model = SliceStudent(
                ckpt_path=self.ckpt_path, # if None: scratch, if not, we use different path for dinov3 and meddinov3
                n_slices=config.n_slices,
                lora_rank=config.lora_rank,
                num_classes=num_classes,
                frozen=frozen,
                backbone_name=enc,
                reg_head=True,
                drop_path_rate=drop_path_rate,
            )
        # 2. GPU
        self.model_to_gpu()

        # 3. loss
        self.criterion = nn.MSELoss()
        self.use_atlas_mixed_loss = str(config.dataset_name).upper() == "ATLAS"
        self.atlas_loss_alpha = float(getattr(config, "atlas_loss_alpha", 0.2))
        self.atlas_huber_delta = float(getattr(config, "atlas_huber_delta", 1.0))
        self.mae_criterion = nn.L1Loss()
        self.huber_criterion = nn.HuberLoss(delta=self.atlas_huber_delta)
        self.last_loss_components = None

        # 4. optimizer
        self.optimizer_weight_decay = float(getattr(config, "optimizer_weight_decay", 1e-4))
        self.optimizer_name = str(getattr(config, "optimizer_name", "adamw")).lower()
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.lr,
                weight_decay=self.optimizer_weight_decay
            )
        elif self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=config.lr,
                weight_decay=self.optimizer_weight_decay
            )
        else:
            raise ValueError(
                f"Unsupported optimizer_name='{self.optimizer_name}'. "
                "Use one of: ['adam', 'adamw']."
            )


        # 5. scheduler
        self.scheduler_mode = str(getattr(config, "scheduler_mode", "max"))
        self.scheduler_factor = float(getattr(config, "scheduler_factor", 0.5))
        self.scheduler_patience = int(getattr(config, "scheduler_patience", 10))
        if config.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.scheduler_mode,
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                verbose=True
            )
        else:
            self.scheduler = None
        self.grad_accum_steps = max(1, int(getattr(config, "grad_accum_steps", 1)))
        self.use_amp = bool(getattr(config, "use_amp", False)) and self.use_cuda
        amp_dtype_name = str(getattr(config, "amp_dtype", "bf16")).lower()
        if amp_dtype_name == "bf16" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        print(f"[Trainer] Loaded {len(self.train_dataloader.dataset)} train volumes")
        print(f"[Trainer] Loaded {len(self.eval_dataloader.dataset)} val volumes")
        print(f"[Trainer] Regression target transform: {self.target_transform}")
        if self.use_atlas_mixed_loss:
            print(
                "[Trainer] ATLAS mixed loss enabled: "
                "alpha*MSE + (1-alpha)*(0.5*MAE + 0.5*Huber), "
                f"alpha={self.atlas_loss_alpha:.3f}, huber_delta={self.atlas_huber_delta:.3f}"
            )
        print(
            f"[Trainer] hparams: optimizer={self.optimizer_name}, drop_path={drop_path_rate}, "
            f"weight_decay={self.optimizer_weight_decay}, "
            f"scheduler_mode={self.scheduler_mode}, scheduler_factor={self.scheduler_factor}, "
            f"scheduler_patience={self.scheduler_patience}, "
            f"use_amp={self.use_amp}, amp_dtype={self.amp_dtype}, grad_accum_steps={self.grad_accum_steps}"
        )

    def _compute_regression_loss(self, pred, target_train):
        mse = self.criterion(pred, target_train)
        if not self.use_atlas_mixed_loss:
            self.last_loss_components = {"mse": float(mse.detach().item())}
            return mse

        mae = self.mae_criterion(pred, target_train)
        huber = self.huber_criterion(pred, target_train)
        alpha = self.atlas_loss_alpha
        mixed = alpha * mse + (1.0 - alpha) * (0.5 * mae + 0.5 * huber)
        self.last_loss_components = {
            "mse": float(mse.detach().item()),
            "mae": float(mae.detach().item()),
            "huber": float(huber.detach().item()),
            "mixed": float(mixed.detach().item()),
        }
        return mixed

    def _target_to_train_space_torch(self, target):
        """Map original target (days) to training space."""
        if self.use_log1p_target:
            return torch.log1p(torch.clamp(target, min=0.0))
        return target

    def _pred_to_metric_space_torch(self, pred):
        """Map model prediction from training space back to original target space."""
        if self.use_log1p_target:
            return torch.expm1(torch.clamp(pred, min=-20.0, max=20.0))
        return pred

    def _pred_to_metric_space_numpy(self, pred):
        if self.use_log1p_target:
            return np.expm1(np.clip(pred, -20.0, 20.0))
        return pred

    # ---------------------------
    # 🔥 override: set input
    # ---------------------------
    def set_input(self, sample):
        """
        sample: (volume, label, index)
        volume shape: (B,1,Nslices,H,W)
        """
        self.volume = sample[0].to(self.device)
        self.target = sample[1].to(self.device)

    # ---------------------------
    # 🔥 override: forward
    # ---------------------------
    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def forward(self):
        with self._autocast_context():
            pred = self.model(self.volume).squeeze(-1)  # (B,)
            target = self.target.float().view(-1)       # (B,)
            target_train = self._target_to_train_space_torch(target)

            self.loss = self._compute_regression_loss(pred, target_train)
            self.pred = pred


    def backward(self):
        if self.scaler.is_enabled():
            self.scaler.scale(self.loss).backward()
        else:
            self.loss.backward()

    def evaluate(self, save_predictions=False):
        """Evaluate on validation set and optionally save per-case predictions."""
        self.model.eval()

        all_preds = []
        all_targets = []
        all_ids = []

        with torch.no_grad():
            for vol, gt, pid in tqdm(self.eval_dataloader, desc="Evaluating"):
                vol = vol.to(self.device)
                gt = gt.to(self.device).float()

                with self._autocast_context():
                    pred = self.model(vol).squeeze(-1)
                pred_metric = self._pred_to_metric_space_torch(pred).float()

                all_preds.extend(pred_metric.cpu().numpy())
                all_targets.extend(gt.cpu().numpy())
                all_ids.extend(pid)

        all_preds = np.asarray(all_preds)
        all_targets = np.asarray(all_targets)

        errors = all_preds - all_targets
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        try:
            r2 = float(metrics.r2_score(all_targets, all_preds))
        except Exception:
            r2 = 0.0

        if save_predictions:
            df = pd.DataFrame({
                "patient_id": list(all_ids),
                "true_age": all_targets.tolist(),
                "predicted_age": all_preds.tolist(),
                "error": errors.tolist(),
                "abs_error": abs_errors.tolist(),
            })
            save_path = os.path.join(self.recorder.save_dir, "per_case_predictions.csv")
            df.to_csv(save_path, index=False)
            print(f"[Trainer] Saved per-case predictions to: {save_path}")

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    # ---------------------------
    # 🔥 main training loop
    # ---------------------------
    def train(self):
        avg_train_losses = []
        avg_val_mae = []

        best_metric = float("inf")
        num_epoch_no_improvement = 0

        # ----------------------------------
        # open .out log file
        # ----------------------------------
        out_path = os.path.join(self.recorder.save_dir, "train.out")
        print(f"[Trainer] Logging training progress to: {out_path}")
        self.out_f = open(out_path, "a", buffering=1)  # line-buffered

        train_metric_name = "Train_Loss" if self.use_atlas_mixed_loss else "Train_MSE"
        self.out_f.write(f"# Epoch | {train_metric_name} | Val_MAE | Best_MAE\n")
        self.out_f.flush()

        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            train_losses = []
            self.optimizer.zero_grad(set_to_none=True)

            train_bar = tqdm(self.train_dataloader)
            for itr, sample in enumerate(train_bar):
                self.set_input(sample)
                self.forward()
                train_losses.append(self.loss.item())

                loss_for_backward = self.loss / self.grad_accum_steps
                if self.scaler.is_enabled():
                    self.scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                should_step = ((itr + 1) % self.grad_accum_steps == 0) or ((itr + 1) == len(self.train_dataloader))
                if should_step:
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            train_loss = float(np.mean(train_losses))
            avg_train_losses.append(train_loss)

            train_scalar_name = "Train/Loss" if self.use_atlas_mixed_loss else "Train/MSE"
            self.recorder.writer.add_scalar(train_scalar_name, train_loss, epoch)

            # -----------------------------
            # Validation
            # -----------------------------
            if epoch % self.config.val_epoch == 0:
                self.model.eval()
                gts, preds = [], []

                with torch.no_grad():
                    for vol, gt, _ in self.eval_dataloader:
                        vol = vol.to(self.device)
                        gt = gt.to(self.device).float()

                        with self._autocast_context():
                            pred = self.model(vol).squeeze(-1)
                        pred_metric = self._pred_to_metric_space_torch(pred).float()
                        preds.append(pred_metric.cpu().numpy())
                        gts.append(gt.cpu().numpy())

                preds = np.concatenate(preds)
                gts = np.concatenate(gts)

                mae = float(np.mean(np.abs(preds - gts)))
                avg_val_mae.append(mae)

                self.recorder.writer.add_scalar("Val/MAE", mae, epoch)

                # logger
                log_msg = (
                    f"Epoch {epoch+1:03d} | "
                    f"{train_metric_name.replace('_', ' ')}={train_loss:.4f} | "
                    f"Val MAE={mae:.4f} | "
                    f"Best MAE={best_metric:.4f}"
                )
                self.recorder.logger.info(log_msg)

                # ----------------------------------
                # write to .out file
                # ----------------------------------
                self.out_f.write(
                    f"{epoch+1:03d} "
                    f"{train_loss:.6f} "
                    f"{mae:.6f} "
                    f"{best_metric:.6f}\n"
                )
                self.out_f.flush()

                # save csv
                pd.DataFrame({
                    train_metric_name: avg_train_losses,
                    "Val_MAE": avg_val_mae
                }).to_csv(
                    os.path.join(self.recorder.save_dir, "results.csv"),
                    index_label="epoch"
                )

                # early stopping
                if mae < best_metric:
                    best_metric = mae
                    num_epoch_no_improvement = 0
                    self.save_state_dict(
                        epoch + 1,
                        os.path.join(self.recorder.save_dir, "model_best.pth")
                    )
                else:
                    num_epoch_no_improvement += 1

                if num_epoch_no_improvement >= self.config.patience:
                    self.recorder.logger.info("Early stopping triggered")
                    self.out_f.write("# Early stopping triggered\n")
                    self.out_f.flush()
                    break

                if self.scheduler is not None:
                    self.scheduler.step(mae)

        # ----------------------------------
        # cleanup
        # ----------------------------------
        self.out_f.write("# Training finished\n")
        self.out_f.close()

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return

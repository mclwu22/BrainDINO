"""
Multi-modality survival trainer with risk stratification.

Features:
1. Multi-modality fusion (4 modalities → averaged features)
2. Binary classification for 1-year mortality prediction
3. Risk score extraction for stratification
4. Kaplan-Meier analysis on test set
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics

sys.path.insert(0, '/path/to/Med_DINOv3/Finetune/Large-Scale-Medical/Downstream/monai/LUNA16')

from trainers.base_trainer import BaseTrainer
from SurvivalRiskStratification_KaplanMeier.datasets.survival_multimodal_dataset import SurvivalMultiModalDataset
from SurvivalRiskStratification_KaplanMeier.utils.cox_loss import CoxPHLoss, concordance_index
from networks.SliceStudent_MultiModal import SliceStudent_MultiModal
from networks.SliceStudent_MultiModal_2mod import SliceStudent_MultiModal_2mod
from networks.SliceStudent_comparisons import SliceStudent_comparisons


BACKBONE_REGISTRY = {
    "meddinov3": {
        "feat_dim": 768,
        "ckpt": "/path/to/Med_DINOv3/Models/Meddinov3/high_res/57999/model.pth",
    },
    "dinov3": {
        "feat_dim": 768,
        "ckpt": "/path/to/Med_DINOv3/pretrained_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    },
    "BrainMVP": {
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
        "ckpt": None,
    }
}


class SurvivalMultiModalTrainer(BaseTrainer):
    """
    Trainer for multi-modality survival risk stratification.
    """

    def __init__(self, config):
        super().__init__(config, init_dataloader=False)

        self.data_root = config.data_root
        self.encoder_name = config.encoder_name

        # Get backbone info
        backbone_info = BACKBONE_REGISTRY[self.encoder_name]
        self.ckpt_path = backbone_info["ckpt"]

        # Hard-coded optimization for meddinov3
        # Skip this override in quick-probe mode so caller can enforce tiny 1-epoch settings.
        quick_probe = bool(getattr(config, "quick_probe", False))
        if self.encoder_name.lower() == "meddinov3" and not quick_probe:
            print(f"[SurvivalMultiModalTrainer] 🔥 Using optimized hyperparameters for meddinov3:")
            print(f"  batch_size: {config.batch_size} -> 12")
            print(f"  val_batch: {config.val_batch} -> 16")
            print(f"  lr: {config.lr} -> 3e-4")
            print(f"  patience: {config.patience} -> 20")

            config.batch_size = 12
            config.val_batch = 16
            config.lr = 3e-4
            config.patience = 20
        elif self.encoder_name.lower() == "meddinov3" and quick_probe:
            print("[SurvivalMultiModalTrainer] quick_probe=True, skip meddinov3 hyperparameter override.")

        # Build datasets
        print(f"[SurvivalMultiModalTrainer] Building datasets...")
        train_ratio = getattr(config, 'train_ratio', 1.0)

        self.train_dataset = SurvivalMultiModalDataset(
            config,
            self.data_root,
            flag='train',
            train_ratio=train_ratio
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )

        self.eval_dataset = SurvivalMultiModalDataset(
            config,
            self.data_root,
            flag='valid',
            train_ratio=1.0  # Always use full validation set
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=config.val_batch,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        # Build model
        print(f"[SurvivalMultiModalTrainer] Building model: {self.encoder_name}")

        TEACHERS = ["brainmvp", "bm_mae", "brainiac"]
        enc = self.encoder_name.lower()

        if enc in TEACHERS:
            # Use comparison encoder for BrainMVP/bm_mae
            self.model = SliceStudent_MultiModal(
                n_slices=config.n_slices,
                lora_rank=config.lora_rank,
                ckpt_path=self.ckpt_path,
                embed_dim=768,
                num_classes=2,  # Not used, will use reg_head for survival
                frozen=True,
                backbone_name=enc,
                reg_head=True  # Use regression head for survival risk score
            )
        else:
            # Use standard SliceStudent for meddinov3/dinov3/scratch
            self.model = SliceStudent_MultiModal(
                n_slices=config.n_slices,
                lora_rank=config.lora_rank,
                ckpt_path=self.ckpt_path,
                embed_dim=768,
                num_classes=2,  # Not used, will use reg_head for survival
                frozen=True,
                backbone_name=enc,
                reg_head=True  # Use regression head for survival risk score
            )

        # Move to GPU
        self.model_to_gpu()

        # Loss: Cox Partial Likelihood for survival learning
        self.criterion = CoxPHLoss()

        # Optimizer (only trainable parameters)
        # Increased weight_decay from 1e-5 to 1e-3 to reduce overfitting
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=1e-3
        )

        # Scheduler
        if config.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None

        self.start_epoch = 0

        print(f"[SurvivalMultiModalTrainer] Train: {len(self.train_dataset)} samples")
        print(f"[SurvivalMultiModalTrainer] Val: {len(self.eval_dataset)} samples")

    def set_input(self, sample):
        """
        sample: (multi_modal_volume, label, survival_time, event, patient_id)
        multi_modal_volume: (B, 4, 128, 224, 224)

        For survival learning, we use (T, E) instead of binary label:
        - survival_time (T): time to event or censoring
        - event (E): 1=death, 0=censored
        """
        self.volume = sample[0].to(self.device)  # (B, 4, 128, 224, 224)
        self.survival_time = sample[2].to(self.device).float()  # (B,) - T
        self.event = sample[3].to(self.device).float()          # (B,) - E

    def forward(self):
        """
        Forward pass for survival learning.

        Model outputs continuous risk score (not probability).
        Loss is Cox partial likelihood over (risk_score, T, E).
        """
        risk_score = self.model(self.volume).squeeze(-1)  # (B,) - continuous risk score
        self.loss = self.criterion(risk_score, self.survival_time, self.event)
        self.pred = risk_score  # Store for evaluation

    def backward(self):
        """Backward pass"""
        self.loss.backward()

    def train(self):
        """Main training loop"""
        train_losses = []
        avg_train_losses = []
        avg_valid_metrics = []

        best_metric = 0
        num_epoch_no_improvement = 0

        for epoch in range(self.start_epoch, self.config.epochs):
            train_losses = []

            self.model.train()
            self.recorder.logger.info(
                'Epoch: %d/%d lr %e',
                epoch, self.config.epochs,
                self.optimizer.param_groups[0]['lr']
            )

            train_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
            for itr, sample in enumerate(train_bar):
                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(self.loss.item())
                train_bar.set_postfix({'loss': self.loss.item()})

            self.recorder.writer.add_scalar('Train/total_loss', np.mean(train_losses), epoch)

            # Validation
            if epoch % self.config.val_epoch == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.recorder.logger.info("Validating...")

                    risk_scores = []
                    survival_times = []
                    events = []

                    for vol, label, surv_time, event, pat_id in self.eval_dataloader:
                        vol = vol.to(self.device)

                        risk_score = self.model(vol).squeeze(-1)  # (B,) - continuous risk score

                        risk_scores.extend(risk_score.cpu().numpy())
                        survival_times.extend(surv_time.numpy())
                        events.extend(event.numpy())

                    risk_scores = np.asarray(risk_scores)  # (N,)
                    survival_times = np.asarray(survival_times)  # (N,)
                    events = np.asarray(events)  # (N,)

                    # Compute C-index (concordance index)
                    c_index = concordance_index(
                        torch.from_numpy(risk_scores),
                        torch.from_numpy(survival_times),
                        torch.from_numpy(events)
                    )
                    valid_metric = c_index * 100.0  # Convert to percentage for consistency

                    self.recorder.writer.add_scalar('Val/C-index', valid_metric, epoch)

                # Logging
                train_loss = np.mean(train_losses)
                avg_train_losses.append(train_loss)
                avg_valid_metrics.append(valid_metric)

                self.recorder.logger.info(
                    "Epoch {}, validation C-index is {:.4f}, training loss is {:.4f}".format(
                        epoch + 1, valid_metric, train_loss
                    )
                )

                # Save results
                data_frame = pd.DataFrame({
                    'Train_Loss': avg_train_losses,
                    'Val_C_index': avg_valid_metrics,
                })
                data_frame.to_csv(
                    os.path.join(self.recorder.save_dir, "results.csv"),
                    index_label='epoch'
                )

                # Early stopping
                if valid_metric > best_metric:
                    best_metric = valid_metric
                    num_epoch_no_improvement = 0
                    self.save_state_dict(epoch + 1, os.path.join(self.recorder.save_dir, "model_best.pth"))
                    self.recorder.logger.info(f"✓ New best C-index: {best_metric:.4f}")
                else:
                    num_epoch_no_improvement += 1

                if num_epoch_no_improvement == self.config.patience:
                    self.recorder.logger.info("Early stopping triggered")
                    break

                if self.scheduler is not None:
                    self.scheduler.step(valid_metric)

        self.recorder.logger_shutdown()
        self.recorder.writer.close()

    def predict(self, dataloader):
        """
        Extract risk scores and survival info from dataloader.

        Returns:
            risk_scores: (N,) array of continuous risk scores
            survival_times: (N,) array of survival times
            events: (N,) array of event indicators
            patient_ids: (N,) list of patient IDs
        """
        self.model.eval()

        risk_scores = []
        survival_times = []
        events = []
        patient_ids = []

        with torch.no_grad():
            for vol, label, surv_time, event, pat_id in tqdm(dataloader, desc="Predicting"):
                vol = vol.to(self.device)

                risk_score = self.model(vol).squeeze(-1)  # (B,) - continuous risk score
                risk = risk_score.cpu().numpy()

                risk_scores.extend(risk)
                survival_times.extend(surv_time.numpy())
                events.extend(event.numpy())
                patient_ids.extend(pat_id)

        return (
            np.array(risk_scores),
            np.array(survival_times),
            np.array(events),
            patient_ids
        )

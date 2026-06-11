import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import DataLoader

# Add LUNA16 to path
sys.path.insert(0, '/path/to/Med_DINOv3/Finetune/Large-Scale-Medical/Downstream/monai/LUNA16')

from networks.SliceStudent import SliceStudent
from networks.SliceStudent_comparisons import SliceStudent_comparisons
from trainers.base_trainer import BaseTrainer


class SurvivalTrainer(BaseTrainer):
    """
    Trainer for survival risk stratification using frozen encoders.

    Binary classification:
    - Input: 768-d features from encoder
    - Output: 2-class logits (low risk vs high risk)
    - Loss: CrossEntropyLoss
    - Only classification head is trainable
    """

    def __init__(self, config, train_dataset, val_dataset):
        super().__init__(config, init_dataloader=False)

        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Build dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        self.eval_dataloader = DataLoader(
            val_dataset,
            batch_size=config.val_batch,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Build model
        self.model = self._build_model(config)
        self.model_to_gpu()

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer (only trainable params)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=1e-5
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

        print(f"[SurvivalTrainer] Train samples: {len(train_dataset)}")
        print(f"[SurvivalTrainer] Val samples: {len(val_dataset)}")

    def _build_model(self, config):
        """Build model based on encoder name"""
        encoder_name = config.encoder_name.lower()

        # Encoder registry (from dinov3_volume2d_trainer_general.py)
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
            "scratch": {
                "feat_dim": 768,
                "ckpt": None,
            }
        }

        if encoder_name not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown encoder: {encoder_name}")

        backbone_info = BACKBONE_REGISTRY[encoder_name]
        ckpt_path = backbone_info["ckpt"]

        TEACHERS = ["brainmvp", "bm_mae"]

        if encoder_name in TEACHERS:
            model = SliceStudent_comparisons(
                ckpt_path=ckpt_path,
                n_slices=config.n_slices,
                lora_rank=config.lora_rank,
                num_classes=config.num_classes,
                teacher_name=encoder_name
            )
        else:
            model = SliceStudent(
                ckpt_path=ckpt_path,
                n_slices=config.n_slices,
                lora_rank=config.lora_rank,
                num_classes=config.num_classes,
                backbone_name=encoder_name,
            )

        print(f"[SurvivalTrainer] Using encoder: {encoder_name}")
        return model

    def set_input(self, batch):
        """
        batch: (volume, label, survival_time, event, patient_id)
        """
        self.volume = batch[0].to(self.device)
        self.target = batch[1].to(self.device)
        # survival_time and event are for evaluation only
        self.survival_time = batch[2]
        self.event = batch[3]
        self.patient_ids = batch[4]

    def forward(self):
        """Forward pass"""
        logits = self.model(self.volume)  # (B, 2)
        self.target = self.target.long().view(-1)
        self.loss = self.criterion(logits, self.target)
        self.pred = logits

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

        for epoch in range(0, self.config.epochs):
            train_losses = []

            self.model.train()
            self.recorder.logger.info(
                'Epoch: %d/%d lr %e',
                epoch, self.config.epochs,
                self.optimizer.param_groups[0]['lr']
            )

            train_bar = tqdm(self.train_dataloader)
            for batch in train_bar:
                self.set_input(batch)
                self.optimize_parameters()
                train_losses.append(self.loss.item())

            self.recorder.writer.add_scalar('Train/total_loss', np.mean(train_losses), epoch)

            # Validation
            if epoch % self.config.val_epoch == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.recorder.logger.info("validating....")

                    gts = []
                    preds = []

                    for batch in self.eval_dataloader:
                        vol = batch[0].to(self.device)
                        gt = batch[1].to(self.device)

                        pred = self.model(vol)  # (B, num_classes)
                        prob = torch.softmax(pred, dim=1)  # (B, num_classes)

                        gts.extend(gt.cpu().numpy())
                        preds.extend(prob.cpu().numpy())

                    gts = np.asarray(gts)
                    preds = np.asarray(preds)

                    # AUC (binary classification)
                    gts_onehot = np.zeros_like(preds)
                    gts_onehot[np.arange(len(gts)), gts] = 1

                    auc = 100.0 * metrics.roc_auc_score(
                        gts_onehot,
                        preds,
                        multi_class="ovr",
                        average="macro"
                    )
                    valid_metric = auc

                    self.recorder.writer.add_scalar('Val/AUC', valid_metric, epoch)

                # Logging
                train_loss = np.mean(train_losses)
                avg_train_losses.append(train_loss)
                avg_valid_metrics.append(valid_metric)

                self.recorder.logger.info(
                    "Epoch {}, validation AUC is {:.4f}, training loss is {:.4f}".format(
                        epoch + 1, valid_metric, train_loss
                    )
                )

                # Save results
                data_frame = pd.DataFrame(
                    data={
                        'Train_Loss': avg_train_losses,
                        'Val_AUC': avg_valid_metrics,
                    }
                )
                data_frame.to_csv(
                    os.path.join(self.recorder.save_dir, "results.csv"),
                    index_label='epoch'
                )

                # Early stopping
                if valid_metric > best_metric:
                    best_metric = valid_metric
                    num_epoch_no_improvement = 0
                    self.save_state_dict(epoch + 1, os.path.join(self.recorder.save_dir, "best_model.pth"))
                else:
                    num_epoch_no_improvement += 1

                if num_epoch_no_improvement == self.config.patience:
                    self.recorder.logger.info("Early Stopping")
                    break

                if self.scheduler is not None:
                    self.scheduler.step(valid_metric)

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return

    def predict(self, dataloader):
        """
        Inference to extract risk scores, survival times, events, and patient IDs.

        Returns:
        - risk_scores: np.array of risk scores (probability of class 1 = high risk)
        - survival_times: np.array of survival times in days
        - events: np.array of event indicators (1=deceased, 0=censored)
        - patient_ids: list of patient identifiers
        """
        self.model.eval()

        risk_scores = []
        survival_times = []
        events = []
        patient_ids = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                vol = batch[0].to(self.device)
                survival_time = batch[2].numpy()
                event = batch[3].numpy()
                patient_id = batch[4]

                pred = self.model(vol)  # (B, 2)
                prob = torch.softmax(pred, dim=1)  # (B, 2)
                risk = prob[:, 1].cpu().numpy()  # Probability of class 1 (high risk)

                risk_scores.extend(risk)
                survival_times.extend(survival_time)
                events.extend(event)
                patient_ids.extend(patient_id)

        return (
            np.array(risk_scores),
            np.array(survival_times),
            np.array(events),
            patient_ids
        )

    def load_best_model(self):
        """Load the best model checkpoint"""
        ckpt_path = os.path.join(self.recorder.save_dir, "best_model.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(self.recorder.save_dir, "model_best.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Best model not found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
        if state_dict is None:
            raise KeyError("Checkpoint missing model state dict")
        self.model.load_state_dict(state_dict)
        print(f"[SurvivalTrainer] Loaded best model from {ckpt_path}")

"""
UCSF-PDGM Mutation Classification Trainer (Multi-Modality)

Binary classification: IDH wildtype (0) vs non-wildtype (1)
Uses 2 modalities: T1c and FLAIR

Based on dinov3_volume2d_trainer_general.py but adapted for:
- 2-modality input (T1c + FLAIR)
- Binary classification
- UCSF-PDGM dataset
"""

from trainers.base_trainer import BaseTrainer
from sklearn import metrics
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import torch.nn as nn

# Import dataset
from datasets_3D.Classification.mutation_ucsf_classification import MutationUCSFClassificationSet

# Import multi-modality model
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


class MutationUCSFTrainer(BaseTrainer):
    """
    Trainer for UCSF-PDGM mutation classification.

    Binary classification with multi-modality fusion (T1c + FLAIR).
    """

    def __init__(self, config):
        super().__init__(config, init_dataloader=False)

        # Get dataset config
        data_root = config.data_root
        num_classes = 2  # Binary classification

        # Get backbone info
        backbone_info = BACKBONE_REGISTRY[config.encoder_name]
        print(f"[MutationUCSFTrainer] Using backbone: {config.encoder_name}")
        self.ckpt_path = backbone_info["ckpt"]

        # Build datasets
        self.train_dataset = MutationUCSFClassificationSet(
            config,
            data_root,
            flag="train",
            train_ratio=config.train_ratio,
            teacher=None
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        self.eval_dataset = MutationUCSFClassificationSet(
            config,
            data_root,
            flag="valid",
            teacher=None
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=config.val_batch,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        self.start_epoch = 0

        # Build model
        TEACHERS = ["brainmvp", "bm_mae", "brainiac"]
        enc = config.encoder_name.lower()

        print(f"[MutationUCSFTrainer] Building 2-modality model...")
        self.model = SliceStudent_MultiModal_2mod(
            n_slices=config.n_slices,
            lora_rank=config.lora_rank,
            ckpt_path=self.ckpt_path,
            embed_dim=768,
            num_classes=num_classes,
            frozen=True,
            backbone_name=enc
        )

        # Move to GPU
        self.model_to_gpu()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer (only trainable parameters)
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

        print(f"[MutationUCSFTrainer] Loaded {len(self.train_dataloader.dataset)} train samples")
        print(f"[MutationUCSFTrainer] Loaded {len(self.eval_dataloader.dataset)} val samples")

    def set_input(self, sample):
        """
        sample: (multi_modal_volume, label, patient_id)
        multi_modal_volume shape: (B, 2, 128, 224, 224)
        """
        self.volume = sample[0].to(self.device)
        self.target = sample[1].to(self.device)

    def forward(self):
        """Forward pass for binary classification"""
        logits = self.model(self.volume)          # (B, 2)
        self.target = self.target.long().view(-1)  # CE needs long
        self.loss = self.criterion(logits, self.target)
        self.pred = logits

    def backward(self):
        self.loss.backward()

    def train(self):
        """Main training loop"""
        train_losses = []
        avg_train_losses = []
        avg_valid_metrics = []
        best_acc = 0.0
        num_epoch_no_improvement = 0

        for epoch in range(self.start_epoch, self.config.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"{'='*60}")

            # Training phase
            self.model.train()
            epoch_losses = []

            pbar = tqdm(self.train_dataloader, desc=f"Training")
            for sample in pbar:
                self.set_input(sample)
                self.optimizer.zero_grad()
                self.forward()
                self.backward()
                self.optimizer.step()

                epoch_losses.append(self.loss.item())
                pbar.set_postfix({'loss': f'{self.loss.item():.4f}'})

            avg_train_loss = np.mean(epoch_losses)
            avg_train_losses.append(avg_train_loss)

            print(f"Train Loss: {avg_train_loss:.4f}")

            # Validation phase
            if (epoch + 1) % self.config.val_epoch == 0:
                val_metrics = self.evaluate()

                print(f"Validation Results:")
                print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  AUC: {val_metrics['auc']:.4f}")
                print(f"  F1: {val_metrics['f1']:.4f}")
                print(f"  Precision: {val_metrics['precision']:.4f}")
                print(f"  Recall: {val_metrics['recall']:.4f}")

                avg_valid_metrics.append(val_metrics)

                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step(val_metrics['accuracy'])

                # Save best model and early stopping
                if val_metrics['accuracy'] > best_acc:
                    best_acc = val_metrics['accuracy']
                    num_epoch_no_improvement = 0
                    save_path = os.path.join(self.recorder.save_dir, "model_best.pth")
                    self.save_state_dict(epoch, save_path)
                    print(f"  ✓ New best accuracy: {best_acc:.4f}")
                else:
                    num_epoch_no_improvement += 1

                # Early stopping
                if num_epoch_no_improvement >= self.config.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"No improvement for {self.config.patience} epochs")
                    break

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                save_path = os.path.join(self.recorder.save_dir, f"model_epoch_{epoch+1}.pth")
                self.save_state_dict(epoch, save_path)

        # Save final results
        self.save_training_curves(avg_train_losses, avg_valid_metrics)
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {best_acc:.4f}")
        print(f"{'='*60}")

    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for sample in tqdm(self.eval_dataloader, desc="Validating"):
                self.set_input(sample)
                self.forward()

                probs = torch.softmax(self.pred, dim=1)  # (B, 2)
                preds = torch.argmax(self.pred, dim=1)   # (B,)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(self.target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Compute metrics
        accuracy = metrics.accuracy_score(all_targets, all_preds)
        auc = metrics.roc_auc_score(all_targets, all_probs)
        f1 = metrics.f1_score(all_targets, all_preds, average='binary')
        precision = metrics.precision_score(all_targets, all_preds, average='binary')
        recall = metrics.recall_score(all_targets, all_preds, average='binary')

        # Confusion matrix
        cm = metrics.confusion_matrix(all_targets, all_preds)
        print(f"\nConfusion Matrix:")
        print(cm)

        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }

    def save_training_curves(self, train_losses, val_metrics):
        """Save training curves to CSV"""
        results = {
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses
        }

        if val_metrics:
            results['val_accuracy'] = [m['accuracy'] for m in val_metrics]
            results['val_auc'] = [m['auc'] for m in val_metrics]
            results['val_f1'] = [m['f1'] for m in val_metrics]

        df = pd.DataFrame(results)
        save_path = os.path.join(self.recorder.save_dir, "training_curves.csv")
        df.to_csv(save_path, index=False)
        print(f"Training curves saved to: {save_path}")

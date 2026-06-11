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
from contextlib import nullcontext

# ★ add your dataloader
from datasets_3D.Classification.abide_classification import ABIDEClassificationSet
from datasets_3D.Classification.adni_classification import ADNIClassificationSet
from datasets_3D.Classification.brats_2023_classification import BraTS_2023_ClassificationSet
from datasets_3D.Classification.Survival_upenn_classification import Survival_upenn_ClassificationSet
from datasets_3D.Classification.oasis_classification import OASISClassificationSet
from networks.SliceStudent import SliceStudent,SliceStudent_attn_pooling
from networks.SliceStudent_comparisons import SliceStudent_comparisons

# NOTE: set these to wherever you place each dataset locally.
# Each data_root is expected to contain the train/valid (and optional test) CSVs.
DATASET_REGISTRY = {
    "ABIDE": {
        "dataset": ABIDEClassificationSet,
        "num_classes": 2,
        "data_root": "/path/to/datasets/ABIDE/",
        "csv_format": "two_csv"  # train_0.csv + train_1.csv
    },

    "ADNI": {
        "dataset": ADNIClassificationSet,
        "num_classes": 3,
        "data_root": "/path/to/datasets/ADNI/",
        "csv_format": "single_csv"
    },

    "BraTS": {
        "dataset": BraTS_2023_ClassificationSet,
        "num_classes": 4,
        "data_root": "/path/to/datasets/BraTS_2023/",
        "csv_format": "single_csv"
    },

    "UPENN": {
        "dataset": Survival_upenn_ClassificationSet,
        "num_classes": 2,
        "data_root": "/path/to/datasets/UPENN/",
        "csv_format": "single_csv"
    },

    "OASIS": {
        "dataset": OASISClassificationSet,
        "num_classes": 2,
        "data_root": "/path/to/datasets/OASIS/",
        "csv_format": "single_csv"
    }
}
# NOTE: download each pretrained backbone yourself and point "ckpt" to the local file.
BACKBONE_REGISTRY = {
    "meddinov3": {
        "feat_dim": 768,
        "ckpt": "/path/to/weights/braindino.pth",
    },
    "dinov3": {
        "feat_dim": 768,
        "ckpt": "/path/to/weights/dinov3_vitb16.pth",
    },
    "brainmvp": {
        "feat_dim": 768,
        "ckpt": "/path/to/weights/BrainMVP.pt",
    },
    "bm_mae": {
        "feat_dim": 768,
        "ckpt": "/path/to/weights/bmmae.pth",
    },
    "brainiac": {
        "feat_dim": 768,
        "ckpt": "/path/to/weights/BrainIAC.ckpt",
    },
    "scratch": {
        "feat_dim": 768,
        "ckpt": None,      # train from scratch (no checkpoint)
    }
}


class dinov3_volume2d_trainer(BaseTrainer):
    """
    A trainer for 2.5D models:
    3D volume → slice sampler → 2D encoder → aggregation → classifier.
    Compatible with BaseTrainer.
    """

    def __init__(self, config):
        super().__init__(config, init_dataloader=False)  # ★ 禁用 BaseTrainer dataloader

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
        # ★ Optional: stratified split of training data for in-training validation
        # When enabled, ~5% of training data is used for early stopping,
        # with a minimum number of samples per class in validation.
        # Toggle via config.use_train_val_split (default: False)
        # -----------------------------------------
        self.holdout_eval_dataloader = None  # default: not used
        use_split = getattr(config, 'use_train_val_split', False)
        self.use_train_val_split = bool(use_split)
        use_predefined_split = bool(getattr(config, 'use_predefined_oasis_split', False))
        split_ratio = getattr(config, 'train_val_split_ratio', 0.05)
        min_val_per_class = int(getattr(config, 'train_val_min_per_class', 2))
        min_train_after_split = int(getattr(config, 'train_min_per_class_after_split', 1))

        # Standard early stopping only: any non-standard low-data policies are disabled.
        self.use_low_ratio_policy = False
        self.low_ratio_disable_early_stopping = False

        if use_predefined_split:
            # Predefined split mode:
            # - train loader already uses config.oasis_train_csv
            # - eval loader already uses config.oasis_valid_csv
            # - build holdout loader from config.oasis_test_csv
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
                f"[Trainer] ★ Predefined split enabled: "
                f"{len(self.train_dataloader.dataset)} train, "
                f"{len(self.eval_dataloader.dataset)} val, "
                f"{len(self.holdout_eval_dataloader.dataset)} holdout-val"
            )

        elif use_split:
            from torch.utils.data import Subset
            from collections import defaultdict, Counter
            full_train = self.train_dataset
            labels = full_train.all_labels

            # Group indices by class
            class_indices = defaultdict(list)
            for i, lbl in enumerate(labels):
                class_indices[lbl].append(i)

            split_ok = len(class_indices) >= 2
            split_fail_reason = ""
            split_seed = getattr(config, "split_seed", 0)  # PLACEHOLDER: set your own split seed
            rng = np.random.RandomState(split_seed)
            val_indices = []
            train_indices = []

            if split_ok:
                for cls in sorted(class_indices.keys()):
                    idx = class_indices[cls][:]
                    rng.shuffle(idx)
                    n_cls = len(idx)

                    if n_cls <= min_train_after_split:
                        split_ok = False
                        split_fail_reason = (
                            f"class {cls} has only {n_cls} sample(s), "
                            f"cannot keep >= {min_train_after_split} for train"
                        )
                        break

                    n_cls_val = int(np.ceil(n_cls * split_ratio))
                    n_cls_val = max(min_val_per_class, n_cls_val)
                    n_cls_val = min(n_cls_val, n_cls - min_train_after_split)

                    if n_cls_val < 1:
                        split_ok = False
                        split_fail_reason = (
                            f"class {cls} produced invalid val count {n_cls_val}"
                        )
                        break

                    val_indices.extend(idx[:n_cls_val])
                    train_indices.extend(idx[n_cls_val:])

            if split_ok:
                val_labels = [labels[i] for i in val_indices]
                if len(set(val_labels)) < 2:
                    split_ok = False
                    split_fail_reason = "in-train validation split still has a single class"

            if split_ok:
                train_subset = Subset(full_train, train_indices)
                val_subset = Subset(full_train, val_indices)

                self.train_dataloader = DataLoader(
                    train_subset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                    pin_memory=True,
                )

                # Save the real held-out val for final evaluation
                self.holdout_eval_dataloader = self.eval_dataloader

                # Use stratified train-val split for in-training early stopping
                self.eval_dataloader = DataLoader(
                    val_subset,
                    batch_size=config.val_batch,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=True,
                )

                val_dist = dict(Counter(val_labels))
                train_dist = dict(Counter([labels[i] for i in train_indices]))
                print(
                    f"[Trainer] ★ Stratified train-val split: {len(train_indices)} train "
                    f"(classes: {train_dist}), {len(val_indices)} in-train-val "
                    f"(classes: {val_dist}), min_val_per_class={min_val_per_class}, "
                    f"{len(self.holdout_eval_dataloader.dataset)} holdout-val"
                )
            else:
                print(
                    "[Trainer] ⚠ Skip in-train split; keep original holdout validation. "
                    f"Reason: {split_fail_reason or 'not enough class diversity'}"
                )

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
                drop_path_rate=drop_path_rate,
            )
        # 2. GPU
        self.model_to_gpu()

        # 3. loss
        self.label_smoothing = float(getattr(config, "label_smoothing", 0.0))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # 4. optimizer
        self.optimizer_weight_decay = float(getattr(config, "optimizer_weight_decay", 1e-5))
        self.optimizer_name = str(getattr(config, "optimizer_name", "adam")).lower()
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=config.lr,
                weight_decay=self.optimizer_weight_decay
            )
        elif self.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
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
        print(
            f"[Trainer] hparams: optimizer={self.optimizer_name}, drop_path={drop_path_rate}, "
            f"label_smoothing={self.label_smoothing}, "
            f"weight_decay={self.optimizer_weight_decay}, scheduler_mode={self.scheduler_mode}, "
            f"scheduler_factor={self.scheduler_factor}, scheduler_patience={self.scheduler_patience}, "
            f"use_amp={self.use_amp}, amp_dtype={self.amp_dtype}, grad_accum_steps={self.grad_accum_steps}"
        )

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
            logits = self.model(self.volume)          # (B,2)
            self.target = self.target.long().view(-1) # CE 需要 long
            self.loss = self.criterion(logits, self.target)
            self.pred = logits

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
        all_probs = []
        all_ids = []

        with torch.no_grad():
            for vol, gt, pid in tqdm(self.eval_dataloader, desc="Evaluating"):
                vol = vol.to(self.device)
                gt = gt.to(self.device)

                with self._autocast_context():
                    logits = self.model(vol)
                probs = torch.softmax(logits, dim=1).float()
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(gt.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_ids.extend(pid)

        all_preds = np.asarray(all_preds)
        all_targets = np.asarray(all_targets)
        all_probs = np.asarray(all_probs)

        # metrics
        accuracy = metrics.accuracy_score(all_targets, all_preds)

        num_classes = all_probs.shape[1]
        if num_classes == 2:
            auc = metrics.roc_auc_score(all_targets, all_probs[:, 1])
            f1 = metrics.f1_score(all_targets, all_preds, average="binary")
            precision = metrics.precision_score(all_targets, all_preds, average="binary")
            recall = metrics.recall_score(all_targets, all_preds, average="binary")
        else:
            gts_onehot = np.zeros_like(all_probs)
            gts_onehot[np.arange(len(all_targets)), all_targets] = 1
            auc = metrics.roc_auc_score(
                gts_onehot,
                all_probs,
                multi_class="ovr",
                average="macro"
            )
            f1 = metrics.f1_score(all_targets, all_preds, average="macro")
            precision = metrics.precision_score(all_targets, all_preds, average="macro")
            recall = metrics.recall_score(all_targets, all_preds, average="macro")

        cm = metrics.confusion_matrix(all_targets, all_preds)

        if save_predictions:
            # per-case predictions CSV
            data = {
                "patient_id": list(all_ids),
                "true_label": all_targets.tolist(),
                "predicted_label": all_preds.tolist(),
            }
            for c in range(num_classes):
                data[f"prob_class_{c}"] = all_probs[:, c].tolist()

            df = pd.DataFrame(data)
            save_path = os.path.join(self.recorder.save_dir, "per_case_predictions.csv")
            df.to_csv(save_path, index=False)
            print(f"[Trainer] Saved per-case predictions to: {save_path}")

        return {
            "accuracy": float(accuracy),
            "auc": float(auc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": cm,
        }
    # ---------------------------
    # 🔥 main training loop
    # ---------------------------
    def train(self):
        train_losses = []
        avg_train_losses = []
        avg_valid_metrics = []
        avg_valid_metric_names = []

        best_metric = 0
        num_epoch_no_improvement = 0
        sys.stdout.flush()

        for epoch in range(self.start_epoch, self.config.epochs):
            # reset
            train_losses = []
            
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            self.recorder.logger.info(
                'Epoch: %d/%d lr %e',
                epoch, self.config.epochs,
                self.optimizer.param_groups[0]['lr']
            )

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

            self.recorder.writer.add_scalar('Train/total_loss', np.mean(train_losses), epoch)

            # ---------------------------
            # 🔥 Validation
            # ---------------------------
            if epoch % self.config.val_epoch == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.recorder.logger.info("validating....")

                    gts = []
                    preds = []

                    for itr, (vol, gt, _) in enumerate(self.eval_dataloader):
                        vol = vol.to(self.device)
                        gt = gt.to(self.device)

                        with self._autocast_context():
                            pred = self.model(vol)                # (B, num_classes)
                        prob = torch.softmax(pred, dim=1).float() # (B, num_classes)

                        gts.extend(gt.cpu().numpy())
                        preds.extend(prob.cpu().numpy())

                    gts = np.asarray(gts)         # (N,)
                    preds = np.asarray(preds)     # (N, C)

                    # ======================================================
                    # 🔥 Universal AUC for binary & multi-class
                    # ======================================================
                    if self.use_low_ratio_policy:
                        pred_cls = np.argmax(preds, axis=1)
                        valid_metric = 100.0 * metrics.accuracy_score(gts, pred_cls)
                        valid_metric_name = "ACC"
                    elif len(np.unique(gts)) < 2 and self.use_train_val_split:
                        # Holdout mode safety fallback for extreme low-data in-train val.
                        pred_cls = np.argmax(preds, axis=1)
                        valid_metric = 100.0 * metrics.accuracy_score(gts, pred_cls)
                        valid_metric_name = "ACC_FALLBACK"
                        self.recorder.logger.warning(
                            "Validation has a single class; AUC undefined. "
                            "Holdout split is ON, so use ACC %.4f as early-stopping metric.",
                            valid_metric
                        )
                    else:
                        # build one-hot ground truth: (N, C)
                        gts_onehot = np.zeros_like(preds)
                        gts_onehot[np.arange(len(gts)), gts] = 1

                        auc = 100.0 * metrics.roc_auc_score(
                            gts_onehot,
                            preds,
                            multi_class="ovr",
                            average="macro"
                        )
                        valid_metric = auc
                        valid_metric_name = "AUC"

                    self.recorder.writer.add_scalar('Val/metric', valid_metric, epoch)

                # logging
                train_loss = np.mean(train_losses)
                avg_train_losses.append(train_loss)
                avg_valid_metrics.append(valid_metric)
                avg_valid_metric_names.append(valid_metric_name)

                self.recorder.logger.info(
                    "Epoch {}, validation {} is {:.4f}, training loss is {:.4f}".format(
                        epoch + 1, valid_metric_name, valid_metric, train_loss
                    )
                )

                # Save results
                data_frame = pd.DataFrame(
                    data={
                        'Train_Loss': avg_train_losses,
                        'Val_AUC': avg_valid_metrics,  # keep old column name for compatibility
                        'Val_Metric_Name': avg_valid_metric_names,
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
                    self.save_state_dict(epoch + 1, os.path.join(self.recorder.save_dir, "model_best.pth"))
                else:
                    if not (self.use_low_ratio_policy and self.low_ratio_disable_early_stopping):
                        num_epoch_no_improvement += 1

                if (
                    not (self.use_low_ratio_policy and self.low_ratio_disable_early_stopping)
                    and num_epoch_no_improvement == self.config.patience
                ):
                    self.recorder.logger.info("Early Stopping")
                    break

                if self.scheduler is not None:
                    self.scheduler.step(valid_metric)



        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return

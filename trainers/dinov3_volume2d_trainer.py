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

# ★ add your dataloader
from datasets_3D.Classification.abide_classification import ABIDEClassificationSet
from datasets_3D.Classification.adni_classification import ADNIClassificationSet
from datasets_3D.Classification.brats_2023_classification import BraTS_2023_ClassificationSet
from datasets_3D.Classification.Survival_upenn_classification import Survival_upenn_ClassificationSet
from networks.SliceStudent import SliceStudent

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
        self.train_dataset = ABIDEClassificationSet(
            config, 
            config.data_root,
            flag="train"
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.eval_dataset = ABIDEClassificationSet(
            config, 
            config.data_root,
            flag="valid"
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=config.val_batch,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.start_epoch = 0
        self.model = SliceStudent(
            ckpt_path=config.ckpt_path,
            n_slices=config.n_slices,
            lora_rank=config.lora_rank,
            num_classes=2
        )
        # 2. GPU
        self.model_to_gpu()

        # 3. loss
        self.criterion = nn.CrossEntropyLoss()

        # 4. optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=1e-5
        )

        # 5. scheduler
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
        print(f"[Trainer] Loaded {len(self.train_dataloader.dataset)} train volumes")
        print(f"[Trainer] Loaded {len(self.eval_dataloader.dataset)} val volumes")

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
    def forward(self):
        logits = self.model(self.volume)          # (B,2)
        self.target = self.target.long().view(-1) # CE 需要 long
        self.loss = self.criterion(logits, self.target)
        self.pred = logits

    def backward(self):
        self.loss.backward()
    # ---------------------------
    # 🔥 main training loop
    # ---------------------------
    def train(self):
        train_losses = []
        avg_train_losses = []
        avg_valid_metrics = []

        best_metric = 0
        num_epoch_no_improvement = 0
        sys.stdout.flush()

        for epoch in range(self.start_epoch, self.config.epochs):
            # reset
            train_losses = []
            
            self.model.train()
            self.recorder.logger.info(
                'Epoch: %d/%d lr %e',
                epoch, self.config.epochs,
                self.optimizer.param_groups[0]['lr']
            )

            train_bar = tqdm(self.train_dataloader)
            for itr, sample in enumerate(train_bar):
                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(self.loss.item())

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

                        pred = self.model(vol)

                        gts.extend(gt.cpu().numpy())
                        prob = torch.softmax(pred, dim=1)[:, 1]  # 正类 prob
                        preds.extend(prob.cpu().numpy())



                    gts = np.asarray(gts)
                    preds = np.asarray(preds)

                    fpr, tpr, thresholds = metrics.roc_curve(gts, preds, pos_label=1)
                    auc = 100.0 * metrics.auc(fpr, tpr)
                    valid_metric = auc

                    self.recorder.writer.add_scalar('Val/metric', valid_metric, epoch)

                # logging
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
                    self.save_state_dict(epoch + 1, os.path.join(self.recorder.save_dir, "model_best.pth"))

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

class dinov3_volume2d_trainer_ADNI(BaseTrainer):
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
        self.train_dataset = ADNIClassificationSet(
            config, 
            config.data_root,
            flag="train"
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.eval_dataset = ADNIClassificationSet(
            config, 
            config.data_root,
            flag="valid"
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=config.val_batch,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.start_epoch = 0
        self.model = SliceStudent(
            ckpt_path=config.ckpt_path,
            n_slices=config.n_slices,
            lora_rank=config.lora_rank,
            num_classes=3
        )
        # 2. GPU
        self.model_to_gpu()

        # 3. loss
        self.criterion = nn.CrossEntropyLoss()

        # 4. optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=1e-5
        )

        # 5. scheduler
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
        print(f"[Trainer] Loaded {len(self.train_dataloader.dataset)} train volumes")
        print(f"[Trainer] Loaded {len(self.eval_dataloader.dataset)} val volumes")

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
    def forward(self):
        self.pred = self.model(self.volume)      # (B,3)
        self.target = self.target.long().view(-1)
        self.loss = self.criterion(self.pred, self.target)

    def backward(self):
        self.loss.backward()
    # ---------------------------
    # 🔥 main training loop
    # ---------------------------
    def train(self):
        train_losses = []
        avg_train_losses = []
        avg_valid_metrics = []

        best_metric = 0
        num_epoch_no_improvement = 0
        sys.stdout.flush()

        for epoch in range(self.start_epoch, self.config.epochs):
            # reset
            train_losses = []
            
            self.model.train()
            self.recorder.logger.info(
                'Epoch: %d/%d lr %e',
                epoch, self.config.epochs,
                self.optimizer.param_groups[0]['lr']
            )

            train_bar = tqdm(self.train_dataloader)
            for itr, sample in enumerate(train_bar):
                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(self.loss.item())

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

                        # (B,3) logits
                        logits = self.model(vol)

                        # record gt
                        gts.extend(gt.cpu().numpy())

                        # softmax to probabilities (B,3)
                        prob = torch.softmax(logits, dim=1)
                        preds.extend(prob.cpu().numpy())

                    gts = np.asarray(gts)          # (N,)
                    preds = np.asarray(preds)      # (N,3)

                    # ---- [新增] one-hot ground truth for multi-class AUC ----
                    gts_onehot = np.zeros_like(preds)
                    gts_onehot[np.arange(len(gts)), gts] = 1

                    # ---- [新增] 计算三分类 macro-AUC ----
                    auc = 100.0 * metrics.roc_auc_score(
                        gts_onehot,
                        preds,
                        multi_class="ovr",
                        average="macro"
                    )
                    valid_metric = auc

                    self.recorder.writer.add_scalar('Val/metric', valid_metric, epoch)

                # logging
                train_loss = np.mean(train_losses)
                avg_train_losses.append(train_loss)
                avg_valid_metrics.append(valid_metric)

                self.recorder.logger.info(
                    "Epoch {}, validation AUC is {:.4f}, training loss is {:.4f}".format(
                        epoch + 1, valid_metric, train_loss
                    )
                )

                # save results
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
                    self.save_state_dict(epoch + 1, os.path.join(self.recorder.save_dir, "model_best.pth"))
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

class dinov3_volume2d_trainer_BraTS_modality(BaseTrainer):
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
        self.train_dataset = BraTS_2023_ClassificationSet(
            config, 
            config.data_root,
            flag="train"
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.eval_dataset = BraTS_2023_ClassificationSet(
            config, 
            config.data_root,
            flag="valid"
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=config.val_batch,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.start_epoch = 0
        self.model = SliceStudent(
            ckpt_path=config.ckpt_path,
            n_slices=config.n_slices,
            lora_rank=config.lora_rank,
            num_classes=4 # change this for different tasks
        )
        # 2. GPU
        self.model_to_gpu()

        # 3. loss
        self.criterion = nn.CrossEntropyLoss()

        # 4. optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=1e-5
        )

        # 5. scheduler
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
        print(f"[Trainer] Loaded {len(self.train_dataloader.dataset)} train volumes")
        print(f"[Trainer] Loaded {len(self.eval_dataloader.dataset)} val volumes")

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
    def forward(self):
        self.pred = self.model(self.volume)      # (B,3)
        self.target = self.target.long().view(-1)
        self.loss = self.criterion(self.pred, self.target)

    def backward(self):
        self.loss.backward()
    # ---------------------------
    # 🔥 main training loop
    # ---------------------------
    def train(self):
        train_losses = []
        avg_train_losses = []
        avg_valid_metrics = []

        best_metric = 0
        num_epoch_no_improvement = 0
        sys.stdout.flush()

        for epoch in range(self.start_epoch, self.config.epochs):
            # reset
            train_losses = []
            
            self.model.train()
            self.recorder.logger.info(
                'Epoch: %d/%d lr %e',
                epoch, self.config.epochs,
                self.optimizer.param_groups[0]['lr']
            )

            train_bar = tqdm(self.train_dataloader)
            for itr, sample in enumerate(train_bar):
                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(self.loss.item())

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

                        # (B,4) logits
                        logits = self.model(vol)

                        # record gt
                        gts.extend(gt.cpu().numpy())

                        # softmax → probabilities (B,4)
                        prob = torch.softmax(logits, dim=1)
                        preds.extend(prob.cpu().numpy())

                    gts = np.asarray(gts)           # (N,)
                    preds = np.asarray(preds)       # (N,4)

                    # ---- AUC (macro) ----
                    gts_onehot = np.zeros_like(preds)
                    gts_onehot[np.arange(len(gts)), gts] = 1

                    auc = 100.0 * metrics.roc_auc_score(
                        gts_onehot,
                        preds,
                        multi_class="ovr",
                        average="macro"
                    )

                    # ---- Accuracy ----
                    pred_class = np.argmax(preds, axis=1)   # (N,)
                    acc = 100.0 * metrics.accuracy_score(gts, pred_class)

                    # ---- Macro-F1 ----
                    f1 = metrics.f1_score(gts, pred_class, average="macro") * 100.0

                    valid_metric = auc  # 主 metric，继续用 AUC

                    # ---- TensorBoard ----
                    self.recorder.writer.add_scalar('Val/AUC', auc, epoch)
                    self.recorder.writer.add_scalar('Val/Accuracy', acc, epoch)
                    self.recorder.writer.add_scalar('Val/F1', f1, epoch)

                # logging
                train_loss = np.mean(train_losses)
                avg_train_losses.append(train_loss)
                avg_valid_metrics.append(valid_metric)

                self.recorder.logger.info(
                    "Epoch {} | AUC {:.2f} | Acc {:.2f}% | F1 {:.2f}% | Train Loss {:.4f}".format(
                        epoch + 1, auc, acc, f1, train_loss
                    )
                )

                # save results
                data_frame = pd.DataFrame(
                    data={
                        'Train_Loss': avg_train_losses,
                        'Val_AUC': avg_valid_metrics,
                    }
                )
                data_frame["Val_Accuracy"] = acc
                data_frame["Val_F1"] = f1

                data_frame.to_csv(
                    os.path.join(self.recorder.save_dir, "results.csv"),
                    index_label='epoch'
                )
        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return

class dinov3_volume2d_trainer_Survival_upenn(BaseTrainer):
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
        self.train_dataset = Survival_upenn_ClassificationSet(
            config, 
            config.data_root,
            flag="train",
            fold=None
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.eval_dataset = Survival_upenn_ClassificationSet(
            config, 
            config.data_root,
            flag="valid"
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=config.val_batch,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.start_epoch = 0
        self.model = SliceStudent(
            ckpt_path=config.ckpt_path,
            n_slices=config.n_slices,
            lora_rank=config.lora_rank,
            num_classes=2
        )
        # 2. GPU
        self.model_to_gpu()

        # 3. loss
        self.criterion = nn.CrossEntropyLoss()

        # 4. optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=1e-5
        )

        # 5. scheduler
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
        print(f"[Trainer] Loaded {len(self.train_dataloader.dataset)} train volumes")
        print(f"[Trainer] Loaded {len(self.eval_dataloader.dataset)} val volumes")

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
    def forward(self):
        logits = self.model(self.volume)          # (B,2)
        self.target = self.target.long().view(-1) # CE 需要 long
        self.loss = self.criterion(logits, self.target)
        self.pred = logits

    def backward(self):
        self.loss.backward()
    # ---------------------------
    # 🔥 main training loop
    # ---------------------------
    def train(self):
        train_losses = []
        avg_train_losses = []
        avg_valid_metrics = []

        best_metric = 0
        num_epoch_no_improvement = 0
        sys.stdout.flush()

        for epoch in range(self.start_epoch, self.config.epochs):
            # reset
            train_losses = []
            
            self.model.train()
            self.recorder.logger.info(
                'Epoch: %d/%d lr %e',
                epoch, self.config.epochs,
                self.optimizer.param_groups[0]['lr']
            )

            train_bar = tqdm(self.train_dataloader)
            for itr, sample in enumerate(train_bar):
                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(self.loss.item())

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

                        pred = self.model(vol)

                        gts.extend(gt.cpu().numpy())
                        prob = torch.softmax(pred, dim=1)[:, 1]  # 正类 prob
                        preds.extend(prob.cpu().numpy())



                    gts = np.asarray(gts)
                    preds = np.asarray(preds)

                    fpr, tpr, thresholds = metrics.roc_curve(gts, preds, pos_label=1)
                    auc = 100.0 * metrics.auc(fpr, tpr)
                    valid_metric = auc

                    self.recorder.writer.add_scalar('Val/metric', valid_metric, epoch)

                # logging
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
                    self.save_state_dict(epoch + 1, os.path.join(self.recorder.save_dir, "model_best.pth"))

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

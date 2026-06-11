"""Survival task launcher (multi-modal Cox-style risk model on UPENN-GBM).

How the task is done:
  * Inputs: multi-modal MRI; preprocessing handled by the survival loader.
  * Model pipeline: a frozen pretrained encoder produces per-slice CLS tokens that are
    mean-pooled into a volume embedding; a lightweight risk head is trained on top while
    the encoder stays frozen.
  * Loss: Cox partial-likelihood (risk regression).

This is a minimal launch shell. It only builds the task config and runs training;
risk-score thresholding, Kaplan-Meier stratification, log-rank testing, and any other
evaluation/reporting are left to the user.
"""
from __future__ import annotations

from pathlib import Path

from downstream_tasks.bootstrap import bootstrap_paths
from downstream_tasks.config_utils import (
    build_save_dir,
    serialize_config,
    to_multimodal_encoder_name,
    write_json,
)

bootstrap_paths()

from SurvivalRiskStratification_KaplanMeier.configs.survival_config import SurvivalConfig


def _build_config(spec, args):
    encoder = to_multimodal_encoder_name(args.encoder or spec.supported_encoders[0])
    save_root = build_save_dir(args.output_root, spec.task_id, encoder.lower(), args.train_ratio)
    epochs = args.epochs if args.epochs is not None else (2 if args.quick else 20)
    config = SurvivalConfig()
    config.encoder_name = encoder
    config.train_ratio = args.train_ratio
    config.batch_size = args.batch_size if args.batch_size is not None else 2
    config.val_batch = args.val_batch if args.val_batch is not None else 4
    config.lr = args.lr if args.lr is not None else 1e-4
    config.epochs = epochs
    config.val_epoch = 1
    config.patience = 1 if args.quick else min(10, max(3, epochs // 2))
    config.num_workers = args.num_workers
    config.gpu_ids = [args.gpu]
    config.manualseed = args.seed
    config.save_root = save_root
    config.note = f"{spec.task_id}_{encoder.lower()}"
    config.quick_probe = bool(args.quick)
    return config, encoder.lower()


def run_cox_task(spec, args):
    config, encoder = _build_config(spec, args)
    save_dir = Path(config.save_root)
    manifest = {
        "task_id": spec.task_id,
        "category": spec.category,
        "paper_task": spec.paper_task,
        "dataset": spec.dataset,
        "metric": spec.metric,
        "status": spec.status,
        "code_entry": spec.code_entry,
        "config": serialize_config(config),
    }
    write_json(save_dir / "run_manifest.json", manifest)
    if args.dry_run:
        return {
            "status": "dry_run",
            "task_id": spec.task_id,
            "encoder": encoder,
            "save_dir": str(save_dir),
            "config": serialize_config(config),
        }

    from SurvivalRiskStratification_KaplanMeier.trainers.survival_multimodal_trainer import (
        SurvivalMultiModalTrainer,
    )

    trainer = SurvivalMultiModalTrainer(config)
    trainer.train()

    return {
        "status": "trained",
        "task_id": spec.task_id,
        "encoder": encoder,
        "save_dir": str(save_dir),
    }

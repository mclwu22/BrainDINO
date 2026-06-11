"""Mutation-prediction task launcher (IDH status on UCSF-PDGM).

How the task is done:
  * Inputs: two MRI modalities (T1c + FLAIR); preprocessing handled by the UCSF loader.
  * Model pipeline: a frozen pretrained encoder produces per-slice CLS tokens that are
    mean-pooled per modality; a lightweight head is trained on top while the encoder
    stays frozen.
  * Loss: cross-entropy.
  * Metric: macro-AUC.

This is a minimal launch shell. It only builds the task config and runs training;
checkpoint selection, evaluation, and result reporting are left to the user.
"""
from __future__ import annotations

from pathlib import Path

from downstream_tasks.bootstrap import bootstrap_paths
from downstream_tasks.config_utils import (
    build_save_dir,
    serialize_config,
    to_mutation_encoder_name,
    write_json,
)

bootstrap_paths()

from configs.mutation_ucsf_config import MutationUCSFConfig


def _build_config(spec, args):
    encoder = to_mutation_encoder_name(args.encoder or spec.supported_encoders[0])
    save_root = build_save_dir(args.output_root, spec.task_id, encoder.lower(), args.train_ratio)
    epochs = args.epochs if args.epochs is not None else (2 if args.quick else 8)
    batch_size = args.batch_size if args.batch_size is not None else 4
    val_batch = args.val_batch if args.val_batch is not None else 4
    config = MutationUCSFConfig()
    config.encoder_name = encoder
    config.train_ratio = args.train_ratio
    config.batch_size = batch_size
    config.val_batch = val_batch
    config.lr = args.lr if args.lr is not None else 1e-4
    config.epochs = epochs
    config.val_epoch = 1
    config.patience = 1 if args.quick else min(4, max(2, epochs // 2))
    config.num_workers = args.num_workers
    config.gpu_ids = [args.gpu]
    config.manualseed = args.seed
    config.save_root = save_root
    config.note = f"{spec.task_id}_{encoder.lower()}"
    return config, encoder.lower()


def run_task(spec, args):
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

    from trainers.mutation_ucsf_trainer import MutationUCSFTrainer

    trainer = MutationUCSFTrainer(config)
    trainer.train()

    return {
        "status": "trained",
        "task_id": spec.task_id,
        "encoder": encoder,
        "save_dir": str(save_dir),
    }

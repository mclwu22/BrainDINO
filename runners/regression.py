from __future__ import annotations

import json
from pathlib import Path

from downstream_tasks.bootstrap import bootstrap_paths
from downstream_tasks.config_utils import (
    SimpleConfig,
    build_save_dir,
    serialize_config,
    to_general_encoder_name,
    write_json,
)

bootstrap_paths()


def _build_config(spec, args):
    encoder = to_general_encoder_name(args.encoder or spec.supported_encoders[0])
    save_root = build_save_dir(args.output_root, spec.task_id, encoder, args.train_ratio)
    epochs = args.epochs if args.epochs is not None else (2 if args.quick else 8)
    batch_size = args.batch_size if args.batch_size is not None else 4
    val_batch = args.val_batch if args.val_batch is not None else 4
    patience = 1 if args.quick else min(4, max(2, epochs // 2))
    is_atlas = spec.dataset_name == "ATLAS"
    config = SimpleConfig(
        dataset_name=spec.dataset_name,
        encoder_name=encoder,
        n_slices=128,
        lora_rank=8,
        attr="regression",
        manualseed=args.seed,
        model="slice_student",
        teacher="brainiac" if encoder == "brainiac" else None,
        use_augmentation=bool(args.use_augmentation),
        batch_size=batch_size,
        val_batch=val_batch,
        lr=args.lr if args.lr is not None else (1e-3 if is_atlas else 1e-3),
        epochs=epochs,
        val_epoch=1,
        patience=patience,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        gpu_ids=[args.gpu],
        scheduler="ReduceLROnPlateau",
        save_root=save_root,
        note=f"{spec.task_id}_{encoder}",
        benchmark=False,
        frozen=True,
        drop_path_rate=0.2,
        teacher_dropout_prob=0.2,
        optimizer_name="adamw" if is_atlas else "adamw",
        optimizer_weight_decay=1e-4,
        scheduler_mode="min",
        scheduler_factor=0.5,
        scheduler_patience=10,
        use_amp=False,
        amp_dtype="bf16",
        grad_accum_steps=1,
        regression_target_transform="log1p" if is_atlas else "none",
        atlas_loss_alpha=0.2,
        atlas_huber_delta=1.0,
    )
    return config, encoder


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

    from trainers.dinov3_volume2d_trainer_general_regression import (
        dinov3_volume2d_trainer_general_regression,
    )

    trainer = dinov3_volume2d_trainer_general_regression(config)
    trainer.train()

    best_model = save_dir / "model_best.pth"
    if best_model.exists():
        trainer.load_model_state_dict(str(best_model))

    metrics = trainer.evaluate(save_predictions=True)
    summary = {
        "task_id": spec.task_id,
        "encoder": encoder,
        "save_dir": str(save_dir),
        "metrics": metrics,
    }
    write_json(save_dir / "summary.json", summary)
    (save_dir / "summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

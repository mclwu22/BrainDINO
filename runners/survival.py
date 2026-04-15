from __future__ import annotations

import json
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

    import numpy as np
    import pandas as pd

    from SurvivalRiskStratification_KaplanMeier.trainers.survival_multimodal_trainer import (
        SurvivalMultiModalTrainer,
    )
    from SurvivalRiskStratification_KaplanMeier.utils.survival_utils import (
        compute_logrank_pvalue,
        compute_median_risk_threshold,
        plot_kaplan_meier,
        stratify_patients,
    )

    trainer = SurvivalMultiModalTrainer(config)
    trainer.train()

    best_model = save_dir / "model_best.pth"
    if best_model.exists():
        trainer.load_model_state_dict(str(best_model))

    train_risk_scores, train_survival, train_events, train_ids = trainer.predict(trainer.train_dataloader)
    val_risk_scores, val_survival, val_events, val_ids = trainer.predict(trainer.eval_dataloader)

    threshold = compute_median_risk_threshold(train_risk_scores)
    val_groups = stratify_patients(val_risk_scores, threshold)
    p_value = compute_logrank_pvalue(val_survival, val_events, val_groups)

    km_plot = save_dir / "kaplan_meier_validation.png"
    plot_kaplan_meier(
        survival_times=val_survival,
        events=val_events,
        risk_groups=val_groups,
        save_path=str(km_plot),
        title=f"Kaplan-Meier Curve - {encoder}",
    )

    risk_df = pd.DataFrame(
        {
            "patient_id": val_ids,
            "risk_score": val_risk_scores,
            "risk_group": val_groups,
            "survival_time": val_survival,
            "event": val_events,
        }
    )
    risk_csv = save_dir / "risk_stratification_results.csv"
    risk_df.to_csv(risk_csv, index=False)

    low_mask = val_groups == 0
    high_mask = val_groups == 1
    summary = {
        "task_id": spec.task_id,
        "encoder": encoder,
        "save_dir": str(save_dir),
        "metrics": {
            "threshold": float(threshold),
            "p_value": float(p_value),
            "low_risk_n": int(low_mask.sum()),
            "high_risk_n": int(high_mask.sum()),
            "low_risk_median_survival": float(np.median(val_survival[low_mask])) if low_mask.any() else None,
            "high_risk_median_survival": float(np.median(val_survival[high_mask])) if high_mask.any() else None,
            "train_sample_count": int(len(train_ids)),
            "val_sample_count": int(len(val_ids)),
        },
    }
    write_json(save_dir / "summary.json", summary)
    (save_dir / "summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

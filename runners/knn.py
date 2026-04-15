from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from downstream_tasks.config_utils import (
    build_save_dir,
    write_json,
)


VISUALIZATION_ROOT = Path(
    "/home/exx/Desktop/Med_DINOv3/Foundation_model_paper_contents/Visualization"
)
KNN_SCRIPT = VISUALIZATION_ROOT / "run_knn_eval.py"
FEATURE_CACHE_DIR = VISUALIZATION_ROOT / "feature_cache"

ENCODER_TO_KNN_MODEL = {
    "meddinov3": "BrainDINO",
    "dinov3": "DINOv3",
    "brainmvp": "BrainMVP",
    "bm_mae": "BM-MAE",
    "brainiac": "BrainIAC",
}


def _canonical_encoder(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in ENCODER_TO_KNN_MODEL:
        raise ValueError(
            f"Unsupported kNN encoder '{name}'. "
            f"Supported: {sorted(ENCODER_TO_KNN_MODEL)}"
        )
    return normalized


def _build_command(spec, args, output_csv: Path) -> tuple[list[str], str]:
    encoder = _canonical_encoder(args.encoder or "meddinov3")
    model_name = ENCODER_TO_KNN_MODEL[encoder]
    batch_size = args.batch_size if args.batch_size is not None else 32
    ks = ["1", "5"] if args.quick else ["1", "3", "5", "10", "20", "50"]
    command = [
        sys.executable,
        str(KNN_SCRIPT),
        "--datasets",
        "ADNI",
        "OASIS",
        "--models",
        model_name,
        "--ks",
        *ks,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(args.num_workers),
        "--output-csv",
        str(output_csv),
        "--cache-dir",
        str(FEATURE_CACHE_DIR),
    ]
    return command, encoder


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def run_task(spec, args):
    encoder = _canonical_encoder(args.encoder or "meddinov3")
    save_root = Path(build_save_dir(args.output_root, spec.task_id, encoder, 1.0))
    output_csv = save_root / "knn_results.csv"
    command, _ = _build_command(spec, args, output_csv)
    manifest = {
        "task_id": spec.task_id,
        "category": spec.category,
        "paper_task": spec.paper_task,
        "dataset": spec.dataset,
        "metric": spec.metric,
        "status": spec.status,
        "code_entry": spec.code_entry,
        "feature_cache_dir": str(FEATURE_CACHE_DIR),
        "command": command,
    }
    write_json(save_root / "run_manifest.json", manifest)

    if args.dry_run:
        return {
            "status": "dry_run",
            "task_id": spec.task_id,
            "encoder": encoder,
            "save_dir": str(save_root),
            "command": command,
        }

    completed = subprocess.run(
        command,
        check=True,
        cwd=str(VISUALIZATION_ROOT),
        text=True,
    )
    _ = completed

    rows = _load_rows(output_csv)
    summary = {
        "task_id": spec.task_id,
        "encoder": encoder,
        "save_dir": str(save_root),
        "results_csv": str(output_csv),
        "rows": rows,
    }
    write_json(save_root / "summary.json", summary)
    return summary

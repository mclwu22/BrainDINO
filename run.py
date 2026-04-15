from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from downstream_tasks.registry import get_task_spec, iter_task_specs
from downstream_tasks.runners import RUNNER_TARGETS


DEFAULT_OUTPUT_ROOT = str(
    Path(__file__).resolve().parents[1] / "downstream_tasks_runs"
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal organized launcher for downstream tasks in LUNA16.",
    )
    parser.add_argument("--list", action="store_true", help="List organized tasks and exit.")
    parser.add_argument("--task", type=str, help="Task id to run.")
    parser.add_argument("--encoder", type=str, default="meddinov3", help="Encoder alias.")
    parser.add_argument("--train-ratio", type=float, default=1.0, help="Training data ratio.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size.")
    parser.add_argument("--val-batch", type=int, default=None, help="Override validation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--gpu", type=int, default=0, help="Single GPU id.")
    parser.add_argument("--seed", type=int, default=111, help="Random seed.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for new organized runs.",
    )
    parser.add_argument("--quick", action="store_true", help="Use a tiny sanity-run config.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve task/config without training.")
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        help="Enable dataset-side augmentation for classification/regression tasks.",
    )
    return parser


def print_task_table() -> None:
    header = f"{'task_id':24s} {'status':12s} {'category':14s} {'dataset':22s} metric"
    print(header)
    print("-" * len(header))
    for spec in iter_task_specs():
        print(
            f"{spec.task_id:24s} {spec.status:12s} {spec.category:14s} "
            f"{spec.dataset[:22]:22s} {spec.metric}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print_task_table()
        return

    if not args.task:
        parser.error("--task is required unless --list is used.")

    spec = get_task_spec(args.task)
    if spec.status != "runnable" or spec.runner_name is None:
        raise SystemExit(
            f"Task '{spec.task_id}' is marked as '{spec.status}'. {spec.notes}"
        )

    module_name, func_name = RUNNER_TARGETS[spec.runner_name]
    module = importlib.import_module(module_name)
    runner = getattr(module, func_name)
    result = runner(spec, args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

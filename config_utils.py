from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ENCODER_ALIASES = {
    "braindino": "meddinov3",
    "brain_dino": "meddinov3",
    "meddinov3": "meddinov3",
    "dinov3": "dinov3",
    "brainmvp": "brainmvp",
    "bm-mae": "bm_mae",
    "bm_mae": "bm_mae",
    "brainiac": "brainiac",
    "scratch": "scratch",
}

GENERAL_ENCODERS = ("meddinov3", "dinov3", "brainmvp", "bm_mae", "brainiac", "scratch")


def normalize_encoder_name(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    if key not in ENCODER_ALIASES:
        raise ValueError(
            f"Unsupported encoder '{name}'. "
            f"Supported aliases: {sorted(ENCODER_ALIASES)}"
        )
    return ENCODER_ALIASES[key]


def to_general_encoder_name(name: str) -> str:
    encoder = normalize_encoder_name(name)
    if encoder not in GENERAL_ENCODERS:
        raise ValueError(f"Encoder '{name}' is not supported by the general runner.")
    return encoder


def to_mutation_encoder_name(name: str) -> str:
    encoder = normalize_encoder_name(name)
    return {"brainmvp": "BrainMVP"}.get(encoder, encoder)


def to_multimodal_encoder_name(name: str) -> str:
    encoder = normalize_encoder_name(name)
    return {"brainmvp": "BrainMVP"}.get(encoder, encoder)


def format_ratio_tag(train_ratio: float) -> str:
    return f"ratio_{int(round(float(train_ratio) * 100))}"


def build_save_dir(output_root: str, task_id: str, encoder: str, train_ratio: float) -> str:
    base = Path(output_root)
    save_dir = base / task_id / encoder / format_ratio_tag(train_ratio)
    save_dir.mkdir(parents=True, exist_ok=True)
    return str(save_dir)


def serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_value(v) for v in value]
    if hasattr(value, "__dict__"):
        return serialize_config(value)
    return str(value)


def serialize_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "__dict__"):
        return {
            key: serialize_value(val)
            for key, val in vars(config).items()
            if not key.startswith("_")
        }
    raise TypeError(f"Unsupported config type: {type(config)!r}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


class SimpleConfig:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def display(self, logger=None) -> None:
        output = logger.info if logger is not None else print
        output("=" * 60)
        output(f"Config: {getattr(self, 'note', 'unnamed_run')}")
        output("=" * 60)
        for key in sorted(vars(self)):
            output(f"{key:24s}: {getattr(self, key)}")
        output("=" * 60)

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DINO_ROOT = PROJECT_ROOT / "networks" / "dinov3"


def bootstrap_paths() -> None:
    for path in (PROJECT_ROOT, DINO_ROOT):
        as_str = str(path)
        if as_str not in sys.path:
            sys.path.insert(0, as_str)


bootstrap_paths()

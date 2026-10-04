"""Locate the source tree used for manifests and CLI artifact checks."""

import os
from pathlib import Path


def source_root() -> Path:
    configured = os.environ.get("POORSTAR_SOURCE_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "poorstar/__init__.py").is_file():
            raise ValueError("POORSTAR_SOURCE_ROOT must contain the poorstar package")
        return root
    for root in Path(__file__).resolve().parents:
        if (root / "pyproject.toml").is_file() and (root / "poorstar/__init__.py").is_file():
            return root
    return Path(__file__).resolve().parents[2]

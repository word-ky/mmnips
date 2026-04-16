from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def autodl_dir() -> Path:
    return repo_root() / ".autodl"


def remote_runs_dir() -> Path:
    return repo_root() / "remote-runs"


def newest_subdirectory(path: Path) -> Path | None:
    candidates = [item for item in path.iterdir() if item.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

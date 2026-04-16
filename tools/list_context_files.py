from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools._common import repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List tracked context and prompt files in this repository."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    return parser.parse_args()


def collect_files() -> list[str]:
    root = repo_root()
    matches: list[str] = []
    for directory_name in ("context", "prompts"):
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            matches.append(str(path.relative_to(root)))
    return matches


def main() -> None:
    args = parse_args()
    matches = collect_files()
    if args.json:
        print(json.dumps(matches, indent=2))
        return

    for relative_path in matches:
        print(relative_path)


if __name__ == "__main__":
    main()

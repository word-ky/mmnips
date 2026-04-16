from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools._common import remote_runs_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List fetched runs stored under remote-runs/.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    return parser.parse_args()


def describe_run(path: Path) -> dict[str, object]:
    summary_path = path / "artifacts" / "summary.json"
    return {
        "runId": path.name,
        "hasSummary": summary_path.exists(),
        "trainLog": str((path / "train.log").relative_to(path.parents[1])),
    }


def main() -> None:
    args = parse_args()
    base_dir = remote_runs_dir()
    runs = []
    if base_dir.exists():
        runs = [describe_run(path) for path in sorted(base_dir.iterdir()) if path.is_dir()]

    if args.json:
        print(json.dumps(runs, indent=2))
        return

    for run in runs:
        suffix = "summary" if run["hasSummary"] else "no-summary"
        print(f'{run["runId"]} [{suffix}]')


if __name__ == "__main__":
    main()

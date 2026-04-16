from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools._common import newest_subdirectory, read_json, remote_runs_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a fetched run from remote-runs/<runId>."
    )
    parser.add_argument("--run-id", help="Specific fetched run directory to inspect.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    return parser.parse_args()


def resolve_run_dir(run_id: str | None) -> Path:
    base_dir = remote_runs_dir()
    if run_id:
        run_dir = base_dir / run_id
        if not run_dir.exists():
            print(f"Run directory not found: {run_dir}", file=sys.stderr)
            raise SystemExit(1)
        return run_dir

    newest = newest_subdirectory(base_dir)
    if newest is None:
        print("No fetched runs found under remote-runs/.", file=sys.stderr)
        raise SystemExit(1)
    return newest


def collect_summary(run_dir: Path) -> dict[str, object]:
    meta_path = run_dir / "meta.json"
    summary_path = run_dir / "artifacts" / "summary.json"

    result: dict[str, object] = {
        "runId": run_dir.name,
        "metaPath": str(meta_path.relative_to(run_dir.parents[1])) if meta_path.exists() else None,
        "summaryPath": (
            str(summary_path.relative_to(run_dir.parents[1])) if summary_path.exists() else None
        ),
    }

    if meta_path.exists():
        result["meta"] = read_json(meta_path)
    if summary_path.exists():
        result["summary"] = read_json(summary_path)
    return result


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_id)
    payload = collect_summary(run_dir)

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f'runId: {payload["runId"]}')
    if payload["metaPath"]:
        print(f'metaPath: {payload["metaPath"]}')
    if payload["summaryPath"]:
        print(f'summaryPath: {payload["summaryPath"]}')

    meta = payload.get("meta")
    if isinstance(meta, dict):
        command = meta.get("command")
        if command:
            print(f"command: {command}")

    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("message", "steps", "finishedAt", "releaseId"):
            value = summary.get(key)
            if value not in (None, ""):
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()

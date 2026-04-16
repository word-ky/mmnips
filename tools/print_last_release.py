from __future__ import annotations

import argparse
import json
import sys

from tools._common import autodl_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the last deployed AutoDL release id.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_path = autodl_dir() / "last-release"
    if not release_path.exists():
        print("No .autodl/last-release file found.", file=sys.stderr)
        raise SystemExit(1)

    release_id = release_path.read_text(encoding="utf-8").strip()
    if args.json:
        print(json.dumps({"releaseId": release_id}, indent=2))
        return

    print(release_id)


if __name__ == "__main__":
    main()

"""Fetch pinned public dependencies without resetting an existing checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def setup(project: Path, names=None):
    lock = json.loads((project / "config/upstream-lock.json").read_text(encoding="utf-8"))
    for name, item in lock.items():
        if names and name not in names:
            continue
        target = project / item["directory"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", item["url"], str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", item["commit"]], check=True)
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != item["commit"]:
            raise RuntimeError(f"Existing {name} checkout has a different version; review it before changing the pin")
        print(f"VERIFIED {name} {actual}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=["scienceclaw", "vep"])
    args = parser.parse_args()
    setup(Path(__file__).resolve().parents[1], args.only)

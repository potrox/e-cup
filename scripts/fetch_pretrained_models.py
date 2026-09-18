from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch pinned upstream model snapshots")
    parser.add_argument("--config", type=Path, default=Path("configs/pretrained_models.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", action="append", dest="models")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    config_path = args.config if args.config.is_absolute() else root / args.config
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("models"), list):
        raise ValueError("invalid pretrained-model config")
    selected = set(args.models or [])
    known = {entry["key"] for entry in payload["models"]}
    if selected - known:
        raise ValueError(f"unknown model keys: {sorted(selected - known)}")
    for entry in payload["models"]:
        if selected and entry["key"] not in selected:
            continue
        destination = root / entry["local_path"]
        snapshot_download(
            repo_id=entry["name"],
            revision=entry["revision"],
            local_dir=destination,
        )
        print(destination)


if __name__ == "__main__":
    main()

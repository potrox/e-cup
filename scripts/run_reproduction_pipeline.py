from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Task 1 reproduction pipeline")
    parser.add_argument("--pipeline", type=Path, default=Path("configs/pipeline.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--from-stage")
    parser.add_argument("--to-stage")
    parser.add_argument("--stage", action="append", dest="stages")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_pipeline(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("pipeline schema_version must be 1")
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("pipeline must contain stages")
    names: list[str] = []
    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(stage.get("name"), str):
            raise ValueError("each pipeline stage requires a name")
        command = stage.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(value, str) for value in command
        ):
            raise ValueError(f"stage {stage['name']} has an invalid command")
        names.append(stage["name"])
    if len(names) != len(set(names)):
        raise ValueError("pipeline stage names must be unique")
    return stages


def _selected_stages(
    stages: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    names = [stage["name"] for stage in stages]
    if args.stages:
        unknown = sorted(set(args.stages) - set(names))
        if unknown:
            raise ValueError(f"unknown pipeline stages: {unknown}")
        selected = set(args.stages)
        return [stage for stage in stages if stage["name"] in selected]
    start = names.index(args.from_stage) if args.from_stage else 0
    end = names.index(args.to_stage) + 1 if args.to_stage else len(stages)
    if start >= end:
        raise ValueError("invalid stage range")
    return stages[start:end]


def main() -> None:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    pipeline_path = args.pipeline if args.pipeline.is_absolute() else root / args.pipeline
    selected = _selected_stages(_load_pipeline(pipeline_path.resolve(strict=True)), args)
    for stage in selected:
        command = [sys.executable if value == "{python}" else value for value in stage["command"]]
        print(json.dumps({"stage": stage["name"], "command": command}, ensure_ascii=False))
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()

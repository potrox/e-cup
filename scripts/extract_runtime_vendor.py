from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path, PurePosixPath


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the audited vendored runtime payload from an organizer-owned archive"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    with zipfile.ZipFile(args.archive) as archive:
        members = []
        seen: set[str] = set()
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or info.filename in seen:
                raise ValueError(f"unsafe or duplicate ZIP entry: {info.filename}")
            seen.add(info.filename)
            if info.filename.startswith("vendor/") and not info.is_dir():
                members.append(info.filename)
        if archive.testzip() is not None:
            raise ValueError("source archive CRC validation failed")
        if not members:
            raise ValueError("source archive has no vendor payload")
        for member in members:
            relative = PurePosixPath(member).relative_to("vendor")
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as reader, destination.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
    print(args.output)


if __name__ == "__main__":
    main()

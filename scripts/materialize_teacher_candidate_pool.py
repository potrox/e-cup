from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

DEFAULT_WEAK_CATEGORIES = ("Одежда", "Обувь", "Ювелирные изделия")


def _parse_categories(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("categories must be unique and non-empty")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a label-blind teacher candidate pool")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=300_000)
    parser.add_argument(
        "--weak-categories",
        type=_parse_categories,
        default=DEFAULT_WEAK_CATEGORIES,
    )
    parser.add_argument("--weak-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _escaped(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quotas(
    categories: list[str],
    weak: tuple[str, ...],
    rows: int,
    fraction: float,
) -> dict[str, int]:
    if rows < len(categories):
        raise ValueError("rows must cover every category")
    unknown = sorted(set(weak) - set(categories))
    if unknown:
        raise ValueError(f"unknown weak categories: {unknown}")
    strong = [category for category in categories if category not in weak]
    weak_rows = round(rows * fraction)
    weak_base, weak_extra = divmod(weak_rows, len(weak))
    strong_base, strong_extra = divmod(rows - weak_rows, len(strong))
    result: dict[str, int] = {}
    for index, category in enumerate(sorted(weak)):
        result[category] = weak_base + int(index < weak_extra)
    for index, category in enumerate(sorted(strong)):
        result[category] = strong_base + int(index < strong_extra)
    if sum(result.values()) != rows:
        raise AssertionError("quota calculation lost rows")
    return result


def main() -> None:
    args = _parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.rows < 20 or not 0.0 < args.weak_fraction < 1.0:
        raise ValueError("invalid rows or weak fraction")
    manifest_path = args.output.with_suffix(".manifest.json")
    if not args.overwrite and (args.output.exists() or manifest_path.exists()):
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    connection = duckdb.connect()
    try:
        input_sql = _escaped(args.input)
        categories = [
            str(row[0])
            for row in connection.execute(
                f"SELECT DISTINCT category FROM read_parquet('{input_sql}') ORDER BY category"
            ).fetchall()
        ]
        if len(categories) != 20:
            raise ValueError(f"expected 20 categories, got {len(categories)}")
        quotas = _quotas(categories, args.weak_categories, args.rows, args.weak_fraction)
        quota_values = ",\n".join(
            f"({_sql_string(category)}, {quota})" for category, quota in sorted(quotas.items())
        )
        output_sql = _escaped(args.output)
        connection.execute(
            f"""
            COPY (
                WITH quotas(category, quota) AS (VALUES {quota_values}),
                labeled AS (
                    SELECT
                        source.*,
                        CASE
                            WHEN target <= (2.0 / 9.0) THEN 0
                            WHEN target >= (7.0 / 9.0) THEN 1
                            ELSE 2
                        END AS candidate_stratum
                    FROM read_parquet('{input_sql}') AS source
                ),
                within_stratum AS (
                    SELECT
                        *,
                        row_number() OVER (
                            PARTITION BY category, candidate_stratum
                            ORDER BY
                                CASE WHEN candidate_stratum = 0
                                    THEN surface_similarity END DESC NULLS LAST,
                                CASE WHEN candidate_stratum = 1
                                    THEN surface_similarity END ASC NULLS LAST,
                                CASE WHEN candidate_stratum = 2
                                    THEN abs(target - 0.5) END ASC NULLS LAST,
                                hash(id1, id2, {args.seed})
                        ) AS stratum_row
                    FROM labeled
                ),
                interleaved AS (
                    SELECT
                        within_stratum.*,
                        quotas.quota,
                        row_number() OVER (
                            PARTITION BY within_stratum.category
                            ORDER BY stratum_row, candidate_stratum,
                                     hash(id1, id2, {args.seed + 1})
                        ) AS category_row
                    FROM within_stratum
                    JOIN quotas USING (category)
                )
                SELECT * EXCLUDE (quota, category_row, stratum_row)
                FROM interleaved
                WHERE category_row <= quota
                ORDER BY hash(id1, id2, {args.seed + 2})
            ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        connection.close()

    frame = pl.read_parquet(
        args.output,
        columns=["id1", "id2", "target", "category", "candidate_stratum"],
    )
    if frame.height != args.rows or frame["category"].n_unique() != 20:
        raise RuntimeError("candidate pool coverage mismatch")
    if frame.select(pl.struct("id1", "id2").n_unique()).item() != frame.height:
        raise RuntimeError("candidate pool contains duplicate pairs")
    summary = (
        frame.group_by("category", "candidate_stratum")
        .agg(pl.len().alias("rows"), pl.col("target").mean().alias("target_mean"))
        .sort("category", "candidate_stratum")
        .to_dicts()
    )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": (
            "Label-blind category quotas with round-robin organizer-negative, organizer-positive, "
            "and ambiguous strata. Negative rows prioritize high surface similarity; positive rows "
            "prioritize low similarity; no model score or human validation label is used."
        ),
        "source": {"path": str(args.input.resolve()), "sha256": _sha256(args.input)},
        "config": {
            "rows": args.rows,
            "weak_categories": list(args.weak_categories),
            "weak_fraction": args.weak_fraction,
            "seed": args.seed,
            "category_quotas": quotas,
        },
        "audit": {
            "rows": frame.height,
            "unique_pairs": frame.select(pl.struct("id1", "id2").n_unique()).item(),
            "categories": frame["category"].n_unique(),
            "strata": summary,
        },
        "output": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": _sha256(args.output),
        },
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()

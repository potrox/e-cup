"""Leakage-safe pair dataset preparation for neural product matching."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb


@dataclass(frozen=True, slots=True)
class NeuralDataConfig:
    """Pinned parameters that define the derived neural datasets."""

    llm_validation_modulus: int = 5
    llm_validation_bucket: int = 0
    llm_train_sample_modulus: int = 4
    sample_seed: int = 2026
    row_group_size: int = 32_768
    duckdb_memory_limit: str = "3GB"
    include_surface_similarity: bool = False

    def validate(self) -> None:
        if self.llm_validation_modulus < 2:
            raise ValueError("llm_validation_modulus must be at least 2")
        if not 0 <= self.llm_validation_bucket < self.llm_validation_modulus:
            raise ValueError("llm_validation_bucket is outside the modulus")
        if self.llm_train_sample_modulus < 1:
            raise ValueError("llm_train_sample_modulus must be positive")
        if self.row_group_size < 1:
            raise ValueError("row_group_size must be positive")


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def llm_split_expression(
    *,
    id1: str = "m.id1",
    id2: str = "m.id2",
    modulus: int = 5,
    validation_bucket: int = 0,
) -> str:
    """Return the DuckDB expression for a deterministic item-disjoint split.

    A pair is validation only when both endpoint items are assigned to the validation
    bucket. It is train only when neither endpoint is in that bucket. Cross-boundary
    pairs are deliberately dropped. Therefore train and validation cannot share items.
    """

    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    if not 0 <= validation_bucket < modulus:
        raise ValueError("validation_bucket is outside the modulus")
    left = f"hash({id1}) % {modulus} = {validation_bucket}"
    right = f"hash({id2}) % {modulus} = {validation_bucket}"
    return (
        f"CASE WHEN {left} AND {right} THEN 'validation' "
        f"WHEN NOT ({left}) AND NOT ({right}) THEN 'train' "
        "ELSE 'dropped_cross' END"
    )


def surface_similarity_expression(
    *,
    left_name: str = "left_item.name",
    right_name: str = "right_item.name",
) -> str:
    """Return a label-free lexical similarity expression for hard-pair diagnostics."""

    left = f"lower(coalesce({left_name}, ''))"
    right = f"lower(coalesce({right_name}, ''))"
    return (
        f"CASE WHEN length({left}) = 0 OR length({right}) = 0 THEN 0.0 "
        f"WHEN length({left}) < 2 OR length({right}) < 2 "
        f"THEN CAST({left} = {right} AS DOUBLE) "
        f"ELSE jaccard({left}, {right}) END"
    )


def _pair_projection(
    pair_relation: str,
    items_path: Path,
    *,
    include_surface_similarity: bool,
    extra_columns: str = "",
) -> str:
    items = _sql_path(items_path)
    surface_column = ""
    if include_surface_similarity:
        surface_column = f", CAST({surface_similarity_expression()} AS FLOAT) AS surface_similarity"
    return f"""
        SELECT
            p.id1,
            p.id2,
            p.target,
            left_item.category AS category,
            right_item.category AS right_category,
            left_item.name AS left_name,
            left_item.attributes AS left_attributes,
            right_item.name AS right_name,
            right_item.attributes AS right_attributes
            {surface_column}
            {extra_columns}
        FROM ({pair_relation}) AS p
        INNER JOIN read_parquet('{items}') AS left_item ON p.id1 = left_item.id
        INNER JOIN read_parquet('{items}') AS right_item ON p.id2 = right_item.id
    """


def _copy_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
    row_group_size: int,
) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    connection.execute(
        f"""
        COPY ({query}) TO '{_sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {row_group_size})
        """
    )
    return time.perf_counter() - started


def _dataset_stats(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    include_surface_similarity: bool,
) -> dict[str, Any]:
    source = _sql_path(path)
    surface_summary = ""
    if include_surface_similarity:
        surface_summary = """,
            min(surface_similarity) AS min_surface_similarity,
            avg(surface_similarity) AS mean_surface_similarity,
            max(surface_similarity) AS max_surface_similarity,
            sum(CASE WHEN NOT isfinite(surface_similarity)
                     OR surface_similarity < 0 OR surface_similarity > 1
                     THEN 1 ELSE 0 END) AS invalid_surface_similarity_rows
        """
    summary = connection.execute(
        f"""
        SELECT
            count(*) AS rows,
            count(DISTINCT id1) AS unique_left_items,
            count(DISTINCT id2) AS unique_right_items,
            avg(target) AS mean_target,
            sum(target >= 5.0 / 9.0) AS majority_positive_rows,
            sum(CASE WHEN category IS NULL OR left_name IS NULL OR right_name IS NULL
                     THEN 1 ELSE 0 END) AS required_null_rows,
            sum(CASE WHEN category IS DISTINCT FROM right_category THEN 1 ELSE 0 END)
                AS category_mismatch_rows
            {surface_summary}
        FROM read_parquet('{source}')
        """
    ).fetchone()
    if summary is None:
        raise RuntimeError(f"failed to summarize {path}")

    categories = connection.execute(
        f"""
        SELECT
            CAST(category AS VARCHAR) AS category,
            count(*) AS rows,
            avg(target) AS mean_target,
            sum(target >= 5.0 / 9.0) AS majority_positive_rows
        FROM read_parquet('{source}')
        GROUP BY category
        ORDER BY category
        """
    ).fetchall()
    result = {
        "rows": int(summary[0]),
        "unique_left_items": int(summary[1]),
        "unique_right_items": int(summary[2]),
        "mean_target": float(summary[3]),
        "majority_positive_rows": int(summary[4]),
        "required_null_rows": int(summary[5]),
        "category_mismatch_rows": int(summary[6]),
        "categories": [
            {
                "category": str(category),
                "rows": int(rows),
                "mean_target": float(mean_target),
                "majority_positive_rows": int(positive_rows),
            }
            for category, rows, mean_target, positive_rows in categories
        ],
    }
    if include_surface_similarity:
        result["surface_similarity"] = {
            "min": float(summary[7]),
            "mean": float(summary[8]),
            "max": float(summary[9]),
            "invalid_rows": int(summary[10]),
        }
    return result


def prepare_neural_pair_datasets(
    *,
    matches_llm_path: Path,
    items_path: Path,
    human_folds_path: Path,
    items_human_path: Path,
    output_dir: Path,
    config: NeuralDataConfig,
    overwrite: bool = False,
    max_rows_per_dataset: int | None = None,
) -> Path:
    """Materialize bounded pair-text datasets and a reproducibility manifest."""

    config.validate()
    for path in (matches_llm_path, items_path, human_folds_path, items_human_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if max_rows_per_dataset is not None and max_rows_per_dataset < 1:
        raise ValueError("max_rows_per_dataset must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    llm_train_filename = (
        "llm_train.parquet" if config.llm_train_sample_modulus == 1 else "llm_train_sample.parquet"
    )
    output_paths = {
        "llm_train": output_dir / llm_train_filename,
        "llm_validation": output_dir / "llm_validation.parquet",
        "human_all": output_dir / "human_all.parquet",
    }
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (*output_paths.values(), manifest_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"derived outputs already exist: {names}")

    temp_dir = output_dir / "duckdb_tmp"
    temp_dir.mkdir(exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{config.duckdb_memory_limit}'")
    connection.execute(f"SET temp_directory='{_sql_path(temp_dir)}'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET threads=16")

    split = llm_split_expression(
        modulus=config.llm_validation_modulus,
        validation_bucket=config.llm_validation_bucket,
    )
    llm = _sql_path(matches_llm_path)
    human = _sql_path(human_folds_path)
    limit = f" LIMIT {max_rows_per_dataset}" if max_rows_per_dataset is not None else ""

    llm_train_pairs = f"""
        SELECT id1, id2, target
        FROM read_parquet('{llm}') AS m
        WHERE ({split}) = 'train'
          AND hash(m.id1, m.id2, {config.sample_seed}) % {config.llm_train_sample_modulus} = 0
        {limit}
    """
    llm_validation_pairs = f"""
        SELECT id1, id2, target
        FROM read_parquet('{llm}') AS m
        WHERE ({split}) = 'validation'
        {limit}
    """
    human_pairs = f"""
        SELECT id1, id2, target, component_id, fold
        FROM read_parquet('{human}')
        {limit}
    """

    queries = {
        "llm_train": _pair_projection(
            llm_train_pairs,
            items_path,
            include_surface_similarity=config.include_surface_similarity,
        ),
        "llm_validation": _pair_projection(
            llm_validation_pairs,
            items_path,
            include_surface_similarity=config.include_surface_similarity,
        ),
        "human_all": _pair_projection(
            human_pairs,
            items_human_path,
            include_surface_similarity=config.include_surface_similarity,
            extra_columns=(
                ", CAST(p.component_id AS BIGINT) AS component_id, CAST(p.fold AS TINYINT) AS fold"
            ),
        ),
    }

    timings: dict[str, float] = {}
    stats: dict[str, dict[str, Any]] = {}
    try:
        for name, query in queries.items():
            timings[name] = _copy_parquet(
                connection,
                query,
                output_paths[name],
                config.row_group_size,
            )
            stats[name] = _dataset_stats(
                connection,
                output_paths[name],
                include_surface_similarity=config.include_surface_similarity,
            )
            if stats[name]["required_null_rows"]:
                raise ValueError(f"{name} contains missing category or product names")
            if stats[name]["category_mismatch_rows"]:
                raise ValueError(f"{name} contains cross-category product pairs")
            if (
                config.include_surface_similarity
                and stats[name]["surface_similarity"]["invalid_rows"]
            ):
                raise ValueError(f"{name} contains invalid surface similarities")

        split_counts = connection.execute(
            f"""
            SELECT ({split}) AS split, count(*) AS rows
            FROM read_parquet('{llm}') AS m
            GROUP BY split
            ORDER BY split
            """
        ).fetchall()
        overlap_summary = connection.execute(
            f"""
            WITH llm_items AS (
                SELECT id1 AS id FROM read_parquet('{llm}')
                UNION
                SELECT id2 AS id FROM read_parquet('{llm}')
            ),
            human_items AS (
                SELECT id1 AS id FROM read_parquet('{human}')
                UNION
                SELECT id2 AS id FROM read_parquet('{human}')
            )
            SELECT
                (SELECT count(*) FROM llm_items),
                (SELECT count(*) FROM human_items),
                (SELECT count(*) FROM llm_items INNER JOIN human_items USING (id))
            """
        ).fetchone()
    finally:
        connection.close()

    if overlap_summary is None:
        raise RuntimeError("failed to audit LLM/human item overlap")
    llm_unique_items, human_unique_items, overlap_items = map(int, overlap_summary)
    if overlap_items:
        raise ValueError(
            "LLM and human sources share items; build unified connected components before training"
        )

    manifest = {
        "schema_version": 2 if config.include_surface_similarity else 1,
        "created_unix_seconds": time.time(),
        "duckdb_version": duckdb.__version__,
        "config": asdict(config),
        "max_rows_per_dataset": max_rows_per_dataset,
        "split_contract": (
            "Validation contains pairs whose two item hashes are in the held-out bucket; "
            "train contains pairs whose two item hashes are outside it; cross-boundary pairs "
            "are dropped. Train and validation item sets are disjoint by construction."
        ),
        "llm_source_split_rows": {str(name): int(rows) for name, rows in split_counts},
        "source_item_overlap_audit": {
            "llm_unique_items": llm_unique_items,
            "human_unique_items": human_unique_items,
            "overlap_items": overlap_items,
            "gate": "pass" if overlap_items == 0 else "fail",
        },
        "sources": {
            "matches_llm": {
                "path": str(matches_llm_path.resolve()),
                "sha256": _sha256(matches_llm_path),
            },
            "items": {"path": str(items_path.resolve()), "sha256": _sha256(items_path)},
            "human_folds": {
                "path": str(human_folds_path.resolve()),
                "sha256": _sha256(human_folds_path),
            },
            "items_human": {
                "path": str(items_human_path.resolve()),
                "sha256": _sha256(items_human_path),
            },
        },
        "datasets": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "build_seconds": timings[name],
                **stats[name],
            }
            for name, path in output_paths.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path

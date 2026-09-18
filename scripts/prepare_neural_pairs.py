from __future__ import annotations

import argparse
from pathlib import Path

from ecup_matching.neural_data import NeuralDataConfig, prepare_neural_pair_datasets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe neural pair datasets")
    parser.add_argument("--matches-llm", type=Path, default=Path("matches_llm.parquet"))
    parser.add_argument("--items", type=Path, default=Path("items.parquet"))
    parser.add_argument(
        "--human-folds",
        type=Path,
        default=Path("data_derived/splits/human_folds.parquet"),
    )
    parser.add_argument("--items-human", type=Path, default=Path("items_human.parquet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_derived/neural/pairs_v1"),
    )
    parser.add_argument("--llm-validation-modulus", type=int, default=5)
    parser.add_argument("--llm-train-sample-modulus", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--duckdb-memory-limit", default="3GB")
    parser.add_argument("--include-surface-similarity", action="store_true")
    parser.add_argument("--max-rows-per-dataset", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = prepare_neural_pair_datasets(
        matches_llm_path=args.matches_llm,
        items_path=args.items,
        human_folds_path=args.human_folds,
        items_human_path=args.items_human,
        output_dir=args.output_dir,
        config=NeuralDataConfig(
            llm_validation_modulus=args.llm_validation_modulus,
            llm_train_sample_modulus=args.llm_train_sample_modulus,
            sample_seed=args.sample_seed,
            duckdb_memory_limit=args.duckdb_memory_limit,
            include_surface_similarity=args.include_surface_similarity,
        ),
        overwrite=args.overwrite,
        max_rows_per_dataset=args.max_rows_per_dataset,
    )
    print(manifest)


if __name__ == "__main__":
    main()

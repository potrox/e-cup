# Reproducibility contract

## Verification targets

This repository supports four independent checks:

1. **Exact trained release.** Six ordered parts reconstruct the organizer-selected release with
   SHA-256 `bac701732a04c6b74f5dbbcb0d4fe969a7c80aacc0255a5c3b47f7d02b5095e5`.
2. **Exact LLM-derived data.** Raw Qwen scores, calibrated probabilities and the replay consumed by
   student training are included with row counts, schemas and SHA-256 records.
3. **Teacher reproducibility.** A deterministic, category-balanced 2,000-pair sample can be scored
   again with the pinned Qwen revision and compared against the included reference outputs.
4. **Fresh training reproduction.** Organizer data, pinned upstream model revisions, historical
   source profiles and executable configs reproduce the complete training and inference procedure.

GPU kernels and stochastic optimization do not guarantee byte-identical newly trained weights.
The included historical release provides exact deployed weights; fresh runs are judged with the
included metric, frozen split and parity protocol.

## Environment

Create a Python 3.12 environment:

```text
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
```

Verify every included binary artifact before use:

```text
python scripts/verify_review_artifacts.py
```

## Exact selected release and trained models

The split parts are ordinary consecutive byte ranges, not a new model format. Check them without
creating another 3 GB file:

```text
python scripts/reassemble_review_artifact.py --check-only
```

Reconstruct and verify the exact archive:

```text
python scripts/reassemble_review_artifact.py
python scripts/verify_selected_release.py --archive-dir artifacts/trained_models --release ECUP2026_task1_final_full_postblend_h100_v1.zip
```

The archive contains these loadable model directories:

- `models/rumodernbert_base/`;
- `models/lamar_600m/`;
- `models/teacher_student/`.

The second selected release uses byte-identical versions of all three weight files. Its distinct
inference code and frozen manifest are versioned under
`release_profiles/teacher_distilled_k17_v1/`.

## Organizer inputs

Place the organizer files in the repository root:

- `items.parquet`;
- `items_human.parquet`;
- `matches.parquet`;
- `matches_llm.parquet`.

Expected row counts and SHA-256 values are in `evidence/data_lineage.json`. Training runners reject
a row-count, data-hash or source-code mismatch before starting a GPU run. Do not commit these four
source files to the review repository.

## Upstream open models

Download only the pinned revisions:

```text
python scripts/fetch_pretrained_models.py
```

The exact repositories, revisions and license evidence are in `configs/pretrained_models.json`,
`licenses/MODEL_LICENSE_EVIDENCE.json` and `THIRD_PARTY.md`.

## LLM labeling reproduction

The complete historical outputs are under `artifacts/llm_labels/`. To reproduce the bounded sample,
score it with the pinned local Qwen checkpoint:

```text
python scripts/score_causal_teacher.py --model models/pretrained/qwen35_9b --input artifacts/llm_labels/teacher_repro_sample_2000.parquet --output artifacts/llm_labels/teacher_repro_sample_2000_rescored.parquet --prompt-mode category_rules --batch-size 10 --max-length 768 --max-attribute-characters 1600 --bidirectional --seed 2026
python scripts/verify_teacher_repro_sample.py --expected artifacts/llm_labels/teacher_repro_sample_2000.parquet --actual artifacts/llm_labels/teacher_repro_sample_2000_rescored.parquet --output artifacts/llm_labels/teacher_repro_sample_verification.json
```

The scorer performs deterministic next-token logit evaluation for binary tokens in both pair
orientations; it does not sample generated text. The verification gate requires preserved ranking,
at least 99% binary-sign agreement and mean probability drift no greater than 0.005.

## End-to-end training

Inspect the complete ordered command graph:

```text
python scripts/run_reproduction_pipeline.py --dry-run
```

Execute the complete graph:

```text
python scripts/run_reproduction_pipeline.py
```

Execute one stage or a bounded range:

```text
python scripts/run_reproduction_pipeline.py --stage rumodernbert_gold_consolidation
python scripts/run_reproduction_pipeline.py --from-stage human_item_disjoint_folds --to-stage lamar_human_gold
```

The eight `teacher_score_shard_*` stages are independent and may run concurrently on separate
GPUs. They must all finish before `teacher_combine_shards`. Base and LAMAR use the byte-exact
historical sources under `source_profiles/base_lamar_v1`; later stages use the current source tree.
Every training config verifies the source files and input artifacts it names.

## Fresh release construction

After training, build the three component archives with `scripts/build_neural_submission.py` and
apply either frozen release graph with `scripts/build_frozen_ensemble_submission.py`. Exact command
arguments, model directories, lineage manifests, batching and serialization settings are defined in
the training configs and release-profile manifests; no leaderboard-dependent value is selected at
build time.

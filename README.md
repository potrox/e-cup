# E-CUP Task 1: product matching

> Private jury-review repository. It contains competition-derived product text and trained model
> weights. Keep access restricted to the participant and authorized E-CUP reviewers.

This repository contains the source, immutable configurations, LLM-generated labels, trained
models, historical source profiles and release contracts needed to inspect and reproduce the two
selected final solutions.

The model returns a real-valued duplicate score for every candidate pair. The official metric is
the unweighted mean of `sklearn.metrics.average_precision_score` over the 20 product categories.

## Selected solution

The stronger selected archive has this inference graph:

1. `deepvk/RuModernBERT-base`: organizer LLM soft-label stage, human-gold stage, then one low-rate
   human-gold consolidation pass.
2. `nlpai-lab/LAMAR-600m`: organizer LLM soft-label stage followed by human-gold fine-tuning.
3. A fixed category router blends raw logits from the two models.
4. Three categories use a longer RuModernBERT context.
5. A compact RuModernBERT student trained with Qwen-corrected hard-pair replay contributes only to
   the jewelry category.
6. A bounded category-rank LAMAR postblend produces the final score ordering.

The exact deployed graph is frozen under `release_profiles/full_postblend_h100_v1/`. The second
selected profile is under `release_profiles/teacher_distilled_k17_v1/`. Both profiles use the same
three trained weight payloads.

## Included review artifacts

- `artifacts/llm_labels/` contains the complete 80,000-row Qwen output, calibrated labels, the
  exact 78,000-row replay consumed by student training and a balanced 2,000-row reproducibility
  sample.
- `artifacts/trained_models/` contains the exact selected release, byte-split into six parts for
  large-file repository transport. The release contains all three trained checkpoints and the
  audited offline runtime.
- `configs/` and `scripts/` define the complete 28-stage data, training, labeling and release
  pipeline.
- `licenses/` and `THIRD_PARTY.md` record the pinned open-license model and dependency evidence.
- `PROVENANCE.md` records the participant declaration and external-input boundary.

Organizer source files are not redistributed. Reviewers use their canonical copies of
`items.parquet`, `items_human.parquet`, `matches.parquet` and `matches_llm.parquet`.

## Fast verification

Use Python 3.12:

```text
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
python scripts/verify_review_artifacts.py
python scripts/reassemble_review_artifact.py --check-only
python -m pytest
python scripts/run_reproduction_pipeline.py --dry-run
```

Materialize the exact selected release:

```text
python scripts/reassemble_review_artifact.py
python scripts/verify_selected_release.py --archive-dir artifacts/trained_models --release ECUP2026_task1_final_full_postblend_h100_v1.zip
```

Complete training and teacher-label reproduction instructions are in `REPRODUCIBILITY.md`.

## Private repository import

Extract this package into a new private repository. Keep the supplied `.gitattributes`; it routes
the large model and parquet artifacts through Git LFS. Do not add organizer source parquet files,
credentials, identity documents, local caches or generated experiment logs to the repository.

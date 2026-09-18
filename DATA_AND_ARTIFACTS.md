# Data and artifact policy

This is a private jury-review repository. The included Qwen-derived artifacts contain text derived
from organizer product cards and must not be published or reused outside E-CUP review.

## Included LLM-derived artifacts

`artifacts/llm_labels/ARTIFACT_MANIFEST.json` is the machine-readable source of truth for schemas,
row counts and SHA-256 values.

| Artifact | Role |
|---|---|
| `teacher_scored_80k_qwen35_9b.parquet` | Complete raw bidirectional Qwen scores selected for hard-pair correction |
| `teacher_scored_80k_qwen35_9b_calibrated.parquet` | Same rows with frozen calibrated probabilities |
| `replay_calibrated_w100_78k.parquet` | Exact replay consumed by teacher-student training |
| `teacher_repro_sample_2000.parquet` | Balanced bounded reference for relabeling reproducibility |

## Included trained models

`artifacts/trained_models/` contains ordered chunks of the exact selected release. Its artifact
manifest fixes every chunk hash, reconstructed archive hash and the three contained model-weight
records. `scripts/reassemble_review_artifact.py` verifies or materializes it.

The second selected inference profile uses the same three trained weight files, so they are stored
once rather than duplicated.

## Organizer-owned inputs

Organizer source parquet files are not redistributed. Their expected hashes and all deterministic
derived training-input hashes remain in `evidence/data_lineage.json`. The complete pipeline verifies
those values before training.

OOF predictions, rejected experiments, caches, logs, notebooks, resumable checkpoints and personal
documents are not part of the review repository.

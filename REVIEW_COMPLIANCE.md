# Jury review map

| Review requirement | Repository evidence |
|---|---|
| Work created during the competition | `PROVENANCE.md` plus immutable selected-release hashes verifiable against the organizer copy |
| No restricted proprietary models | `THIRD_PARTY.md`, `licenses/MODEL_LICENSE_EVIDENCE.json`, pinned revisions and standard license texts |
| Reproducible training | `configs/pipeline.json`, executable training configs, historical source profiles and `REPRODUCIBILITY.md` |
| Reproducible LLM labeling | `score_causal_teacher.py`, frozen prompt/scoring configuration, complete raw labels and the 2,000-pair reference sample |
| LLM-generated data is available | `artifacts/llm_labels/` with row-count, schema and SHA-256 manifests |
| Trained models are available | `artifacts/trained_models/` with exact release chunks, model records and reassembly verifier |
| Selected runtime is inspectable | both frozen `release_profiles/`, exact runtime code and the reconstructed selected release |

This package is intended only for a private jury repository because its derived artifacts retain
organizer product-card content.

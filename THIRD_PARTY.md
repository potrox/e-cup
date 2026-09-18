# Third-party software and models

The selected solution and its included LLM-derived labels use only models with unrestricted open
licenses. No proprietary model or remote inference service contributed labels, weights or runtime
predictions to the selected solution.

| Role | Model | Size | Pinned revision | License |
|---|---|---:|---|---|
| primary classifier | `deepvk/RuModernBERT-base` | 150M | `8aee6f818db9070b635a549580dc64dcfa03e39a` | Apache-2.0 |
| independent classifier | `nlpai-lab/LAMAR-600m` | 600M | `3af1f97945392da8ef46217640641355928aeb59` | MIT |
| hard-pair label teacher | `Qwen/Qwen3.5-9B` | 9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Apache-2.0 |

All are below the competition limit of 400B parameters. Qwen is used only during offline labeling
and is not shipped inside the inference release. Exact model-card evidence hashes and extracted
license declarations are recorded in `licenses/MODEL_LICENSE_EVIDENCE.json`. Standard license texts
are in `licenses/Apache-2.0.txt` and `licenses/MIT.txt`.

## Direct dependencies

| Dependency | License family |
|---|---|
| DuckDB | MIT |
| huggingface-hub | Apache-2.0 |
| LightGBM | MIT |
| NumPy | BSD-3-Clause |
| orjson | Apache-2.0 OR MIT |
| pandas | BSD-3-Clause |
| Polars | MIT |
| psutil | BSD-3-Clause |
| Apache Arrow | Apache-2.0 |
| RapidFuzz | MIT |
| SciPy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| PyTorch | BSD-3-Clause |
| Transformers | Apache-2.0 |

Exact resolved training versions are in `requirements.lock.txt`. The competition runtime dependency
contract and bundled license records are frozen under `configs/runtime/`, `release_profiles/` and
the included selected release.

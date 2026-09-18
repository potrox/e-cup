from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_neural_submission import (  # noqa: E402
    _MODEL_ARTIFACT_NAMES,
    _validate_neural_model,
    _validate_orjson_vendor,
    _validate_release_training_contract,
    _validate_training_lineage,
)


def _model_checkpoint(
    root: Path,
    *,
    model_type: str,
    architecture: str,
    label_count: int = 1,
) -> Path:
    root.mkdir(parents=True)
    config = {
        "model_type": model_type,
        "architectures": [architecture],
        "id2label": {str(index): f"LABEL_{index}" for index in range(label_count)},
    }
    for name in _MODEL_ARTIFACT_NAMES:
        path = root / name
        if name == "config.json":
            path.write_text(json.dumps(config), encoding="utf-8")
        else:
            path.write_bytes(f"artifact:{name}".encode())
    return root


def _checkpoint(root: Path) -> list[dict[str, object]]:
    root.mkdir(parents=True)
    artifacts: list[dict[str, object]] = []
    for name in _MODEL_ARTIFACT_NAMES:
        path = root / name
        path.write_bytes(f"artifact:{name}".encode())
        artifacts.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return artifacts


def _training_manifest(
    path: Path,
    *,
    model_source: Path,
    checkpoint: Path,
    artifacts: list[dict[str, object]],
) -> Path:
    payload = {
        "schema_version": 1,
        "config": {
            "serialization_mode": "pair_v2",
            "max_length": 256,
            "max_attribute_characters": 2048,
            "seed": 2026,
            "checkpoint_selection": "last_epoch",
        },
        "model_source": str(model_source.resolve()),
        "serialization_version": "pair_v2.1",
        "train": {"sha256": "train", "available_rows": 10, "used_rows": 10},
        "validation": {"sha256": "validation"},
        "best_epoch": 1,
        "best_checkpoint": str(checkpoint.resolve()),
        "best_checkpoint_artifacts": artifacts,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_training_lineage_requires_connected_exact_checkpoints(tmp_path: Path) -> None:
    pretrained = {
        "name": "deepvk/RuModernBERT-small",
        "revision": "cdfe49388e1f5261c69e4b1c8c41fa9e60b9ebef",
    }
    base_source = (
        tmp_path / "models--deepvk--RuModernBERT-small" / "snapshots" / pretrained["revision"]
    )
    llm_checkpoint = tmp_path / "llm" / "best"
    gold_checkpoint = tmp_path / "gold" / "best"
    llm_manifest = _training_manifest(
        tmp_path / "llm.json",
        model_source=base_source,
        checkpoint=llm_checkpoint,
        artifacts=_checkpoint(llm_checkpoint),
    )
    gold_manifest = _training_manifest(
        tmp_path / "gold.json",
        model_source=llm_checkpoint,
        checkpoint=gold_checkpoint,
        artifacts=_checkpoint(gold_checkpoint),
    )

    lineage = _validate_training_lineage(
        [llm_manifest, gold_manifest],
        final_model_dir=gold_checkpoint,
        pretrained_model=pretrained,
    )

    assert len(lineage) == 2
    assert lineage[0]["manifest_sha256"]
    assert lineage[1]["train"]["used_rows"] == 10

    disconnected = json.loads(gold_manifest.read_text(encoding="utf-8"))
    disconnected["model_source"] = str((tmp_path / "wrong-checkpoint").resolve())
    gold_manifest.write_text(json.dumps(disconnected), encoding="utf-8")
    with pytest.raises(ValueError, match="lineage is disconnected"):
        _validate_training_lineage(
            [llm_manifest, gold_manifest],
            final_model_dir=gold_checkpoint,
            pretrained_model=pretrained,
        )


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("modernbert", "ModernBertForSequenceClassification"),
        ("xlm-roberta", "XLMRobertaForSequenceClassification"),
    ],
)
def test_neural_model_validation_allows_release_supported_architectures(
    tmp_path: Path,
    model_type: str,
    architecture: str,
) -> None:
    checkpoint = _model_checkpoint(
        tmp_path / model_type,
        model_type=model_type,
        architecture=architecture,
    )

    artifacts, validated_model_type = _validate_neural_model(checkpoint)

    assert tuple(path.name for path in artifacts) == _MODEL_ARTIFACT_NAMES
    assert validated_model_type == model_type


def test_neural_model_validation_rejects_unsupported_or_inconsistent_model(
    tmp_path: Path,
) -> None:
    unsupported = _model_checkpoint(
        tmp_path / "unsupported",
        model_type="bert",
        architecture="BertForSequenceClassification",
    )
    with pytest.raises(ValueError, match="supported single-logit"):
        _validate_neural_model(unsupported)

    wrong_architecture = _model_checkpoint(
        tmp_path / "wrong-architecture",
        model_type="xlm-roberta",
        architecture="ModernBertForSequenceClassification",
    )
    with pytest.raises(ValueError, match="supported single-logit"):
        _validate_neural_model(wrong_architecture)

    multiple_labels = _model_checkpoint(
        tmp_path / "multiple-labels",
        model_type="xlm-roberta",
        architecture="XLMRobertaForSequenceClassification",
        label_count=2,
    )
    with pytest.raises(ValueError, match="supported single-logit"):
        _validate_neural_model(multiple_labels)


def test_release_contract_must_match_every_training_stage() -> None:
    matching = {
        "serializer": {
            "mode": "pair_v2",
            "version": "pair_v2.1",
            "max_length": 256,
            "max_attribute_characters": 2048,
        }
    }
    _validate_release_training_contract(
        [matching, matching],
        serialization_mode="pair_v2",
        serialization_version="pair_v2.1",
        max_length=256,
        max_attribute_characters=2048,
    )

    mismatched = {
        "serializer": {
            **matching["serializer"],
            "max_length": 384,
        }
    }
    with pytest.raises(ValueError, match="serializer contract mismatch"):
        _validate_release_training_contract(
            [matching, mismatched],
            serialization_mode="pair_v2",
            serialization_version="pair_v2.1",
            max_length=256,
            max_attribute_characters=2048,
        )


def test_orjson_vendor_requires_linux_binary_and_license(tmp_path: Path) -> None:
    package = tmp_path / "orjson"
    package.mkdir()
    (package / "orjson.cpython-312-x86_64-linux-gnu.so").write_bytes(b"binary")
    metadata = tmp_path / "orjson-3.12.0.dist-info"
    licenses = metadata / "licenses"
    licenses.mkdir(parents=True)
    (licenses / "LICENSE-MIT").write_text("license", encoding="utf-8")

    assert _validate_orjson_vendor(tmp_path) == (package, metadata)

    (licenses / "LICENSE-MIT").unlink()
    with pytest.raises(ValueError, match="license evidence"):
        _validate_orjson_vendor(tmp_path)

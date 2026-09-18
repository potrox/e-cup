# Participant provenance declaration

All task-specific source code, derived Qwen labels, fine-tuning runs, model-selection logic,
ensemble routing and release packaging represented in this repository were produced by the
participant team during the E-CUP 2026 online competition stage.

Pre-existing external inputs were limited to:

- organizer-provided Task 1 data and the organizer runtime contract;
- the pinned open-license pretrained models listed in `THIRD_PARTY.md`;
- the open-source software dependencies listed in `requirements.lock.txt`.

The exact selected release is identified by immutable SHA-256 in
`artifacts/trained_models/ARTIFACT_MANIFEST.json` and `SELECTED_RELEASES.json`. The organizer already
holds the originally submitted copy and can compare it byte-for-byte with the reconstructed
artifact supplied here.

The repository contains no identity documents, credentials, personal contact data, development
workstation paths or version-control history. Participant identity and eligibility documents are
provided only through the organizer's private form.

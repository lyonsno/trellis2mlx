"""Focused contract for the final shape-upsample width-64 LayerNorm."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from scripts.decoder_full_hash_ledger_contract import (
    FULL_DECODER_HASH_BOUNDARY_NAMES,
    validate_decoder_full_hash_ledger,
)


WITNESS_SCHEMA = "trellis2mlx.decoder_level3_norm2_width64_witness.v1"
WITNESS_ARRAY_NAMES = (
    "level3_upsample_h_c2s",
    "level3_upsample_norm2_candidate",
)
SOURCE_SCHEMA = "trellis2mlx.source_cuda_shape_slat_grid_decode.v1"
SOURCE_ROUTE = "official-source-cuda-shape-decoder-full-hash-ledger"
CANONICAL_SOURCE_REPORT_SHA256 = (
    "f4f28b9f060a2b8477e449ba8adc909dbfbe1592001550edcd9af99b882fe38d"
)
CANONICAL_SOURCE_INPUT_SHA256 = (
    "917776fe1b92b50e655a462eebc71f903895c1f1bb0c2d273b3ae98ec02af63f"
)
CANONICAL_SOURCE_CANDIDATE_SHA256 = (
    "1ceb1b3e976b57d65fea56e8f90c1d4afc73c7113c29d46935a1f1d9ea439c85"
)
INPUT_BOUNDARY = "level3_upsample_h_c2s"
CANDIDATE_BOUNDARY = "level3_upsample_norm2"
INPUT_BOUNDARY_INDEX = FULL_DECODER_HASH_BOUNDARY_NAMES.index(INPUT_BOUNDARY)
CANDIDATE_BOUNDARY_INDEX = FULL_DECODER_HASH_BOUNDARY_NAMES.index(
    CANDIDATE_BOUNDARY
)


def validate_decoder_level3_norm2_width64_witness(
    arrays: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    missing = sorted(set(WITNESS_ARRAY_NAMES) - set(arrays))
    extra = sorted(set(arrays) - set(WITNESS_ARRAY_NAMES))
    if missing or extra:
        raise ValueError(
            "decoder width-64 norm2 witness arrays mismatch: "
            f"missing={missing}, extra={extra}"
        )

    validated = {}
    rows = None
    for name in WITNESS_ARRAY_NAMES:
        values = np.asarray(arrays[name])
        if values.dtype != np.dtype(np.float16):
            raise ValueError(f"{name} must have dtype float16, got {values.dtype}")
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 64:
            raise ValueError(
                f"{name} must have nonempty shape [N, 64], got {values.shape}"
            )
        if rows is None:
            rows = int(values.shape[0])
        elif values.shape[0] != rows:
            raise ValueError("width-64 witness input and candidate rows differ")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        validated[name] = np.ascontiguousarray(values)
    return validated


def write_decoder_level3_norm2_width64_witness_npz(
    path: Path,
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    validated = validate_decoder_level3_norm2_width64_witness(arrays)
    path.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp.npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez(
            temporary_path,
            **{name: validated[name] for name in WITNESS_ARRAY_NAMES},
        )
        reopened = load_decoder_level3_norm2_width64_witness(temporary_path)
        for name in WITNESS_ARRAY_NAMES:
            if not np.array_equal(reopened[name], validated[name]):
                raise ValueError(
                    f"decoder width-64 norm2 array {name!r} changed after write"
                )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "schema": WITNESS_SCHEMA,
        "rows": int(validated[INPUT_BOUNDARY].shape[0]),
        "channels": 64,
        "array_names": list(WITNESS_ARRAY_NAMES),
        "reopened_exact": True,
    }


def load_decoder_level3_norm2_width64_witness(
    path: Path,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        duplicate_names = sorted(
            {
                name
                for name in archive.files
                if archive.files.count(name) > 1
            }
        )
        if duplicate_names:
            raise ValueError(
                "decoder width-64 norm2 witness contains duplicate members: "
                + ", ".join(duplicate_names)
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return validate_decoder_level3_norm2_width64_witness(arrays)


def load_source_width64_contract(report: Any) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise ValueError("source full-ledger report must be an object")
    route = report.get("effective_route")
    requested_route = report.get("requested_route")
    if (
        report.get("schema") != SOURCE_SCHEMA
        or report.get("status") != "done"
        or not isinstance(route, Mapping)
        or route.get("route") != SOURCE_ROUTE
        or route.get("device_type") != "cuda"
        or route.get("full_decoder_hash_ledger") is not True
        or route.get("decoder_level1_trace") is not True
        or route.get("one_model_load") is not True
    ):
        raise ValueError(
            "source full-ledger report does not prove the official CUDA route"
        )
    if not isinstance(requested_route, Mapping):
        raise ValueError("source requested route is missing")
    expected_route_fields = {
        "route": SOURCE_ROUTE,
        "full_decoder_hash_ledger": True,
        "decoder_level1_trace": True,
        "one_model_load": True,
        "decoder_output_head_backend": "torch-sparse-linear-fp32",
    }
    for name, expected in expected_route_fields.items():
        requested = requested_route.get(name)
        effective = route.get(name)
        if requested != effective:
            raise ValueError(
                "source requested/effective route disagreement at "
                f"{name}: requested={requested!r}, effective={effective!r}"
            )
        if effective != expected:
            raise ValueError(
                f"source route field {name} must be {expected!r}, "
                f"got {effective!r}"
            )
    artifacts = report.get("decoder_trace_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError(
            "source full-ledger report must name exactly one decoder trace artifact"
        )
    artifact = artifacts[0]
    if not isinstance(artifact, Mapping) or artifact.get("status") != "written":
        raise ValueError("source decoder trace artifact is not written")
    ledger = validate_decoder_full_hash_ledger(
        artifact.get("full_decoder_hash_ledger")
    )
    input_entry = ledger["entries"][INPUT_BOUNDARY_INDEX]
    candidate_entry = ledger["entries"][CANDIDATE_BOUNDARY_INDEX]
    if (
        input_entry["sha256"] != CANONICAL_SOURCE_INPUT_SHA256
        or candidate_entry["sha256"] != CANONICAL_SOURCE_CANDIDATE_SHA256
    ):
        raise ValueError(
            "source ledger does not match the canonical width-64 boundary "
            "identities"
        )
    return {
        "source_schema": SOURCE_SCHEMA,
        "source_route": SOURCE_ROUTE,
        "cuda_device": route.get("cuda_device"),
        "artifact": {
            "path": artifact.get("path"),
            "sha256": artifact.get("sha256"),
            "status": "written",
        },
        "full_decoder_hash_ledger": ledger,
        "input_entry": input_entry,
        "expected_candidate_entry": candidate_entry,
    }


def compare_decoder_level3_norm2_width64_witness(
    source_ledger: Any,
    local_prefix_entries: Any,
    local_candidate_entry: Any,
) -> dict[str, Any]:
    source = validate_decoder_full_hash_ledger(source_ledger)
    if not isinstance(local_prefix_entries, list):
        raise ValueError("local width-64 witness prefix must be a list")
    source_prefix = source["entries"][: INPUT_BOUNDARY_INDEX + 1]
    local_names = [
        entry.get("name") if isinstance(entry, Mapping) else None
        for entry in local_prefix_entries
    ]
    expected_names = [entry["name"] for entry in source_prefix]
    if local_names != expected_names:
        raise ValueError(
            "local width-64 witness prefix does not contain the exact "
            "ordered source boundaries through its input"
        )

    comparisons = []
    first_nonexact = None
    for source_entry, local_entry in zip(source_prefix, local_prefix_entries):
        if not isinstance(local_entry, Mapping):
            raise ValueError("local width-64 witness prefix entry is not an object")
        if (
            local_entry.get("dtype") != source_entry["dtype"]
            or local_entry.get("shape") != source_entry["shape"]
        ):
            raise ValueError(
                f"local width-64 witness prefix metadata differs at "
                f"{source_entry['name']}"
            )
        exact = local_entry.get("sha256") == source_entry["sha256"]
        if first_nonexact is None and not exact:
            first_nonexact = source_entry["name"]
        comparisons.append(
            {
                "name": source_entry["name"],
                "dtype": source_entry["dtype"],
                "shape": source_entry["shape"],
                "source_sha256": source_entry["sha256"],
                "local_sha256": local_entry.get("sha256"),
                "exact": exact,
            }
        )
    if first_nonexact is not None:
        raise ValueError(
            f"width-64 witness prefix first diverges at {first_nonexact}"
        )

    expected_candidate = source["entries"][CANDIDATE_BOUNDARY_INDEX]
    if (
        not isinstance(local_candidate_entry, Mapping)
        or local_candidate_entry.get("name") != CANDIDATE_BOUNDARY
        or local_candidate_entry.get("dtype") != expected_candidate["dtype"]
        or local_candidate_entry.get("shape") != expected_candidate["shape"]
    ):
        raise ValueError("local width-64 candidate entry metadata is invalid")
    candidate_exact = (
        local_candidate_entry.get("sha256") == expected_candidate["sha256"]
    )
    return {
        "schema": "trellis2mlx.decoder_level3_norm2_width64_comparison.v1",
        "status": "done",
        "prefix_exact": True,
        "input_exact": comparisons[-1]["exact"],
        "candidate_exact": candidate_exact,
        "prefix": comparisons,
        "candidate": {
            "name": CANDIDATE_BOUNDARY,
            "dtype": expected_candidate["dtype"],
            "shape": expected_candidate["shape"],
            "source_sha256": expected_candidate["sha256"],
            "local_sha256": local_candidate_entry.get("sha256"),
            "exact": candidate_exact,
        },
    }

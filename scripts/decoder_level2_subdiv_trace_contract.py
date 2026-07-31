"""Focused contract for the decoder level-two subdivision projection."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

try:
    from scripts.decoder_level2_block0_trace_contract import (
        decoder_boundary_hash_entry,
    )
    from scripts.decoder_level1_trace_contract import (
        decoder_level1_hash_entry,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from decoder_level2_block0_trace_contract import (  # type: ignore[no-redef]
        decoder_boundary_hash_entry,
    )
    from decoder_level1_trace_contract import (  # type: ignore[no-redef]
        decoder_level1_hash_entry,
    )


TRACE_SCHEMA = "trellis2mlx.decoder_level2_subdiv_trace.v1"
COMPARISON_SCHEMA = "trellis2mlx.decoder_level2_subdiv_comparison.v1"
BLOCK0_COMPARISON_SCHEMA = (
    "trellis2mlx.decoder_level2_block0_trace_comparison.v2"
)
LEDGER_COMPARISON_SCHEMA = "trellis2mlx.decoder_level1_trace_comparison.v1"
BLOCK7_BOUNDARY = "level2_block7_output"
LOGITS_BOUNDARY = "level2_upsample_subdiv_logits"
PROJECTION_DISPOSITIONS = (
    "historical-turing-fda",
    "projection-candidate",
)
TRACE_ARRAY_NAMES = (
    "level2_child_coords",
    "level2_block0_output",
    "level2_block7_output",
    "level2_upsample_subdiv_weight",
    "level2_upsample_subdiv_bias",
    "level2_upsample_subdiv_logits",
)


def validate_decoder_level2_subdiv_trace(
    arrays: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    missing = sorted(set(TRACE_ARRAY_NAMES) - set(arrays))
    extra = sorted(set(arrays) - set(TRACE_ARRAY_NAMES))
    if missing:
        raise ValueError(
            "decoder level-two subdivision trace missing required arrays: "
            + ", ".join(missing)
        )
    if extra:
        raise ValueError(
            "decoder level-two subdivision trace contains extra arrays: "
            + ", ".join(extra)
        )

    coords = np.asarray(arrays["level2_child_coords"])
    if coords.dtype != np.dtype(np.int32):
        raise ValueError(
            "level2_child_coords must have dtype int32, "
            f"got {coords.dtype}"
        )
    if coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] == 0:
        raise ValueError(
            "level2_child_coords must have nonempty shape [N, 4], "
            f"got {coords.shape}"
        )
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("level2_child_coords contains duplicate rows")

    rows = int(coords.shape[0])
    specs = {
        "level2_block0_output": (rows, 256),
        BLOCK7_BOUNDARY: (rows, 256),
        "level2_upsample_subdiv_weight": (8, 256),
        "level2_upsample_subdiv_bias": (8,),
        LOGITS_BOUNDARY: (rows, 8),
    }
    validated = {"level2_child_coords": np.ascontiguousarray(coords)}
    for name, shape in specs.items():
        values = np.asarray(arrays[name])
        if values.dtype != np.dtype(np.float16):
            raise ValueError(
                f"{name} must have dtype float16, got {values.dtype}"
            )
        if values.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        validated[name] = np.ascontiguousarray(values)
    return validated


def write_decoder_level2_subdiv_trace_npz(
    path: Path,
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    path.unlink(missing_ok=True)
    validated = validate_decoder_level2_subdiv_trace(arrays)
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
            **{name: validated[name] for name in TRACE_ARRAY_NAMES},
        )
        reopened = load_decoder_level2_subdiv_trace(temporary_path)
        for name in TRACE_ARRAY_NAMES:
            if not np.array_equal(reopened[name], validated[name]):
                raise ValueError(
                    f"decoder level-two subdivision array {name!r} "
                    "changed after write"
                )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "rows": int(validated["level2_child_coords"].shape[0]),
        "input_channels": 256,
        "output_channels": 8,
        "array_names": list(TRACE_ARRAY_NAMES),
        "reopened_exact": True,
    }


def load_decoder_level2_subdiv_trace(
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
                "decoder level-two subdivision trace contains duplicate "
                "members: " + ", ".join(duplicate_names)
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return validate_decoder_level2_subdiv_trace(arrays)


def _require_hash_row(
    ledger_comparison: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in ledger_comparison.get("hash_boundaries", ())
        if isinstance(row, Mapping) and row.get("name") == name
    ]
    if len(rows) != 1:
        raise ValueError(
            f"ledger comparison must contain exactly one {name} row"
        )
    return dict(rows[0])


def _assert_entry_equal(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    for key in ("name", "dtype", "shape", "sha256"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"{label} {key} mismatch")


def validate_parent_evidence(
    parent_arrays: Mapping[str, Any],
    block0_comparison: Mapping[str, Any],
    ledger_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    parent_names = {"level2_child_coords", "level2_block0_output"}
    if set(parent_arrays) != parent_names:
        missing = sorted(parent_names - set(parent_arrays))
        extra = sorted(set(parent_arrays) - parent_names)
        raise ValueError(
            "compact block0 parent arrays mismatch: "
            f"missing={missing}, extra={extra}"
        )
    coords = np.asarray(parent_arrays["level2_child_coords"])
    block0 = np.asarray(parent_arrays["level2_block0_output"])
    validate_decoder_level2_subdiv_trace(
        {
            "level2_child_coords": coords,
            "level2_block0_output": block0,
            BLOCK7_BOUNDARY: block0,
            "level2_upsample_subdiv_weight": np.zeros(
                (8, 256),
                dtype=np.float16,
            ),
            "level2_upsample_subdiv_bias": np.zeros(8, dtype=np.float16),
            LOGITS_BOUNDARY: np.zeros((coords.shape[0], 8), dtype=np.float16),
        }
    )

    if block0_comparison.get("schema") != BLOCK0_COMPARISON_SCHEMA:
        raise ValueError("block0 comparison schema mismatch")
    if block0_comparison.get("status") != "done":
        raise ValueError("block0 comparison is not done")
    if block0_comparison.get("first_nonexact_boundary") is not None:
        raise ValueError("block0 comparison is not exact")
    disposition = block0_comparison.get("parent_fork_disposition")
    if not isinstance(disposition, Mapping):
        raise ValueError("block0 comparison lacks fork disposition")
    if (
        disposition.get("requested") != "corrected-child-exact-to-source"
        or disposition.get("effective")
        != "corrected-child-exact-to-source"
        or disposition.get("all_block_boundaries_exact_to_source") is not True
    ):
        raise ValueError("block0 comparison correction disposition mismatch")

    coord_entry = decoder_boundary_hash_entry("level2_child_coords", coords)
    block0_entry = decoder_boundary_hash_entry("level2_block0_output", block0)
    child_inputs = block0_comparison.get("child_input_identity")
    child_outputs = block0_comparison.get("child_output_identity")
    if not isinstance(child_inputs, Mapping) or not isinstance(
        child_outputs,
        Mapping,
    ):
        raise ValueError("block0 comparison lacks child identities")
    for label in ("source", "local"):
        input_identity = child_inputs.get(label)
        output_identity = child_outputs.get(label)
        if not isinstance(input_identity, Mapping) or not isinstance(
            output_identity,
            Mapping,
        ):
            raise ValueError(f"block0 comparison lacks {label} identity")
        reported_coords = input_identity.get("level2_child_coords")
        if not isinstance(reported_coords, Mapping):
            raise ValueError(f"block0 comparison lacks {label} coordinates")
        _assert_entry_equal(
            coord_entry,
            reported_coords,
            f"{label} block0 parent coordinates",
        )
        _assert_entry_equal(
            block0_entry,
            output_identity,
            f"{label} block0 output",
        )

    if ledger_comparison.get("schema") != LEDGER_COMPARISON_SCHEMA:
        raise ValueError("ledger comparison schema mismatch")
    if ledger_comparison.get("status") != "done":
        raise ValueError("ledger comparison is not done")
    if ledger_comparison.get("first_nonexact_hash_boundary") != LOGITS_BOUNDARY:
        raise ValueError(
            "ledger comparison first nonexact hash boundary mismatch"
        )
    block7_row = _require_hash_row(ledger_comparison, BLOCK7_BOUNDARY)
    logits_row = _require_hash_row(ledger_comparison, LOGITS_BOUNDARY)
    if (
        block7_row.get("exact") is not True
        or block7_row.get("source_sha256")
        != block7_row.get("local_sha256")
    ):
        raise ValueError("ledger block7 row is not exact")
    if (
        logits_row.get("exact") is not False
        or logits_row.get("source_sha256")
        == logits_row.get("local_sha256")
    ):
        raise ValueError("ledger subdivision logits row is not a fork")
    return {
        "level2_child_coords": coord_entry,
        "level2_block0_output": block0_entry,
        BLOCK7_BOUNDARY: block7_row,
        LOGITS_BOUNDARY: logits_row,
    }


def _numeric_delta(source: np.ndarray, local: np.ndarray) -> dict[str, Any]:
    delta = local.astype(np.float64) - source.astype(np.float64)
    absolute = np.abs(delta)
    return {
        "dtype": str(source.dtype),
        "shape": [int(value) for value in source.shape],
        "mean_abs": float(absolute.mean()),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(absolute.max()),
        "nonzero_count": int(np.count_nonzero(delta)),
    }


def compare_decoder_level2_subdiv_traces(
    source_arrays: Mapping[str, Any],
    local_arrays: Mapping[str, Any],
    *,
    block0_comparison: Mapping[str, Any],
    ledger_comparison: Mapping[str, Any],
    projection_disposition: str,
) -> dict[str, Any]:
    if projection_disposition not in PROJECTION_DISPOSITIONS:
        raise ValueError(
            f"unknown projection disposition {projection_disposition!r}"
        )
    source = validate_decoder_level2_subdiv_trace(source_arrays)
    local = validate_decoder_level2_subdiv_trace(local_arrays)
    parent = validate_parent_evidence(
        {
            "level2_child_coords": source["level2_child_coords"],
            "level2_block0_output": source["level2_block0_output"],
        },
        block0_comparison,
        ledger_comparison,
    )
    for name in (
        "level2_child_coords",
        "level2_block0_output",
        BLOCK7_BOUNDARY,
        "level2_upsample_subdiv_weight",
        "level2_upsample_subdiv_bias",
    ):
        if not np.array_equal(source[name], local[name]):
            raise ValueError(f"source/local {name} mismatch")

    source_block7 = decoder_level1_hash_entry(
        BLOCK7_BOUNDARY,
        source[BLOCK7_BOUNDARY],
    )
    local_block7 = decoder_level1_hash_entry(
        BLOCK7_BOUNDARY,
        local[BLOCK7_BOUNDARY],
    )
    block7_row = parent[BLOCK7_BOUNDARY]
    for label, entry in (
        ("source block7", source_block7),
        ("local block7", local_block7),
    ):
        expected = {
            "name": block7_row["name"],
            "dtype": block7_row["dtype"],
            "shape": block7_row["shape"],
            "sha256": block7_row[
                "source_sha256" if label.startswith("source") else "local_sha256"
            ],
        }
        _assert_entry_equal(entry, expected, label)

    source_logits = decoder_level1_hash_entry(
        LOGITS_BOUNDARY,
        source[LOGITS_BOUNDARY],
    )
    local_logits = decoder_level1_hash_entry(
        LOGITS_BOUNDARY,
        local[LOGITS_BOUNDARY],
    )
    logits_row = parent[LOGITS_BOUNDARY]
    expected_source = {
        "name": logits_row["name"],
        "dtype": logits_row["dtype"],
        "shape": logits_row["shape"],
        "sha256": logits_row["source_sha256"],
    }
    _assert_entry_equal(source_logits, expected_source, "source logits")
    if (
        projection_disposition == "historical-turing-fda"
        and local_logits["sha256"] != logits_row["local_sha256"]
    ):
        raise ValueError("historical local logits hash mismatch")

    delta = _numeric_delta(source[LOGITS_BOUNDARY], local[LOGITS_BOUNDARY])
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "done",
        "first_nonexact_boundary": (
            LOGITS_BOUNDARY if delta["nonzero_count"] else None
        ),
        "projection_disposition": {
            "requested": projection_disposition,
            "effective": projection_disposition,
            "source_sha256": source_logits["sha256"],
            "historical_local_sha256": logits_row["local_sha256"],
            "current_local_sha256": local_logits["sha256"],
        },
        "parent_identity": {
            key: parent[key]
            for key in (
                "level2_child_coords",
                "level2_block0_output",
                BLOCK7_BOUNDARY,
                LOGITS_BOUNDARY,
            )
        },
        "stages": {LOGITS_BOUNDARY: delta},
    }

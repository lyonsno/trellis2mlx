"""Pure-Numpy hash contract for the remaining shape-decoder frontier."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np


FULL_DECODER_HASH_LEDGER_SCHEMA = (
    "trellis2mlx.decoder_full_hash_ledger.v1"
)
FULL_DECODER_HASH_BOUNDARY_NAMES = (
    "level2_upsample_output",
    "level3_block0_conv",
    "level3_block0_norm",
    "level3_block0_mlp_fc1",
    "level3_block0_silu",
    "level3_block0_mlp_fc2",
    "level3_block0_output",
    "level3_block1_output",
    "level3_block2_output",
    "level3_block3_output",
    "level3_upsample_subdiv_logits",
    "level3_upsample_norm1",
    "level3_upsample_silu1",
    "level3_upsample_conv1",
    "level4_child_coords",
    "level3_upsample_h_c2s",
    "level3_upsample_skip_c2s",
    "level3_upsample_skip_repeated",
    "level3_upsample_norm2",
    "level3_upsample_silu2",
    "level3_upsample_conv2",
    "level3_upsample_output",
    "decoder_final_layernorm",
    "decoder_output",
)


def decoder_full_hash_entry(
    name: str,
    values: Any,
) -> dict[str, Any]:
    if name not in FULL_DECODER_HASH_BOUNDARY_NAMES:
        raise ValueError(f"unknown full-decoder hash boundary {name!r}")
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(name.encode("ascii") + b"\0")
    digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
    digest.update(
        ",".join(str(value) for value in contiguous.shape).encode("ascii")
        + b"\0"
    )
    digest.update(contiguous.tobytes())
    return {
        "name": name,
        "dtype": str(contiguous.dtype),
        "shape": [int(value) for value in contiguous.shape],
        "sha256": digest.hexdigest(),
    }


def _ledger_row_count(entries: list[Any], name: str) -> int:
    entry = entries[FULL_DECODER_HASH_BOUNDARY_NAMES.index(name)]
    shape = entry.get("shape") if isinstance(entry, Mapping) else None
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not isinstance(shape[0], int)
        or shape[0] <= 0
    ):
        raise ValueError(
            f"full-decoder hash ledger row count for {name} is invalid"
        )
    return shape[0]


def _expected_specs(
    level3_rows: int,
    level4_rows: int,
) -> dict[str, tuple[tuple[int, ...], str]]:
    specs: dict[str, tuple[tuple[int, ...], str]] = {
        "level2_upsample_output": ((level3_rows, 128), "float16"),
        "level3_block0_conv": ((level3_rows, 128), "float16"),
        "level3_block0_norm": ((level3_rows, 128), "float16"),
        "level3_block0_mlp_fc1": ((level3_rows, 512), "float16"),
        "level3_block0_silu": ((level3_rows, 512), "float16"),
        "level3_block0_mlp_fc2": ((level3_rows, 128), "float16"),
        "level3_block0_output": ((level3_rows, 128), "float16"),
        "level3_block1_output": ((level3_rows, 128), "float16"),
        "level3_block2_output": ((level3_rows, 128), "float16"),
        "level3_block3_output": ((level3_rows, 128), "float16"),
        "level3_upsample_subdiv_logits": ((level3_rows, 8), "float16"),
        "level3_upsample_norm1": ((level3_rows, 128), "float16"),
        "level3_upsample_silu1": ((level3_rows, 128), "float16"),
        "level3_upsample_conv1": ((level3_rows, 512), "float16"),
        "level4_child_coords": ((level4_rows, 4), "int32"),
        "level3_upsample_h_c2s": ((level4_rows, 64), "float16"),
        "level3_upsample_skip_c2s": ((level4_rows, 16), "float16"),
        "level3_upsample_skip_repeated": ((level4_rows, 64), "float16"),
        "level3_upsample_norm2": ((level4_rows, 64), "float16"),
        "level3_upsample_silu2": ((level4_rows, 64), "float16"),
        "level3_upsample_conv2": ((level4_rows, 64), "float16"),
        "level3_upsample_output": ((level4_rows, 64), "float16"),
        "decoder_final_layernorm": ((level4_rows, 64), "float32"),
        "decoder_output": ((level4_rows, 7), "float32"),
    }
    return specs


def validate_decoder_full_hash_ledger(
    ledger: Any,
) -> dict[str, Any]:
    if not isinstance(ledger, Mapping):
        raise ValueError("full-decoder hash ledger must be an object")
    if set(ledger) != {"schema", "entries"}:
        raise ValueError(
            "full-decoder hash ledger must contain only schema and entries"
        )
    if ledger.get("schema") != FULL_DECODER_HASH_LEDGER_SCHEMA:
        raise ValueError("full-decoder hash ledger schema mismatch")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("full-decoder hash ledger entries must be a list")
    names = [
        entry.get("name") if isinstance(entry, Mapping) else None
        for entry in entries
    ]
    if names != list(FULL_DECODER_HASH_BOUNDARY_NAMES):
        raise ValueError(
            "full-decoder hash ledger must contain the exact ordered boundaries"
        )

    specs = _expected_specs(
        _ledger_row_count(entries, "level2_upsample_output"),
        _ledger_row_count(entries, "level4_child_coords"),
    )
    normalized_entries = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(
                "full-decoder hash ledger entry must be an object"
            )
        if set(entry) != {"name", "dtype", "shape", "sha256"}:
            raise ValueError(
                "full-decoder hash ledger entries require "
                "name, dtype, shape, and sha256"
            )
        name = entry["name"]
        expected_shape, expected_dtype = specs[name]
        if entry["shape"] != list(expected_shape):
            raise ValueError(
                f"full-decoder hash boundary {name} shape mismatch: "
                f"expected={expected_shape}, actual={entry['shape']}"
            )
        if entry["dtype"] != expected_dtype:
            raise ValueError(
                f"full-decoder hash boundary {name} dtype mismatch: "
                f"expected={expected_dtype}, actual={entry['dtype']}"
            )
        digest = entry["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"full-decoder hash boundary {name} SHA256 is not canonical"
            )
        normalized_entries.append(
            {
                "name": name,
                "dtype": expected_dtype,
                "shape": list(expected_shape),
                "sha256": digest,
            }
        )
    return {
        "schema": FULL_DECODER_HASH_LEDGER_SCHEMA,
        "entries": normalized_entries,
    }


def build_decoder_full_hash_ledger(
    boundaries: Mapping[str, Any],
) -> dict[str, Any]:
    missing = sorted(set(FULL_DECODER_HASH_BOUNDARY_NAMES) - set(boundaries))
    extra = sorted(set(boundaries) - set(FULL_DECODER_HASH_BOUNDARY_NAMES))
    if missing or extra:
        raise ValueError(
            "full-decoder hash ledger boundaries mismatch: "
            f"missing={missing}, extra={extra}"
        )
    ledger = {
        "schema": FULL_DECODER_HASH_LEDGER_SCHEMA,
        "entries": [
            decoder_full_hash_entry(name, boundaries[name])
            for name in FULL_DECODER_HASH_BOUNDARY_NAMES
        ],
    }
    return validate_decoder_full_hash_ledger(ledger)


def compare_decoder_full_hash_ledgers(
    source_ledger: Any,
    local_ledger: Any,
    *,
    source_parent_entry: Any,
    local_parent_entry: Any,
) -> dict[str, Any]:
    source = validate_decoder_full_hash_ledger(source_ledger)
    local = validate_decoder_full_hash_ledger(local_ledger)
    for label, ledger, parent_entry in (
        ("source", source, source_parent_entry),
        ("local", local, local_parent_entry),
    ):
        if not isinstance(parent_entry, Mapping):
            raise ValueError(f"{label} parent entry must be an object")
        if dict(parent_entry) != ledger["entries"][0]:
            raise ValueError(
                f"{label} full ledger parent does not match the "
                "authenticated level-two output entry"
            )
    if source["entries"][0]["sha256"] != local["entries"][0]["sha256"]:
        raise ValueError("source and local full-decoder parent hashes differ")

    first_nonexact_boundary = None
    boundaries = []
    for source_entry, local_entry in zip(
        source["entries"],
        local["entries"],
    ):
        exact = source_entry["sha256"] == local_entry["sha256"]
        if first_nonexact_boundary is None and not exact:
            first_nonexact_boundary = source_entry["name"]
        boundaries.append(
            {
                "name": source_entry["name"],
                "dtype": source_entry["dtype"],
                "shape": source_entry["shape"],
                "source_sha256": source_entry["sha256"],
                "local_sha256": local_entry["sha256"],
                "exact": exact,
            }
        )
    return {
        "schema": "trellis2mlx.decoder_full_hash_comparison.v1",
        "status": "done",
        "parent_exact": True,
        "first_nonexact_boundary": first_nonexact_boundary,
        "boundaries": boundaries,
    }

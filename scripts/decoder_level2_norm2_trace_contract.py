"""Focused contract for decoder level-two post-upsample LayerNorm."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


TRACE_SCHEMA = "trellis2mlx.decoder_level2_norm2_trace.v1"
TRACE_ARRAY_NAMES = (
    "level3_child_coords",
    "level2_upsample_h_c2s",
    "level2_upsample_norm2",
)


def validate_decoder_level2_norm2_trace(
    arrays: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    missing = sorted(set(TRACE_ARRAY_NAMES) - set(arrays))
    extra = sorted(set(arrays) - set(TRACE_ARRAY_NAMES))
    if missing:
        raise ValueError(
            "decoder level-two norm2 trace missing required arrays: "
            + ", ".join(missing)
        )
    if extra:
        raise ValueError(
            "decoder level-two norm2 trace contains extra arrays: "
            + ", ".join(extra)
        )

    coords = np.asarray(arrays["level3_child_coords"])
    if coords.dtype != np.dtype(np.int32):
        raise ValueError(
            "level3_child_coords must have dtype int32, "
            f"got {coords.dtype}"
        )
    if coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] == 0:
        raise ValueError(
            "level3_child_coords must have nonempty shape [N, 4], "
            f"got {coords.shape}"
        )
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("level3_child_coords contains duplicate rows")

    rows = int(coords.shape[0])
    validated = {"level3_child_coords": np.ascontiguousarray(coords)}
    for name in (
        "level2_upsample_h_c2s",
        "level2_upsample_norm2",
    ):
        values = np.asarray(arrays[name])
        if values.dtype != np.dtype(np.float16):
            raise ValueError(
                f"{name} must have dtype float16, got {values.dtype}"
            )
        if values.shape != (rows, 128):
            raise ValueError(
                f"{name} must have shape [N, 128], got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        validated[name] = np.ascontiguousarray(values)
    return validated


def write_decoder_level2_norm2_trace_npz(
    path: Path,
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    validated = validate_decoder_level2_norm2_trace(arrays)
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
            **{name: validated[name] for name in TRACE_ARRAY_NAMES},
        )
        reopened = load_decoder_level2_norm2_trace(temporary_path)
        for name in TRACE_ARRAY_NAMES:
            if not np.array_equal(reopened[name], validated[name]):
                raise ValueError(
                    f"decoder level-two norm2 array {name!r} "
                    "changed after write"
                )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "schema": TRACE_SCHEMA,
        "rows": int(validated["level3_child_coords"].shape[0]),
        "channels": 128,
        "array_names": list(TRACE_ARRAY_NAMES),
        "reopened_exact": True,
    }


def load_decoder_level2_norm2_trace(
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
                "decoder level-two norm2 trace contains duplicate members: "
                + ", ".join(duplicate_names)
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return validate_decoder_level2_norm2_trace(arrays)

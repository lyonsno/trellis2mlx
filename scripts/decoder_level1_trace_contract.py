"""Pure-Numpy artifact contract for the first decoder upsample and level one."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


PARENT_TRACE_NAMES = (
    "level0_output",
    "upsample_subdiv_logits",
    "upsample_norm1",
    "upsample_silu1",
    "upsample_conv1",
)
CHILD_TRACE_NAMES = (
    "upsample_h_c2s",
    "upsample_skip_c2s",
    "upsample_skip_repeated",
    "upsample_norm2",
    "upsample_silu2",
    "upsample_conv2",
    "upsample_output",
    "level1_block0_conv",
    "level1_block0_norm",
    "level1_block0_mlp_fc1",
    "level1_block0_silu",
    "level1_block0_mlp_fc2",
    "level1_block0_output",
)
TRACE_NAMES = PARENT_TRACE_NAMES + CHILD_TRACE_NAMES
REQUIRED_ARRAYS = ("parent_coords", "child_coords") + TRACE_NAMES


def decoder_level1_trace_input_sha256(
    feats: np.ndarray,
    coords: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, values in (
        ("level0_output", np.asarray(feats)),
        ("parent_coords", np.asarray(coords)),
    ):
        contiguous = np.ascontiguousarray(values)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
        digest.update(
            ",".join(str(value) for value in contiguous.shape).encode("ascii")
            + b"\0"
        )
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_coords(name: str, values: Any) -> np.ndarray:
    coords = np.asarray(values)
    if coords.dtype != np.dtype(np.int32):
        raise ValueError(f"{name} must have dtype int32, got {coords.dtype}")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N, 4], got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{name} must be nonempty")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError(f"{name} contains duplicate rows")
    return np.ascontiguousarray(coords)


def _expected_child_coords(
    parent_coords: np.ndarray,
    subdiv_mask: np.ndarray,
) -> np.ndarray:
    parent_indices, child_indices = np.nonzero(subdiv_mask)
    if parent_indices.size == 0:
        raise ValueError("upsample subdivision mask selects no children")
    expected = parent_coords[parent_indices].copy()
    expected[:, 1:] *= 2
    expected[:, 1] += child_indices % 2
    expected[:, 2] += (child_indices // 2) % 2
    expected[:, 3] += child_indices // 4
    return np.ascontiguousarray(expected, dtype=np.int32)


def validate_decoder_level1_trace(
    arrays: Mapping[str, Any],
    *,
    parent_channels: int = 1024,
    child_channels: int = 512,
    torso_dtype: np.dtype[Any] | type[np.generic] = np.float16,
) -> dict[str, np.ndarray]:
    missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
    if missing:
        raise KeyError(
            "decoder level-one trace missing required arrays: "
            + ", ".join(missing)
        )
    extra = sorted(set(arrays) - set(REQUIRED_ARRAYS))
    if extra:
        raise KeyError(
            "decoder level-one trace contains unexpected arrays: "
            + ", ".join(extra)
        )
    if parent_channels != child_channels * 2:
        raise ValueError(
            "first decoder upsample requires parent_channels == "
            f"2 * child_channels, got {parent_channels} and {child_channels}"
        )
    if parent_channels % 8:
        raise ValueError(
            f"parent_channels must be divisible by 8, got {parent_channels}"
        )

    parent_coords = _validate_coords("parent_coords", arrays["parent_coords"])
    child_coords = _validate_coords("child_coords", arrays["child_coords"])
    parent_rows = int(parent_coords.shape[0])
    child_rows = int(child_coords.shape[0])
    expected_dtype = np.dtype(torso_dtype)

    expected_specs = {
        "level0_output": (parent_rows, parent_channels),
        "upsample_subdiv_logits": (parent_rows, 8),
        "upsample_norm1": (parent_rows, parent_channels),
        "upsample_silu1": (parent_rows, parent_channels),
        "upsample_conv1": (parent_rows, child_channels * 8),
        "upsample_h_c2s": (child_rows, child_channels),
        "upsample_skip_c2s": (child_rows, parent_channels // 8),
        "upsample_skip_repeated": (child_rows, child_channels),
        "upsample_norm2": (child_rows, child_channels),
        "upsample_silu2": (child_rows, child_channels),
        "upsample_conv2": (child_rows, child_channels),
        "upsample_output": (child_rows, child_channels),
        "level1_block0_conv": (child_rows, child_channels),
        "level1_block0_norm": (child_rows, child_channels),
        "level1_block0_mlp_fc1": (child_rows, child_channels * 4),
        "level1_block0_silu": (child_rows, child_channels * 4),
        "level1_block0_mlp_fc2": (child_rows, child_channels),
        "level1_block0_output": (child_rows, child_channels),
    }
    validated: dict[str, np.ndarray] = {
        "parent_coords": parent_coords,
        "child_coords": child_coords,
    }
    for name in TRACE_NAMES:
        values = np.asarray(arrays[name])
        if values.dtype != expected_dtype:
            raise ValueError(
                f"{name} must have dtype {expected_dtype}, got {values.dtype}"
            )
        expected_shape = expected_specs[name]
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        validated[name] = np.ascontiguousarray(values)

    subdiv_mask = validated["upsample_subdiv_logits"] > 0
    expected_child_coords = _expected_child_coords(parent_coords, subdiv_mask)
    if not np.array_equal(child_coords, expected_child_coords):
        raise ValueError(
            "child_coords do not exactly match parent-major, ascending-child "
            "channel-to-spatial expansion"
        )
    expected_h = validated["upsample_conv1"].reshape(
        parent_rows,
        8,
        child_channels,
    )[subdiv_mask]
    if not np.array_equal(validated["upsample_h_c2s"], expected_h):
        raise ValueError(
            "upsample_h_c2s does not exactly match conv1 channel slices"
        )
    expected_skip = validated["level0_output"].reshape(
        parent_rows,
        8,
        parent_channels // 8,
    )[subdiv_mask]
    if not np.array_equal(validated["upsample_skip_c2s"], expected_skip):
        raise ValueError(
            "upsample_skip_c2s does not exactly match level0 channel slices"
        )
    repeat_factor = child_channels // (parent_channels // 8)
    expected_repeated = np.repeat(expected_skip, repeat_factor, axis=1)
    if not np.array_equal(
        validated["upsample_skip_repeated"],
        expected_repeated,
    ):
        raise ValueError(
            "upsample_skip_repeated does not exactly match source repeat order"
        )
    return validated


def load_decoder_level1_trace(
    path: Path,
    *,
    parent_channels: int = 1024,
    child_channels: int = 512,
    torso_dtype: np.dtype[Any] | type[np.generic] = np.float16,
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
                "decoder level-one trace contains duplicate NPZ members: "
                + ", ".join(duplicate_names)
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return validate_decoder_level1_trace(
        arrays,
        parent_channels=parent_channels,
        child_channels=child_channels,
        torso_dtype=torso_dtype,
    )


def write_decoder_level1_trace_npz(
    path: Path,
    arrays: Mapping[str, Any],
    *,
    parent_channels: int = 1024,
    child_channels: int = 512,
    torso_dtype: np.dtype[Any] | type[np.generic] = np.float16,
) -> dict[str, Any]:
    path = Path(path)
    validated = validate_decoder_level1_trace(
        arrays,
        parent_channels=parent_channels,
        child_channels=child_channels,
        torso_dtype=torso_dtype,
    )
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
            **{name: validated[name] for name in REQUIRED_ARRAYS},
        )
        reopened = load_decoder_level1_trace(
            temporary_path,
            parent_channels=parent_channels,
            child_channels=child_channels,
            torso_dtype=torso_dtype,
        )
        for name in REQUIRED_ARRAYS:
            if not np.array_equal(reopened[name], validated[name]):
                raise ValueError(
                    f"decoder level-one trace array {name!r} changed after write"
                )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "parent_rows": int(validated["parent_coords"].shape[0]),
        "child_rows": int(validated["child_coords"].shape[0]),
        "parent_channels": int(parent_channels),
        "child_channels": int(child_channels),
        "torso_dtype": str(np.dtype(torso_dtype)),
        "trace_names": list(TRACE_NAMES),
        "reopened_exact": True,
        "child_expansion_exact": True,
    }

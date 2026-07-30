"""Pure-Numpy artifact contract for shape-decoder level-zero traces."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


TRACE_NAMES = (
    "input_feats",
    "from_latent_fp32",
    "torso_input",
    "block0_conv",
    "block0_norm",
    "block0_mlp_fc1",
    "block0_silu",
    "block0_mlp_fc2",
    "block0_output",
    "block1_output",
    "block2_output",
    "block3_output",
    "level0_subdiv_logits",
)
REQUIRED_ARRAYS = ("coords",) + TRACE_NAMES


def decoder_trace_input_sha256(
    feats: np.ndarray,
    coords: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, values in (
        ("feats", np.asarray(feats)),
        ("coords", np.asarray(coords)),
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


def validate_decoder_level0_trace(
    arrays: Mapping[str, Any],
    *,
    latent_channels: int,
    channels: int,
    torso_dtype: np.dtype[Any] | type[np.generic],
) -> dict[str, np.ndarray]:
    missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
    if missing:
        raise KeyError(
            "decoder level-zero trace missing required arrays: "
            + ", ".join(missing)
        )
    extra = sorted(set(arrays) - set(REQUIRED_ARRAYS))
    if extra:
        raise KeyError(
            "decoder level-zero trace contains unexpected arrays: "
            + ", ".join(extra)
        )

    coords = np.asarray(arrays["coords"])
    if coords.dtype != np.dtype(np.int32):
        raise ValueError(f"coords must have dtype int32, got {coords.dtype}")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords must have shape [N, 4], got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError("coords must be nonempty")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("coords contains duplicate rows")
    rows = int(coords.shape[0])

    expected_dtype = np.dtype(torso_dtype)
    expected_specs = {
        "input_feats": (np.dtype(np.float32), latent_channels),
        "from_latent_fp32": (np.dtype(np.float32), channels),
        "torso_input": (expected_dtype, channels),
        "block0_conv": (expected_dtype, channels),
        "block0_norm": (expected_dtype, channels),
        "block0_mlp_fc1": (expected_dtype, channels * 4),
        "block0_silu": (expected_dtype, channels * 4),
        "block0_mlp_fc2": (expected_dtype, channels),
        "block0_output": (expected_dtype, channels),
        "block1_output": (expected_dtype, channels),
        "block2_output": (expected_dtype, channels),
        "block3_output": (expected_dtype, channels),
        "level0_subdiv_logits": (expected_dtype, 8),
    }
    validated: dict[str, np.ndarray] = {
        "coords": np.ascontiguousarray(coords),
    }
    for name in TRACE_NAMES:
        values = np.asarray(arrays[name])
        dtype, width = expected_specs[name]
        if values.dtype != dtype:
            raise ValueError(
                f"{name} must have dtype {dtype}, got {values.dtype}"
            )
        if values.ndim != 2 or values.shape != (rows, width):
            raise ValueError(
                f"{name} must have shape [{rows}, {width}], got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        validated[name] = np.ascontiguousarray(values)
    return validated


def load_decoder_level0_trace(
    path: Path,
    *,
    latent_channels: int,
    channels: int,
    torso_dtype: np.dtype[Any] | type[np.generic],
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
                "decoder level-zero trace contains duplicate NPZ members: "
                + ", ".join(duplicate_names)
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return validate_decoder_level0_trace(
        arrays,
        latent_channels=latent_channels,
        channels=channels,
        torso_dtype=torso_dtype,
    )


def write_decoder_level0_trace_npz(
    path: Path,
    arrays: Mapping[str, Any],
    *,
    latent_channels: int,
    channels: int,
    torso_dtype: np.dtype[Any] | type[np.generic],
) -> dict[str, Any]:
    path = Path(path)
    validated = validate_decoder_level0_trace(
        arrays,
        latent_channels=latent_channels,
        channels=channels,
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
        reopened = load_decoder_level0_trace(
            temporary_path,
            latent_channels=latent_channels,
            channels=channels,
            torso_dtype=torso_dtype,
        )
        for name in REQUIRED_ARRAYS:
            if not np.array_equal(reopened[name], validated[name]):
                raise ValueError(
                    f"decoder level-zero trace array {name!r} changed after write"
                )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "rows": int(validated["coords"].shape[0]),
        "latent_channels": int(latent_channels),
        "channels": int(channels),
        "torso_dtype": str(np.dtype(torso_dtype)),
        "trace_names": list(TRACE_NAMES),
        "reopened_exact": True,
    }

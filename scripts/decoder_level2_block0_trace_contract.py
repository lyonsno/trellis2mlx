"""Focused parent-composed contract for decoder level-two block zero."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

try:
    from scripts.decoder_level1_trace_contract import (
        LEVEL1_HASH_BOUNDARY_NAMES,
        LEVEL1_HASH_LEDGER_SCHEMA,
        validate_decoder_level1_hash_ledger,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from decoder_level1_trace_contract import (  # type: ignore[no-redef]
        LEVEL1_HASH_BOUNDARY_NAMES,
        LEVEL1_HASH_LEDGER_SCHEMA,
        validate_decoder_level1_hash_ledger,
    )


PARENT_OBJECT_COMMIT = "f382af6000d77e48ce105fe7084fa90096ed2a44"
PARENT_FEATURE_BOUNDARY = "level1_upsample_output"
PARENT_COORD_BOUNDARY = "level2_child_coords"
PARENT_FORK_BOUNDARY = "level2_block0_output"
PARENT_RECEIPT_SCHEMA = (
    "trellis2mlx.decoder_level2_block0_parent_receipt.v1"
)
TRACE_RUN_SCHEMA = "trellis2mlx.decoder_level2_block0_trace_run.v1"
COMPARISON_SCHEMA = (
    "trellis2mlx.decoder_level2_block0_trace_comparison.v1"
)
CHILD_ARRAY_NAMES = (
    PARENT_COORD_BOUNDARY,
    PARENT_FEATURE_BOUNDARY,
    "level2_block0_conv",
    "level2_block0_norm",
    "level2_block0_mlp_fc1",
    "level2_block0_silu",
    "level2_block0_mlp_fc2",
    "level2_block0_output",
)
BLOCK_BOUNDARY_NAMES = CHILD_ARRAY_NAMES[2:]
LEVEL2_BLOCK0_NORM_BOUNDARY_ROUTE = {
    "backend": "cuda-welford-turing-t4",
    "algorithm": (
        "pytorch-2.10-vectorized-layernorm-128-thread-welford-"
        "turing-rsqrt-on-metal"
    ),
    "input_dtype": "float16",
    "parameter_dtype": "float16",
    "hidden_width": 256,
    "affine": True,
    "shape_flow_layernorm": False,
    "decoder_layernorm": True,
    "authenticated": True,
}
TURING_RSQRT_LUT_SIZE = 1 << 24
LEVEL2_BLOCK0_AFFINE_LAYERNORM_CONTRACT = {
    "input_dtype": "float16",
    "parameter_dtype": "float16",
    "hidden_width": 256,
    "affine": True,
    "reduction": {
        "threads": 128,
        "warps": 4,
        "vector_width": 4,
        "active_values_per_thread": 4,
        "average_values_per_launched_thread": 2,
        "active_vector_threads": 64,
        "inactive_vector_threads": 64,
        "accumulator_dtype": "float32",
    },
}
DECODER_LAYERNORM_AUTHENTICATED_CONTRACTS = [
    {
        "input_dtype": "float16",
        "parameter_dtype": "float16",
        "hidden_width": 1024,
        "affine": True,
        "reduction": {
            "threads": 128,
            "warps": 4,
            "vector_width": 4,
            "values_per_thread": 8,
            "accumulator_dtype": "float32",
        },
    },
    {
        "input_dtype": "float16",
        "parameter_dtype": "float16",
        "hidden_width": 512,
        "affine": True,
        "reduction": {
            "threads": 128,
            "warps": 4,
            "vector_width": 4,
            "values_per_thread": 4,
            "accumulator_dtype": "float32",
        },
    },
    {
        "input_dtype": "float16",
        "hidden_width": 512,
        "affine": False,
        "reduction": {
            "threads": 128,
            "warps": 4,
            "vector_width": 4,
            "values_per_thread": 4,
            "accumulator_dtype": "float32",
        },
    },
    LEVEL2_BLOCK0_AFFINE_LAYERNORM_CONTRACT,
    {
        "input_dtype": "float16",
        "hidden_width": 256,
        "affine": False,
        "reduction": {
            "threads": 128,
            "warps": 4,
            "vector_width": 4,
            "active_values_per_thread": 4,
            "average_values_per_launched_thread": 2,
            "active_vector_threads": 64,
            "inactive_vector_threads": 64,
            "accumulator_dtype": "float32",
        },
    },
]
DECODER_LAYERNORM_STATIC_IDENTITY = {
    "backend": "cuda-welford-turing-t4",
    "algorithm": (
        "pytorch-2.10-vectorized-layernorm-128-thread-welford-"
        "turing-rsqrt-on-metal"
    ),
    "experimental": True,
    "cuda_source_tag": "pytorch-v2.10.0",
    "cuda_source_kernel": "vectorized_layer_norm_kernel",
    "cuda_architecture": "sm_75",
    "cuda_device_anchor": "Tesla T4",
    "cuda_rsqrt_bit_exact_for_configured_lut": True,
    "authenticated_contract": {
        "input_dtype": "float16",
        "parameter_dtype": "float16",
        "hidden_width": 1024,
        "affine": True,
    },
    "reduction": {
        "threads": 128,
        "warps": 4,
        "vector_width": 4,
        "values_per_thread": 8,
        "accumulator_dtype": "float32",
    },
    "rsqrt": "Turing MUFU.RSQ normalized signed-ULP LUT",
    "turing_rsqrt_lut_entries": TURING_RSQRT_LUT_SIZE,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a canonical SHA256")
    return value


def decoder_boundary_hash_entry(name: str, values: Any) -> dict[str, Any]:
    """Hash an array in the v2 ledger's name/dtype/shape/value domain."""
    if name not in {
        PARENT_FEATURE_BOUNDARY,
        PARENT_COORD_BOUNDARY,
        *BLOCK_BOUNDARY_NAMES,
    }:
        raise ValueError(f"unknown decoder block0 boundary {name!r}")
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


def validate_decoder_level2_block0_trace(
    arrays: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    missing = sorted(set(CHILD_ARRAY_NAMES) - set(arrays))
    extra = sorted(set(arrays) - set(CHILD_ARRAY_NAMES))
    if missing:
        raise KeyError(
            "decoder level-two block0 trace missing required arrays: "
            + ", ".join(missing)
        )
    if extra:
        raise KeyError(
            "decoder level-two block0 trace contains unexpected arrays: "
            + ", ".join(extra)
        )

    coords = np.asarray(arrays[PARENT_COORD_BOUNDARY])
    if coords.dtype != np.dtype(np.int32):
        raise ValueError(
            f"{PARENT_COORD_BOUNDARY} must have dtype int32, got {coords.dtype}"
        )
    if coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] == 0:
        raise ValueError(
            f"{PARENT_COORD_BOUNDARY} must have nonempty shape [N, 4], "
            f"got {coords.shape}"
        )
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError(f"{PARENT_COORD_BOUNDARY} contains duplicate rows")
    rows = int(coords.shape[0])
    specs = {
        PARENT_FEATURE_BOUNDARY: (rows, 256),
        "level2_block0_conv": (rows, 256),
        "level2_block0_norm": (rows, 256),
        "level2_block0_mlp_fc1": (rows, 1024),
        "level2_block0_silu": (rows, 1024),
        "level2_block0_mlp_fc2": (rows, 256),
        "level2_block0_output": (rows, 256),
    }
    validated = {
        PARENT_COORD_BOUNDARY: np.ascontiguousarray(coords),
    }
    for name, shape in specs.items():
        values = np.asarray(arrays[name])
        if values.dtype != np.dtype(np.float16):
            raise ValueError(f"{name} must have dtype float16, got {values.dtype}")
        if values.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        validated[name] = np.ascontiguousarray(values)
    return validated


def load_decoder_level2_block0_trace(path: Path) -> dict[str, np.ndarray]:
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
                "decoder level-two block0 trace contains duplicate members: "
                + ", ".join(duplicate_names)
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return validate_decoder_level2_block0_trace(arrays)


def write_decoder_level2_block0_trace_npz(
    path: Path,
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    validated = validate_decoder_level2_block0_trace(arrays)
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
            **{name: validated[name] for name in CHILD_ARRAY_NAMES},
        )
        reopened = load_decoder_level2_block0_trace(temporary_path)
        for name in CHILD_ARRAY_NAMES:
            if not np.array_equal(reopened[name], validated[name]):
                raise ValueError(
                    f"decoder level-two block0 array {name!r} changed after write"
                )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "rows": int(validated[PARENT_COORD_BOUNDARY].shape[0]),
        "channels": 256,
        "torso_dtype": "float16",
        "array_names": list(CHILD_ARRAY_NAMES),
        "reopened_exact": True,
    }


def _load_json_receipt(
    descriptor: Any,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "path",
        "sha256",
    }:
        raise ValueError(f"{label} descriptor must contain path and sha256")
    path = Path(descriptor["path"])
    expected = _canonical_sha256(descriptor["sha256"], f"{label} SHA256")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}"
        )
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return path, payload, actual


def authenticate_parent_receipt_file(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    path = Path(path).resolve()
    expected = _canonical_sha256(
        expected_sha256,
        "expected parent receipt SHA256",
    )
    if not path.is_file():
        raise ValueError("parent receipt file is missing")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            "parent receipt SHA256 mismatch: "
            f"expected={expected}, actual={actual}"
        )
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("parent receipt must contain a JSON object")
    return payload, {
        "path": str(path),
        "sha256": actual,
    }


def _resolve_reported_path(value: Any, report_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("reported primary path is invalid")
    path = Path(value)
    if not path.is_absolute():
        path = report_path.parent / path
    return path.resolve()


def _ledger_entries(ledger: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    validated = validate_decoder_level1_hash_ledger(ledger)
    return {entry["name"]: entry for entry in validated["entries"]}


def _source_parent(
    report_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        report.get("schema")
        != "trellis2mlx.source_cuda_shape_slat_grid_decode.v1"
        or report.get("status") != "done"
    ):
        raise ValueError("source parent report is not a done source-CUDA report")
    route = report.get("effective_route")
    expected = {
        "route": "official-source-cuda-shape-decoder-level1-trace",
        "device_type": "cuda",
        "decoder_level1_trace": True,
        "sparse_conv_backend": "none",
    }
    if not isinstance(route, Mapping) or any(
        route.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("source parent report route mismatch")
    device = route.get("cuda_device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("source parent report omits CUDA device")
    artifacts = report.get("decoder_trace_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("source parent report must contain one trace artifact")
    return _parent_primary(report_path, artifacts[0], label="source")


def _local_parent(
    report_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        report.get("schema") != "trellis2mlx.decoder_level1_trace_run.v1"
        or report.get("status") != "done"
    ):
        raise ValueError("local parent report is not done")
    route = report.get("effective_route")
    expected = {
        "route": "mlx-shape-decoder-level1-trace",
        "device_type": "metal",
        "decoder_linear_backend": "turing_fda",
        "sparse_conv_matmul_backend": "turing_fda",
    }
    if not isinstance(route, Mapping) or any(
        route.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("local parent report route mismatch")
    if "gpu" not in str(route.get("device", "")).lower():
        raise ValueError("local parent report is not a Metal GPU route")
    if route.get("decoder_layernorm", {}).get("backend") != (
        "cuda-welford-turing-t4"
    ):
        raise ValueError("local parent LayerNorm route mismatch")
    if route.get("decoder_silu", {}).get("backend") != (
        "cuda-turing-t4-fp16-lut"
    ):
        raise ValueError("local parent SiLU route mismatch")
    return _parent_primary(report_path, report.get("primary"), label="local")


def _parent_primary(
    report_path: Path,
    primary: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(primary, Mapping):
        raise ValueError(f"{label} parent primary is missing")
    validation = primary.get("validation")
    if (
        primary.get("status") != "written"
        or not isinstance(validation, Mapping)
        or validation.get("reopened_exact") is not True
        or validation.get("child_expansion_exact") is not True
    ):
        raise ValueError(f"{label} parent primary validation is incomplete")
    digest = _canonical_sha256(primary.get("sha256"), "parent primary SHA256")
    path = _resolve_reported_path(primary.get("path"), report_path)
    if not path.is_file():
        raise ValueError(f"{label} parent primary is missing: {path}")
    if _sha256_file(path) != digest:
        raise ValueError("parent primary live bytes do not match report")
    return {
        "path": path,
        "sha256": digest,
        "entries": _ledger_entries(primary.get("hash_ledger")),
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != "gpu-greenroom.command.v1":
        raise ValueError("local command manifest schema mismatch")
    argv = manifest.get("argv")
    if not isinstance(argv, list):
        raise ValueError("local command manifest argv is invalid")
    matches = [
        argv[index + 1]
        for index, value in enumerate(argv[:-1])
        if value == "--expected-repo-commit"
    ]
    if matches != [PARENT_OBJECT_COMMIT]:
        raise ValueError(
            "local command manifest parent object commit mismatch"
        )


def _validate_parent_comparison(
    comparison: Mapping[str, Any],
    source: Mapping[str, Any],
    local: Mapping[str, Any],
) -> None:
    if (
        comparison.get("schema")
        != "trellis2mlx.decoder_level1_trace_comparison.v1"
        or comparison.get("status") != "done"
    ):
        raise ValueError("parent strict comparison is not done")
    if comparison.get("first_nonexact_boundary") is not None:
        raise ValueError("parent detailed comparison has an earlier fork")
    if comparison.get("first_nonexact_hash_boundary") != PARENT_FORK_BOUNDARY:
        raise ValueError(
            "parent first nonexact hash boundary is not level2_block0_output"
        )
    rows = comparison.get("hash_boundaries")
    if not isinstance(rows, list):
        raise ValueError("parent strict comparison omits hash boundaries")
    names = [row.get("name") if isinstance(row, Mapping) else None for row in rows]
    if names != list(LEVEL1_HASH_BOUNDARY_NAMES):
        raise ValueError("parent strict comparison boundary order mismatch")
    fork_index = names.index(PARENT_FORK_BOUNDARY)
    for row in rows[:fork_index]:
        if (
            row.get("exact") is not True
            or row.get("source_sha256") != row.get("local_sha256")
        ):
            raise ValueError(
                f"parent predecessor boundary {row.get('name')} is not exact"
            )
    fork = rows[fork_index]
    source_fork_sha = source["entries"][PARENT_FORK_BOUNDARY]["sha256"]
    local_fork_sha = local["entries"][PARENT_FORK_BOUNDARY]["sha256"]
    if (
        fork.get("exact") is not False
        or fork.get("source_sha256") != source_fork_sha
        or fork.get("local_sha256") != local_fork_sha
        or source_fork_sha == local_fork_sha
    ):
        raise ValueError(
            "parent fork boundary is not a ledger-bound nonexact row"
        )
    artifacts = comparison.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("parent strict comparison omits artifacts")
    for label, parent in (("source", source), ("local", local)):
        artifact = artifacts.get(label)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"parent comparison omits {label} artifact")
        if (
            _resolve_reported_path(artifact.get("path"), Path.cwd())
            != parent["path"]
            or artifact.get("sha256") != parent["sha256"]
        ):
            raise ValueError(
                f"parent comparison {label} primary custody mismatch"
            )
    by_name = {row["name"]: row for row in rows}
    for name in (PARENT_COORD_BOUNDARY, PARENT_FEATURE_BOUNDARY):
        row = by_name[name]
        if (
            row["source_sha256"] != source["entries"][name]["sha256"]
            or row["local_sha256"] != local["entries"][name]["sha256"]
        ):
            raise ValueError(
                f"parent comparison {name} ledger custody mismatch"
            )


def validate_parent_receipt(
    receipt: Any,
    *,
    receipt_file: Mapping[str, str],
    source_arrays: Mapping[str, np.ndarray],
    local_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise ValueError("parent receipt must be an object")
    required = {
        "schema",
        "parent_object_commit",
        "parent_contract_schema",
        "boundary_names",
        "source_parent_report",
        "local_parent_report",
        "local_command_manifest",
        "parent_strict_comparison",
    }
    if set(receipt) != required:
        raise ValueError("parent receipt fields mismatch")
    if receipt.get("schema") != PARENT_RECEIPT_SCHEMA:
        raise ValueError("parent receipt schema mismatch")
    if receipt.get("parent_object_commit") != PARENT_OBJECT_COMMIT:
        raise ValueError("parent object commit mismatch")
    if receipt.get("parent_contract_schema") != LEVEL1_HASH_LEDGER_SCHEMA:
        raise ValueError("parent contract schema mismatch")
    expected_boundaries = {
        "features": PARENT_FEATURE_BOUNDARY,
        "coordinates": PARENT_COORD_BOUNDARY,
    }
    if receipt.get("boundary_names") != expected_boundaries:
        raise ValueError("parent receipt boundary names mismatch")

    source_path, source_report, source_report_sha = _load_json_receipt(
        receipt["source_parent_report"],
        "source parent report",
    )
    local_path, local_report, local_report_sha = _load_json_receipt(
        receipt["local_parent_report"],
        "local parent report",
    )
    _, manifest, manifest_sha = _load_json_receipt(
        receipt["local_command_manifest"],
        "local command manifest",
    )
    _, comparison, comparison_sha = _load_json_receipt(
        receipt["parent_strict_comparison"],
        "parent strict comparison",
    )
    source = _source_parent(source_path, source_report)
    local = _local_parent(local_path, local_report)
    _validate_manifest(manifest)
    _validate_parent_comparison(comparison, source, local)

    for name in (PARENT_COORD_BOUNDARY, PARENT_FEATURE_BOUNDARY):
        if not np.array_equal(source_arrays[name], local_arrays[name]):
            raise ValueError(
                f"source and local child inputs differ at {name}"
            )

    child_identity: dict[str, dict[str, dict[str, Any]]] = {}
    for label, arrays in (("source", source_arrays), ("local", local_arrays)):
        entries = {}
        for name in (PARENT_COORD_BOUNDARY, PARENT_FEATURE_BOUNDARY):
            computed = decoder_boundary_hash_entry(name, arrays[name])
            if (
                computed != source["entries"][name]
                or computed != local["entries"][name]
            ):
                raise ValueError(
                    f"{label} child {name} parent boundary hash mismatch"
                )
            entries[name] = computed
        child_identity[label] = entries
    child_output_identity = {
        "source": decoder_boundary_hash_entry(
            PARENT_FORK_BOUNDARY,
            source_arrays[PARENT_FORK_BOUNDARY],
        ),
        "local": decoder_boundary_hash_entry(
            PARENT_FORK_BOUNDARY,
            local_arrays[PARENT_FORK_BOUNDARY],
        ),
    }
    if child_output_identity["source"] != source["entries"][
        PARENT_FORK_BOUNDARY
    ]:
        raise ValueError(
            "source child block0 output parent boundary hash mismatch"
        )
    if child_output_identity["local"] != local["entries"][
        PARENT_FORK_BOUNDARY
    ]:
        raise ValueError(
            "local child block0 output parent boundary hash mismatch"
        )
    return {
        "schema": PARENT_RECEIPT_SCHEMA,
        "parent_object_commit": PARENT_OBJECT_COMMIT,
        "parent_contract_schema": LEVEL1_HASH_LEDGER_SCHEMA,
        "boundary_names": expected_boundaries,
        "receipt_file": dict(receipt_file),
        "receipt_sha256": {
            "source_parent_report": source_report_sha,
            "local_parent_report": local_report_sha,
            "local_command_manifest": manifest_sha,
            "parent_strict_comparison": comparison_sha,
        },
        "child_input_identity": child_identity,
        "child_output_identity": child_output_identity,
    }


def _load_child_report(
    label: str,
    report_path: Path,
    primary_path: Path,
    *,
    turing_rsqrt_lut_path: Path | None = None,
    expected_turing_rsqrt_lut_sha256: str | None = None,
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text())
    if label == "source" and report.get("schema") == (
        "trellis2mlx.source_cuda_shape_slat_grid_decode.v1"
    ):
        if report.get("status") != "done":
            raise ValueError("source child report is not done")
        matching = [
            artifact
            for artifact in report.get("decoder_trace_artifacts", [])
            if _resolve_reported_path(
                artifact.get("path"),
                Path(report_path),
            )
            == Path(primary_path).resolve()
        ]
        if len(matching) != 1:
            raise ValueError(
                "source child report does not identify exactly one primary"
            )
        primary = matching[0]
        equality = primary.get("manual_natural_equality")
        route = report.get("effective_route")
        requested_route = report.get("requested_route")
    else:
        if (
            report.get("schema") != TRACE_RUN_SCHEMA
            or report.get("status") != "done"
        ):
            raise ValueError(f"{label} child report is not done")
        primary = report.get("primary")
        equality = report.get("manual_natural_equality")
        route = report.get("effective_route")
        requested_route = report.get("requested_route")
    if not isinstance(primary, Mapping) or primary.get("status") != "written":
        raise ValueError(f"{label} child primary is not written")
    if _resolve_reported_path(primary.get("path"), Path(report_path)) != (
        Path(primary_path).resolve()
    ):
        raise ValueError(f"{label} child primary path mismatch")
    if primary.get("sha256") != _sha256_file(primary_path):
        raise ValueError(f"{label} child primary SHA256 mismatch")
    validation = primary.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("reopened_exact") is not True
        or validation.get("array_names") != list(CHILD_ARRAY_NAMES)
    ):
        raise ValueError(f"{label} child primary validation is incomplete")
    if (
        not isinstance(equality, Mapping)
        or equality.get("features") is not True
        or equality.get("coordinates") is not True
    ):
        raise ValueError(f"{label} manual and natural block forward differ")
    if not isinstance(route, Mapping):
        raise ValueError(f"{label} child route is missing")
    if label == "source":
        expected = {
            "route": "official-source-cuda-shape-decoder-level2-block0-trace",
            "device_type": "cuda",
            "decoder_level2_block0_trace": True,
            "sparse_conv_backend": "none",
        }
        if any(route.get(key) != value for key, value in expected.items()):
            raise ValueError("source child route mismatch")
        if not str(route.get("cuda_device", "")).strip():
            raise ValueError("source child CUDA device is missing")
        requested_expected = {
            "route": expected["route"],
            "decoder_level2_block0_trace": True,
            "raw_meshes": False,
            "mesh_conversion": False,
        }
        if not isinstance(requested_route, Mapping) or any(
            requested_route.get(key) != value
            for key, value in requested_expected.items()
        ):
            raise ValueError("source child requested route mismatch")
    else:
        expected = {
            "route": "mlx-shape-decoder-level2-block0-trace",
            "device_type": "metal",
            "decoder_linear_backend": "turing_fda",
            "sparse_conv_matmul_backend": "turing_fda",
        }
        if any(route.get(key) != value for key, value in expected.items()):
            raise ValueError("local child route mismatch")
        if "gpu" not in str(route.get("device", "")).lower():
            raise ValueError("local child route is not Metal GPU")
        if route.get("decoder_silu", {}).get("backend") != (
            "cuda-turing-t4-fp16-lut"
        ):
            raise ValueError("local child SiLU route mismatch")
        layernorm = route.get("decoder_layernorm")
        if not isinstance(layernorm, Mapping):
            raise ValueError("local child predecessor LayerNorm route mismatch")
        contracts = layernorm.get("authenticated_contracts")
        if contracts != DECODER_LAYERNORM_AUTHENTICATED_CONTRACTS:
            if (
                not isinstance(contracts, list)
                or LEVEL2_BLOCK0_AFFINE_LAYERNORM_CONTRACT not in contracts
            ):
                raise ValueError(
                    "local child LayerNorm route omits the authenticated "
                    "affine width-256 contract"
                )
            raise ValueError(
                "local child global LayerNorm identity has an incomplete "
                "authenticated contract ledger"
            )
        for field, value in DECODER_LAYERNORM_STATIC_IDENTITY.items():
            if layernorm.get(field) != value:
                raise ValueError(
                    "local child global LayerNorm identity field "
                    f"{field!r} mismatch"
                )
        if (
            turing_rsqrt_lut_path is None
            or expected_turing_rsqrt_lut_sha256 is None
        ):
            raise ValueError(
                "local child comparison omits caller-bound Turing rsqrt LUT"
            )
        expected_lut_sha256 = _canonical_sha256(
            expected_turing_rsqrt_lut_sha256,
            "expected Turing rsqrt LUT SHA256",
        )
        lut_path = Path(turing_rsqrt_lut_path).resolve()
        if not lut_path.is_file():
            raise FileNotFoundError(
                f"Turing rsqrt LUT does not exist: {lut_path}"
            )
        if _sha256_file(lut_path) != expected_lut_sha256:
            raise ValueError("Turing rsqrt LUT artifact SHA256 mismatch")
        lut = route.get("decoder_layernorm_lut")
        if not isinstance(lut, Mapping):
            raise ValueError("local child LayerNorm route omits rsqrt LUT")
        if _resolve_reported_path(lut.get("path"), Path(report_path)) != lut_path:
            raise ValueError("local child LayerNorm rsqrt LUT path mismatch")
        if (
            lut.get("sha256") != expected_lut_sha256
            or layernorm.get(
                "turing_rsqrt_lut_artifact_sha256_attested"
            )
            != expected_lut_sha256
            or lut.get("entries") != TURING_RSQRT_LUT_SIZE
            or lut.get("dtype") != "int8"
            or layernorm.get("turing_rsqrt_lut_entries")
            != TURING_RSQRT_LUT_SIZE
        ):
            raise ValueError("local child LayerNorm rsqrt LUT identity mismatch")
        try:
            with np.load(lut_path, allow_pickle=False) as archive:
                if "normalized_delta" not in archive.files:
                    raise ValueError(
                        "Turing rsqrt LUT omits normalized_delta"
                    )
                normalized_delta = np.asarray(archive["normalized_delta"])
        except (OSError, ValueError) as error:
            if "Turing rsqrt LUT" in str(error):
                raise
            raise ValueError(
                "Turing rsqrt LUT is not a valid NPZ artifact"
            ) from error
        if (
            normalized_delta.dtype != np.dtype(np.int8)
            or normalized_delta.shape != (TURING_RSQRT_LUT_SIZE,)
        ):
            raise ValueError("Turing rsqrt LUT payload schema mismatch")
        content_sha256 = hashlib.sha256(
            np.ascontiguousarray(normalized_delta).tobytes()
        ).hexdigest()
        if (
            lut.get("normalized_delta_sha256") != content_sha256
            or layernorm.get("turing_rsqrt_lut_content_sha256")
            != content_sha256
        ):
            raise ValueError("Turing rsqrt LUT payload identity mismatch")
        norm = route.get("boundary_routes", {}).get("level2_block0_norm")
        if norm != LEVEL2_BLOCK0_NORM_BOUNDARY_ROUTE:
            raise ValueError(
                "local level2_block0_norm boundary route mismatch"
            )
        requested_expected = {
            "route": expected["route"],
            "device_type": "metal",
            "decoder_linear_backend": "turing_fda",
            "sparse_conv_matmul_backend": "turing_fda",
            "decoder_layernorm_backend": "cuda-welford-turing-t4",
            "decoder_silu_backend": "cuda-turing-t4-fp16-lut",
            "boundary_routes": {
                "level2_block0_norm": LEVEL2_BLOCK0_NORM_BOUNDARY_ROUTE,
            },
        }
        if not isinstance(requested_route, Mapping) or any(
            requested_route.get(key) != value
            for key, value in requested_expected.items()
        ):
            raise ValueError("local child requested route mismatch")
    return {"report": report, "route": dict(route)}


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


def compare_decoder_level2_block0_traces(
    *,
    source_path: Path,
    source_report_path: Path,
    local_path: Path,
    local_report_path: Path,
    parent_receipt_path: Path,
    expected_parent_receipt_sha256: str,
    turing_rsqrt_lut_path: Path,
    expected_turing_rsqrt_lut_sha256: str,
) -> dict[str, Any]:
    parent_receipt, receipt_file = authenticate_parent_receipt_file(
        parent_receipt_path,
        expected_parent_receipt_sha256,
    )
    paths = {
        "source": Path(source_path),
        "local": Path(local_path),
    }
    reports = {
        "source": _load_child_report(
            "source",
            Path(source_report_path),
            paths["source"],
        ),
        "local": _load_child_report(
            "local",
            Path(local_report_path),
            paths["local"],
            turing_rsqrt_lut_path=Path(turing_rsqrt_lut_path),
            expected_turing_rsqrt_lut_sha256=(
                expected_turing_rsqrt_lut_sha256
            ),
        ),
    }
    arrays = {
        label: load_decoder_level2_block0_trace(path)
        for label, path in paths.items()
    }
    receipt = validate_parent_receipt(
        parent_receipt,
        receipt_file=receipt_file,
        source_arrays=arrays["source"],
        local_arrays=arrays["local"],
    )
    stages = {}
    first_nonexact = None
    for name in BLOCK_BOUNDARY_NAMES:
        delta = _numeric_delta(arrays["source"][name], arrays["local"][name])
        stages[name] = delta
        if first_nonexact is None and delta["nonzero_count"]:
            first_nonexact = name
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "done",
        "first_nonexact_boundary": first_nonexact,
        "parent_receipt": {
            key: receipt[key]
            for key in (
                "schema",
                "parent_object_commit",
                "parent_contract_schema",
                "boundary_names",
                "receipt_file",
                "receipt_sha256",
            )
        },
        "child_input_identity": receipt["child_input_identity"],
        "child_output_identity": receipt["child_output_identity"],
        "artifacts": {
            label: {
                "path": str(paths[label].resolve()),
                "sha256": _sha256_file(paths[label]),
                "effective_route": reports[label]["route"],
            }
            for label in ("source", "local")
        },
        "stages": stages,
    }

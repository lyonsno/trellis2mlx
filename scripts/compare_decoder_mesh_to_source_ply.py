"""Compare an authenticated decoder checkpoint with a source raw-mesh PLY."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decoder_full_hash_ledger_contract import (
    decoder_full_hash_entry,
)
from trellmlx.mesh_extract import decoder_output_to_mesh


SCHEMA = "trellis2mlx.decoder_mesh_source_ply_comparison.v1"
EXPECTED_LINEAR_BACKEND = "turing_fda"
EXPECTED_SPARSE_CONV_BACKEND = "turing_fda"
EXPECTED_LAYERNORM_BACKEND = "cuda-welford-turing-t4"
EXPECTED_SILU_BACKEND = "cuda-turing-t4-fp16-lut"
EXPECTED_OUTPUT_HEAD_BACKEND = "mlx-native-fp32"
_FACE_DTYPE = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])


def compare_decoder_mesh_to_source_ply(
    *,
    decoder_checkpoint: Path,
    expected_decoder_checkpoint_sha256: str,
    expected_decoder_feats_sha256: str,
    expected_decoder_coords_sha256: str,
    expected_decoder_rsqrt_sha256: str,
    expected_decoder_silu_sha256: str,
    source_ply: Path,
    source_report: Path,
    expected_source_ply_sha256: str,
    expected_source_report_sha256: str,
    expected_source_mesh_override_sha256: str,
    resolution: int,
    output_ply: Path,
    report_json: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    decoder_checkpoint = Path(decoder_checkpoint).resolve()
    source_ply = Path(source_ply).resolve()
    source_report = Path(source_report).resolve()
    output_ply = Path(output_ply).resolve()
    report_json = Path(report_json).resolve()
    protected_paths = {
        decoder_checkpoint,
        source_ply,
        source_report,
        output_ply,
    }
    effective_report_json = (
        _safe_failure_report_path(report_json, protected_paths)
        if report_json in protected_paths
        else report_json
    )
    requested_hashes = {
        "decoder_checkpoint": expected_decoder_checkpoint_sha256,
        "decoder_feats": expected_decoder_feats_sha256,
        "decoder_coords": expected_decoder_coords_sha256,
        "decoder_rsqrt": expected_decoder_rsqrt_sha256,
        "decoder_silu": expected_decoder_silu_sha256,
        "source_ply": expected_source_ply_sha256,
        "source_report": expected_source_report_sha256,
        "source_mesh_override": expected_source_mesh_override_sha256,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "primary_output_status": "not_started",
        "stale_output_removed": False,
        "requested_route": {
            "decoder_linear_backend": EXPECTED_LINEAR_BACKEND,
            "decoder_sparse_conv_matmul_backend": (
                EXPECTED_SPARSE_CONV_BACKEND
            ),
            "decoder_layernorm_backend": EXPECTED_LAYERNORM_BACKEND,
            "decoder_silu_backend": EXPECTED_SILU_BACKEND,
            "decoder_output_head_backend": EXPECTED_OUTPUT_HEAD_BACKEND,
            "resolution": int(resolution),
            "expected_hashes": requested_hashes,
        },
        "inputs": {
            "decoder_checkpoint": str(decoder_checkpoint),
            "source_ply": str(source_ply),
            "source_report": str(source_report),
        },
        "output": {
            "path": str(output_ply),
            "report_path": str(effective_report_json),
            "requested_report_path": str(report_json),
            "effective_report_path": str(effective_report_json),
            "report_path_rerouted": effective_report_json != report_json,
        },
    }
    phase = "request_validation"
    try:
        report["code_identity"] = _code_identity()
        expected_hashes = {
            "decoder_checkpoint": _canonical_sha256(
                expected_decoder_checkpoint_sha256,
                "expected decoder checkpoint SHA256",
            ),
            "decoder_feats": _canonical_sha256(
                expected_decoder_feats_sha256,
                "expected decoder feats SHA256",
            ),
            "decoder_coords": _canonical_sha256(
                expected_decoder_coords_sha256,
                "expected decoder coords SHA256",
            ),
            "decoder_rsqrt": _canonical_sha256(
                expected_decoder_rsqrt_sha256,
                "expected decoder rsqrt SHA256",
            ),
            "decoder_silu": _canonical_sha256(
                expected_decoder_silu_sha256,
                "expected decoder SiLU SHA256",
            ),
            "source_ply": _canonical_sha256(
                expected_source_ply_sha256,
                "expected source PLY SHA256",
            ),
            "source_report": _canonical_sha256(
                expected_source_report_sha256,
                "expected source report SHA256",
            ),
            "source_mesh_override": _canonical_sha256(
                expected_source_mesh_override_sha256,
                "expected source mesh override SHA256",
            ),
        }
        report["requested_route"]["expected_hashes"] = expected_hashes
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if output_ply in {decoder_checkpoint, source_ply, source_report}:
            raise ValueError("output PLY must not replace an input artifact")
        if report_json in {
            decoder_checkpoint,
            source_ply,
            source_report,
            output_ply,
        }:
            raise ValueError("report JSON must have a distinct path")
        if output_ply.exists():
            output_ply.unlink()
            report["stale_output_removed"] = True
        report["last_trustworthy_phase"] = "request_validated"

        phase = "decoder_validation"
        decoder_identity = _validate_decoder_checkpoint(
            decoder_checkpoint,
            expected_hashes=expected_hashes,
        )
        feats = decoder_identity.pop("feats")
        coords = decoder_identity.pop("coords")
        effective_route = decoder_identity.pop("effective_route")
        report["decoder"] = decoder_identity
        report["effective_route"] = {
            "decoder_linear_backend": effective_route[
                "decoder_linear_backend"
            ],
            "decoder_sparse_conv_matmul_backend": effective_route[
                "sparse_conv_matmul_backend"
            ],
            "decoder_layernorm_backend": effective_route[
                "decoder_layernorm"
            ]["backend"],
            "decoder_silu_backend": effective_route["decoder_silu"][
                "backend"
            ],
            "decoder_output_head_backend": effective_route[
                "decoder_output_head_backend"
            ],
            "decoder_rsqrt_sha256": expected_hashes["decoder_rsqrt"],
            "decoder_silu_sha256": expected_hashes["decoder_silu"],
            "resolution": int(resolution),
        }
        report["last_trustworthy_phase"] = "decoder_validated"

        phase = "source_validation"
        source_identity = _validate_source(
            source_ply=source_ply,
            source_report=source_report,
            expected_hashes=expected_hashes,
        )
        source_vertices = source_identity.pop("vertices")
        source_faces = source_identity.pop("faces")
        report["source"] = source_identity
        report["effective_route"]["source_mesh_override_sha256"] = (
            expected_hashes["source_mesh_override"]
        )
        report["last_trustworthy_phase"] = "source_validated"

        phase = "mesh_extraction"
        local_vertices, local_faces = decoder_output_to_mesh(
            feats,
            coords,
            resolution=resolution,
        )
        local_vertices = np.ascontiguousarray(
            np.asarray(local_vertices, dtype="<f4")
        )
        local_faces = np.ascontiguousarray(
            np.asarray(local_faces, dtype="<i4")
        )
        _validate_mesh_arrays(local_vertices, local_faces, "local mesh")
        report["local_mesh"] = _mesh_identity(local_vertices, local_faces)
        report["last_trustworthy_phase"] = "mesh_extracted"

        phase = "output_write"
        _write_binary_ply_atomic(output_ply, local_vertices, local_faces)
        report["primary_output_status"] = "written_unvalidated"

        phase = "output_validation"
        reopened_vertices, reopened_faces = _read_binary_ply_strict(output_ply)
        if not np.array_equal(reopened_vertices, local_vertices):
            raise ValueError("reopened output PLY vertices differ from extraction")
        if not np.array_equal(reopened_faces, local_faces):
            raise ValueError("reopened output PLY faces differ from extraction")
        report["primary_output_status"] = "validated"
        report["output"].update(
            {
                "sha256": _sha256_file(output_ply),
                "size_bytes": output_ply.stat().st_size,
                "reopened_exact": True,
            }
        )
        report["last_trustworthy_phase"] = "output_validated"

        phase = "comparison"
        vertex_comparison = _compare_vertices(
            source_vertices,
            reopened_vertices,
        )
        face_comparison = _compare_faces(source_faces, reopened_faces)
        topology_exact = face_comparison["exact"]
        vertices_exact = vertex_comparison["exact"]
        report["vertex_comparison"] = vertex_comparison
        report["face_comparison"] = face_comparison
        report["comparison"] = {
            "mesh_exact": bool(topology_exact and vertices_exact),
            "topology_exact": bool(topology_exact),
            "vertices_exact": bool(vertices_exact),
        }
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "comparison_complete",
                "elapsed_seconds": _elapsed(started),
            }
        )
        _write_json_atomic(effective_report_json, report)
        return report
    except Exception as exc:
        if phase == "output_validation":
            report["primary_output_status"] = "failed_validation"
        if output_ply.exists() and phase == "output_validation":
            output_ply.unlink()
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": _elapsed(started),
            }
        )
        _write_json_atomic(effective_report_json, report)
        raise


def _safe_failure_report_path(
    requested_path: Path,
    protected_paths: set[Path],
) -> Path:
    index = 0
    while True:
        suffix = ".failure.json" if index == 0 else f".failure.{index}.json"
        candidate = requested_path.with_name(requested_path.name + suffix)
        if candidate not in protected_paths:
            if not candidate.exists() or _is_owned_rerouted_report(
                candidate,
                requested_path,
            ):
                return candidate
        index += 1


def _is_owned_rerouted_report(path: Path, requested_path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    output = payload.get("output")
    if not isinstance(output, Mapping):
        return False
    return (
        payload.get("schema") == SCHEMA
        and payload.get("status") == "failed"
        and payload.get("failure_phase") == "request_validation"
        and payload.get("last_trustworthy_phase") == "request_received"
        and payload.get("primary_output_status") == "not_started"
        and isinstance(payload.get("error_type"), str)
        and isinstance(payload.get("error"), str)
        and output.get("requested_report_path") == str(requested_path)
        and output.get("report_path") == str(path)
        and output.get("effective_report_path") == str(path)
        and output.get("report_path_rerouted") is True
    )


def _validate_decoder_checkpoint(
    path: Path,
    *,
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    actual_file_sha256 = _require_file_sha256(
        path,
        expected_hashes["decoder_checkpoint"],
        "decoder checkpoint",
    )
    with np.load(path, allow_pickle=False) as arrays:
        missing = {"feats", "coords", "decoder_route_json"} - set(
            arrays.files
        )
        if missing:
            raise ValueError(
                f"decoder checkpoint is missing arrays {sorted(missing)}"
            )
        feats = np.ascontiguousarray(arrays["feats"])
        coords = np.ascontiguousarray(arrays["coords"])
        route_array = arrays["decoder_route_json"]
    if feats.ndim != 2 or feats.shape[0] <= 0 or feats.shape[1] != 7:
        raise ValueError(
            f"decoder feats must have nonempty shape [N, 7], got {feats.shape}"
        )
    if feats.dtype != np.float32:
        raise ValueError(
            f"decoder feats dtype must be float32, got {feats.dtype}"
        )
    if not np.isfinite(feats).all():
        raise ValueError("decoder feats must contain only finite values")
    if (
        coords.ndim != 2
        or coords.shape != (feats.shape[0], 4)
        or coords.dtype != np.int32
    ):
        raise ValueError(
            "decoder coords must have dtype int32 and shape "
            f"[{feats.shape[0]}, 4], got {coords.shape} {coords.dtype}"
        )
    feats_entry = decoder_full_hash_entry("decoder_output", feats)
    coords_entry = decoder_full_hash_entry("level4_child_coords", coords)
    if feats_entry["sha256"] != expected_hashes["decoder_feats"]:
        raise ValueError("decoder feats SHA256 mismatch")
    if coords_entry["sha256"] != expected_hashes["decoder_coords"]:
        raise ValueError("decoder coords SHA256 mismatch")
    if route_array.shape != ():
        raise ValueError("decoder route JSON must be a scalar array")
    try:
        route = json.loads(str(route_array.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("decoder route JSON is invalid") from exc
    if not isinstance(route, Mapping):
        raise ValueError("decoder route JSON must contain an object")
    _validate_decoder_route(route, expected_hashes)
    return {
        "path": str(path),
        "sha256": actual_file_sha256,
        "size_bytes": path.stat().st_size,
        "feats": feats,
        "coords": coords,
        "feats_identity": feats_entry,
        "coords_identity": coords_entry,
        "effective_route": dict(route),
    }


def _validate_decoder_route(
    route: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
) -> None:
    if route.get("decoder_linear_backend") != EXPECTED_LINEAR_BACKEND:
        raise ValueError("decoder linear backend is not turing_fda")
    if (
        route.get("sparse_conv_matmul_backend")
        != EXPECTED_SPARSE_CONV_BACKEND
    ):
        raise ValueError("decoder sparse-convolution backend is not turing_fda")
    layernorm = route.get("decoder_layernorm")
    if not isinstance(layernorm, Mapping):
        raise ValueError("decoder LayerNorm route is missing")
    if layernorm.get("backend") != EXPECTED_LAYERNORM_BACKEND:
        raise ValueError("decoder LayerNorm backend mismatch")
    if (
        layernorm.get("turing_rsqrt_lut_artifact_sha256_attested")
        != expected_hashes["decoder_rsqrt"]
    ):
        raise ValueError("decoder rsqrt artifact SHA256 mismatch")
    silu = route.get("decoder_silu")
    if not isinstance(silu, Mapping):
        raise ValueError("decoder SiLU route is missing")
    if silu.get("backend") != EXPECTED_SILU_BACKEND:
        raise ValueError("decoder SiLU backend mismatch")
    for field in (
        "output_lut_artifact_sha256_attested",
        "output_lut_artifact_sha256_effective",
    ):
        if silu.get(field) != expected_hashes["decoder_silu"]:
            raise ValueError(f"decoder SiLU {field} mismatch")
    if (
        route.get("decoder_output_head_backend")
        != EXPECTED_OUTPUT_HEAD_BACKEND
    ):
        raise ValueError("decoder output-head backend mismatch")


def _validate_source(
    *,
    source_ply: Path,
    source_report: Path,
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    report_sha256 = _require_file_sha256(
        source_report,
        expected_hashes["source_report"],
        "source report",
    )
    try:
        payload = json.loads(source_report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("source report JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("source report must contain an object")
    actual_ply_sha256 = _require_file_sha256(
        source_ply,
        expected_hashes["source_ply"],
        "source PLY",
    )
    matching = []
    for artifact in payload.get("mesh_artifacts", []):
        if not isinstance(artifact, Mapping):
            continue
        reported_path = Path(str(artifact.get("path", "")))
        if not reported_path.is_absolute():
            reported_path = source_report.parent / reported_path
        if reported_path.resolve() == source_ply:
            matching.append(artifact)
    if len(matching) != 1:
        raise ValueError(
            "source report must identify exactly one requested source PLY"
        )
    artifact = matching[0]
    if artifact.get("status") != "written" or artifact.get("variant") != "raw":
        raise ValueError("source report PLY must be a written raw artifact")
    if artifact.get("sha256") != actual_ply_sha256:
        raise ValueError("source report PLY SHA256 mismatch")
    if artifact.get("size_bytes") != source_ply.stat().st_size:
        raise ValueError("source report PLY size mismatch")
    mesh_override = payload.get("mesh_override")
    effective_route = payload.get("effective_route")
    nested_mesh_override = (
        effective_route.get("mesh_override")
        if isinstance(effective_route, Mapping)
        else None
    )
    if nested_mesh_override is not None:
        if mesh_override is not None:
            raise ValueError(
                "source report has ambiguous duplicate mesh override identity"
            )
        raise ValueError(
            "source report mesh override identity must be top-level"
        )
    if not isinstance(mesh_override, Mapping):
        raise ValueError("source report mesh override identity is missing")
    if (
        mesh_override.get("sha256")
        != expected_hashes["source_mesh_override"]
    ):
        raise ValueError("source report mesh override SHA256 mismatch")
    vertices, faces = _read_binary_ply_strict(source_ply)
    reported_summary = artifact.get("mesh_summary")
    if not isinstance(reported_summary, Mapping):
        raise ValueError("source report PLY mesh summary is missing")
    if reported_summary.get("vertices") != vertices.shape[0]:
        raise ValueError("source report PLY vertex count mismatch")
    if reported_summary.get("faces") != faces.shape[0]:
        raise ValueError("source report PLY face count mismatch")
    return {
        "ply_path": str(source_ply),
        "ply_sha256": actual_ply_sha256,
        "ply_size_bytes": source_ply.stat().st_size,
        "report_path": str(source_report),
        "report_sha256": report_sha256,
        "mesh_override_sha256": expected_hashes["source_mesh_override"],
        "mesh": _mesh_identity(vertices, faces),
        "vertices": vertices,
        "faces": faces,
    }


def _read_binary_ply_strict(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY ended before end_header")
            try:
                decoded = line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError("PLY header is not ASCII") from exc
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError("only binary_little_endian PLY is supported")
        vertex_count = _header_count(header_lines, "vertex")
        face_count = _header_count(header_lines, "face")
        vertex_bytes = handle.read(vertex_count * 3 * 4)
        if len(vertex_bytes) != vertex_count * 3 * 4:
            raise ValueError("PLY ended before all vertices were read")
        face_bytes = handle.read(face_count * _FACE_DTYPE.itemsize)
        if len(face_bytes) != face_count * _FACE_DTYPE.itemsize:
            raise ValueError("PLY ended before all faces were read")
        if handle.read(1):
            raise ValueError("PLY contains trailing bytes after declared faces")
    vertices = (
        np.frombuffer(vertex_bytes, dtype="<f4")
        .reshape(vertex_count, 3)
        .copy()
    )
    face_records = np.frombuffer(face_bytes, dtype=_FACE_DTYPE)
    if not np.all(face_records["count"] == 3):
        raise ValueError("only triangular PLY faces are supported")
    faces = np.asarray(face_records["indices"], dtype="<i4").copy()
    _validate_mesh_arrays(vertices, faces, "PLY mesh")
    return vertices, faces


def _write_binary_ply_atomic(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {vertices.shape[0]}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                f"element face {faces.shape[0]}\n"
                "property list uchar int vertex_indices\n"
                "end_header\n"
            ).encode("ascii")
            records = np.empty(faces.shape[0], dtype=_FACE_DTYPE)
            records["count"] = 3
            records["indices"] = faces
            handle.write(header)
            handle.write(np.ascontiguousarray(vertices, dtype="<f4").tobytes())
            handle.write(records.tobytes())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _mesh_identity(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, Any]:
    return {
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "vertices_dtype": str(vertices.dtype),
        "faces_dtype": str(faces.dtype),
        "vertices_sha256": _array_sha256(vertices),
        "faces_sha256": _array_sha256(faces),
    }


def _compare_vertices(
    source: np.ndarray,
    local: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "source_shape": list(source.shape),
        "local_shape": list(local.shape),
        "source_sha256": _array_sha256(source),
        "local_sha256": _array_sha256(local),
        "exact": False,
        "mismatched_rows": None,
        "nonzero_values": None,
        "mean_abs": None,
        "max_abs": None,
    }
    if source.shape != local.shape:
        return report
    mismatched_rows = 0
    nonzero_values = 0
    sum_abs = 0.0
    max_abs = 0.0
    for start in range(0, source.shape[0], 1_000_000):
        end = min(start + 1_000_000, source.shape[0])
        delta = np.abs(
            source[start:end].astype(np.float64)
            - local[start:end].astype(np.float64)
        )
        mismatched_rows += int(np.count_nonzero(np.any(delta != 0, axis=1)))
        nonzero_values += int(np.count_nonzero(delta))
        sum_abs += float(delta.sum(dtype=np.float64))
        if delta.size:
            max_abs = max(max_abs, float(delta.max()))
    report.update(
        {
            "exact": nonzero_values == 0,
            "mismatched_rows": mismatched_rows,
            "nonzero_values": nonzero_values,
            "mean_abs": sum_abs / source.size,
            "max_abs": max_abs,
        }
    )
    return report


def _compare_faces(
    source: np.ndarray,
    local: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "source_shape": list(source.shape),
        "local_shape": list(local.shape),
        "source_sha256": _array_sha256(source),
        "local_sha256": _array_sha256(local),
        "exact": False,
        "mismatched_rows": None,
        "nonzero_values": None,
    }
    if source.shape != local.shape:
        return report
    mismatched_rows = 0
    nonzero_values = 0
    for start in range(0, source.shape[0], 1_000_000):
        end = min(start + 1_000_000, source.shape[0])
        unequal = source[start:end] != local[start:end]
        mismatched_rows += int(np.count_nonzero(np.any(unequal, axis=1)))
        nonzero_values += int(np.count_nonzero(unequal))
    report.update(
        {
            "exact": nonzero_values == 0,
            "mismatched_rows": mismatched_rows,
            "nonzero_values": nonzero_values,
        }
    )
    return report


def _validate_mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    label: str,
) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"{label} vertices must have shape [N, 3], got {vertices.shape}"
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"{label} faces must have shape [F, 3], got {faces.shape}"
        )
    if not np.isfinite(vertices).all():
        raise ValueError(f"{label} vertices must be finite")
    if faces.size and (
        int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]
    ):
        raise ValueError(f"{label} faces contain out-of-range indices")


def _header_count(header_lines: list[str], element: str) -> int:
    prefix = f"element {element} "
    matches = [line for line in header_lines if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"PLY must declare exactly one {element} element")
    value = int(matches[0].split()[-1])
    if value < 0:
        raise ValueError(f"PLY {element} count must be nonnegative")
    return value


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _canonical_sha256(value: str, label: str) -> str:
    normalized = str(value).lower()
    if (
        len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{label} must be a canonical SHA256")
    return normalized


def _require_file_sha256(
    path: Path,
    expected: str,
    label: str,
) -> str:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return actual


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_identity() -> dict[str, Any]:
    comparator_path = Path(__file__).resolve()
    extractor_path = REPO_ROOT / "trellmlx" / "mesh_extract.py"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        "comparator": {
            "path": str(comparator_path),
            "sha256": _sha256_file(comparator_path),
        },
        "extractor": {
            "path": str(extractor_path),
            "sha256": _sha256_file(extractor_path),
        },
        "repo_root": str(REPO_ROOT),
        "repo_commit": commit,
        "repo_dirty": bool(status),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-decoder-checkpoint-sha256", required=True)
    parser.add_argument("--expected-decoder-feats-sha256", required=True)
    parser.add_argument("--expected-decoder-coords-sha256", required=True)
    parser.add_argument("--expected-decoder-rsqrt-sha256", required=True)
    parser.add_argument("--expected-decoder-silu-sha256", required=True)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--expected-source-ply-sha256", required=True)
    parser.add_argument("--expected-source-report-sha256", required=True)
    parser.add_argument(
        "--expected-source-mesh-override-sha256",
        required=True,
    )
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        compare_decoder_mesh_to_source_ply(
            decoder_checkpoint=args.decoder_checkpoint,
            expected_decoder_checkpoint_sha256=(
                args.expected_decoder_checkpoint_sha256
            ),
            expected_decoder_feats_sha256=args.expected_decoder_feats_sha256,
            expected_decoder_coords_sha256=(
                args.expected_decoder_coords_sha256
            ),
            expected_decoder_rsqrt_sha256=(
                args.expected_decoder_rsqrt_sha256
            ),
            expected_decoder_silu_sha256=args.expected_decoder_silu_sha256,
            source_ply=args.source_ply,
            source_report=args.source_report,
            expected_source_ply_sha256=args.expected_source_ply_sha256,
            expected_source_report_sha256=(
                args.expected_source_report_sha256
            ),
            expected_source_mesh_override_sha256=(
                args.expected_source_mesh_override_sha256
            ),
            resolution=args.resolution,
            output_ply=args.output_ply,
            report_json=args.report_json,
        )
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

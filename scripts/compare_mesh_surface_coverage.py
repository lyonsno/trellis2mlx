#!/usr/bin/env python3
"""Measure source-triangle coverage by authenticated simplifier outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time
from typing import Any
import uuid

import igl
import numpy as np

from trellmlx.glb_aabb_crop import open_triangle_glb, sha256_file


ROUTE = "authenticated-source-surface-coverage-v1"
HARNESS_PATH = Path(__file__).resolve()
DISTANCE_RATIOS = (0.005, 0.01, 0.02, 0.05, 0.1)
NORMAL_COSINES = (0.5, 0.8, 0.9, 0.95)


SAMPLE_BARYCENTRICS = np.array(
    [[0.50, 0.25, 0.25], [0.25, 0.50, 0.25], [0.25, 0.25, 0.50]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SurfaceCoverage:
    coverage_fraction: np.ndarray
    max_normalized_distance: np.ndarray
    min_normal_agreement: np.ndarray
    sample_normalized_distance: np.ndarray
    sample_normal_agreement: np.ndarray
    sample_candidate_face_index: np.ndarray
    candidate_degenerate_faces: int


class CoverageError(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


@dataclass
class ReportCustody:
    owned_path: Path
    file_descriptor: int
    requested_path: Path
    effective_path: Path
    rerouted: bool
    invocation_id: str
    requested_linked: bool
    lost_effective_path: Path | None = None

    def _assert_path(self, path: Path, role: str) -> None:
        try:
            path_stat = path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            if role == "requested":
                self.lost_effective_path = path
            raise CoverageError("report_custody", f"{role} report path disappeared: {path}") from exc
        descriptor_stat = os.fstat(self.file_descriptor)
        if not stat.S_ISREG(path_stat.st_mode) or (path_stat.st_dev, path_stat.st_ino) != (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ):
            if role == "requested":
                self.lost_effective_path = path
            raise CoverageError("report_custody", f"{role} report path changed ownership: {path}")

    def write(self, payload: dict[str, Any], *, require_requested_link: bool = True) -> None:
        self._assert_path(self.owned_path, "owned")
        if require_requested_link and self.requested_linked:
            self._assert_path(self.requested_path, "requested")
        payload["report"] = {
            "requested_path": str(self.requested_path),
            "effective_path": str(self.effective_path),
            "owned_path": str(self.owned_path),
            "lost_effective_path": (
                str(self.lost_effective_path) if self.lost_effective_path is not None else None
            ),
            "rerouted": self.rerouted,
            "invocation_id": self.invocation_id,
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        os.lseek(self.file_descriptor, 0, os.SEEK_SET)
        os.ftruncate(self.file_descriptor, 0)
        written = 0
        while written < len(encoded):
            written += os.write(self.file_descriptor, encoded[written:])
        os.fsync(self.file_descriptor)
        self._assert_path(self.owned_path, "owned")
        if require_requested_link and self.requested_linked:
            self._assert_path(self.requested_path, "requested")

    def close(self) -> None:
        if self.file_descriptor >= 0:
            os.close(self.file_descriptor)
            self.file_descriptor = -1


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray, label: str) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{label} vertices must be an Nx3 array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"{label} faces must be an Mx3 integer array")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"{label} mesh must contain vertices and faces")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{label} vertices contain non-finite coordinates")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError(f"{label} faces index outside the vertex array")


def _triangle_geometry(
    vertices: np.ndarray, faces: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_mesh(vertices, faces, label)
    triangles = np.asarray(vertices[faces], dtype=np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError(f"{label} mesh contains degenerate triangles")
    normals /= lengths[:, None]
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    return triangles, normals, edge_lengths.max(axis=1)


def measure_surface_coverage(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    candidate_vertices: np.ndarray,
    candidate_faces: np.ndarray,
    *,
    distance_ratio: float,
    normal_cosine: float,
    chunk_faces: int,
) -> SurfaceCoverage:
    if not np.isfinite(distance_ratio) or distance_ratio < 0.0:
        raise ValueError("distance ratio must be finite and nonnegative")
    if not np.isfinite(normal_cosine) or not 0.0 <= normal_cosine <= 1.0:
        raise ValueError("normal cosine must be finite and between zero and one")
    if chunk_faces <= 0:
        raise ValueError("chunk face count must be positive")

    source_triangles, source_normals, source_max_edge = _triangle_geometry(
        source_vertices, source_faces, "source"
    )
    _validate_mesh(candidate_vertices, candidate_faces, "candidate")
    candidate_vertices64 = np.asarray(candidate_vertices, dtype=np.float64)
    candidate_faces64 = np.asarray(candidate_faces, dtype=np.int64)
    candidate_triangles = candidate_vertices64[candidate_faces64]
    candidate_cross = np.cross(
        candidate_triangles[:, 1] - candidate_triangles[:, 0],
        candidate_triangles[:, 2] - candidate_triangles[:, 0],
    )
    candidate_lengths = np.linalg.norm(candidate_cross, axis=1)
    candidate_surface = candidate_lengths > 0.0
    candidate_degenerate_faces = int(np.count_nonzero(~candidate_surface))
    candidate_surface_index = np.flatnonzero(candidate_surface).astype(np.int64, copy=False)
    candidate_faces64 = candidate_faces64[candidate_surface]
    candidate_normals = candidate_cross[candidate_surface] / candidate_lengths[
        candidate_surface, None
    ]
    if len(candidate_faces64) == 0:
        raise ValueError("candidate mesh contains no nondegenerate surface triangles")
    tree = igl.AABB()
    tree.init(candidate_vertices64, candidate_faces64)

    face_count = len(source_faces)
    coverage_fraction = np.empty(face_count, dtype=np.float32)
    max_normalized_distance = np.empty(face_count, dtype=np.float32)
    min_normal_agreement = np.empty(face_count, dtype=np.float32)
    sample_normalized_distance = np.empty((face_count, 3), dtype=np.float32)
    sample_normal_agreement = np.empty((face_count, 3), dtype=np.float32)
    sample_candidate_face_index = np.empty((face_count, 3), dtype=np.int64)
    for start in range(0, face_count, chunk_faces):
        stop = min(start + chunk_faces, face_count)
        triangles = source_triangles[start:stop]
        query = np.einsum("sk,fkj->fsj", SAMPLE_BARYCENTRICS, triangles).reshape(-1, 3)
        squared_distance, candidate_index, _closest = tree.squared_distance(
            candidate_vertices64, candidate_faces64, query
        )
        normalized_distance = np.sqrt(squared_distance).reshape(-1, 3) / source_max_edge[
            start:stop, None
        ]
        candidate_normal = candidate_normals[candidate_index].reshape(-1, 3, 3)
        normal_agreement = np.abs(
            np.einsum("fsj,fj->fs", candidate_normal, source_normals[start:stop])
        )
        covered = (normalized_distance <= distance_ratio) & (
            normal_agreement >= normal_cosine
        )
        coverage_fraction[start:stop] = covered.mean(axis=1, dtype=np.float64)
        max_normalized_distance[start:stop] = normalized_distance.max(axis=1)
        min_normal_agreement[start:stop] = normal_agreement.min(axis=1)
        sample_normalized_distance[start:stop] = normalized_distance
        sample_normal_agreement[start:stop] = normal_agreement
        sample_candidate_face_index[start:stop] = candidate_surface_index[candidate_index].reshape(
            -1, 3
        )

    return SurfaceCoverage(
        coverage_fraction=coverage_fraction,
        max_normalized_distance=max_normalized_distance,
        min_normal_agreement=min_normal_agreement,
        sample_normalized_distance=sample_normalized_distance,
        sample_normal_agreement=sample_normal_agreement,
        sample_candidate_face_index=sample_candidate_face_index,
        candidate_degenerate_faces=candidate_degenerate_faces,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, nargs=2, metavar=("GLB", "EXPECTED_SHA256"))
    parser.add_argument(
        "--candidate", required=True, nargs=3, metavar=("LABEL", "GLB", "EXPECTED_SHA256")
    )
    parser.add_argument(
        "--survival-count", required=True, nargs=2, metavar=("NPY", "EXPECTED_SHA256")
    )
    parser.add_argument(
        "--survival-report", required=True, nargs=2, metavar=("JSON", "EXPECTED_SHA256")
    )
    parser.add_argument(
        "--route-count",
        required=True,
        type=int,
        help="Authenticated assay route universe used to interpret survival counts",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--chunk-faces", type=int, default=100_000)
    return parser.parse_args(argv)


def _validate_digest(digest: str, label: str) -> None:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise CoverageError("validate_request", f"{label} SHA256 must be lowercase hex")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _input_paths(args: argparse.Namespace) -> set[Path]:
    return {
        Path(args.source[0]).resolve(),
        Path(args.candidate[1]).resolve(),
        Path(args.survival_count[0]).resolve(),
        Path(args.survival_report[0]).resolve(),
    }


def _reserve_owned(args: argparse.Namespace, invocation_id: str, kind: str) -> tuple[Path, int]:
    requested = args.report
    parent = args.output_dir.parent if _is_within(requested, args.output_dir) else requested.parent
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        path = parent / f"{requested.name}.surface-coverage-{kind}.{invocation_id}.json"
        if path.resolve() in _input_paths(args) or _is_within(path, args.output_dir):
            raise CoverageError("validate_paths", "owned report alias falls inside protected custody")
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            return path, descriptor
        except FileExistsError:
            invocation_id = uuid.uuid4().hex


def _reserve_report(args: argparse.Namespace) -> ReportCustody:
    invocation_id = uuid.uuid4().hex
    requested = args.report.resolve()
    unsafe = requested in _input_paths(args) or _is_within(requested, args.output_dir)
    if not unsafe:
        owned, descriptor = _reserve_owned(args, invocation_id, "invocation")
        try:
            os.link(owned, args.report)
            return ReportCustody(
                owned, descriptor, args.report, args.report, False, invocation_id, True
            )
        except FileExistsError:
            os.close(descriptor)
            owned.unlink()
    owned, descriptor = _reserve_owned(args, invocation_id, "error")
    return ReportCustody(owned, descriptor, args.report, owned, True, invocation_id, False)


def _validate_request(args: argparse.Namespace, custody: ReportCustody) -> None:
    label = args.candidate[0]
    if not label.strip() or not label.replace("-", "").replace("_", "").isalnum():
        raise CoverageError("validate_request", "candidate label has invalid characters")
    if args.chunk_faces <= 0:
        raise CoverageError("validate_request", "chunk face count must be positive")
    if args.route_count <= 0:
        raise CoverageError("validate_request", "route count must be positive")
    for digest, name in (
        (args.source[1], "source"),
        (args.candidate[2], "candidate"),
        (args.survival_count[1], "survival count"),
        (args.survival_report[1], "survival report"),
    ):
        _validate_digest(digest, name)
    if len(_input_paths(args)) != 4:
        raise CoverageError(
            "validate_request", "source, candidate, survival count, and survival report paths must differ"
        )
    if args.output_dir.exists():
        raise CoverageError("validate_paths", f"output directory already exists: {args.output_dir}")
    if custody.owned_path.resolve() in _input_paths(args) or _is_within(
        custody.owned_path, args.output_dir
    ):
        raise CoverageError("validate_paths", "report path collides with input or output custody")


def _authenticate(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise CoverageError("authenticate_inputs", f"{label} input does not exist: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise CoverageError(
            "authenticate_inputs", f"{label} SHA256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _validate_survival_provenance(
    payload: Any,
    *,
    source_digest: str,
    survival_path: Path,
    survival_digest: str,
    route_count: int,
) -> None:
    if not isinstance(payload, dict):
        raise CoverageError("validate_provenance", "survival report root must be an object")
    if payload.get("schema") != "trellis2mlx.exact_source_face_survival.v1" or payload.get(
        "route"
    ) != "authenticated-exact-source-face-survival-v1":
        raise CoverageError("validate_provenance", "survival report route identity is not admitted")
    if payload.get("status") != "completed" or payload.get("primary_output_status") != "validated":
        raise CoverageError("validate_provenance", "survival report is not completed and validated")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("sha256") != source_digest:
        raise CoverageError("validate_provenance", "survival report source does not match source mesh")
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict) or len(candidates) != route_count:
        raise CoverageError(
            "validate_provenance", "survival report candidate count does not match route count"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise CoverageError("validate_provenance", "survival report artifact inventory is missing")
    survival_resolved = survival_path.resolve()
    matching = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("sha256") == survival_digest
        and Path(str(artifact.get("path", ""))).resolve() == survival_resolved
    ]
    if len(matching) != 1:
        raise CoverageError(
            "validate_provenance", "survival count is not uniquely admitted by the survival report"
        )


def _class_summary(hit: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    selected = hit[mask]
    return {
        "source_faces": count,
        "all_three_samples_covered_faces": int(np.count_nonzero(selected.all(axis=1))),
        "at_least_two_of_three_samples_covered_faces": int(
            np.count_nonzero(selected.sum(axis=1) >= 2)
        ),
        "covered_sample_fraction": float(selected.mean()) if count else None,
    }


def _threshold_matrix(
    result: SurfaceCoverage, survival_count: np.ndarray, route_count: int
) -> list[dict[str, Any]]:
    classes = {
        "survives-none": survival_count == 0,
        "strict-subset": (survival_count > 0) & (survival_count < route_count),
        "survives-all": survival_count == route_count,
    }
    rows = []
    for distance_ratio in DISTANCE_RATIOS:
        for normal_cosine in NORMAL_COSINES:
            hit = (result.sample_normalized_distance <= distance_ratio) & (
                result.sample_normal_agreement >= normal_cosine
            )
            rows.append({
                "distance_ratio": distance_ratio,
                "normal_cosine": normal_cosine,
                "classes": {
                    label: _class_summary(hit, mask) for label, mask in classes.items()
                },
            })
    return rows


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _validate_saved_array(path: Path, expected: np.ndarray) -> None:
    try:
        observed = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise CoverageError(
            "validate_outputs", f"cannot reload output array {path.name}: {exc}"
        ) from exc
    if observed.shape != expected.shape:
        raise CoverageError(
            "validate_outputs",
            f"output array {path.name} shape mismatch: expected {expected.shape}, "
            f"observed {observed.shape}",
        )
    if observed.dtype != expected.dtype:
        raise CoverageError(
            "validate_outputs",
            f"output array {path.name} dtype mismatch: expected {expected.dtype}, "
            f"observed {observed.dtype}",
        )
    if not np.array_equal(observed, expected):
        raise CoverageError("validate_outputs", f"output array {path.name} values changed on disk")


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "trellis2mlx.source_surface_coverage.v1",
        "route": ROUTE,
        "harness": {"path": str(HARNESS_PATH), "sha256": sha256_file(HARNESS_PATH)},
        "status": "running",
        "failure_phase": "validate_request",
        "error": None,
        "primary_output_status": "not_started",
        "requested": {
            "source": {"path": args.source[0], "expected_sha256": args.source[1]},
            "candidate": {
                "label": args.candidate[0],
                "path": args.candidate[1],
                "expected_sha256": args.candidate[2],
            },
            "survival_count": {
                "path": args.survival_count[0],
                "expected_sha256": args.survival_count[1],
            },
            "survival_report": {
                "path": args.survival_report[0],
                "expected_sha256": args.survival_report[1],
            },
            "route_count": args.route_count,
            "output_dir": str(args.output_dir),
            "report": str(args.report),
        },
        "effective_config": {
            "sample_barycentrics": SAMPLE_BARYCENTRICS.tolist(),
            "coverage_claim": "three interior point samples per source triangle; not whole-face proof",
            "distance_normalization": "source triangle maximum edge length",
            "normal_agreement": "absolute cosine; winding-neutral",
            "distance_ratios": list(DISTANCE_RATIOS),
            "normal_cosines": list(NORMAL_COSINES),
            "chunk_faces": args.chunk_faces,
            "result_cap": None,
        },
        "runtime": {
            "surface_query_backend": "libigl.AABB.squared_distance",
            "libigl_distribution_version": importlib.metadata.version("libigl"),
            "numpy_version": np.__version__,
        },
        "source": None,
        "candidate": None,
        "survival_count": None,
        "survival_report": None,
        "threshold_matrix": None,
        "artifacts": [],
    }


def run(args: argparse.Namespace) -> int:
    custody = _reserve_report(args)
    report = _base_report(args)
    started = time.perf_counter()
    temporary_dir: Path | None = None
    try:
        _validate_request(args, custody)
        if custody.rerouted:
            raise CoverageError("validate_paths", "requested report collides with protected evidence")
        custody.write(report)
        report["failure_phase"] = "authenticate_inputs"
        source_path = Path(args.source[0])
        candidate_path = Path(args.candidate[1])
        survival_path = Path(args.survival_count[0])
        survival_report_path = Path(args.survival_report[0])
        source_digest = _authenticate(source_path, args.source[1], "source")
        candidate_digest = _authenticate(candidate_path, args.candidate[2], "candidate")
        survival_digest = _authenticate(survival_path, args.survival_count[1], "survival count")
        survival_report_digest = _authenticate(
            survival_report_path, args.survival_report[1], "survival report"
        )

        report["failure_phase"] = "load_inputs"
        survival_count = np.load(survival_path, allow_pickle=False)
        if survival_count.ndim != 1 or not np.issubdtype(survival_count.dtype, np.integer):
            raise CoverageError("load_inputs", "survival count must be a one-dimensional integer NPY")
        with open_triangle_glb(source_path) as source_view:
            source_vertices = np.asarray(source_view.vertices, dtype=np.float32).copy()
            source_faces = np.asarray(source_view.faces, dtype=np.int64).copy()
        with open_triangle_glb(candidate_path) as candidate_view:
            candidate_vertices = np.asarray(candidate_view.vertices, dtype=np.float32).copy()
            candidate_faces = np.asarray(candidate_view.faces, dtype=np.int64).copy()
        if len(survival_count) != len(source_faces):
            raise CoverageError("load_inputs", "survival count length does not equal source faces")
        if np.any(survival_count < 0) or np.any(survival_count > args.route_count):
            raise CoverageError(
                "load_inputs", "survival count falls outside the explicit route count universe"
            )
        try:
            survival_report_payload = json.loads(survival_report_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CoverageError("load_inputs", f"cannot load survival report: {exc}") from exc
        report["failure_phase"] = "validate_provenance"
        _validate_survival_provenance(
            survival_report_payload,
            source_digest=source_digest,
            survival_path=survival_path,
            survival_digest=survival_digest,
            route_count=args.route_count,
        )

        report["failure_phase"] = "measure_coverage"
        result = measure_surface_coverage(
            source_vertices,
            source_faces,
            candidate_vertices,
            candidate_faces,
            distance_ratio=DISTANCE_RATIOS[0],
            normal_cosine=NORMAL_COSINES[0],
            chunk_faces=args.chunk_faces,
        )
        report["source"] = {
            "path": str(source_path), "sha256": source_digest,
            "vertices": int(len(source_vertices)), "faces": int(len(source_faces)),
        }
        report["candidate"] = {
            "label": args.candidate[0], "path": str(candidate_path),
            "sha256": candidate_digest, "vertices": int(len(candidate_vertices)),
            "faces": int(len(candidate_faces)),
            "degenerate_faces_excluded": result.candidate_degenerate_faces,
        }
        report["survival_count"] = {
            "path": str(survival_path), "sha256": survival_digest,
            "shape": list(survival_count.shape), "dtype": str(survival_count.dtype),
            "routes": args.route_count,
        }
        report["survival_report"] = {
            "path": str(survival_report_path),
            "sha256": survival_report_digest,
            "schema": survival_report_payload["schema"],
            "route": survival_report_payload["route"],
        }
        report["threshold_matrix"] = _threshold_matrix(
            result, survival_count, args.route_count
        )

        report["failure_phase"] = "prepare_outputs"
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
        )
        np.save(temporary_dir / "sample-normalized-distance.npy", result.sample_normalized_distance)
        np.save(temporary_dir / "sample-normal-agreement.npy", result.sample_normal_agreement)
        np.save(temporary_dir / "sample-candidate-face-index.npy", result.sample_candidate_face_index)

        report["failure_phase"] = "validate_outputs"
        _validate_saved_array(
            temporary_dir / "sample-normalized-distance.npy",
            result.sample_normalized_distance,
        )
        _validate_saved_array(
            temporary_dir / "sample-normal-agreement.npy",
            result.sample_normal_agreement,
        )
        _validate_saved_array(
            temporary_dir / "sample-candidate-face-index.npy",
            result.sample_candidate_face_index,
        )
        for path, digest, label in (
            (source_path, source_digest, "source"),
            (candidate_path, candidate_digest, "candidate"),
            (survival_path, survival_digest, "survival count"),
            (survival_report_path, survival_report_digest, "survival report"),
        ):
            if sha256_file(path) != digest:
                raise CoverageError("validate_outputs", f"{label} changed during analysis")
        temporary_artifacts = [_artifact(path) for path in sorted(temporary_dir.iterdir())]
        if len(temporary_artifacts) != 3:
            raise CoverageError("validate_outputs", "unexpected primary artifact inventory")

        report["failure_phase"] = "publish"
        temporary_dir.replace(args.output_dir)
        temporary_dir = None
        report["artifacts"] = [
            _artifact(args.output_dir / Path(item["path"]).name) for item in temporary_artifacts
        ]
        report["status"] = "completed"
        report["failure_phase"] = None
        report["primary_output_status"] = "validated"
        report["timing_seconds"] = time.perf_counter() - started
        custody.write(report)
        return 0
    except Exception as exc:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        report["status"] = "failed"
        report["failure_phase"] = getattr(exc, "phase", report["failure_phase"])
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["timing_seconds"] = time.perf_counter() - started
        try:
            custody.write(report, require_requested_link=False)
        except Exception as report_exc:
            print(f"failure report could not be written: {report_exc}", file=sys.stderr)
        print(f"{report['failure_phase']}: {report['error']}", file=sys.stderr)
        return 1
    finally:
        custody.close()


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare exact source-face survival across authenticated simplifier outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

import numpy as np

from trellmlx.glb_aabb_crop import open_triangle_glb, sha256_file


ROUTE = "authenticated-exact-source-face-survival-v1"
HARNESS_PATH = Path(__file__).resolve()


@dataclass(frozen=True)
class ExactFaceMatches:
    source_survival_mask: np.ndarray
    candidate_source_face_index: np.ndarray
    source_duplicate_face_keys: int
    candidate_duplicate_face_keys: int


@dataclass(frozen=True)
class SourceFaceIndex:
    face_count: int
    source_order: np.ndarray
    sorted_keys: np.ndarray
    duplicate_face_keys: int


class SurvivalError(RuntimeError):
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

    def _assert_owned_alias(self) -> None:
        try:
            path_stat = self.owned_path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise SurvivalError(
                "report_custody", f"owned report alias disappeared: {self.owned_path}"
            ) from exc
        descriptor_stat = os.fstat(self.file_descriptor)
        if not stat.S_ISREG(path_stat.st_mode):
            raise SurvivalError(
                "report_custody", f"owned report alias is no longer regular: {self.owned_path}"
            )
        if (path_stat.st_dev, path_stat.st_ino) != (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ):
            raise SurvivalError(
                "report_custody", f"owned report alias changed ownership: {self.owned_path}"
            )

    def _assert_requested_link(self) -> None:
        if not self.requested_linked:
            return
        try:
            path_stat = self.requested_path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            self.lost_effective_path = self.requested_path
            raise SurvivalError(
                "report_custody", f"requested report path disappeared: {self.requested_path}"
            ) from exc
        descriptor_stat = os.fstat(self.file_descriptor)
        if (path_stat.st_dev, path_stat.st_ino) != (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ):
            self.lost_effective_path = self.requested_path
            raise SurvivalError(
                "report_custody", f"requested report path changed ownership: {self.requested_path}"
            )

    def write(self, payload: dict[str, Any], *, require_requested_link: bool = True) -> None:
        self._assert_owned_alias()
        if require_requested_link:
            self._assert_requested_link()
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
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        os.lseek(self.file_descriptor, 0, os.SEEK_SET)
        os.ftruncate(self.file_descriptor, 0)
        written = 0
        while written < len(encoded):
            written += os.write(self.file_descriptor, encoded[written:])
        os.fsync(self.file_descriptor)
        self._assert_owned_alias()
        if require_requested_link:
            self._assert_requested_link()

    def close(self) -> None:
        if self.file_descriptor >= 0:
            os.close(self.file_descriptor)
            self.file_descriptor = -1


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray, label: str) -> None:
    if vertices.dtype != np.float32 or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{label} vertices must be an Nx3 float32 array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"{label} faces must be an Mx3 integer array")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{label} vertices contain non-finite coordinates")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"{label} mesh must contain vertices and faces")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError(f"{label} faces index outside the vertex array")


def _canonical_face_keys(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Encode each triangle as three lexically sorted float32 bit triples."""
    triangles = np.ascontiguousarray(vertices[faces], dtype="<f4")
    bits = triangles.view("<u4")
    vertex_dtype = np.dtype([("x", "<u4"), ("y", "<u4"), ("z", "<u4")])
    structured_vertices = bits.reshape(-1, 3).view(vertex_dtype).reshape(len(faces), 3)
    ordered = np.sort(structured_vertices, axis=1, order=("x", "y", "z"))
    ordered_bits = np.ascontiguousarray(ordered).view("<u4").reshape(len(faces), 9)
    return ordered_bits.view(np.dtype((np.void, ordered_bits.dtype.itemsize * 9))).reshape(-1)


def _duplicate_key_groups(sorted_keys: np.ndarray) -> int:
    if len(sorted_keys) < 2:
        return 0
    equal_previous = sorted_keys[1:] == sorted_keys[:-1]
    starts = equal_previous & np.concatenate(([True], ~equal_previous[:-1]))
    return int(np.count_nonzero(starts))


def build_source_face_index(vertices: np.ndarray, faces: np.ndarray) -> SourceFaceIndex:
    _validate_mesh(vertices, faces, "source")
    keys = _canonical_face_keys(vertices, faces)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    return SourceFaceIndex(
        face_count=len(faces),
        source_order=order,
        sorted_keys=sorted_keys,
        duplicate_face_keys=_duplicate_key_groups(sorted_keys),
    )


def match_candidate_to_source_index(
    source_index: SourceFaceIndex,
    candidate_vertices: np.ndarray,
    candidate_faces: np.ndarray,
) -> ExactFaceMatches:
    _validate_mesh(candidate_vertices, candidate_faces, "candidate")
    candidate_keys = _canonical_face_keys(candidate_vertices, candidate_faces)
    candidate_order = np.argsort(candidate_keys, kind="stable")
    sorted_candidate = candidate_keys[candidate_order]
    sorted_source = source_index.sorted_keys

    left = np.searchsorted(sorted_source, sorted_candidate, side="left")
    new_group = np.concatenate(([True], sorted_candidate[1:] != sorted_candidate[:-1]))
    group_start = np.maximum.accumulate(
        np.where(new_group, np.arange(len(sorted_candidate), dtype=np.int64), 0)
    )
    duplicate_rank = np.arange(len(sorted_candidate), dtype=np.int64) - group_start
    source_position = left + duplicate_rank
    in_bounds = source_position < len(sorted_source)
    matched_sorted = np.zeros(len(sorted_candidate), dtype=np.bool_)
    matched_sorted[in_bounds] = (
        sorted_source[source_position[in_bounds]] == sorted_candidate[in_bounds]
    )

    candidate_source_face_index = np.full(len(candidate_faces), -1, dtype=np.int64)
    matched_candidate = candidate_order[matched_sorted]
    matched_source = source_index.source_order[source_position[matched_sorted]]
    candidate_source_face_index[matched_candidate] = matched_source
    source_survival_mask = np.zeros(source_index.face_count, dtype=np.bool_)
    source_survival_mask[matched_source] = True
    return ExactFaceMatches(
        source_survival_mask=source_survival_mask,
        candidate_source_face_index=candidate_source_face_index,
        source_duplicate_face_keys=source_index.duplicate_face_keys,
        candidate_duplicate_face_keys=_duplicate_key_groups(sorted_candidate),
    )


def match_exact_source_faces(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    candidate_vertices: np.ndarray,
    candidate_faces: np.ndarray,
) -> ExactFaceMatches:
    """Return multiplicity-aware, winding-neutral exact float32 face matches."""
    return match_candidate_to_source_index(
        build_source_face_index(source_vertices, source_faces),
        candidate_vertices,
        candidate_faces,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        nargs=2,
        metavar=("GLB", "EXPECTED_SHA256"),
    )
    parser.add_argument(
        "--candidate",
        required=True,
        action="append",
        nargs=3,
        metavar=("LABEL", "GLB", "EXPECTED_SHA256"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--chunk-faces", type=int, default=250_000)
    return parser.parse_args(argv)


def _validate_digest(digest: str, label: str) -> None:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SurvivalError("validate_request", f"{label} SHA256 must be lowercase hex")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _input_paths(args: argparse.Namespace) -> set[Path]:
    return {
        Path(args.source[0]).resolve(),
        *(Path(path).resolve() for _label, path, _digest in args.candidate),
    }


def _open_exclusive(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)


def _reserve_owned_alias(
    args: argparse.Namespace, invocation_id: str, *, failure: bool
) -> tuple[Path, int]:
    requested = args.report
    output = args.output_dir.resolve()
    parent = args.output_dir.parent if _is_within(requested, args.output_dir) else requested.parent
    parent.mkdir(parents=True, exist_ok=True)
    kind = "error" if failure else "invocation"
    prefix = f"{requested.name}.face-survival-{kind}."
    while True:
        candidate = parent / f"{prefix}{invocation_id}.json"
        if candidate.resolve() in _input_paths(args) or _is_within(candidate, output):
            raise SurvivalError("validate_paths", "owned report alias falls inside protected custody")
        try:
            return candidate, _open_exclusive(candidate)
        except FileExistsError:
            invocation_id = uuid.uuid4().hex


def _reserve_report_custody(args: argparse.Namespace) -> ReportCustody:
    requested = args.report.resolve()
    invocation_id = uuid.uuid4().hex
    unsafe = requested in _input_paths(args) or _is_within(requested, args.output_dir)
    if not unsafe:
        owned_path, descriptor = _reserve_owned_alias(
            args, invocation_id, failure=False
        )
        try:
            os.link(owned_path, args.report)
            return ReportCustody(
                owned_path=owned_path,
                file_descriptor=descriptor,
                requested_path=args.report,
                effective_path=args.report,
                rerouted=False,
                invocation_id=invocation_id,
                requested_linked=True,
            )
        except FileExistsError:
            os.close(descriptor)
            owned_path.unlink()
    path, descriptor = _reserve_owned_alias(args, invocation_id, failure=True)
    return ReportCustody(
        owned_path=path,
        file_descriptor=descriptor,
        requested_path=args.report,
        effective_path=path,
        rerouted=True,
        invocation_id=invocation_id,
        requested_linked=False,
    )


def _validate_request(args: argparse.Namespace, custody: ReportCustody) -> None:
    labels = [label for label, _path, _digest in args.candidate]
    if len(labels) != len(set(labels)):
        raise SurvivalError("validate_request", "candidate labels must be unique")
    if any(not label.strip() or not label.replace("-", "").replace("_", "").isalnum() for label in labels):
        raise SurvivalError(
            "validate_request",
            "candidate labels must be nonempty alphanumeric, hyphen, or underscore text",
        )
    if args.chunk_faces <= 0:
        raise SurvivalError("validate_request", "chunk face count must be positive")
    source_path = Path(args.source[0]).resolve()
    candidate_paths = [Path(path).resolve() for _label, path, _digest in args.candidate]
    input_paths = [source_path, *candidate_paths]
    if len(input_paths) != len(set(input_paths)):
        raise SurvivalError("validate_request", "source and candidate paths must be unique")
    _validate_digest(args.source[1], "source")
    for label, _path, digest in args.candidate:
        _validate_digest(digest, label)
    if args.output_dir.exists():
        raise SurvivalError("validate_paths", f"output directory already exists: {args.output_dir}")
    report = custody.owned_path.resolve()
    if report in input_paths or _is_within(report, args.output_dir):
        raise SurvivalError("validate_paths", "report path collides with input or output custody")


def _authenticate(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise SurvivalError("authenticate_inputs", f"{label} input does not exist: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise SurvivalError(
            "authenticate_inputs",
            f"{label} SHA256 mismatch: expected {expected}, observed {observed}",
        )
    return observed


def _face_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    chunk_faces: int,
) -> dict[str, np.ndarray]:
    face_count = len(faces)
    area = np.empty(face_count, dtype=np.float64)
    centroid = np.empty((face_count, 3), dtype=np.float32)
    maximum = np.empty(face_count, dtype=np.float64)
    minimum = np.empty(face_count, dtype=np.float64)
    quality = np.empty(face_count, dtype=np.float64)
    for start in range(0, face_count, chunk_faces):
        end = min(start + chunk_faces, face_count)
        triangles = np.asarray(vertices[faces[start:end]], dtype=np.float64)
        edge_squared = np.stack(
            (
                np.sum((triangles[:, 1] - triangles[:, 0]) ** 2, axis=1),
                np.sum((triangles[:, 2] - triangles[:, 1]) ** 2, axis=1),
                np.sum((triangles[:, 0] - triangles[:, 2]) ** 2, axis=1),
            ),
            axis=1,
        )
        chunk_area = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        area[start:end] = chunk_area
        centroid[start:end] = triangles.mean(axis=1).astype(np.float32)
        maximum[start:end] = np.sqrt(edge_squared.max(axis=1))
        minimum[start:end] = np.sqrt(edge_squared.min(axis=1))
        denominator = edge_squared.sum(axis=1)
        quality[start:end] = np.divide(
            4.0 * np.sqrt(3.0) * chunk_area,
            denominator,
            out=np.zeros_like(chunk_area),
            where=denominator > 0,
        )

    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(np.asarray(edges, dtype=np.uint32), axis=1)
    edge_keys = np.ascontiguousarray(edges).view(np.dtype((np.void, 8))).reshape(-1)
    _unique, inverse, counts = np.unique(edge_keys, return_inverse=True, return_counts=True)
    incidence = counts[inverse].reshape(3, face_count).T
    return {
        "area": area,
        "centroid": centroid,
        "max_edge_length": maximum,
        "min_edge_length": minimum,
        "triangle_quality": quality,
        "boundary_face_mask": np.any(incidence == 1, axis=1),
        "nonmanifold_face_mask": np.any(incidence > 2, axis=1),
    }


def _numeric_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(values[mask], dtype=np.float64)
    if len(selected) == 0:
        return {"count": 0}
    quantiles = np.quantile(selected, [0.05, 0.5, 0.95])
    return {
        "count": int(len(selected)),
        "min": float(selected.min()),
        "q05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "mean": float(selected.mean()),
        "q95": float(quantiles[2]),
        "max": float(selected.max()),
    }


def _survival_by_decile(values: np.ndarray, mask: np.ndarray) -> list[dict[str, Any]]:
    boundaries = np.quantile(values, np.linspace(0.0, 1.0, 11))
    bins = np.searchsorted(boundaries[1:-1], values, side="right")
    records = []
    for index in range(10):
        selected = bins == index
        count = int(np.count_nonzero(selected))
        survived = int(np.count_nonzero(mask & selected))
        records.append(
            {
                "decile": index,
                "lower": float(boundaries[index]),
                "upper": float(boundaries[index + 1]),
                "source_faces": count,
                "surviving_faces": survived,
                "survival_fraction": float(survived / count) if count else None,
            }
        )
    return records


def _route_summary(
    matches: ExactFaceMatches,
    metrics: dict[str, np.ndarray],
    candidate_face_count: int,
) -> dict[str, Any]:
    survived = matches.source_survival_mask
    removed = ~survived
    exact = int(np.count_nonzero(matches.candidate_source_face_index >= 0))
    numeric_names = ("area", "max_edge_length", "min_edge_length", "triangle_quality")
    feature_summary = {
        name: {
            "survived": _numeric_summary(values, survived),
            "removed": _numeric_summary(values, removed),
            "survival_by_source_decile": _survival_by_decile(values, survived),
        }
        for name, values in metrics.items()
        if name in numeric_names
    }
    topology = {}
    for name in ("boundary_face_mask", "nonmanifold_face_mask"):
        category = metrics[name]
        count = int(np.count_nonzero(category))
        category_survived = int(np.count_nonzero(category & survived))
        topology[name] = {
            "source_faces": count,
            "surviving_faces": category_survived,
            "survival_fraction": float(category_survived / count) if count else None,
        }
    return {
        "faces": int(candidate_face_count),
        "exact_source_faces": exact,
        "generated_or_modified_faces": int(candidate_face_count - exact),
        "candidate_exact_source_fraction": float(exact / candidate_face_count),
        "source_faces_surviving": int(np.count_nonzero(survived)),
        "source_survival_fraction": float(np.mean(survived)),
        "source_duplicate_face_keys": matches.source_duplicate_face_keys,
        "candidate_duplicate_face_keys": matches.candidate_duplicate_face_keys,
        "feature_summary": feature_summary,
        "topology_summary": topology,
    }


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "trellis2mlx.exact_source_face_survival.v1",
        "route": ROUTE,
        "harness": {
            "path": str(HARNESS_PATH),
            "sha256": sha256_file(HARNESS_PATH),
        },
        "status": "running",
        "failure_phase": "validate_request",
        "error": None,
        "primary_output_status": "not_started",
        "requested": {
            "source": {"path": args.source[0], "expected_sha256": args.source[1]},
            "candidates": [
                {"label": label, "path": path, "expected_sha256": digest}
                for label, path, digest in args.candidate
            ],
            "output_dir": str(args.output_dir),
            "report": str(args.report),
        },
        "effective_config": {
            "identity": "winding-neutral exact float32 coordinate triples with one-to-one multiplicity",
            "chunk_faces": args.chunk_faces,
            "result_cap": None,
        },
        "source": None,
        "candidates": None,
        "pairwise": None,
        "consensus": None,
        "artifacts": [],
        "timing_seconds": {},
    }


def run(args: argparse.Namespace) -> int:
    custody = _reserve_report_custody(args)
    report = _base_report(args)
    started = time.perf_counter()
    temporary_dir: Path | None = None
    try:
        _validate_request(args, custody)
        if custody.rerouted:
            raise SurvivalError(
                "validate_paths", "requested report path collides with an input or existing evidence"
            )
        custody.write(report)
        report["failure_phase"] = "authenticate_inputs"
        source_path = Path(args.source[0])
        source_digest = _authenticate(source_path, args.source[1], "source")
        authenticated_candidates = []
        for label, raw_path, expected in args.candidate:
            path = Path(raw_path)
            authenticated_candidates.append((label, path, _authenticate(path, expected, label)))

        report["failure_phase"] = "load_source"
        source_loaded = time.perf_counter()
        with open_triangle_glb(source_path) as source_view:
            source_vertices = np.asarray(source_view.vertices, dtype=np.float32)
            source_faces = np.asarray(source_view.faces, dtype=np.uint32)
            report["source"] = {
                "path": str(source_path),
                "sha256": source_digest,
                "vertices": int(len(source_vertices)),
                "faces": int(len(source_faces)),
            }
            report["failure_phase"] = "index_source"
            source_index = build_source_face_index(source_vertices, source_faces)
            report["timing_seconds"]["source_index"] = time.perf_counter() - source_loaded
            report["failure_phase"] = "measure_source"
            metrics = _face_metrics(source_vertices, source_faces, args.chunk_faces)

        report["failure_phase"] = "prepare_outputs"
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
        )
        metrics_path = temporary_dir / "source-face-metrics.npz"
        np.savez(metrics_path, **metrics)
        route_masks: dict[str, np.ndarray] = {}
        candidate_reports: dict[str, Any] = {}
        for label, path, digest in authenticated_candidates:
            candidate_started = time.perf_counter()
            report["failure_phase"] = f"match_candidate:{label}"
            with open_triangle_glb(path) as candidate_view:
                candidate_vertices = np.asarray(candidate_view.vertices, dtype=np.float32)
                candidate_faces = np.asarray(candidate_view.faces, dtype=np.uint32)
                matches = match_candidate_to_source_index(
                    source_index, candidate_vertices, candidate_faces
                )
                candidate_reports[label] = {
                    "path": str(path),
                    "sha256": digest,
                    "vertices": int(len(candidate_vertices)),
                    **_route_summary(matches, metrics, len(candidate_faces)),
                    "timing_seconds": time.perf_counter() - candidate_started,
                }
            route_masks[label] = matches.source_survival_mask
            np.save(temporary_dir / f"{label}.source-survival-mask.npy", matches.source_survival_mask)
            np.save(
                temporary_dir / f"{label}.candidate-source-face-index.npy",
                matches.candidate_source_face_index,
            )

        report["failure_phase"] = "compare_routes"
        labels = list(route_masks)
        pairwise = []
        for left_index, left_label in enumerate(labels):
            for right_label in labels[left_index + 1 :]:
                left_mask = route_masks[left_label]
                right_mask = route_masks[right_label]
                intersection = int(np.count_nonzero(left_mask & right_mask))
                union = int(np.count_nonzero(left_mask | right_mask))
                pairwise.append({
                    "left_label": left_label,
                    "right_label": right_label,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": float(intersection / union) if union else 1.0,
                    "left_only": int(np.count_nonzero(left_mask & ~right_mask)),
                    "right_only": int(np.count_nonzero(right_mask & ~left_mask)),
                })
        survival_count = np.zeros(source_index.face_count, dtype=np.uint16)
        for mask in route_masks.values():
            survival_count += mask
        np.save(temporary_dir / "source-surviving-route-count.npy", survival_count)
        histogram = np.bincount(survival_count, minlength=len(labels) + 1)
        report["candidates"] = candidate_reports
        report["pairwise"] = pairwise
        report["consensus"] = {
            "surviving_route_count_histogram": {
                str(index): int(count) for index, count in enumerate(histogram)
            },
            "survives_all_routes": int(histogram[len(labels)]),
            "survives_no_routes": int(histogram[0]),
        }

        report["failure_phase"] = "validate_outputs"
        if sha256_file(source_path) != source_digest:
            raise SurvivalError("validate_outputs", "source changed during analysis")
        for label, path, digest in authenticated_candidates:
            if sha256_file(path) != digest:
                raise SurvivalError("validate_outputs", f"candidate {label} changed during analysis")
        temporary_artifacts = [_artifact(path) for path in sorted(temporary_dir.iterdir())]
        if len(temporary_artifacts) != 2 * len(labels) + 2:
            raise SurvivalError("validate_outputs", "unexpected primary artifact inventory")

        report["failure_phase"] = "publish"
        temporary_dir.replace(args.output_dir)
        temporary_dir = None
        report["artifacts"] = [
            _artifact(args.output_dir / Path(item["path"]).name) for item in temporary_artifacts
        ]
        report["status"] = "completed"
        report["failure_phase"] = None
        report["primary_output_status"] = "validated"
        report["timing_seconds"]["total"] = time.perf_counter() - started
        custody.write(report)
        return 0
    except Exception as exc:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        report["status"] = "failed"
        report["failure_phase"] = getattr(exc, "phase", report["failure_phase"])
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["timing_seconds"]["total"] = time.perf_counter() - started
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

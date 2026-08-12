#!/usr/bin/env python3
"""Compare exact source-face survival across authenticated simplifier outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_digest(digest: str, label: str) -> None:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SurvivalError("validate_request", f"{label} SHA256 must be lowercase hex")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _effective_report_path(args: argparse.Namespace) -> tuple[Path, bool]:
    requested = args.report.resolve()
    inputs = {
        Path(args.source[0]).resolve(),
        *(Path(path).resolve() for _label, path, _digest in args.candidate),
    }
    if requested not in inputs and not args.report.exists():
        return args.report, False
    candidate = args.report.with_name(args.report.name + ".face-survival-error.json")
    while candidate.resolve() in inputs or candidate.exists() or _is_within(candidate, args.output_dir):
        candidate = candidate.with_name(candidate.name + ".face-survival-error.json")
    return candidate, True


def _validate_request(args: argparse.Namespace, report_path: Path) -> None:
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
    report = report_path.resolve()
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


def _base_report(
    args: argparse.Namespace, report_path: Path, report_rerouted: bool
) -> dict[str, Any]:
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
        "report": {
            "requested_path": str(args.report),
            "effective_path": str(report_path),
            "rerouted": report_rerouted,
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
    report_path, report_rerouted = _effective_report_path(args)
    report = _base_report(args, report_path, report_rerouted)
    started = time.perf_counter()
    temporary_dir: Path | None = None
    try:
        _validate_request(args, report_path)
        if report_rerouted:
            raise SurvivalError(
                "validate_paths", "requested report path collides with an input or existing evidence"
            )
        _write_json(report_path, report)
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
        pairwise = {}
        for left_index, left_label in enumerate(labels):
            for right_label in labels[left_index + 1 :]:
                left_mask = route_masks[left_label]
                right_mask = route_masks[right_label]
                intersection = int(np.count_nonzero(left_mask & right_mask))
                union = int(np.count_nonzero(left_mask | right_mask))
                pairwise[f"{left_label}__{right_label}"] = {
                    "intersection": intersection,
                    "union": union,
                    "jaccard": float(intersection / union) if union else 1.0,
                    "left_only": int(np.count_nonzero(left_mask & ~right_mask)),
                    "right_only": int(np.count_nonzero(right_mask & ~left_mask)),
                }
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
        _write_json(report_path, report)
        return 0
    except Exception as exc:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        report["status"] = "failed"
        report["failure_phase"] = getattr(exc, "phase", report["failure_phase"])
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["timing_seconds"]["total"] = time.perf_counter() - started
        try:
            _write_json(report_path, report)
        except Exception as report_exc:
            print(f"failure report could not be written: {report_exc}", file=sys.stderr)
        print(f"{report['failure_phase']}: {report['error']}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())

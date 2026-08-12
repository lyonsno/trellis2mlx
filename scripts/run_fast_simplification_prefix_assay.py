#!/usr/bin/env python3
"""Measure repeat stability and cross-target collapse prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any

import fast_simplification
import numpy as np

from trellmlx.glb_aabb_crop import open_triangle_glb, sha256_file, write_geometry_glb


ROUTE = "fast-simplification-collapse-prefix-v1"


class AssayError(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--target-faces", required=True, nargs="+", type=int)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--aggressiveness", type=float, default=7.0)
    return parser.parse_args(argv)


def _same_or_within(path: Path, directory: Path) -> bool:
    resolved = path.resolve()
    root = directory.resolve()
    return resolved == root or resolved.is_relative_to(root)


def _report_temporary_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _effective_report_path(args: argparse.Namespace) -> tuple[Path, bool]:
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    requested = args.report.resolve()
    requested_temporary = _report_temporary_path(args.report).resolve()
    if (
        requested != source
        and requested_temporary != source
        and not _same_or_within(requested, output_dir)
    ):
        return args.report, False
    if requested_temporary == source:
        candidate = args.report.with_name(args.report.name + ".assay-error.json")
    else:
        candidate = args.input.with_name(args.input.name + ".assay-error.json")
    while (
        candidate.resolve() == source
        or _report_temporary_path(candidate).resolve() == source
        or _same_or_within(candidate, output_dir)
    ):
        candidate = candidate.with_name(candidate.name + ".assay-error.json")
    return candidate, True


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


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_array(path: Path, array: np.ndarray) -> str:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    temporary.replace(path)
    return sha256_file(path)


def _request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "report": str(args.report),
        "target_faces": list(args.target_faces),
        "repeats": args.repeats,
        "aggressiveness": args.aggressiveness,
    }


def _validate_paths(args: argparse.Namespace, report_path: Path) -> None:
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if source == output_dir or source.is_relative_to(output_dir):
        raise AssayError("validate_paths", "output directory must not contain the input GLB")
    if report_path.resolve() == source:
        raise AssayError("validate_paths", "effective report must not replace the input GLB")
    if _report_temporary_path(report_path).resolve() == source:
        raise AssayError("validate_paths", "effective report temporary must not replace the input GLB")
    if _same_or_within(report_path, output_dir):
        raise AssayError("validate_paths", "effective report must not be inside output directory")
    if args.report.resolve() == source:
        raise AssayError("validate_paths", "requested report must not replace the input GLB")
    if _report_temporary_path(args.report).resolve() == source:
        raise AssayError(
            "validate_paths", "requested report temporary must not replace the input GLB"
        )


def _clean_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or path.is_symlink():
            raise AssayError("prepare_outputs", f"output surface is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _load_source(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with open_triangle_glb(path) as view:
            return (
                np.asarray(view.vertices, dtype=np.float32).copy(),
                np.asarray(view.faces, dtype=np.int32).copy(),
            )
    except Exception as exc:
        if isinstance(exc, AssayError):
            raise
        raise AssayError("load_source", str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    requested_report = args.report
    effective_report, report_rerouted = _effective_report_path(args)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "failed",
        "failure_phase": "validate_paths",
        "primary_output_status": "not_started",
        "route": {
            "id": ROUTE,
            "package": "fast-simplification",
            "version": fast_simplification.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "request": _request(args),
        "report": {
            "requested_path": str(requested_report),
            "effective_path": str(effective_report),
            "rerouted": report_rerouted,
        },
        "effective_config": {
            "aggressiveness": args.aggressiveness,
            "face_limit": None,
            "fresh_source_reload_per_run": True,
            "collapse_history": "complete-uncapped",
        },
        "source": None,
        "targets": None,
        "target_contract": None,
        "repeat_stability": None,
        "prefix_relations": None,
        "last_trustworthy_evidence": None,
        "partial_output_cleanup": None,
        "elapsed_seconds": None,
    }

    try:
        _validate_paths(args, effective_report)
        _write_json(effective_report, report)

        report["failure_phase"] = "prepare_outputs"
        _clean_output_dir(args.output_dir)

        report["failure_phase"] = "load_source"
        source_sha256 = sha256_file(args.input)
        vertices, faces = _load_source(args.input)
        source_faces = int(len(faces))
        report["source"] = {
            "path": str(args.input),
            "sha256": source_sha256,
            "vertices": int(len(vertices)),
            "faces": source_faces,
        }
        del vertices, faces
        _write_json(effective_report, report)

        report["failure_phase"] = "validate_targets"
        if args.repeats < 2:
            raise AssayError("validate_targets", "repeats must be at least two")
        if not np.isfinite(args.aggressiveness) or args.aggressiveness < 0:
            raise AssayError("validate_targets", "aggressiveness must be finite and nonnegative")
        if len(set(args.target_faces)) != len(args.target_faces):
            raise AssayError("validate_targets", "target face counts must be unique")
        if any(target <= 0 or target >= source_faces for target in args.target_faces):
            raise AssayError(
                "validate_targets",
                f"every target must be positive and below source face count {source_faces}",
            )
        targets = sorted(args.target_faces, reverse=True)

        report["failure_phase"] = "simplify"
        target_records: list[dict[str, Any]] = []
        representative_collapses: dict[int, np.ndarray] = {}
        all_repeat_exact = True
        report["primary_output_status"] = "partial"

        for target in targets:
            runs: list[dict[str, Any]] = []
            for repeat in range(1, args.repeats + 1):
                if sha256_file(args.input) != source_sha256:
                    raise AssayError("source_identity", "input GLB changed during the assay")
                vertices, faces = _load_source(args.input)
                run_started = time.perf_counter()
                out_vertices, out_faces, collapses = fast_simplification.simplify(
                    vertices,
                    faces,
                    target_count=target,
                    agg=args.aggressiveness,
                    return_collapses=True,
                )
                elapsed = time.perf_counter() - run_started
                out_vertices = np.asarray(out_vertices, dtype=np.float32)
                out_faces = np.asarray(out_faces, dtype=np.uint32)
                collapses = np.asarray(collapses, dtype=np.int64)
                stem = f"target-{target}-repeat-{repeat}"
                mesh_path = args.output_dir / f"{stem}.glb"
                collapse_path = args.output_dir / f"{stem}.collapses.npy"
                write_geometry_glb(mesh_path, out_vertices, out_faces)
                collapse_file_sha256 = _write_array(collapse_path, collapses)
                runs.append(
                    {
                        "repeat": repeat,
                        "requested_faces": target,
                        "achieved_vertices": int(len(out_vertices)),
                        "achieved_faces": int(len(out_faces)),
                        "target_satisfied": int(len(out_faces)) <= target,
                        "collapses": int(len(collapses)),
                        "collapse_sha256": _array_sha256(collapses),
                        "collapse_file_sha256": collapse_file_sha256,
                        "collapse_path": str(collapse_path),
                        "mesh_sha256": sha256_file(mesh_path),
                        "mesh_path": str(mesh_path),
                        "elapsed_seconds": elapsed,
                    }
                )
                report["last_trustworthy_evidence"] = {
                    "target": target,
                    "repeat": repeat,
                    "achieved_vertices": int(len(out_vertices)),
                    "achieved_faces": int(len(out_faces)),
                    "target_satisfied": int(len(out_faces)) <= target,
                    "collapses": int(len(collapses)),
                    "collapse_sha256": _array_sha256(collapses),
                    "collapse_file_sha256": collapse_file_sha256,
                    "mesh_sha256": sha256_file(mesh_path),
                }
                if repeat == 1:
                    representative_collapses[target] = collapses

            identity_fields = (
                "achieved_vertices",
                "achieved_faces",
                "collapses",
                "collapse_sha256",
                "collapse_file_sha256",
                "mesh_sha256",
            )
            repeat_exact = all(
                all(run[field] == runs[0][field] for field in identity_fields)
                for run in runs[1:]
            )
            all_repeat_exact = all_repeat_exact and repeat_exact
            target_records.append(
                {
                    "requested_faces": target,
                    "target_satisfied": all(run["target_satisfied"] for run in runs),
                    "repeat_exact": repeat_exact,
                    "runs": runs,
                }
            )

        report["failure_phase"] = "analyze_prefix"
        prefix_relations: list[dict[str, Any]] = []
        for higher, lower in zip(targets, targets[1:], strict=False):
            high_history = representative_collapses[higher]
            low_history = representative_collapses[lower]
            common = min(len(high_history), len(low_history))
            mismatch = np.nonzero(np.any(high_history[:common] != low_history[:common], axis=1))[0]
            exact_prefix = len(high_history) <= len(low_history) and len(mismatch) == 0
            prefix_relations.append(
                {
                    "higher_target": higher,
                    "lower_target": lower,
                    "higher_collapses": int(len(high_history)),
                    "lower_collapses": int(len(low_history)),
                    "exact_prefix": exact_prefix,
                    "first_mismatch_index": int(mismatch[0]) if len(mismatch) else None,
                }
            )

        if sha256_file(args.input) != source_sha256:
            raise AssayError("source_identity", "input GLB changed before report publication")
        unsatisfied_targets = [
            target["requested_faces"]
            for target in target_records
            if not target["target_satisfied"]
        ]
        report.update(
            {
                "status": "completed",
                "failure_phase": None,
                "primary_output_status": "validated",
                "targets": target_records,
                "target_contract": {
                    "all_targets_satisfied": not unsatisfied_targets,
                    "unsatisfied_targets": unsatisfied_targets,
                },
                "repeat_stability": {"all_exact": all_repeat_exact},
                "prefix_relations": prefix_relations,
            }
        )
    except BaseException as exc:
        report["status"] = "failed"
        report["failure_phase"] = getattr(exc, "phase", report["failure_phase"])
        report["error"] = f"{type(exc).__name__}: {exc}"
        output_dir_existed = args.output_dir.is_dir() and not args.output_dir.is_symlink()
        artifact_count = (
            sum(1 for path in args.output_dir.rglob("*") if path.is_file())
            if output_dir_existed
            else 0
        )
        cleanup_error = None
        removed = False
        if output_dir_existed:
            try:
                shutil.rmtree(args.output_dir)
                removed = True
            except Exception as cleanup_exc:  # pragma: no cover - filesystem-specific
                cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        report["partial_output_cleanup"] = {
            "output_dir_existed": output_dir_existed,
            "artifact_count": artifact_count,
            "removed": removed,
            "error": cleanup_error,
        }
        if artifact_count:
            report["primary_output_status"] = (
                "partial_removed" if removed else "partial_cleanup_failed"
            )
        else:
            report["primary_output_status"] = "not_started"
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_json(effective_report, report)

    if report["status"] != "completed":
        print(f"{report['failure_phase']}: {report['error']}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

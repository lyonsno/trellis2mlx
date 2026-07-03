#!/usr/bin/env python3
"""Probe mesh extraction variants from a saved decoder_output.npz."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROUTE = "trellis2mlx_decoder_extraction_probe"
REPORT_NAME = "decoder_extraction_probe_report.json"
FORBIDDEN_TO_PROVE = [
    "full_trellis2_parity",
    "microsoft_cuda_parity",
    "production_winding_closure",
    "postprocess_or_texture_bake_behavior",
]
PANELS = ("front_xz", "side_yz", "top_xy")
PANEL_FRONT_FACE = {
    "front_xz": "ccw",
    "side_yz": "cw",
    "top_xy": "cw",
}


class ProbeError(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")


def _git_identity() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "cwd": str(Path.cwd()),
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty_short": run_git("status", "--short"),
    }


def _last_trustworthy_evidence(decoder_output: Path, output_dir: Path) -> dict[str, Any]:
    report = {
        "decoder_output_exists": decoder_output.exists(),
        "decoder_output_size_bytes": decoder_output.stat().st_size if decoder_output.exists() else None,
        "output_dir_exists": output_dir.exists(),
        "report_exists": (output_dir / REPORT_NAME).exists(),
    }
    variants = output_dir / "variants"
    if variants.exists():
        report["variant_files"] = sorted(path.name for path in variants.iterdir())
    return report


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "route": ROUTE,
        "decoder_output": str(args.decoder_output),
        "metadata_json": str(args.metadata_json) if args.metadata_json else None,
        "baseline_raw_mesh": str(args.baseline_raw_mesh) if args.baseline_raw_mesh else None,
        "output_dir": str(args.output_dir),
        "image_size": int(args.image_size),
        "pixel_metric": args.pixel_metric,
        "primary_output_status": "not_produced",
        "forbidden_to_prove": FORBIDDEN_TO_PROVE,
        "matched_variables": {
            "decoder_output": str(args.decoder_output),
            "no_generation": True,
            "no_postprocess": True,
            "no_texture_decode": True,
            "no_texture_bake": True,
        },
        "intentional_differences": [
            "controlled_axis_permutation_variants",
            "optional_face_reversal_variant",
            "not_a_reference_cuda_route",
        ],
        "route_identity": {
            "script": str(Path(__file__).resolve()),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "git": _git_identity(),
        },
    }


def _failure_report(args: argparse.Namespace, phase: str, error: str) -> dict[str, Any]:
    report = _base_report(args)
    report.update(
        {
            "status": "error",
            "phase": phase,
            "error": error,
            "last_trustworthy_evidence": _last_trustworthy_evidence(args.decoder_output, args.output_dir),
        }
    )
    return report


def _load_decoder_output(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise ProbeError("load_inputs", f"decoder output does not exist: {path}")
    try:
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
    except Exception as exc:
        raise ProbeError("load_inputs", f"failed to load decoder output: {exc}") from exc

    if "feats" not in arrays or "coords" not in arrays:
        raise ProbeError("load_inputs", "decoder output must contain feats and coords")
    feats = np.asarray(arrays["feats"])
    coords = np.asarray(arrays["coords"])
    if feats.ndim != 2 or feats.shape[1] != 7:
        raise ProbeError("validate_inputs", f"expected feats shape [N,7], got {feats.shape}")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ProbeError("validate_inputs", f"expected coords shape [N,4], got {coords.shape}")
    if feats.shape[0] != coords.shape[0]:
        raise ProbeError("validate_inputs", "feats and coords row counts differ")
    if not np.isfinite(feats).all():
        raise ProbeError("validate_inputs", "feats contain non-finite values")
    return arrays


def _load_resolution(args: argparse.Namespace) -> int:
    if args.resolution is not None:
        return int(args.resolution)
    if args.metadata_json is not None and args.metadata_json.exists():
        try:
            metadata = json.loads(args.metadata_json.read_text())
            if "mesh_grid_size" in metadata:
                return int(metadata["mesh_grid_size"])
        except Exception as exc:
            raise ProbeError("load_inputs", f"failed to read metadata json: {exc}") from exc
    return 512


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0, x)


def _permutation_parity(perm: tuple[int, int, int]) -> int:
    inversions = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            inversions += int(perm[i] > perm[j])
    return inversions % 2


def _variant_inputs(
    *,
    feats: np.ndarray,
    coords: np.ndarray,
    perm: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    variant_feats = np.asarray(feats, dtype=np.float32).copy()
    variant_coords = np.asarray(coords, dtype=np.int32).copy()
    variant_coords[:, 1:4] = variant_coords[:, 1:4][:, perm]
    variant_feats[:, 0:3] = variant_feats[:, 0:3][:, perm]
    variant_feats[:, 3:6] = variant_feats[:, 3:6][:, perm]
    return variant_feats, variant_coords


def _inverse_permute_vertices(vertices: np.ndarray, perm: tuple[int, int, int]) -> np.ndarray:
    inverse = np.argsort(np.asarray(perm))
    return np.asarray(vertices, dtype=np.float32)[:, inverse]


def _export_vertices_for_panel(vertices: np.ndarray) -> np.ndarray:
    export = np.asarray(vertices, dtype=np.float64).copy()
    export[:, 1], export[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()
    return export


def _panel_axes(panel: str) -> tuple[int, int]:
    if panel == "front_xz":
        return 0, 2
    if panel == "side_yz":
        return 1, 2
    if panel == "top_xy":
        return 0, 1
    raise ValueError(panel)


def _signed_areas(vertices: np.ndarray, faces: np.ndarray, panel: str) -> np.ndarray:
    axis_a, axis_b = _panel_axes(panel)
    coords = vertices[:, [axis_a, axis_b]]
    tri = coords[faces]
    return 0.5 * (
        tri[:, 0, 0] * (tri[:, 1, 1] - tri[:, 2, 1])
        + tri[:, 1, 0] * (tri[:, 2, 1] - tri[:, 0, 1])
        + tri[:, 2, 0] * (tri[:, 0, 1] - tri[:, 1, 1])
    )


def _orientation_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    export_vertices = _export_vertices_for_panel(vertices)
    result: dict[str, Any] = {}
    total_front = 0
    total_back = 0
    total_deg = 0
    for panel in PANELS:
        signed = _signed_areas(export_vertices, faces, panel)
        degenerate = int(np.count_nonzero(np.abs(signed) < 1e-12))
        if PANEL_FRONT_FACE[panel] == "ccw":
            front = int(np.count_nonzero(signed > 0))
            back = int(np.count_nonzero(signed < 0))
        else:
            front = int(np.count_nonzero(signed < 0))
            back = int(np.count_nonzero(signed > 0))
        total_front += front
        total_back += back
        total_deg += degenerate
        result[panel] = {
            "effective_front_face": PANEL_FRONT_FACE[panel],
            "front_faces": front,
            "back_faces": back,
            "degenerate_faces": degenerate,
            "front_ratio": float(front / (front + back)) if (front + back) else 0.0,
            "back_ratio": float(back / (front + back)) if (front + back) else 0.0,
        }
    result["summary"] = {
        "front_faces": total_front,
        "back_faces": total_back,
        "degenerate_faces": total_deg,
        "front_ratio": float(total_front / (total_front + total_back)) if (total_front + total_back) else 0.0,
        "back_ratio": float(total_back / (total_front + total_back)) if (total_front + total_back) else 0.0,
    }
    return result


def _pixel_metrics(vertices: np.ndarray, faces: np.ndarray, image_size: int) -> dict[str, Any]:
    from scripts.mesh_culling_attribution import projected_front_face_missing_attribution

    export_vertices = _export_vertices_for_panel(vertices)
    panels: dict[str, Any] = {}
    missing = 0
    reference = 0
    for panel in PANELS:
        panel_report = projected_front_face_missing_attribution(
            vertices=export_vertices,
            faces=faces,
            panel=panel,
            image_size=image_size,
            front_face=PANEL_FRONT_FACE[panel],
        )
        panel_report = {key: value for key, value in panel_report.items() if key != "missing_pixels_by_face"}
        panels[panel] = panel_report
        missing += int(panel_report["missing_pixels"])
        reference += int(panel_report["double_sided_pixels"])
    return {
        "image_size": int(image_size),
        "total_missing_pixels": missing,
        "total_double_sided_pixels": reference,
        "total_missing_ratio": float(missing / reference) if reference else 0.0,
        "panels": panels,
    }


def _load_baseline_raw(path: Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    if not path.exists():
        raise ProbeError("load_inputs", f"baseline raw mesh does not exist: {path}")
    with np.load(path) as data:
        return np.asarray(data["vertices"], dtype=np.float32), np.asarray(data["faces"], dtype=np.int64)


def _mesh_match_report(
    vertices: np.ndarray,
    faces: np.ndarray,
    baseline: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any] | None:
    if baseline is None:
        return None
    baseline_vertices, baseline_faces = baseline
    result = {
        "vertices_shape_equal": list(vertices.shape) == list(baseline_vertices.shape),
        "faces_shape_equal": list(faces.shape) == list(baseline_faces.shape),
        "faces_equal": bool(np.array_equal(faces, baseline_faces)) if faces.shape == baseline_faces.shape else False,
        "vertices_allclose": bool(np.allclose(vertices, baseline_vertices, atol=1e-6, rtol=0.0))
        if vertices.shape == baseline_vertices.shape
        else False,
    }
    if vertices.shape == baseline_vertices.shape:
        result["vertices_max_abs_error"] = float(np.max(np.abs(vertices - baseline_vertices))) if vertices.size else 0.0
    return result


def _variants() -> list[dict[str, Any]]:
    variants = [
        {
            "name": "identity",
            "perm": (0, 1, 2),
            "inverse_vertices": False,
            "reverse_faces": False,
            "description": "current extractor inputs",
        },
        {
            "name": "identity_reverse_faces",
            "perm": (0, 1, 2),
            "inverse_vertices": False,
            "reverse_faces": True,
            "description": "current extractor inputs with face winding reversed after extraction",
        },
    ]
    for perm in itertools.permutations((0, 1, 2)):
        if perm == (0, 1, 2):
            continue
        variants.append(
            {
                "name": "perm_" + "".join(str(axis) for axis in perm),
                "perm": perm,
                "inverse_vertices": True,
                "reverse_faces": False,
                "description": "permute coords/offsets/intersection channels, inverse-permute vertices",
            }
        )
        variants.append(
            {
                "name": "perm_" + "".join(str(axis) for axis in perm) + "_handed",
                "perm": perm,
                "inverse_vertices": True,
                "reverse_faces": bool(_permutation_parity(perm)),
                "description": "permute coords/channels, inverse-permute vertices, correct odd-permutation handedness",
            }
        )
    return variants


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    total_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _load_decoder_output(args.decoder_output)
    resolution = _load_resolution(args)
    baseline_raw = _load_baseline_raw(args.baseline_raw_mesh)

    feats = np.asarray(arrays["feats"], dtype=np.float32)
    coords = np.asarray(arrays["coords"], dtype=np.int32)
    selected = set(args.variants.split(",")) if args.variants else None

    try:
        from trellmlx.mesh_extract import decoder_output_to_mesh
    except Exception as exc:
        raise ProbeError("import_dependencies", f"failed to import local mesh extractor: {exc}") from exc

    variant_dir = args.output_dir / "variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    variant_reports = []
    for spec in _variants():
        if selected is not None and spec["name"] not in selected:
            continue
        name = spec["name"]
        start = time.perf_counter()
        perm = tuple(spec["perm"])
        variant_feats, variant_coords = _variant_inputs(feats=feats, coords=coords, perm=perm)
        vertices, faces = decoder_output_to_mesh(variant_feats, variant_coords, resolution=resolution)
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)
        if spec["inverse_vertices"]:
            vertices = _inverse_permute_vertices(vertices, perm)
        if spec["reverse_faces"]:
            faces = faces[:, ::-1].copy()

        mesh_path = variant_dir / f"{name}.npz"
        np.savez(mesh_path, vertices=vertices, faces=faces)
        elapsed = time.perf_counter() - start
        print(f"[{name}] {len(vertices):,}V {len(faces):,}F in {elapsed:.3f}s", flush=True)

        report: dict[str, Any] = {
            "name": name,
            "description": spec["description"],
            "perm": list(perm),
            "inverse_vertices": bool(spec["inverse_vertices"]),
            "reverse_faces": bool(spec["reverse_faces"]),
            "elapsed_seconds": elapsed,
            "mesh_npz": str(mesh_path),
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "bounds_min": vertices.min(axis=0).tolist() if len(vertices) else None,
            "bounds_max": vertices.max(axis=0).tolist() if len(vertices) else None,
            "orientation": _orientation_metrics(vertices, faces),
            "baseline_raw_match": _mesh_match_report(vertices, faces, baseline_raw),
        }
        if args.pixel_metric == "all" or (args.pixel_metric == "identity" and name in {"identity", "identity_reverse_faces"}):
            pixel_start = time.perf_counter()
            report["pixel_missing"] = _pixel_metrics(vertices, faces, args.image_size)
            report["pixel_metric_elapsed_seconds"] = time.perf_counter() - pixel_start
        variant_reports.append(report)

    if not variant_reports:
        raise ProbeError("run_variants", "no variants were selected")

    ranked_by_back_ratio = sorted(
        variant_reports,
        key=lambda item: item["orientation"]["summary"]["back_ratio"],
    )
    ranked_by_pixel = sorted(
        [item for item in variant_reports if "pixel_missing" in item],
        key=lambda item: item["pixel_missing"]["total_missing_ratio"],
    )

    report = _base_report(args)
    report.update(
        {
            "status": "ok",
            "phase": "complete",
            "primary_output_status": "produced",
            "resolution": resolution,
            "decoder_shape": {
                "feats": list(feats.shape),
                "coords": list(coords.shape),
                "coord_min": coords[:, 1:4].min(axis=0).tolist(),
                "coord_max": coords[:, 1:4].max(axis=0).tolist(),
            },
            "variants": variant_reports,
            "ranked_by_orientation_back_ratio": [
                {
                    "name": item["name"],
                    "back_ratio": item["orientation"]["summary"]["back_ratio"],
                    "front_ratio": item["orientation"]["summary"]["front_ratio"],
                    "faces": item["faces"],
                }
                for item in ranked_by_back_ratio
            ],
            "ranked_by_pixel_missing": [
                {
                    "name": item["name"],
                    "total_missing_ratio": item["pixel_missing"]["total_missing_ratio"],
                    "total_missing_pixels": item["pixel_missing"]["total_missing_pixels"],
                    "total_double_sided_pixels": item["pixel_missing"]["total_double_sided_pixels"],
                }
                for item in ranked_by_pixel
            ],
            "total_elapsed_seconds": time.perf_counter() - total_start,
            "last_trustworthy_evidence": _last_trustworthy_evidence(args.decoder_output, args.output_dir),
        }
    )
    _write_json(args.output_dir / REPORT_NAME, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-output", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--baseline-raw-mesh", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--pixel-metric", choices=("none", "identity", "all"), default="none")
    parser.add_argument("--variants", help="Comma-separated variant names; defaults to all")
    args = parser.parse_args(argv)
    if args.image_size < 128:
        parser.error("--image-size must be at least 128; smaller raw-mesh projections can collapse to zero pixels")
    if args.resolution is not None and args.resolution <= 0:
        parser.error("--resolution must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_probe(args)
        return 0
    except ProbeError as exc:
        report = _failure_report(args, exc.phase, str(exc))
        _write_json(args.output_dir / REPORT_NAME, report)
        print(f"{exc.phase}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        report = _failure_report(args, "unexpected_error", repr(exc))
        _write_json(args.output_dir / REPORT_NAME, report)
        print(f"unexpected_error: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

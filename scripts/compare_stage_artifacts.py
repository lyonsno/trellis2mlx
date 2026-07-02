"""Compare reference and candidate TRELLIS stage artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare TRELLIS stage artifacts")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["conditioning", "sparse_coords", "sparse_flow_step", "sparse_internals", "shape_slat", "decoder_output"],
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compare_stage(args.stage, args.reference, args.candidate)
    _write_json(args.output, report)
    print(json.dumps(_compact_console_summary(report), sort_keys=True))
    return 0


def compare_stage(stage: str, reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    with np.load(reference_path) as ref, np.load(candidate_path) as cand:
        report: dict[str, Any] = {
            "schema": "trellis2mlx.stage_artifact_comparison.v1",
            "stage": stage,
            "reference": _artifact_identity(reference_path),
            "candidate": _artifact_identity(candidate_path),
        }
        if stage == "conditioning":
            report["arrays"] = {
                name: _array_delta(ref[name], cand[name])
                for name in ("cond", "neg_cond")
            }
            return report
        if stage == "sparse_flow_step":
            report["arrays"] = {
                name: _array_delta(ref[name], cand[name])
                for name in (
                    "noise",
                    "pred_pos",
                    "pred_neg",
                    "pred_cfg",
                    "x0_pos",
                    "x0_cfg",
                    "std_pos",
                    "std_cfg",
                    "ratio_raw",
                    "std_ratio",
                    "ratio_effective",
                    "x0_rescaled",
                    "x0_after_rescale",
                    "pred_final",
                    "sample_next",
                    "t",
                    "t_prev",
                )
                if name in ref and name in cand
            }
            return report
        if stage == "sparse_internals":
            report["arrays"] = {
                name: _array_delta(ref[name], cand[name])
                for name in ("z_s", "logits", "decoded", "decoded_ds")
            }
            coords_report, _ref_order, _cand_order = _coord_overlap(ref["coords"], cand["coords"])
            report["coords"] = coords_report
            return report

        coords_report, ref_order, cand_order = _coord_overlap(ref["coords"], cand["coords"])
        report["coords"] = coords_report
        if stage in {"shape_slat", "decoder_output"} and "feats" in ref and "feats" in cand:
            report["features"] = _feature_delta(ref["feats"], cand["feats"], ref_order, cand_order)
        return report


def _array_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    shape_match = reference.shape == candidate.shape
    dtype_match = str(reference.dtype) == str(candidate.dtype)
    summary: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_match": shape_match,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "dtype_match": dtype_match,
    }
    if not shape_match:
        return summary
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    summary.update(_diff_summary(diff))
    return summary


def _coord_overlap(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[dict[str, Any], list[int], list[int]]:
    ref_keys = _coord_index(reference)
    cand_keys = _coord_index(candidate)
    ref_set = set(ref_keys)
    cand_set = set(cand_keys)
    common = sorted(ref_set & cand_set)
    union = ref_set | cand_set
    ref_order = [ref_keys[key] for key in common]
    cand_order = [cand_keys[key] for key in common]
    report = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_count": int(reference.shape[0]),
        "candidate_count": int(candidate.shape[0]),
        "common_count": len(common),
        "reference_only_count": len(ref_set - cand_set),
        "candidate_only_count": len(cand_set - ref_set),
        "union_count": len(union),
        "jaccard": (len(common) / len(union)) if union else 1.0,
    }
    return report, ref_order, cand_order


def _feature_delta(
    reference_feats: np.ndarray,
    candidate_feats: np.ndarray,
    ref_order: list[int],
    cand_order: list[int],
) -> dict[str, Any]:
    if not ref_order:
        return {
            "common_shape": [0, int(reference_feats.shape[1]) if reference_feats.ndim == 2 else 0],
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rms_diff": None,
        }
    ref_common = reference_feats[np.asarray(ref_order)]
    cand_common = candidate_feats[np.asarray(cand_order)]
    diff = np.abs(ref_common.astype(np.float64) - cand_common.astype(np.float64))
    return {
        "common_shape": list(ref_common.shape),
        **_diff_summary(diff),
    }


def _diff_summary(diff: np.ndarray) -> dict[str, Any]:
    return {
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "rms_diff": float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0,
    }


def _coord_index(coords: np.ndarray) -> dict[tuple[int, ...], int]:
    return {tuple(int(v) for v in row): i for i, row in enumerate(coords)}


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256_file(path) if path.exists() else None,
    }


def _compact_console_summary(report: dict[str, Any]) -> dict[str, Any]:
    if "coords" in report:
        summary = {
            "stage": report["stage"],
            "common": report["coords"]["common_count"],
            "jaccard": report["coords"]["jaccard"],
            "reference_count": report["coords"]["reference_count"],
            "candidate_count": report["coords"]["candidate_count"],
        }
        if "features" in report:
            summary["feature_max_abs_diff"] = report["features"]["max_abs_diff"]
            summary["feature_mean_abs_diff"] = report["features"]["mean_abs_diff"]
        return summary
    return {
        "stage": report["stage"],
        "array_max_abs_diff": {
            name: values.get("max_abs_diff")
            for name, values in report.get("arrays", {}).items()
        },
        "array_mean_abs_diff": {
            name: values.get("mean_abs_diff")
            for name, values in report.get("arrays", {}).items()
        },
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())

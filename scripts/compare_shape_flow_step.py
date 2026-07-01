"""Compare first shape-flow step witnesses between Trellis-Mac and trellis2mlx."""

import argparse
import json
import os
import sys

import numpy as np

from compare_shape_slat import (
    _artifact_record,
    _compare_features,
    _coord_summary,
    _fail_report,
    _flag_value,
    _infer_unmatched_variables,
    _route_proof_status,
    _route_record,
    _spatial_coords,
)


REPORT_NAME = "shape_flow_step_comparison_report.json"


def main():
    parser = argparse.ArgumentParser(description="Compare first shape-flow step outputs")
    parser.add_argument("--trellis-mac", required=True, help="Path to Trellis-Mac shape_flow_step0.npz")
    parser.add_argument("--trellis-mlx", required=True, help="Path to trellis2mlx shape_flow_step0.npz")
    parser.add_argument("--trellis-mac-receipt", help="Path to Trellis-Mac Greenroom receipt.json")
    parser.add_argument("--trellis-mlx-receipt", help="Path to trellis2mlx Greenroom receipt.json")
    parser.add_argument("--output-dir", default="/tmp/shape-flow-step-comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "schema": "trellis2mlx.shape_flow_step_comparison.v1",
        "comparison_status": "running",
        "evidence_use_class": "bounded_candidate",
        "artifacts": {
            "trellis_mac": _artifact_record(args.trellis_mac),
            "trellis_mlx": _artifact_record(args.trellis_mlx),
        },
    }
    try:
        report["routes"] = {
            "trellis_mac": _route_record(args.trellis_mac_receipt),
            "trellis_mlx": _route_record(args.trellis_mlx_receipt),
        }
    except Exception as exc:
        return _fail_report_named(args.output_dir, report, "load_receipts", exc)

    report["route_validation"] = {
        "trellis_mac": _validate_route(report["routes"]["trellis_mac"], report["artifacts"]["trellis_mac"]),
        "trellis_mlx": _validate_route(report["routes"]["trellis_mlx"], report["artifacts"]["trellis_mlx"]),
    }
    report["unmatched_variables"] = _infer_unmatched_variables(report["routes"])
    report["route_proof_status"] = _route_proof_status(report["route_validation"], report["unmatched_variables"])
    if report["route_proof_status"] == "rejected":
        report["evidence_use_class"] = "negative_evidence"
        return _fail_report_named(
            args.output_dir,
            report,
            "validate_routes",
            RuntimeError("route validation rejected at least one receipt/artifact pair"),
        )

    try:
        mac = _load_step(args.trellis_mac)
        mlx = _load_step(args.trellis_mlx)
    except Exception as exc:
        return _fail_report_named(args.output_dir, report, "load_artifacts", exc)

    print(f"Trellis-Mac shape-flow step: pred_v {mac['pred_v_feats'].shape}, coords {mac['coords'].shape}")
    print(f"trellis2mlx shape-flow step: pred_v {mlx['pred_v_feats'].shape}, coords {mlx['coords'].shape}")
    report["shapes"] = {
        "trellis_mac": {key: list(value.shape) for key, value in mac.items() if hasattr(value, "shape")},
        "trellis_mlx": {key: list(value.shape) for key, value in mlx.items() if hasattr(value, "shape")},
    }
    _compare_step(report, mac, mlx)
    report["comparison_status"] = "completed"
    _write_report(args.output_dir, report)
    return 0


def _load_step(path):
    data = np.load(path)
    required = ["sample_feats", "pred_v_feats", "pred_x0_feats", "coords"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path}: missing required arrays: {missing}")
    out = {key: data[key] for key in required}
    if "pred_eps_feats" in data.files:
        out["pred_eps_feats"] = data["pred_eps_feats"]
    for key in [k for k in out if k.endswith("_feats")]:
        if out[key].ndim != 2:
            raise ValueError(f"{path}: {key} must be 2D, got {out[key].shape}")
        if out[key].shape[0] != out["coords"].shape[0]:
            raise ValueError(f"{path}: {key}/coords row mismatch")
    if out["coords"].ndim != 2 or out["coords"].shape[1] not in (3, 4):
        raise ValueError(f"{path}: coords must be 2D with 3 or 4 columns, got {out['coords'].shape}")
    return out


def _compare_step(report, mac, mlx):
    mac_spatial = _spatial_coords(mac["coords"])
    mlx_spatial = _spatial_coords(mlx["coords"])
    mac_set = set(map(tuple, mac_spatial))
    mlx_set = set(map(tuple, mlx_spatial))
    common = mac_set & mlx_set
    only_mac = mac_set - mlx_set
    only_mlx = mlx_set - mac_set
    union = mac_set | mlx_set
    jaccard = len(common) / len(union) if union else None
    report["coordinate_comparison"] = {
        "mode": "spatial_set_intersection",
        "coords_match": bool(mac_set == mlx_set),
        "common_voxels": int(len(common)),
        "only_trellis_mac": int(len(only_mac)),
        "only_trellis_mlx": int(len(only_mlx)),
        "jaccard_similarity": float(jaccard) if jaccard is not None else None,
        "trellis_mac": _coord_summary(mac["coords"], mac_spatial, mac_set),
        "trellis_mlx": _coord_summary(mlx["coords"], mlx_spatial, mlx_set),
    }
    if not common:
        report["feature_comparison_status"] = "not_applicable_no_common_voxels"
        return

    mac_idx = {tuple(c): i for i, c in enumerate(mac_spatial)}
    mlx_idx = {tuple(c): i for i, c in enumerate(mlx_spatial)}
    common_list = sorted(common)
    report["feature_comparison"] = {}
    for key in sorted(k for k in mac.keys() if k.endswith("_feats") and k in mlx):
        mac_feats = np.array([mac[key][mac_idx[c]] for c in common_list])
        mlx_feats = np.array([mlx[key][mlx_idx[c]] for c in common_list])
        report["feature_comparison"][key] = _compare_features(
            mac_feats,
            mlx_feats,
            sample_policy="full_common_spatial_set",
        )


def _validate_route(route, artifact):
    if not route.get("receipt_loaded"):
        return {
            "status": "missing",
            "evidence_use_class": "file_comparison_only",
            "reasons": [],
            "warnings": ["receipt_not_provided"],
        }
    reasons = []
    warnings = []
    effective_route = route.get("effective_route") or ""
    if route.get("status") != "done":
        reasons.append("receipt_status_not_done")
    if route.get("exit_code") not in (None, 0):
        reasons.append("receipt_exit_code_nonzero")
    if route.get("failure_phase") is not None:
        reasons.append("receipt_failure_phase_nonnull")
    if not artifact.get("exists"):
        reasons.append("artifact_missing")
    if artifact.get("size_bytes") == 0:
        reasons.append("artifact_zero_size")
    if effective_route and "--stop-after-shape-flow-step" not in effective_route:
        reasons.append("route_missing_stop_after_shape_flow_step")

    output_dir = route.get("output_dir")
    if output_dir and artifact.get("path"):
        output_dir_abs = os.path.abspath(output_dir)
        artifact_path = os.path.abspath(artifact["path"])
        try:
            if os.path.commonpath([output_dir_abs, artifact_path]) != output_dir_abs:
                reasons.append("artifact_outside_receipt_output_dir")
        except ValueError:
            reasons.append("artifact_outside_receipt_output_dir")
    else:
        warnings.append("receipt_output_dir_missing")

    implied_artifact = _route_implied_step_artifact(route)
    if implied_artifact and artifact.get("path"):
        artifact_path = os.path.abspath(artifact["path"])
        if artifact_path != os.path.abspath(implied_artifact):
            reasons.append("artifact_not_implied_by_effective_route")
    elif effective_route:
        warnings.append("route_implied_artifact_unknown")

    if reasons:
        status = "rejected"
        evidence_use_class = "negative_evidence"
    elif warnings:
        status = "accepted_with_warnings"
        evidence_use_class = "bounded_candidate"
    else:
        status = "accepted"
        evidence_use_class = "bounded_candidate"
    return {"status": status, "evidence_use_class": evidence_use_class, "reasons": reasons, "warnings": warnings}


def _route_implied_step_artifact(route):
    effective_route = route.get("effective_route") or ""
    output_dir = _flag_value(effective_route, "--output-dir")
    if output_dir:
        return os.path.join(output_dir, "shape_flow_step0.npz")
    save_checkpoints = _flag_value(effective_route, "--save-checkpoints")
    if save_checkpoints:
        return os.path.join(save_checkpoints, "shape_flow_step0.npz")
    return None


def _write_report(output_dir, report):
    path = os.path.join(output_dir, REPORT_NAME)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"\nSaved shape-flow step comparison report: {path}")


def _fail_report_named(output_dir, report, phase, exc):
    report["comparison_status"] = "failed"
    report["failure_phase"] = phase
    report["failure_message"] = str(exc)
    _write_report(output_dir, report)
    return 1


if __name__ == "__main__":
    sys.exit(main())

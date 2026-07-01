"""Compare shape structured latents between Trellis-Mac and trellis2mlx."""

import argparse
import hashlib
import json
import os
import sys

import numpy as np


REPORT_NAME = "shape_slat_comparison_report.json"


def main():
    parser = argparse.ArgumentParser(description="Compare shape SLat outputs")
    parser.add_argument("--trellis-mac", required=True, help="Path to Trellis-Mac shape_slat.npz")
    parser.add_argument("--trellis-mlx", required=True, help="Path to trellis2mlx shape_slat.npz")
    parser.add_argument("--trellis-mac-receipt", help="Path to Trellis-Mac Greenroom receipt.json")
    parser.add_argument("--trellis-mlx-receipt", help="Path to trellis2mlx Greenroom receipt.json")
    parser.add_argument("--output-dir", default="/tmp/shape-slat-comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "schema": "trellis2mlx.shape_slat_comparison.v1",
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
        return _fail_report(args.output_dir, report, "load_receipts", exc)

    report["route_validation"] = {
        "trellis_mac": _validate_route(report["routes"]["trellis_mac"], report["artifacts"]["trellis_mac"]),
        "trellis_mlx": _validate_route(report["routes"]["trellis_mlx"], report["artifacts"]["trellis_mlx"]),
    }
    report["unmatched_variables"] = _infer_unmatched_variables(report["routes"])
    report["route_proof_status"] = _route_proof_status(
        report["route_validation"],
        report["unmatched_variables"],
    )

    if report["route_proof_status"] == "rejected":
        report["evidence_use_class"] = "negative_evidence"
        return _fail_report(
            args.output_dir,
            report,
            "validate_routes",
            RuntimeError("route validation rejected at least one receipt/artifact pair"),
        )

    try:
        mac_feats, mac_coords = _load_slat_npz(args.trellis_mac)
        mlx_feats, mlx_coords = _load_slat_npz(args.trellis_mlx)
    except Exception as exc:
        return _fail_report(args.output_dir, report, "load_artifacts", exc)

    print(f"Trellis-Mac shape SLat: feats {mac_feats.shape}, coords {mac_coords.shape}")
    print(f"trellis2mlx shape SLat: feats {mlx_feats.shape}, coords {mlx_coords.shape}")
    report["shapes"] = {
        "trellis_mac": {"feats": list(mac_feats.shape), "coords": list(mac_coords.shape)},
        "trellis_mlx": {"feats": list(mlx_feats.shape), "coords": list(mlx_coords.shape)},
    }
    _compare_coordinate_and_features(report, mac_feats, mac_coords, mlx_feats, mlx_coords)
    report["comparison_status"] = "completed"
    _write_report(args.output_dir, report)
    return 0


def _load_slat_npz(path):
    data = np.load(path)
    if "feats" not in data.files or "coords" not in data.files:
        raise ValueError(f"{path}: shape SLat artifact must contain feats and coords arrays")
    feats = data["feats"]
    coords = data["coords"]
    if feats.ndim != 2:
        raise ValueError(f"{path}: feats must be 2D, got {feats.shape}")
    if coords.ndim != 2:
        raise ValueError(f"{path}: coords must be 2D, got {coords.shape}")
    if coords.shape[1] not in (3, 4):
        raise ValueError(f"{path}: coords must have 3 or 4 columns, got {coords.shape}")
    if feats.shape[0] != coords.shape[0]:
        raise ValueError(f"{path}: feats/coords row count mismatch: {feats.shape[0]} vs {coords.shape[0]}")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: coords must be integer dtype, got {coords.dtype}")
    return feats, coords


def _compare_coordinate_and_features(report, mac_feats, mac_coords, mlx_feats, mlx_coords):
    mac_spatial = _spatial_coords(mac_coords)
    mlx_spatial = _spatial_coords(mlx_coords)
    mac_set = set(map(tuple, mac_spatial))
    mlx_set = set(map(tuple, mlx_spatial))
    common = mac_set & mlx_set
    only_mac = mac_set - mlx_set
    only_mlx = mlx_set - mac_set
    union = mac_set | mlx_set
    jaccard = len(common) / len(union) if union else None

    print("\nShape SLat coordinate comparison:")
    print(f"  Common tokens: {len(common):,}")
    print(f"  Only in Trellis-Mac: {len(only_mac):,}")
    print(f"  Only in trellis2mlx: {len(only_mlx):,}")
    if jaccard is not None:
        print(f"  Jaccard similarity: {jaccard:.6f}")

    report["coordinate_comparison"] = {
        "mode": "spatial_set_intersection",
        "coords_match": bool(mac_set == mlx_set),
        "common_voxels": int(len(common)),
        "only_trellis_mac": int(len(only_mac)),
        "only_trellis_mlx": int(len(only_mlx)),
        "jaccard_similarity": float(jaccard) if jaccard is not None else None,
        "trellis_mac": _coord_summary(mac_coords, mac_spatial, mac_set),
        "trellis_mlx": _coord_summary(mlx_coords, mlx_spatial, mlx_set),
    }

    if not common:
        report["feature_comparison_status"] = "not_applicable_no_common_voxels"
        return

    mac_idx = {tuple(c): i for i, c in enumerate(mac_spatial)}
    mlx_idx = {tuple(c): i for i, c in enumerate(mlx_spatial)}
    common_list = sorted(common)
    mac_common_feats = np.array([mac_feats[mac_idx[c]] for c in common_list])
    mlx_common_feats = np.array([mlx_feats[mlx_idx[c]] for c in common_list])
    report["feature_comparison"] = _compare_features(
        mac_common_feats,
        mlx_common_feats,
        sample_policy="full_common_spatial_set",
    )


def _spatial_coords(coords):
    if coords.shape[1] == 4:
        return coords[:, 1:4]
    return coords


def _coord_summary(raw_coords, spatial_coords, spatial_set):
    return {
        "raw_rows": int(raw_coords.shape[0]),
        "raw_columns": int(raw_coords.shape[1]),
        "unique_spatial_voxels": int(len(spatial_set)),
        "duplicate_spatial_voxels": int(raw_coords.shape[0] - len(spatial_set)),
        "spatial_min": spatial_coords.min(axis=0).astype(int).tolist() if len(spatial_coords) else None,
        "spatial_max": spatial_coords.max(axis=0).astype(int).tolist() if len(spatial_coords) else None,
    }


def _compare_features(mac_feats, mlx_feats, *, sample_policy):
    if mac_feats.shape[1] != mlx_feats.shape[1]:
        raise ValueError(f"feature channel mismatch: {mac_feats.shape[1]} vs {mlx_feats.shape[1]}")
    report = {
        "sample_policy": sample_policy,
        "n_common_compared": int(mac_feats.shape[0]),
        "channels": {},
        "correlations": {},
        "global_abs_diff": {},
    }
    all_diff = np.abs(mac_feats - mlx_feats)
    report["global_abs_diff"] = {
        "mean": float(all_diff.mean()),
        "max": float(all_diff.max()),
        "p95": float(np.percentile(all_diff, 95)),
        "p99": float(np.percentile(all_diff, 99)),
    }
    print("\nShape SLat feature comparison:")
    print(
        f"  Global abs diff: mean={report['global_abs_diff']['mean']:.6f}, "
        f"max={report['global_abs_diff']['max']:.6f}"
    )

    for ch in range(mac_feats.shape[1]):
        mac_ch = mac_feats[:, ch]
        mlx_ch = mlx_feats[:, ch]
        diff = np.abs(mac_ch - mlx_ch)
        name = f"channel_{ch:03d}"
        channel_report = {
            "trellis_mac": {
                "min": float(mac_ch.min()),
                "max": float(mac_ch.max()),
                "mean": float(mac_ch.mean()),
                "std": float(mac_ch.std()),
            },
            "trellis_mlx": {
                "min": float(mlx_ch.min()),
                "max": float(mlx_ch.max()),
                "mean": float(mlx_ch.mean()),
                "std": float(mlx_ch.std()),
            },
            "abs_diff": {
                "mean": float(diff.mean()),
                "max": float(diff.max()),
                "p95": float(np.percentile(diff, 95)),
                "p99": float(np.percentile(diff, 99)),
            },
        }
        if mac_ch.std() > 0 and mlx_ch.std() > 0:
            channel_report["correlation"] = float(np.corrcoef(mac_ch, mlx_ch)[0, 1])
            report["correlations"][name] = channel_report["correlation"]
        report["channels"][name] = channel_report
    return report


def _artifact_record(path):
    return {
        "path": os.path.abspath(path),
        "exists": os.path.exists(path),
        "size_bytes": os.path.getsize(path) if os.path.exists(path) else None,
        "sha256": _sha256_file(path) if os.path.exists(path) else None,
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_record(receipt_path):
    if not receipt_path:
        return {
            "receipt_path": None,
            "receipt_loaded": False,
            "job_id": None,
            "job_type": None,
            "status": None,
            "effective_route": None,
            "effective_cwd": None,
            "effective_env": None,
            "effective_defaults": None,
        }
    with open(receipt_path) as f:
        receipt = json.load(f)
    return {
        "receipt_path": os.path.abspath(receipt_path),
        "receipt_loaded": True,
        "job_id": receipt.get("job_id"),
        "job_type": receipt.get("job_type"),
        "status": receipt.get("status"),
        "input_path": receipt.get("input_path"),
        "output_dir": receipt.get("output_dir"),
        "effective_route": receipt.get("effective_route"),
        "effective_cwd": receipt.get("effective_cwd"),
        "effective_env": receipt.get("effective_env"),
        "effective_defaults": receipt.get("effective_defaults"),
        "effective_timeout": receipt.get("effective_timeout"),
        "exit_code": receipt.get("exit_code"),
        "ignored_params": receipt.get("ignored_params"),
        "failure_phase": receipt.get("failure_phase"),
        "error_message": receipt.get("error_message"),
        "warnings": receipt.get("warnings"),
        "started_at": receipt.get("started_at"),
        "finished_at": receipt.get("finished_at"),
    }


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
    if effective_route and "--stop-after-shape-slat" not in effective_route:
        reasons.append("route_missing_stop_after_shape_slat")

    ignored_params = route.get("ignored_params")
    if ignored_params:
        ignored_keys = set(ignored_params)
        parity_keys = {
            "seed", "steps", "shared_noise", "pipeline_type", "resolution",
            "target_faces", "texture_size", "cwd", "output_dir", "input_path",
        }
        if ignored_keys & parity_keys:
            reasons.append("ignored_parity_params")
        else:
            warnings.append("ignored_nonparity_params")

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

    implied_artifact = _route_implied_shape_slat_artifact(route)
    if implied_artifact and artifact.get("path"):
        artifact_path = os.path.abspath(artifact["path"])
        if artifact_path != os.path.abspath(implied_artifact):
            reasons.append("artifact_not_implied_by_effective_route")
    elif effective_route:
        warnings.append("route_implied_artifact_unknown")

    if not route.get("input_path"):
        warnings.append("receipt_input_path_missing")

    if reasons:
        status = "rejected"
        evidence_use_class = "negative_evidence"
    elif warnings:
        status = "accepted_with_warnings"
        evidence_use_class = "bounded_candidate"
    else:
        status = "accepted"
        evidence_use_class = "bounded_candidate"
    return {
        "status": status,
        "evidence_use_class": evidence_use_class,
        "reasons": reasons,
        "warnings": warnings,
    }


def _route_implied_shape_slat_artifact(route):
    effective_route = route.get("effective_route") or ""
    output_dir = _flag_value(effective_route, "--output-dir")
    if output_dir:
        return os.path.join(output_dir, "shape_slat.npz")
    save_checkpoints = _flag_value(effective_route, "--save-checkpoints")
    if save_checkpoints:
        return os.path.join(save_checkpoints, "shape_slat.npz")
    return None


def _route_proof_status(route_validation, unmatched_variables):
    statuses = {v["status"] for v in route_validation.values()}
    if "rejected" in statuses:
        return "rejected"
    if unmatched_variables:
        return "candidate"
    if "missing" in statuses or "accepted_with_warnings" in statuses:
        return "candidate"
    return "accepted"


def _infer_unmatched_variables(routes):
    mac_route = routes["trellis_mac"].get("effective_route") or ""
    mlx_route = routes["trellis_mlx"].get("effective_route") or ""
    unmatched = []
    if "--seed" not in mac_route or "--seed" not in mlx_route:
        unmatched.append("sampler_seed")
    if _flag_value(mac_route, "--steps") != _flag_value(mlx_route, "--steps"):
        unmatched.append("sampler_steps")
    if _flag_value(mac_route, "--pipeline-type") != _mlx_pipeline_mode(mlx_route):
        unmatched.append("pipeline_mode")
    if routes["trellis_mac"].get("effective_env") != routes["trellis_mlx"].get("effective_env"):
        unmatched.append("execution_backend")
    if "run_official_trellis2.py" in mac_route and "generate.py" in mlx_route:
        unmatched.append("implementation_route")
        unmatched.append("image_feature_extraction")
        unmatched.append("flow_sampler_semantics")
    unmatched.append("image_preprocessing")
    unmatched.append("model_weight_identity")
    return sorted(set(unmatched))


def _flag_value(route, flag):
    parts = route.split()
    for i, part in enumerate(parts):
        if part == flag and i + 1 < len(parts):
            return parts[i + 1]
        if part.startswith(flag + "="):
            return part.split("=", 1)[1]
    return None


def _mlx_pipeline_mode(route):
    if "--no-cascade" in route:
        return "512"
    return _flag_value(route, "--resolution")


def _write_report(output_dir, report):
    path = os.path.join(output_dir, REPORT_NAME)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"\nSaved shape SLat comparison report: {path}")


def _fail_report(output_dir, report, phase, exc):
    report["comparison_status"] = "failed"
    report["failure_phase"] = phase
    report["failure_message"] = str(exc)
    _write_report(output_dir, report)
    return 1


if __name__ == "__main__":
    sys.exit(main())

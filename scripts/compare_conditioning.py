"""Compare image-conditioning tensors between Trellis-Mac and trellis2mlx."""

import argparse
import hashlib
import json
import os
import sys

import numpy as np


REPORT_NAME = "conditioning_comparison_report.json"


def main():
    parser = argparse.ArgumentParser(description="Compare image conditioning tensors")
    parser.add_argument("--trellis-mac", required=True, help="Path to Trellis-Mac conditioning.npz")
    parser.add_argument("--trellis-mlx", required=True, help="Path to trellis2mlx conditioning.npz")
    parser.add_argument("--trellis-mac-receipt", help="Path to Trellis-Mac Greenroom receipt.json")
    parser.add_argument("--trellis-mlx-receipt", help="Path to trellis2mlx Greenroom receipt.json")
    parser.add_argument("--output-dir", default="/tmp/conditioning-comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "schema": "trellis2mlx.conditioning_comparison.v1",
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
        mac = _load_conditioning_npz(args.trellis_mac)
        mlx = _load_conditioning_npz(args.trellis_mlx)
    except Exception as exc:
        return _fail_report(args.output_dir, report, "load_artifacts", exc)

    print(f"Trellis-Mac conditioning: {mac['cond'].shape}")
    print(f"trellis2mlx conditioning: {mlx['cond'].shape}")
    report["shapes"] = {
        "trellis_mac": {name: list(value.shape) for name, value in mac.items()},
        "trellis_mlx": {name: list(value.shape) for name, value in mlx.items()},
    }
    report["tensor_comparison"] = _compare_conditioning(mac, mlx)
    report["comparison_status"] = "completed"
    _write_report(args.output_dir, report)
    return 0


def _load_conditioning_npz(path):
    data = np.load(path)
    if "cond" not in data.files or "neg_cond" not in data.files:
        raise ValueError(f"{path}: conditioning artifact must contain cond and neg_cond arrays")
    cond = data["cond"]
    neg_cond = data["neg_cond"]
    if cond.ndim != 3:
        raise ValueError(f"{path}: cond must be 3D [batch,tokens,channels], got {cond.shape}")
    if neg_cond.shape != cond.shape:
        raise ValueError(f"{path}: neg_cond shape {neg_cond.shape} does not match cond {cond.shape}")
    return {"cond": cond, "neg_cond": neg_cond}


def _compare_conditioning(mac, mlx):
    shape_match = mac["cond"].shape == mlx["cond"].shape and mac["neg_cond"].shape == mlx["neg_cond"].shape
    report = {
        "shape_match": bool(shape_match),
        "cond": None,
        "neg_cond": None,
    }
    if not shape_match:
        report["status"] = "shape_mismatch_no_pointwise_tensor_diff"
        return report
    report["cond"] = _tensor_diff(mac["cond"], mlx["cond"])
    report["neg_cond"] = _tensor_diff(mac["neg_cond"], mlx["neg_cond"])
    print(
        "\nConditioning abs diff: "
        f"mean={report['cond']['abs_diff']['mean']:.6f}, "
        f"max={report['cond']['abs_diff']['max']:.6f}"
    )
    return report


def _tensor_diff(a, b):
    diff = np.abs(a - b)
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    result = {
        "abs_diff": {
            "mean": float(diff.mean()),
            "max": float(diff.max()),
            "p95": float(np.percentile(diff, 95)),
            "p99": float(np.percentile(diff, 99)),
        },
        "trellis_mac": {
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "std": float(a.std()),
        },
        "trellis_mlx": {
            "min": float(b.min()),
            "max": float(b.max()),
            "mean": float(b.mean()),
            "std": float(b.std()),
        },
    }
    if flat_a.std() > 0 and flat_b.std() > 0:
        result["correlation"] = float(np.corrcoef(flat_a, flat_b)[0, 1])
    return result


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
    if effective_route and "--stop-after-conditioning" not in effective_route:
        reasons.append("route_missing_stop_after_conditioning")

    ignored_params = route.get("ignored_params")
    if ignored_params:
        ignored_keys = set(ignored_params)
        parity_keys = {"seed", "resolution", "pipeline_type", "cwd", "output_dir", "input_path"}
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

    implied_artifact = _route_implied_conditioning_artifact(route)
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


def _route_implied_conditioning_artifact(route):
    effective_route = route.get("effective_route") or ""
    output_dir = _flag_value(effective_route, "--output-dir")
    if output_dir:
        return os.path.join(output_dir, "conditioning.npz")
    save_checkpoints = _flag_value(effective_route, "--save-checkpoints")
    if save_checkpoints:
        return os.path.join(save_checkpoints, "conditioning.npz")
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
    if _flag_value(mac_route, "--pipeline-type") != _mlx_pipeline_mode(mlx_route):
        unmatched.append("pipeline_mode")
    if routes["trellis_mac"].get("effective_env") != routes["trellis_mlx"].get("effective_env"):
        unmatched.append("execution_backend")
    if "run_official_trellis2.py" in mac_route and "generate.py" in mlx_route:
        unmatched.append("implementation_route")
        unmatched.append("image_feature_extraction")
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
    print(f"\nSaved conditioning comparison report: {path}")


def _fail_report(output_dir, report, phase, exc):
    report["comparison_status"] = "failed"
    report["failure_phase"] = phase
    report["failure_message"] = str(exc)
    _write_report(output_dir, report)
    return 1


if __name__ == "__main__":
    sys.exit(main())

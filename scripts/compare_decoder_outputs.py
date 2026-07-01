"""Compare decoder 7-channel output between Trellis-Mac and trellis2mlx.

Loads decoder_output.npz from each pipeline and reports per-channel
statistics, spatial divergence patterns, and intersection flag agreement.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compare_decoder_outputs.py \
        --trellis-mac /path/to/trellis-mac/decoder_output.npz \
        --trellis-mlx /path/to/trellis2mlx/checkpoints/decoder_output.npz \
        [--output-dir /tmp/decoder-comparison]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np


CHANNEL_NAMES = [
    "vertex_offset_x", "vertex_offset_y", "vertex_offset_z",
    "intersect_flag_x", "intersect_flag_y", "intersect_flag_z",
    "quad_split_weight",
]


def main():
    parser = argparse.ArgumentParser(description="Compare decoder outputs")
    parser.add_argument("--trellis-mac", required=True,
                        help="Path to Trellis-Mac decoder_output.npz")
    parser.add_argument("--trellis-mlx", required=True,
                        help="Path to trellis2mlx decoder_output.npz")
    parser.add_argument("--trellis-mac-receipt",
                        help="Path to Trellis-Mac Greenroom receipt.json")
    parser.add_argument("--trellis-mlx-receipt",
                        help="Path to trellis2mlx Greenroom receipt.json")
    parser.add_argument("--output-dir", default="/tmp/decoder-comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "schema": "trellis2mlx.decoder_comparison.v1",
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
    report["route_proof_status"] = _route_proof_status(report["route_validation"])
    report["unmatched_variables"] = _infer_unmatched_variables(report["routes"])

    if report["route_proof_status"] == "rejected":
        report["evidence_use_class"] = "negative_evidence"
        return _fail_report(
            args.output_dir,
            report,
            "validate_routes",
            RuntimeError("route validation rejected at least one receipt/artifact pair"),
        )

    # Load both
    try:
        mac_feats, mac_coords = _load_decoder_npz(args.trellis_mac)
        mlx_feats, mlx_coords = _load_decoder_npz(args.trellis_mlx)
    except Exception as exc:
        return _fail_report(args.output_dir, report, "load_artifacts", exc)

    print(f"Trellis-Mac: feats {mac_feats.shape}, coords {mac_coords.shape}")
    print(f"trellis2mlx: feats {mlx_feats.shape}, coords {mlx_coords.shape}")
    report["shapes"] = {
        "trellis_mac": {
            "feats": list(mac_feats.shape),
            "coords": list(mac_coords.shape),
        },
        "trellis_mlx": {
            "feats": list(mlx_feats.shape),
            "coords": list(mlx_coords.shape),
        },
    }

    if mac_feats.shape[0] != mlx_feats.shape[0]:
        print(f"\nDIFFERENT VOXEL COUNTS: {mac_feats.shape[0]:,} vs {mlx_feats.shape[0]:,}")
        print("Cannot do pointwise comparison — the decoder produced different "
              "numbers of active voxels. This means the subdivision masks diverge, "
              "which is upstream of the 7-channel features.")
    elif not np.array_equal(mac_coords, mlx_coords):
        print(f"\nCoordinates match: False")

    if mac_feats.shape[0] != mlx_feats.shape[0] or not np.array_equal(mac_coords, mlx_coords):
        _compare_coordinate_sets(report, mac_feats, mac_coords, mlx_feats, mlx_coords)
        report["comparison_status"] = "completed"
        _write_report(args.output_dir, report)
        return 0

    # Same voxel count and coordinates — check if ordering matches.
    coords_match = np.array_equal(mac_coords, mlx_coords)
    print(f"\nCoordinates match: {coords_match}")
    report["coordinate_comparison"] = {
        "mode": "pointwise",
        "coords_match": bool(coords_match),
        "common_voxels": int(mac_feats.shape[0]) if coords_match else None,
        "only_trellis_mac": 0 if coords_match else None,
        "only_trellis_mlx": 0 if coords_match else None,
        "jaccard_similarity": 1.0 if coords_match else None,
    }

    if not coords_match:
        # Try sorting by coordinates to align
        mac_order = np.lexsort(mac_coords[:, ::-1].T)
        mlx_order = np.lexsort(mlx_coords[:, ::-1].T)
        mac_sorted = mac_coords[mac_order]
        mlx_sorted = mlx_coords[mlx_order]
        if np.array_equal(mac_sorted, mlx_sorted):
            print("  Coordinates match after sorting — different ordering only")
            mac_feats = mac_feats[mac_order]
            mlx_feats = mlx_feats[mlx_order]
            report["coordinate_comparison"]["mode"] = "sorted_pointwise"
            report["coordinate_comparison"]["coords_match_after_sorting"] = True
            report["coordinate_comparison"]["common_voxels"] = int(mac_feats.shape[0])
            report["coordinate_comparison"]["only_trellis_mac"] = 0
            report["coordinate_comparison"]["only_trellis_mlx"] = 0
            report["coordinate_comparison"]["jaccard_similarity"] = 1.0
        else:
            print("  Coordinates differ even after sorting!")
            n_diff = (mac_sorted != mlx_sorted).any(axis=1).sum()
            print(f"  {n_diff:,} / {len(mac_sorted):,} coordinates differ")
            report["coordinate_comparison"]["coords_match_after_sorting"] = False
            report["coordinate_comparison"]["differing_sorted_coordinates"] = int(n_diff)

    # Compare features
    print(f"\n=== Per-Channel Feature Comparison ===")
    report["feature_comparison"] = _compare_features(
        mac_feats,
        mlx_feats,
        sample_policy="full_pointwise_set",
    )
    report["comparison_status"] = "completed"
    _write_report(args.output_dir, report)
    return 0


def _load_decoder_npz(path):
    data = np.load(path)
    feats = data['feats']
    coords = data['coords']
    if feats.ndim != 2:
        raise ValueError(f"{path}: feats must be 2D, got {feats.shape}")
    if coords.ndim != 2:
        raise ValueError(f"{path}: coords must be 2D, got {coords.shape}")
    if feats.shape[0] != coords.shape[0]:
        raise ValueError(
            f"{path}: feats/coords row count mismatch: {feats.shape[0]} vs {coords.shape[0]}"
        )
    return feats, coords


def _compare_coordinate_sets(report, mac_feats, mac_coords, mlx_feats, mlx_coords):
    mac_spatial = mac_coords[:, 1:4] if mac_coords.shape[1] == 4 else mac_coords
    mlx_spatial = mlx_coords[:, 1:4] if mlx_coords.shape[1] == 4 else mlx_coords

    mac_set = set(map(tuple, mac_spatial))
    mlx_set = set(map(tuple, mlx_spatial))
    common = mac_set & mlx_set
    only_mac = mac_set - mlx_set
    only_mlx = mlx_set - mac_set
    union = mac_set | mlx_set
    jaccard = len(common) / len(union) if union else None

    print(f"\nCoordinate comparison:")
    print(f"  Common voxels: {len(common):,}")
    print(f"  Only in Trellis-Mac: {len(only_mac):,}")
    print(f"  Only in trellis2mlx: {len(only_mlx):,}")
    if jaccard is not None:
        print(f"  Jaccard similarity: {jaccard:.4f}")

    report["coordinate_comparison"] = {
        "mode": "set_intersection",
        "coords_match": False,
        "common_voxels": len(common),
        "only_trellis_mac": len(only_mac),
        "only_trellis_mlx": len(only_mlx),
        "jaccard_similarity": jaccard,
    }

    if not common:
        report["feature_comparison_status"] = "not_applicable_no_common_voxels"
        return

    mac_idx = {tuple(c): i for i, c in enumerate(mac_spatial)}
    mlx_idx = {tuple(c): i for i, c in enumerate(mlx_spatial)}
    common_list = sorted(common)

    mac_common_feats = np.array([mac_feats[mac_idx[c]] for c in common_list])
    mlx_common_feats = np.array([mlx_feats[mlx_idx[c]] for c in common_list])

    print(f"\nFeature comparison on {len(common_list):,} common voxels:")
    report["feature_comparison"] = _compare_features(
        mac_common_feats,
        mlx_common_feats,
        sample_policy="full_common_set",
    )


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


def _route_proof_status(route_validation):
    statuses = {v["status"] for v in route_validation.values()}
    if "rejected" in statuses:
        return "rejected"
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
    path = os.path.join(output_dir, "comparison_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"\nSaved comparison report: {path}")


def _fail_report(output_dir, report, phase, exc):
    report["comparison_status"] = "failed"
    report["failure_phase"] = phase
    report["failure_message"] = str(exc)
    _write_report(output_dir, report)
    return 1


def _compare_features(mac_feats, mlx_feats, *, sample_policy):
    """Compare 7-channel features between two arrays."""
    report = {
        "sample_policy": sample_policy,
        "n_common_compared": int(mac_feats.shape[0]),
        "channels": {},
        "correlations": {},
    }

    for ch in range(min(mac_feats.shape[1], 7)):
        mac_ch = mac_feats[:, ch]
        mlx_ch = mlx_feats[:, ch]
        diff = np.abs(mac_ch - mlx_ch)
        name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}"

        print(f"\n  {name}:")
        print(f"    Mac range: [{mac_ch.min():.4f}, {mac_ch.max():.4f}], "
              f"mean: {mac_ch.mean():.4f}")
        print(f"    MLX range: [{mlx_ch.min():.4f}, {mlx_ch.max():.4f}], "
              f"mean: {mlx_ch.mean():.4f}")
        print(f"    Abs diff: mean={diff.mean():.6f}, max={diff.max():.6f}, "
              f"p95={np.percentile(diff, 95):.6f}, p99={np.percentile(diff, 99):.6f}")
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

        # For intersection flags (channels 3-5): compare thresholded agreement
        if 3 <= ch <= 5:
            mac_flag = mac_ch > 0
            mlx_flag = mlx_ch > 0
            agree = (mac_flag == mlx_flag).mean()
            n_disagree = (mac_flag != mlx_flag).sum()
            print(f"    Flag agreement (>0): {agree:.4f} ({n_disagree:,} disagree)")
            # Which direction is the disagreement?
            mac_yes_mlx_no = (mac_flag & ~mlx_flag).sum()
            mac_no_mlx_yes = (~mac_flag & mlx_flag).sum()
            print(f"    Mac=yes MLX=no: {mac_yes_mlx_no:,}, Mac=no MLX=yes: {mac_no_mlx_yes:,}")
            channel_report["flag_agreement_gt_0"] = {
                "agreement": float(agree),
                "n_disagree": int(n_disagree),
                "trellis_mac_yes_mlx_no": int(mac_yes_mlx_no),
                "trellis_mac_no_mlx_yes": int(mac_no_mlx_yes),
            }
        report["channels"][name] = channel_report

    # Overall correlation
    for ch in range(min(mac_feats.shape[1], 7)):
        mac_ch = mac_feats[:, ch]
        mlx_ch = mlx_feats[:, ch]
        if mac_ch.std() > 0 and mlx_ch.std() > 0:
            corr = np.corrcoef(mac_ch, mlx_ch)[0, 1]
            name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}"
            print(f"  Correlation {name}: {corr:.6f}")
            report["correlations"][name] = float(corr)
    return report


if __name__ == "__main__":
    sys.exit(main())

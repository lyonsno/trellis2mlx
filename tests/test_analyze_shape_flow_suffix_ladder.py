import hashlib
import json
import copy
import subprocess
import sys

import numpy as np
import pytest


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _refresh_download_report(receipt_path, download_report, result_json, result_npz):
    downloaded = {
        name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for name, path in (
            ("cuda_result.json", result_json),
            ("cuda_result.npz", result_npz),
            ("kaggle_cuda_witness_receipt.json", receipt_path),
        )
    }
    download_report.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.kaggle_cuda_witness.command_report.v1",
                "phase": "kernel_output",
                "status": "done",
                "failure_phase": None,
                "exit_code": 0,
                "downloaded_outputs": downloaded,
            }
        )
        + "\n"
    )


def _refresh_evidence_chain(receipt_path, download_report, result_json, result_npz):
    receipt = json.loads(receipt_path.read_text())
    for name, path in (("cuda_result.json", result_json), ("cuda_result.npz", result_npz)):
        receipt["outputs"][name] = {
            "exists": True,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    receipt_path.write_text(json.dumps(receipt) + "\n")
    _refresh_download_report(receipt_path, download_report, result_json, result_npz)


def _compare(left, right):
    diff = np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32))
    return {
        "shape_match": True,
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "nonzero": int(np.count_nonzero(diff)),
        "exact": bool(np.array_equal(left, right)),
    }


def _write_ladder(tmp_path, values=None, *, receipt_status="done", device="Tesla T4"):
    if values is None:
        values = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    source = np.zeros((2, 3), dtype=np.float32)
    mlx = np.ones((2, 3), dtype=np.float32)
    coords = np.asarray([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=np.int32)
    arrays = {
        "coords": coords,
        "accepted_source_anchor_shape_slat": source,
        "mlx_anchor_shape_slat": mlx,
        "switch_steps": np.arange(9, dtype=np.int32),
    }
    points = []
    endpoints = []
    for step, value in enumerate(values):
        endpoint = np.full(source.shape, value, dtype=np.float32)
        if isinstance(value, (list, tuple, np.ndarray)):
            endpoint = np.asarray(value, dtype=np.float32).reshape(source.shape)
        endpoints.append(endpoint)
        key = f"switch_{step}_shape_slat"
        arrays[key] = endpoint
        source_indices = list(range(step, 8))
        points.append(
            {
                "switch_step": step,
                "source_step_indices": source_indices,
                "source_step_count": len(source_indices),
                "output_key": key,
                "shape": list(endpoint.shape),
                "sha256": _array_sha(endpoint),
                "elapsed_seconds": 1.0,
                "step_elapsed_seconds": [0.1] * len(source_indices),
                "vs_source_anchor": _compare(endpoint, source),
                "vs_mlx_anchor": _compare(endpoint, mlx),
                "nearest_anchor": (
                    "source"
                    if _compare(endpoint, source)["mean_abs"]
                    < _compare(endpoint, mlx)["mean_abs"]
                    else "mlx"
                    if _compare(endpoint, mlx)["mean_abs"]
                    < _compare(endpoint, source)["mean_abs"]
                    else "equidistant"
                ),
            }
        )
    pairwise = {
        f"{left}:{right}": _compare(endpoints[left], endpoints[right])
        for left in range(9)
        for right in range(9)
    }
    route = {
        "route": "official-source-cuda-shape-flow-suffix-ladder-from-exact-mlx-prefixes",
        "device_type": "cuda",
        "cuda_device": device,
        "attention_backend": "sdpa",
        "conv_backend": "none",
        "steps": 8,
        "switch_steps": list(range(9)),
        "one_model_load": True,
        "comparison_class": "exact-mlx-prefix-plus-source-cuda-suffix",
    }
    input_identity = {
        "mlx": {"capture_sha256": "a" * 64},
        "accepted_source": {
            "baseline_sha256": "b" * 64,
            "report_sha256": "c" * 64,
        },
        "source_tar_sha256": "d" * 64,
        "conditioning_sha256": "e" * 64,
    }
    timing = {
        "model_load_seconds": 3.0,
        "suffix_continuation_seconds": 36.0,
        "source_steps_completed": 36,
        "source_steps_requested": 36,
        "switch_points_completed": 9,
        "switch_points_requested": 9,
        "t4_compute_seconds_through_continuation": 40.0,
    }
    metadata = {
        "schema": "trellis2mlx.source_cuda_shape_flow_suffix_ladder.artifact.v1",
        "artifact_status": "computed_pending_serialization",
        "external_report_required": True,
        "effective_route": route,
        "inputs": input_identity,
        "points": points,
        "pairwise": pairwise,
        "timing": timing,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    result_npz = tmp_path / "cuda_result.npz"
    np.savez(result_npz, **arrays)
    result_json = tmp_path / "cuda_result.json"
    report = {
        "schema": "trellis2mlx.source_cuda_shape_flow_suffix_ladder.v1",
        "status": "done",
        "failure_phase": None,
        "last_trustworthy_phase": "all_suffixes_and_exact_boundaries_saved",
        "primary_output_status": "written",
        "primary_output": {
            "path": str(result_npz),
            "sha256": _sha256(result_npz),
            "size_bytes": result_npz.stat().st_size,
            "validation": {"point_arrays_bound": True, "switch_count": 9},
        },
        "effective_route": route,
        "inputs": input_identity,
        "points": points,
        "pairwise": pairwise,
        "timing": timing,
    }
    result_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    packet_manifest = tmp_path / "witness-manifest.json"
    manifest_files = {
        "source_cuda_shape_flow_suffix_ladder.py": {"sha256": "1" * 64, "size_bytes": 10},
        "shape_flow_steps.npz": {"sha256": "a" * 64, "size_bytes": 11},
        "accepted_source_baseline.npz": {"sha256": "b" * 64, "size_bytes": 12},
        "accepted_source_report.json": {"sha256": "c" * 64, "size_bytes": 13},
        "trellis2_source_tarball.bin": {"sha256": "d" * 64, "size_bytes": 14},
        "conditioning.npz": {"sha256": "e" * 64, "size_bytes": 15},
    }
    packet_manifest.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.kaggle_cuda_witness.inputs.v1",
                "dataset_id": "operator/suffix-inputs",
                "kernel_id": "operator/suffix-kernel",
                "accelerator": "NvidiaTeslaT4",
                "entrypoint": "source_cuda_shape_flow_suffix_ladder.py",
                "entrypoint_args": [
                    "--sparse-conv-backend", "none", "--sparse-attn-backend", "sdpa"
                ],
                "outputs": ["cuda_result.json", "cuda_result.npz"],
                "files": manifest_files,
            },
            sort_keys=True,
        )
        + "\n"
    )
    receipt = tmp_path / "kaggle_cuda_witness_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.kaggle_cuda_witness.receipt.v1",
                "status": receipt_status,
                "requested_dataset_id": "operator/suffix-inputs",
                "requested_kernel_id": "operator/suffix-kernel",
                "requested_accelerator": "NvidiaTeslaT4",
                "cuda_available": True,
                "cuda_device": device,
                "torch": "2.10.0+cu128",
                "exit_code": 0,
                "effective_dataset_dir": "/kaggle/input/datasets/operator/suffix-inputs",
                "input_manifest": {
                    "sha256": _sha256(packet_manifest),
                    "size_bytes": packet_manifest.stat().st_size,
                },
                "inputs": manifest_files,
                "effective_command": [
                    "python",
                    "source_cuda_shape_flow_suffix_ladder.py",
                    "--output-json",
                    "cuda_result.json",
                    "--output-npz",
                    "cuda_result.npz",
                    "--sparse-conv-backend",
                    "none",
                    "--sparse-attn-backend",
                    "sdpa",
                ],
                "source_identity": {
                    "dataset_sources": ["operator/suffix-inputs"],
                    "competition_sources": [],
                    "kernel_sources": [],
                    "model_sources": [],
                },
                "mounted_input_snapshot": {
                    "mounted_input_root_exists": True,
                    "mounted_input_dirs": [
                        "/kaggle/input/datasets",
                        "/kaggle/input/datasets/operator",
                        "/kaggle/input/datasets/operator/suffix-inputs",
                    ],
                    "mounted_input_files": [
                        f"/kaggle/input/datasets/operator/suffix-inputs/{name}"
                        for name in sorted((*manifest_files, "witness-manifest.json"))
                    ],
                },
                "outputs": {
                    "cuda_result.json": {
                        "exists": True,
                        "sha256": _sha256(result_json),
                        "size_bytes": result_json.stat().st_size,
                    },
                    "cuda_result.npz": {
                        "exists": True,
                        "sha256": _sha256(result_npz),
                        "size_bytes": result_npz.stat().st_size,
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    download_report = tmp_path / "kernel_output.json"
    _refresh_evidence_chain(receipt, download_report, result_json, result_npz)
    return result_json, result_npz, receipt, download_report, packet_manifest


def test_analyzer_maps_exact_anchor_axis_and_single_separatrix(tmp_path):
    from scripts.analyze_shape_flow_suffix_ladder import analyze_suffix_ladder

    result_json, result_npz, receipt, download_report, manifest = _write_ladder(tmp_path)
    analysis = analyze_suffix_ladder(result_json, result_npz, receipt, download_report, manifest)

    assert analysis["status"] == "done"
    assert analysis["switch_steps"] == list(range(9))
    assert [point["anchor_axis_projection"] for point in analysis["points"]] == pytest.approx(
        [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    )
    assert [point["transverse_l2_ratio"] for point in analysis["points"]] == pytest.approx(
        [0.0] * 9
    )
    assert analysis["separatrix"]["classification_transitions"] == [
        {"left_switch": 4, "right_switch": 5, "left_class": "source", "right_class": "mlx"}
    ]
    assert analysis["separatrix"]["disposition"] == "single_crossing"
    assert analysis["decode_recommendation"]["switch_steps"] == [4, 5]
    assert analysis["decode_recommendation"]["anchor_context_steps"] == [0, 8]


def test_analyzer_preserves_exact_quotients_and_all_nonmonotonic_crossings(tmp_path):
    from scripts.analyze_shape_flow_suffix_ladder import analyze_suffix_ladder

    values = [0.0, 0.2, 0.8, 0.8, 0.3, 0.7, 0.4, 0.9, 1.0]
    result_json, result_npz, receipt, download_report, manifest = _write_ladder(tmp_path, values)
    analysis = analyze_suffix_ladder(result_json, result_npz, receipt, download_report, manifest)

    assert [entry["switch_steps"] for entry in analysis["exact_quotient_classes"]] == [
        [0], [1], [2, 3], [4], [5], [6], [7], [8]
    ]
    assert analysis["separatrix"]["disposition"] == "multiple_crossings"
    assert [
        (item["left_switch"], item["right_switch"])
        for item in analysis["separatrix"]["classification_transitions"]
    ] == [(1, 2), (3, 4), (4, 5), (5, 6), (6, 7)]
    assert analysis["decode_recommendation"]["switch_steps"] == [1, 2, 4, 5, 6, 7]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda report, receipt: receipt.update(status="failed"), "receipt is not done"),
        (lambda report, receipt: receipt.update(cuda_device="CPU"), "Tesla T4"),
        (
            lambda report, receipt: receipt["outputs"]["cuda_result.npz"].update(sha256="f" * 64),
            "receipt output digest cuda_result.npz",
        ),
        (
            lambda report, receipt: report["effective_route"].update(attention_backend="flash_attn"),
            "attention_backend",
        ),
        (lambda report, receipt: report["timing"].update(source_steps_completed=35), "source_steps_completed"),
    ],
)
def test_analyzer_rejects_false_route_and_partial_completion(tmp_path, mutator, message):
    from scripts.analyze_shape_flow_suffix_ladder import analyze_suffix_ladder

    result_json, result_npz, receipt_path, download_report, manifest = _write_ladder(tmp_path)
    report = json.loads(result_json.read_text())
    original_report = copy.deepcopy(report)
    receipt = json.loads(receipt_path.read_text())
    mutator(report, receipt)
    if report != original_report:
        result_json.write_text(json.dumps(report) + "\n")
        receipt["outputs"]["cuda_result.json"] = {
            "exists": True,
            "sha256": _sha256(result_json),
            "size_bytes": result_json.stat().st_size,
        }
    receipt_path.write_text(json.dumps(receipt) + "\n")
    _refresh_download_report(receipt_path, download_report, result_json, result_npz)

    with pytest.raises(ValueError, match=message):
        analyze_suffix_ladder(result_json, result_npz, receipt_path, download_report, manifest)


def test_analyzer_rejects_stale_npz_and_reported_pairwise_lies(tmp_path):
    from scripts.analyze_shape_flow_suffix_ladder import analyze_suffix_ladder

    result_json, result_npz, receipt, download_report, manifest = _write_ladder(tmp_path)
    report = json.loads(result_json.read_text())
    report["primary_output"]["sha256"] = "0" * 64
    result_json.write_text(json.dumps(report) + "\n")
    _refresh_evidence_chain(receipt, download_report, result_json, result_npz)
    with pytest.raises(ValueError, match="primary NPZ digest"):
        analyze_suffix_ladder(result_json, result_npz, receipt, download_report, manifest)

    result_json, result_npz, receipt, download_report, manifest = _write_ladder(tmp_path)
    report = json.loads(result_json.read_text())
    report["pairwise"]["4:5"]["mean_abs"] = 99.0
    result_json.write_text(json.dumps(report) + "\n")
    _refresh_evidence_chain(receipt, download_report, result_json, result_npz)
    with pytest.raises(ValueError, match="metadata pairwise matrix|pairwise 4:5"):
        analyze_suffix_ladder(result_json, result_npz, receipt, download_report, manifest)


def test_analyzer_rejects_nonfinite_endpoint_even_with_rehashed_lie(tmp_path):
    from scripts.analyze_shape_flow_suffix_ladder import analyze_suffix_ladder

    result_json, result_npz, receipt, download_report, manifest = _write_ladder(tmp_path)
    with np.load(result_npz, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["switch_4_shape_slat"] = arrays["switch_4_shape_slat"].copy()
    arrays["switch_4_shape_slat"][0, 0] = np.nan
    np.savez(result_npz, **arrays)
    report = json.loads(result_json.read_text())
    report["primary_output"]["sha256"] = _sha256(result_npz)
    report["points"][4]["sha256"] = _array_sha(arrays["switch_4_shape_slat"])
    result_json.write_text(json.dumps(report) + "\n")
    _refresh_evidence_chain(receipt, download_report, result_json, result_npz)

    with pytest.raises(ValueError, match="invalid dtype or values|non-finite"):
        analyze_suffix_ladder(result_json, result_npz, receipt, download_report, manifest)


def test_analyzer_rejects_duplicate_switch_output_key_before_quotienting(tmp_path):
    from scripts.analyze_shape_flow_suffix_ladder import analyze_suffix_ladder

    result_json, result_npz, receipt, download_report, manifest = _write_ladder(tmp_path)
    report = json.loads(result_json.read_text())
    report["points"][4]["output_key"] = "switch_5_shape_slat"
    result_json.write_text(json.dumps(report) + "\n")
    _refresh_evidence_chain(receipt, download_report, result_json, result_npz)

    with pytest.raises(ValueError, match="canonical output key"):
        analyze_suffix_ladder(result_json, result_npz, receipt, download_report, manifest)


def test_analyzer_binds_receipt_through_download_ledger(tmp_path):
    from scripts.analyze_shape_flow_suffix_ladder import analyze_suffix_ladder

    result_json, result_npz, receipt, download_report, manifest = _write_ladder(tmp_path)
    ledger = json.loads(download_report.read_text())
    ledger["downloaded_outputs"]["kaggle_cuda_witness_receipt.json"]["sha256"] = "0" * 64
    download_report.write_text(json.dumps(ledger) + "\n")

    with pytest.raises(ValueError, match="downloaded receipt digest"):
        analyze_suffix_ladder(result_json, result_npz, receipt, download_report, manifest)


def test_analyzer_rejects_extra_mounted_input_even_when_outputs_are_bound(tmp_path):
    from scripts.analyze_shape_flow_suffix_ladder import analyze_suffix_ladder

    result_json, result_npz, receipt, download_report, manifest = _write_ladder(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["mounted_input_snapshot"]["mounted_input_dirs"].append(
        "/kaggle/input/datasets/operator/shadow-inputs"
    )
    payload["mounted_input_snapshot"]["mounted_input_files"].append(
        "/kaggle/input/datasets/operator/shadow-inputs/override.py"
    )
    receipt.write_text(json.dumps(payload) + "\n")
    _refresh_download_report(receipt, download_report, result_json, result_npz)

    with pytest.raises(ValueError, match="mounted input (directory|file) set"):
        analyze_suffix_ladder(result_json, result_npz, receipt, download_report, manifest)


def test_analyzer_rejects_extra_kernel_source_identity(tmp_path):
    from scripts.analyze_shape_flow_suffix_ladder import analyze_suffix_ladder

    result_json, result_npz, receipt, download_report, manifest = _write_ladder(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["source_identity"]["kernel_sources"] = ["operator/shadow-kernel"]
    receipt.write_text(json.dumps(payload) + "\n")
    _refresh_download_report(receipt, download_report, result_json, result_npz)

    with pytest.raises(ValueError, match="receipt source identity"):
        analyze_suffix_ladder(result_json, result_npz, receipt, download_report, manifest)


def test_cli_failure_removes_stale_analysis_and_writes_durable_report(tmp_path):
    from scripts.analyze_shape_flow_suffix_ladder import main

    output = tmp_path / "analysis.json"
    output.write_text('{"status":"stale"}\n')
    status = main(
        [
            "--result-json", str(tmp_path / "missing-result.json"),
            "--result-npz", str(tmp_path / "missing-result.npz"),
            "--receipt", str(tmp_path / "missing-receipt.json"),
            "--download-report", str(tmp_path / "missing-download.json"),
            "--packet-manifest", str(tmp_path / "missing-manifest.json"),
            "--output-json", str(output),
        ]
    )

    assert status == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "input_validation"
    assert report["analysis_status"] == "missing"
    assert "missing-result.json" in report["error"]


def test_script_entrypoint_is_directly_executable():
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_shape_flow_suffix_ladder.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--result-json" in completed.stdout

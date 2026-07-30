import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.decoder_level1_trace_contract import (
    REQUIRED_ARRAYS,
    decoder_level1_trace_input_sha256,
    load_decoder_level1_trace,
    validate_decoder_level1_trace,
    write_decoder_level1_trace_npz,
)


def _valid_trace():
    parent_coords = np.array(
        [[0, 1, 2, 3], [0, 5, 6, 7]],
        dtype=np.int32,
    )
    parent_rows = parent_coords.shape[0]
    level0_output = np.arange(
        parent_rows * 1024,
        dtype=np.float16,
    ).reshape(parent_rows, 1024)
    subdiv_logits = np.full((parent_rows, 8), -1, dtype=np.float16)
    subdiv_logits[0, [0, 3, 7]] = 1
    subdiv_logits[1, [1, 6]] = 1
    mask = subdiv_logits > 0
    parent_indices, child_indices = np.nonzero(mask)
    child_coords = parent_coords[parent_indices].copy()
    child_coords[:, 1:] *= 2
    child_coords[:, 1] += child_indices % 2
    child_coords[:, 2] += (child_indices // 2) % 2
    child_coords[:, 3] += child_indices // 4
    child_rows = child_coords.shape[0]
    conv1 = np.arange(
        parent_rows * 4096,
        dtype=np.float16,
    ).reshape(parent_rows, 4096)
    h_c2s = conv1.reshape(parent_rows, 8, 512)[mask]
    skip_c2s = level0_output.reshape(parent_rows, 8, 128)[mask]
    skip_repeated = np.repeat(skip_c2s, 4, axis=1)

    arrays = {
        "parent_coords": parent_coords,
        "child_coords": child_coords,
        "level0_output": level0_output,
        "upsample_subdiv_logits": subdiv_logits,
        "upsample_norm1": np.zeros((parent_rows, 1024), dtype=np.float16),
        "upsample_silu1": np.zeros((parent_rows, 1024), dtype=np.float16),
        "upsample_conv1": conv1,
        "upsample_h_c2s": h_c2s,
        "upsample_skip_c2s": skip_c2s,
        "upsample_skip_repeated": skip_repeated,
        "upsample_norm2": np.zeros((child_rows, 512), dtype=np.float16),
        "upsample_silu2": np.zeros((child_rows, 512), dtype=np.float16),
        "upsample_conv2": np.zeros((child_rows, 512), dtype=np.float16),
        "upsample_output": np.zeros((child_rows, 512), dtype=np.float16),
        "level1_block0_conv": np.zeros((child_rows, 512), dtype=np.float16),
        "level1_block0_norm": np.zeros((child_rows, 512), dtype=np.float16),
        "level1_block0_mlp_fc1": np.zeros((child_rows, 2048), dtype=np.float16),
        "level1_block0_silu": np.zeros((child_rows, 2048), dtype=np.float16),
        "level1_block0_mlp_fc2": np.zeros((child_rows, 512), dtype=np.float16),
        "level1_block0_output": np.zeros((child_rows, 512), dtype=np.float16),
    }
    return arrays


def test_level1_trace_contract_round_trips_exactly(tmp_path):
    arrays = _valid_trace()
    output = tmp_path / "trace.npz"

    report = write_decoder_level1_trace_npz(output, arrays)
    reopened = load_decoder_level1_trace(output)

    assert report["reopened_exact"] is True
    assert report["child_expansion_exact"] is True
    assert tuple(reopened) == REQUIRED_ARRAYS
    for name in REQUIRED_ARRAYS:
        np.testing.assert_array_equal(reopened[name], arrays[name])


def test_level1_trace_contract_rejects_child_coordinate_reordering():
    arrays = _valid_trace()
    arrays["child_coords"] = arrays["child_coords"][::-1].copy()

    with pytest.raises(ValueError, match="parent-major"):
        validate_decoder_level1_trace(arrays)


def test_level1_trace_contract_rejects_mislabeled_channel_slice():
    arrays = _valid_trace()
    arrays["upsample_h_c2s"] = arrays["upsample_h_c2s"].copy()
    arrays["upsample_h_c2s"][0, 0] += np.float16(1)

    with pytest.raises(ValueError, match="conv1 channel slices"):
        validate_decoder_level1_trace(arrays)


def test_level1_trace_contract_rejects_wrong_skip_repeat_order():
    arrays = _valid_trace()
    arrays["upsample_skip_repeated"] = np.tile(
        arrays["upsample_skip_c2s"],
        (1, 4),
    )

    with pytest.raises(ValueError, match="source repeat order"):
        validate_decoder_level1_trace(arrays)


def test_level1_trace_contract_rejects_missing_or_extra_arrays():
    missing = _valid_trace()
    missing.pop("level1_block0_silu")
    with pytest.raises(KeyError, match="level1_block0_silu"):
        validate_decoder_level1_trace(missing)

    extra = _valid_trace()
    extra["cached_output"] = extra["level1_block0_output"]
    with pytest.raises(KeyError, match="cached_output"):
        validate_decoder_level1_trace(extra)


def test_level1_trace_input_identity_binds_parent_values_and_coords():
    arrays = _valid_trace()
    baseline = decoder_level1_trace_input_sha256(
        arrays["level0_output"],
        arrays["parent_coords"],
    )
    changed_values = arrays["level0_output"].copy()
    changed_values[0, 0] += np.float16(1)
    changed_coords = arrays["parent_coords"].copy()
    changed_coords[0, 1] += 1

    assert baseline != decoder_level1_trace_input_sha256(
        changed_values,
        arrays["parent_coords"],
    )
    assert baseline != decoder_level1_trace_input_sha256(
        arrays["level0_output"],
        changed_coords,
    )


def _write_comparison_inputs(tmp_path, source_arrays, local_arrays):
    source_path = tmp_path / "source.npz"
    local_path = tmp_path / "local.npz"
    write_decoder_level1_trace_npz(source_path, source_arrays)
    write_decoder_level1_trace_npz(local_path, local_arrays)
    input_identity = decoder_level1_trace_input_sha256(
        source_arrays["level0_output"],
        source_arrays["parent_coords"],
    )
    parent_path = tmp_path / "level0-parent.npz"
    parent_path.write_bytes(b"level-zero-parent")
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    silu_path = tmp_path / "decoder-silu-lut.npz"
    silu_path.write_bytes(b"authenticated-silu")
    silu_sha = hashlib.sha256(silu_path.read_bytes()).hexdigest()
    validation = {
        "reopened_exact": True,
        "child_expansion_exact": True,
    }
    source_report = tmp_path / "source.json"
    source_report.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.source_cuda_shape_slat_grid_decode.v1",
                "status": "done",
                "effective_route": {
                    "route": "official-source-cuda-shape-decoder-level1-trace",
                    "device_type": "cuda",
                    "cuda_device": "Tesla T4",
                    "sparse_conv_backend": "none",
                    "decoder_state_only": False,
                    "decoder_level0_trace": False,
                    "decoder_level1_trace": True,
                    "raw_meshes": False,
                    "post_fill_holes_snapshots": False,
                    "mesh_conversion": False,
                    "one_model_load": True,
                },
                "decoder_trace_artifacts": [
                    {
                        "path": str(source_path),
                        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        "input_tensor_sha256": input_identity,
                        "status": "written",
                        "validation": validation,
                    }
                ],
            }
        )
    )
    local_report = tmp_path / "local.json"
    local_report.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.decoder_level1_trace_run.v1",
                "status": "done",
                "effective_route": {
                    "route": "mlx-shape-decoder-level1-trace",
                    "device_type": "metal",
                    "device": "Device(gpu, 0)",
                    "decoder_linear_backend": "turing_fda",
                    "sparse_conv_matmul_backend": "turing_fda",
                    "decoder_layernorm": {
                        "backend": "mlx-fast-layer-norm",
                        "algorithm": "mlx-fast-layer-norm",
                        "experimental": False,
                    },
                    "decoder_silu": {
                        "backend": "cuda-turing-t4-fp16-lut",
                        "algorithm": "exhaustive-fp16-bit-pattern-output-lookup",
                        "experimental": True,
                        "cuda_architecture": "sm_75",
                        "cuda_device_anchor": "Tesla T4",
                        "cuda_source_operation": "torch.nn.functional.silu",
                        "cuda_source_version": "torch-2.10.0+cu128",
                        "authenticated_contract": {
                            "input_dtype": "float16",
                            "output_dtype": "float16",
                            "domain": "all-65536-bit-patterns",
                        },
                        "output_lut_artifact_path": str(silu_path),
                        "output_lut_artifact_sha256_attested": silu_sha,
                        "output_lut_artifact_sha256_effective": silu_sha,
                    },
                    "parent_state": {
                        "path": str(parent_path),
                        "sha256": parent_sha,
                        "input_tensor_sha256": input_identity,
                    },
                },
                "input_tensor_sha256": input_identity,
                "parent_trace": {
                    "path": str(parent_path),
                    "sha256": parent_sha,
                    "input_tensor_sha256": input_identity,
                },
                "primary": {
                    "path": str(local_path),
                    "sha256": hashlib.sha256(local_path.read_bytes()).hexdigest(),
                    "status": "written",
                    "validation": validation,
                },
            }
        )
    )
    return source_path, source_report, local_path, local_report


def test_level1_comparator_reports_hidden_internal_boundary_first(tmp_path):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    source = _valid_trace()
    local = {name: values.copy() for name, values in source.items()}
    local["level1_block0_silu"][0, 0] += np.float16(1)
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, source, local)
    )

    comparison = compare_level1_traces(
        source_path=source_path,
        source_report_path=source_report,
        local_path=local_path,
        local_report_path=local_report,
    )

    assert comparison["first_nonexact_boundary"] == "level1_block0_silu"
    assert comparison["stages"]["level1_block0_silu"]["nonzero_count"] == 1
    assert comparison["stages"]["level1_block0_output"]["nonzero_count"] == 0


def test_level1_comparator_rejects_wrong_effective_route(tmp_path):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    source = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, source, source)
    )
    report = json.loads(local_report.read_text())
    report["effective_route"]["route"] = "mlx-shape-decoder-level0-trace-fp16"
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="local trace route field"):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_level1_comparator_rejects_under_authenticated_local_route(tmp_path):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    report = json.loads(local_report.read_text())
    report["effective_route"] = {
        "route": "mlx-shape-decoder-level1-trace",
        "device_type": "cpu",
        "device": "cpu",
        "decoder_linear_backend": "native",
        "sparse_conv_matmul_backend": "native",
    }
    report["primary"]["status"] = "not_written"
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="local trace route field|primary status"):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_level1_comparator_failed_run_replaces_stale_output_with_phase_report(
    tmp_path,
):
    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    report = json.loads(local_report.read_text())
    report["effective_route"]["route"] = "wrong-local-route"
    local_report.write_text(json.dumps(report))
    output = tmp_path / "comparison.json"
    output.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.decoder_level1_trace_comparison.v1",
                "status": "done",
                "first_nonexact_boundary": "stale_boundary",
            }
        )
    )
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/compare_decoder_level1_traces.py"),
            "--source",
            str(source_path),
            "--source-report",
            str(source_report),
            "--local",
            str(local_path),
            "--local-report",
            str(local_report),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    failed = json.loads(output.read_text())
    assert completed.returncode != 0
    assert failed["status"] == "failed"
    assert failed["failure_phase"] == "comparison"
    assert failed["last_trustworthy_phase"] == "request_validation"
    assert failed["first_nonexact_boundary"] is None


def test_level1_comparator_collision_preserves_input_and_uses_failure_sibling(
    tmp_path,
):
    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    original_report = local_report.read_bytes()
    fallback = local_report.with_name(local_report.name + ".failure.json")
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/compare_decoder_level1_traces.py"),
            "--source",
            str(source_path),
            "--source-report",
            str(source_report),
            "--local",
            str(local_path),
            "--local-report",
            str(local_report),
            "--output",
            str(local_report),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert local_report.read_bytes() == original_report
    failed = json.loads(fallback.read_text())
    assert failed["status"] == "failed"
    assert failed["failure_phase"] == "request_validation"
    assert failed["requested"]["output"] == str(local_report)
    assert failed["effective_output"] == str(fallback)


def test_level1_trace_script_entry_points_resolve_repo_contract_module():
    repo_root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "scripts/run_mlx_decoder_level1_trace.py",
        "scripts/compare_decoder_level1_traces.py",
        "scripts/source_cuda_postcond_full_decode_timing.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(repo_root / relative_path), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            f"{relative_path} direct entry failed:\n{completed.stderr}"
        )


def test_local_level1_trace_failure_writes_phase_report_and_invalidates_stale_primary(
    tmp_path,
):
    from scripts.run_mlx_decoder_level1_trace import main

    missing_parent = tmp_path / "missing-level0.npz"
    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    silu_lut = tmp_path / "silu.npz"
    silu_lut.write_bytes(b"silu")
    output_npz = tmp_path / "trace.npz"
    output_npz.write_bytes(b"stale-primary")
    output_json = tmp_path / "trace.json"

    rc = main(
        [
            "--level0-trace",
            str(missing_parent),
            "--expected-level0-trace-sha256",
            "a" * 64,
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--decoder-silu-lut",
            str(silu_lut),
            "--expected-decoder-silu-lut-sha256",
            hashlib.sha256(silu_lut.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text())
    assert rc == 1
    assert report["schema"] == "trellis2mlx.decoder_level1_trace_run.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "parent_trace_validation"
    assert report["last_trustworthy_phase"] == "request_validation"
    assert report["effective_route"] is None
    assert report["stale_primary_invalidated"] is True
    assert report["primary"]["status"] == "not_written"
    assert not output_npz.exists()


def test_local_level1_trace_rejects_stale_parent_digest_before_model_load(
    tmp_path,
):
    from scripts.run_mlx_decoder_level1_trace import main

    parent = tmp_path / "level0.npz"
    rows = 2
    np.savez(
        parent,
        coords=np.array(
            [[0, 1, 2, 3], [0, 4, 5, 6]],
            dtype=np.int32,
        ),
        block3_output=np.ones((rows, 1024), dtype=np.float16),
    )

    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    silu_lut = tmp_path / "silu.npz"
    silu_lut.write_bytes(b"silu")
    output_npz = tmp_path / "trace.npz"
    output_json = tmp_path / "trace.json"

    rc = main(
        [
            "--level0-trace",
            str(parent),
            "--expected-level0-trace-sha256",
            "0" * 64,
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--decoder-silu-lut",
            str(silu_lut),
            "--expected-decoder-silu-lut-sha256",
            hashlib.sha256(silu_lut.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text())
    assert rc == 1
    assert report["failure_phase"] == "parent_trace_validation"
    assert "level-zero trace digest mismatch" in report["error"]
    assert report["last_trustworthy_phase"] == "request_validation"
    assert not output_npz.exists()


def test_local_level1_trace_report_collision_preserves_parent_and_uses_sibling(
    tmp_path,
):
    from scripts.run_mlx_decoder_level1_trace import main

    parent = tmp_path / "level0.npz"
    np.savez(
        parent,
        coords=np.array([[0, 1, 2, 3]], dtype=np.int32),
        block3_output=np.ones((1, 1024), dtype=np.float16),
    )
    original_parent = parent.read_bytes()
    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    silu_lut = tmp_path / "silu.npz"
    silu_lut.write_bytes(b"silu")
    output_npz = tmp_path / "trace.npz"
    output_npz.write_bytes(b"stale-primary")
    fallback = parent.with_name(parent.name + ".failure.json")

    rc = main(
        [
            "--level0-trace",
            str(parent),
            "--expected-level0-trace-sha256",
            hashlib.sha256(original_parent).hexdigest(),
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--decoder-silu-lut",
            str(silu_lut),
            "--expected-decoder-silu-lut-sha256",
            hashlib.sha256(silu_lut.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(parent),
        ]
    )

    assert rc == 1
    assert parent.read_bytes() == original_parent
    failed = json.loads(fallback.read_text())
    assert failed["status"] == "failed"
    assert failed["failure_phase"] == "request_validation"
    assert failed["requested_report_path"] == str(parent)
    assert failed["effective_report_path"] == str(fallback)
    assert failed["stale_primary_invalidated"] is True
    assert not output_npz.exists()


def test_local_level1_trace_primary_report_alias_invalidates_stale_primary(
    tmp_path,
):
    from scripts.run_mlx_decoder_level1_trace import main

    parent = tmp_path / "level0.npz"
    np.savez(
        parent,
        coords=np.array([[0, 1, 2, 3]], dtype=np.int32),
        block3_output=np.ones((1, 1024), dtype=np.float16),
    )
    original_parent = parent.read_bytes()
    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    silu_lut = tmp_path / "silu.npz"
    silu_lut.write_bytes(b"silu")
    output_npz = tmp_path / "trace.npz"
    output_npz.write_bytes(b"stale-primary")
    fallback = output_npz.with_name(output_npz.name + ".failure.json")

    rc = main(
        [
            "--level0-trace",
            str(parent),
            "--expected-level0-trace-sha256",
            hashlib.sha256(original_parent).hexdigest(),
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--decoder-silu-lut",
            str(silu_lut),
            "--expected-decoder-silu-lut-sha256",
            hashlib.sha256(silu_lut.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_npz),
        ]
    )

    assert rc == 1
    assert parent.read_bytes() == original_parent
    assert not output_npz.exists()
    failed = json.loads(fallback.read_text())
    assert failed["status"] == "failed"
    assert failed["failure_phase"] == "request_validation"
    assert failed["requested_report_path"] == str(output_npz)
    assert failed["effective_report_path"] == str(fallback)
    assert failed["stale_primary_invalidated"] is True

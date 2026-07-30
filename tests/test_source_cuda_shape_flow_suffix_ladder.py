import hashlib
import json

import numpy as np
import pytest


MODULATION_NPZ_SHA256 = "c" * 64
MODULATION_REPORT_SHA256 = "d" * 64
MODULATION_SOURCE_CHECKPOINT_SHA256 = "e" * 64
PROJECTION_BATCH_MODE = "independent-singletons"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _modulation_identity():
    return {
        "schema": "trellis2mlx.source_cuda_timestep_modulation_lut.v1",
        "route_identity_evidence": True,
        "route": "source-cuda-t4-canonical-shared-adaln-lut",
        "npz_path": "/authenticated/cuda_result.npz",
        "npz_sha256_effective": MODULATION_NPZ_SHA256,
        "report_path": "/authenticated/cuda_result.json",
        "report_sha256_effective": MODULATION_REPORT_SHA256,
        "source_checkpoint_sha256_effective": (
            MODULATION_SOURCE_CHECKPOINT_SHA256
        ),
        "projection_batch_mode": PROJECTION_BATCH_MODE,
        "step_indices": list(range(8)),
        "timestep_float32_bits": [
            "0x447a0000",
            "0x446ea2e9",
            "0x44610000",
            "0x44505555",
            "0x443b8000",
            "0x4420b6db",
            "0x43fa0000",
            "0x43960000",
        ],
        "modulation_shape": [8, 9216],
    }


def _expected_modulation_identity():
    return {
        "npz_sha256_effective": MODULATION_NPZ_SHA256,
        "report_sha256_effective": MODULATION_REPORT_SHA256,
        "source_checkpoint_sha256_effective": (
            MODULATION_SOURCE_CHECKPOINT_SHA256
        ),
    }


def _write_capture(
    tmp_path,
    *,
    recurrence_delta=0.0,
    backend="mlx-metal",
    checkpoint_modulation_identity=None,
    modulation_identity=None,
):
    steps = 8
    if modulation_identity is None:
        modulation_identity = _modulation_identity()
    if checkpoint_modulation_identity is None:
        checkpoint_modulation_identity = modulation_identity
    sample_in = np.zeros((steps, 2, 3), dtype=np.float32)
    pred_final = np.ones_like(sample_in)
    schedule = np.linspace(1, 0, steps + 1)
    schedule = 3.0 * schedule / (1 + 2.0 * schedule)
    t = np.asarray(schedule[:-1], dtype=np.float32)
    t_prev = np.asarray(schedule[1:], dtype=np.float32)
    sample_next = sample_in - (t - t_prev)[:, None, None] * pred_final
    for index in range(1, steps):
        sample_in[index] = sample_next[index - 1]
        sample_next[index] = sample_in[index] - (t[index] - t_prev[index]) * pred_final[index]
    sample_in[3, 0, 0] += np.float32(recurrence_delta)
    capture = tmp_path / "shape_flow_steps.npz"
    np.savez(
        capture,
        noise=sample_in[0],
        sample_feats=sample_in[0],
        coords=np.asarray([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=np.int32),
        coords_3d=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int32),
        sample_in=sample_in,
        pred_final=pred_final,
        pred_v_feats=pred_final,
        sample_next=sample_next,
        t=t,
        t_prev=t_prev,
        steps=np.asarray(steps, dtype=np.int32),
        guidance_strength=np.asarray(7.5, dtype=np.float32),
        guidance_rescale=np.asarray(0.5, dtype=np.float32),
        guidance_interval=np.asarray([0.6, 1.0], dtype=np.float32),
        rescale_t=np.asarray(3.0, dtype=np.float32),
        sigma_min=np.asarray(1e-5, dtype=np.float32),
        shape_flow_block_injection_json=np.asarray(""),
        shape_timestep_modulation_lut_json=np.asarray(
            json.dumps(checkpoint_modulation_identity, sort_keys=True)
        ),
    )
    conditioning = tmp_path / "conditioning.npz"
    np.savez(
        conditioning,
        cond=np.zeros((1, 2, 4), dtype=np.float32),
        neg_cond=np.zeros((1, 2, 4), dtype=np.float32),
    )
    report = tmp_path / "mlx-run-report.json"
    report.write_text(
        json.dumps(
            {
                "status": "done",
                "last_trustworthy_phase": "shape_flow_steps_validated",
                "primary_output_status": "written",
                "primary_output_validation": {
                    "sha256": _sha256(capture),
                    "step_count": 8,
                    "token_count": 2,
                    "channel_count": 3,
                    "recurrence_exact": recurrence_delta == 0.0,
                    "sampler": {
                        "shape_timestep_modulation_route": (
                            modulation_identity
                        ),
                    },
                },
                "route_identity": {
                    "requested_stop": "shape_flow_steps",
                    "route": {
                        "family": "trellis2mlx/mlx",
                        "backend": backend,
                        "attention_backend": "fast",
                        "steps": 8,
                        "cascade": False,
                        "conditioning_sample_sha256": _sha256(conditioning),
                        "shape_flow_noise_sample_sha256": "a" * 64,
                        "shape_slat_support_sample_sha256": "b" * 64,
                        "shape_flow_block_injection_trace_path": None,
                        "shape_flow_block_injection_manifest_path": None,
                        "shape_timestep_modulation_lut_sha256_effective": (
                            MODULATION_NPZ_SHA256
                        ),
                        "shape_timestep_modulation_report_sha256_effective": (
                            MODULATION_REPORT_SHA256
                        ),
                        "shape_timestep_modulation_source_checkpoint_sha256": (
                            MODULATION_SOURCE_CHECKPOINT_SHA256
                        ),
                        "shape_timestep_modulation_identity": (
                            modulation_identity
                        ),
                    },
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return capture, report, conditioning


def test_suffix_step_indices_cover_uncapped_switch_ladder():
    from scripts.source_cuda_shape_flow_suffix_ladder import suffix_step_indices

    assert suffix_step_indices(0, steps=8) == list(range(8))
    assert suffix_step_indices(7, steps=8) == [7]
    assert suffix_step_indices(8, steps=8) == []
    with pytest.raises(ValueError, match="switch step"):
        suffix_step_indices(9, steps=8)


def test_load_mlx_trajectory_binds_admitted_route_and_exact_recurrence(tmp_path):
    from scripts.source_cuda_shape_flow_suffix_ladder import load_mlx_trajectory

    capture, report, conditioning = _write_capture(tmp_path)
    arrays, identity = load_mlx_trajectory(
        capture,
        report,
        conditioning,
        expected_modulation_identity=_expected_modulation_identity(),
    )

    assert arrays["sample_in"].shape == (8, 2, 3)
    assert np.array_equal(arrays["sample_in"][0], arrays["noise"])
    assert identity["backend"] == "mlx-metal"
    assert identity["capture_sha256"] == _sha256(capture)
    assert (
        identity["shape_timestep_modulation_identity"]
        == _modulation_identity()
    )


def test_load_mlx_trajectory_rejects_false_closure_routes_and_recurrence(tmp_path):
    from scripts.source_cuda_shape_flow_suffix_ladder import load_mlx_trajectory

    capture, report, conditioning = _write_capture(tmp_path, backend="cpu")
    with pytest.raises(ValueError, match="backend"):
        load_mlx_trajectory(
            capture,
            report,
            conditioning,
            expected_modulation_identity=_expected_modulation_identity(),
        )

    capture, report, conditioning = _write_capture(tmp_path, recurrence_delta=0.25)
    with pytest.raises(ValueError, match="recurrence"):
        load_mlx_trajectory(
            capture,
            report,
            conditioning,
            expected_modulation_identity=_expected_modulation_identity(),
        )


def test_load_mlx_trajectory_rejects_substituted_modulation_checkpoint_identity(
    tmp_path,
):
    from scripts.source_cuda_shape_flow_suffix_ladder import load_mlx_trajectory

    substituted = _modulation_identity()
    substituted["npz_sha256_effective"] = "f" * 64
    capture, report, conditioning = _write_capture(
        tmp_path,
        checkpoint_modulation_identity=substituted,
    )

    with pytest.raises(
        ValueError,
        match="checkpoint.*timestep modulation identity",
    ):
        load_mlx_trajectory(
            capture,
            report,
            conditioning,
            expected_modulation_identity=_expected_modulation_identity(),
        )


@pytest.mark.parametrize("mode", [None, "batched-eight"])
def test_load_mlx_trajectory_rejects_non_singleton_modulation_identity(
    tmp_path,
    mode,
):
    from scripts.source_cuda_shape_flow_suffix_ladder import load_mlx_trajectory

    identity = _modulation_identity()
    if mode is None:
        identity.pop("projection_batch_mode")
    else:
        identity["projection_batch_mode"] = mode
    capture, report, conditioning = _write_capture(
        tmp_path,
        modulation_identity=identity,
    )

    with pytest.raises(
        ValueError, match="projection_batch_mode.*canonical"
    ):
        load_mlx_trajectory(
            capture,
            report,
            conditioning,
            expected_modulation_identity=_expected_modulation_identity(),
        )


def test_anchor_classification_is_explicit_at_both_anchors_and_ties():
    from scripts.source_cuda_shape_flow_suffix_ladder import classify_anchor

    assert classify_anchor(0.0, 3.0) == "source"
    assert classify_anchor(2.0, 0.0) == "mlx"
    assert classify_anchor(1.0, 1.0) == "equidistant"
    with pytest.raises(ValueError, match="finite"):
        classify_anchor(float("nan"), 1.0)


def test_validate_result_requires_all_switches_and_exact_boundaries():
    from scripts.source_cuda_shape_flow_suffix_ladder import validate_result_manifest

    points = [
        {
            "switch_step": step,
            "source_step_indices": list(range(step, 8)),
            "source_step_count": 8 - step,
            "step_elapsed_seconds": [1.0] * (8 - step),
            "output_key": f"switch_{step}_shape_slat",
            "vs_source_anchor": {"exact": step == 0, "max_abs": 0.0 if step == 0 else 1.0, "nonzero": 0 if step == 0 else 1},
            "vs_mlx_anchor": {"exact": step == 8, "max_abs": 0.0 if step == 8 else 1.0, "nonzero": 0 if step == 8 else 1},
        }
        for step in range(9)
    ]
    payload = {
        "status": "done",
        "effective_route": {
            "device_type": "cuda",
            "attention_backend": "sdpa",
            "conv_backend": "none",
            "steps": 8,
            "one_model_load": True,
            "switch_steps": list(range(9)),
        },
        "points": points,
        "timing": {
            "source_steps_completed": 36,
            "source_steps_requested": 36,
            "switch_points_completed": 9,
            "switch_points_requested": 9,
        },
    }

    validate_result_manifest(payload)
    payload["effective_route"]["conv_backend"] = "spconv"
    with pytest.raises(ValueError, match="conv_backend"):
        validate_result_manifest(payload)
    payload["effective_route"]["conv_backend"] = "none"
    payload["points"] = points[:-1]
    with pytest.raises(ValueError, match="switch points"):
        validate_result_manifest(payload)


def test_validate_result_requires_canonical_distinct_switch_output_keys():
    from scripts.source_cuda_shape_flow_suffix_ladder import validate_result_manifest

    points = [
        {
            "switch_step": step,
            "source_step_indices": list(range(step, 8)),
            "source_step_count": 8 - step,
            "step_elapsed_seconds": [1.0] * (8 - step),
            "output_key": f"switch_{step}_shape_slat",
            "vs_source_anchor": {
                "exact": step == 0,
                "max_abs": 0.0 if step == 0 else 1.0,
                "nonzero": 0 if step == 0 else 1,
            },
            "vs_mlx_anchor": {
                "exact": step == 8,
                "max_abs": 0.0 if step == 8 else 1.0,
                "nonzero": 0 if step == 8 else 1,
            },
        }
        for step in range(9)
    ]
    payload = {
        "status": "done",
        "effective_route": {
            "device_type": "cuda",
            "attention_backend": "sdpa",
            "conv_backend": "none",
            "steps": 8,
            "one_model_load": True,
            "switch_steps": list(range(9)),
        },
        "points": points,
        "timing": {
            "source_steps_completed": 36,
            "source_steps_requested": 36,
            "switch_points_completed": 9,
            "switch_points_requested": 9,
        },
    }
    points[4]["output_key"] = points[5]["output_key"]

    with pytest.raises(ValueError, match="canonical output key"):
        validate_result_manifest(payload)


def test_cli_missing_inputs_remove_stale_primary_and_write_durable_report(tmp_path):
    from scripts.source_cuda_shape_flow_suffix_ladder import main

    report = tmp_path / "result.json"
    output = tmp_path / "result.npz"
    output.write_bytes(b"stale")
    status = main(
        [
            "--output-json", str(report),
            "--output-npz", str(output),
            "--mlx-shape-flow-steps", str(tmp_path / "missing-steps.npz"),
            "--mlx-run-report", str(tmp_path / "missing-run.json"),
            "--mlx-timestep-modulation-route", "default",
            "--conditioning", str(tmp_path / "missing-conditioning.npz"),
            "--accepted-source-baseline", str(tmp_path / "missing-baseline.npz"),
            "--accepted-source-report", str(tmp_path / "missing-source.json"),
            "--source-tar", str(tmp_path / "missing-source.tar"),
            "--no-download",
        ]
    )

    assert status == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["failure_phase"] == "input_validation"
    assert payload["last_trustworthy_phase"] == "request_validation"
    assert payload["primary_output_status"] == "missing"
    assert not output.exists()


def test_cli_refuses_to_delete_an_input_collision(tmp_path):
    from scripts.source_cuda_shape_flow_suffix_ladder import main

    capture, report_input, conditioning = _write_capture(tmp_path)
    report = tmp_path / "result.json"
    status = main(
        [
            "--output-json", str(report),
            "--output-npz", str(capture),
            "--mlx-shape-flow-steps", str(capture),
            "--mlx-run-report", str(report_input),
            "--mlx-timestep-modulation-route", "default",
            "--conditioning", str(conditioning),
            "--accepted-source-baseline", str(tmp_path / "baseline.npz"),
            "--accepted-source-report", str(tmp_path / "source.json"),
            "--source-tar", str(tmp_path / "source.tar"),
        ]
    )

    assert status == 1
    assert capture.exists()
    payload = json.loads(report.read_text())
    assert payload["failure_phase"] == "request_validation"
    assert payload["primary_output_status"] == "not_owned_due_to_path_collision"
    assert "collides with MLX shape-flow steps" in payload["error"]


def test_cli_partial_npz_write_is_removed_and_reported_invalid(tmp_path, monkeypatch):
    from scripts import source_cuda_shape_flow_suffix_ladder as ladder

    capture, report_input, conditioning = _write_capture(tmp_path)
    baseline = tmp_path / "baseline.npz"
    source_report = tmp_path / "source.json"
    source_tar = tmp_path / "source.tar"
    baseline.write_bytes(b"baseline")
    source_report.write_text("{}\n")
    source_tar.write_bytes(b"source")
    report = tmp_path / "result.json"
    output = tmp_path / "result.npz"

    monkeypatch.setattr(
        ladder,
        "_load_source_anchor",
        lambda *args, **kwargs: (
            np.zeros((2, 3), dtype=np.float32),
            {"baseline_sha256": "c" * 64},
        ),
    )

    def partial_save(path, **arrays):
        path.write_bytes(b"partial zip")
        raise OSError("disk full after partial write")

    monkeypatch.setattr(ladder.np, "savez", partial_save)
    monkeypatch.setattr(ladder, "_run_suffix", lambda **kwargs: None)

    # Enter the write phase directly by replacing runtime-heavy imports through a
    # computed-report helper; the exception contract is exercised independently.
    status = ladder._write_primary_artifact(
        output,
        report,
        {"coords": np.zeros((2, 4), dtype=np.int32)},
        {"status": "done"},
    )

    assert status == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["failure_phase"] == "write_outputs"
    assert payload["primary_output_status"] == "invalid_removed"
    assert "disk full" in payload["error"]
    assert not output.exists()


def test_saved_artifact_validation_binds_point_hashes_and_metadata(tmp_path):
    from scripts.source_cuda_shape_flow_suffix_ladder import validate_saved_artifact

    arrays = {"coords": np.zeros((2, 4), dtype=np.int32)}
    points = []
    for step in range(9):
        key = f"switch_{step}_shape_slat"
        value = np.full((2, 3), step, dtype=np.float32)
        arrays[key] = value
        points.append(
            {
                "switch_step": step,
                "output_key": key,
                "shape": [2, 3],
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        )
    metadata = {
        "schema": "trellis2mlx.source_cuda_shape_flow_suffix_ladder.artifact.v1",
        "artifact_status": "computed_pending_serialization",
        "external_report_required": True,
        "effective_route": {"device_type": "cuda"},
        "points": points,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output = tmp_path / "result.npz"
    np.savez(output, **arrays)

    validation = validate_saved_artifact(output, points=points)
    assert validation["switch_count"] == 9
    assert validation["metadata_schema"] == metadata["schema"]

    points[4]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="switch 4 digest"):
        validate_saved_artifact(output, points=points)

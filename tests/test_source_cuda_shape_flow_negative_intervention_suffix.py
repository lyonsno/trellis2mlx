import hashlib
import json

import numpy as np
import pytest


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_intervention(tmp_path, *, stage="norm1", branch="neg"):
    shape = (2, 3)
    coords = np.asarray([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    noise = np.arange(6, dtype=np.float32).reshape(shape)
    pred_pos = np.full(shape, 7.0, dtype=np.float32)
    pred_neg = np.full(shape, 8.0, dtype=np.float32)
    injection = {
        "trace_sha256": "a" * 64,
        "step_index": 0,
        "block_index": 0,
        "branch": branch,
        "stage": stage,
        "scale": 1.0,
    }
    checkpoint = tmp_path / f"{stage}.npz"
    np.savez(
        checkpoint,
        coords=coords,
        noise=noise,
        sample_feats=noise,
        pred_pos=pred_pos,
        pred_neg=pred_neg,
        t=np.asarray(1.0, dtype=np.float32),
        t_prev=np.asarray(0.95454544, dtype=np.float32),
        steps=np.asarray(8, dtype=np.int32),
        guidance_strength=np.asarray(7.5, dtype=np.float32),
        guidance_rescale=np.asarray(0.5, dtype=np.float32),
        guidance_interval=np.asarray([0.6, 1.0], dtype=np.float32),
        rescale_t=np.asarray(3.0, dtype=np.float32),
        shape_flow_block_injection_json=np.asarray(json.dumps(injection)),
    )
    checkpoint_sha = _sha256(checkpoint)
    report = {
        "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
        "status": "done",
        "exit_code": 0,
        "failure_phase": None,
        "last_trustworthy_phase": "shape_flow_step_saved",
        "primary_output_status": "written",
        "artifacts": {
            "shape_flow_step.npz": {
                "sha256": checkpoint_sha,
                "size_bytes": checkpoint.stat().st_size,
            }
        },
        "route_identity": {
            "env": {"TRELLIS2MLX_ATTENTION_BACKEND": "fast"},
            "requested_stop": "shape_flow_step",
            "route": {
                "family": "trellis2mlx/mlx",
                "backend": "mlx-metal",
                "attention_backend": "fast",
                "cascade": False,
                "steps": 8,
                "conditioning_sample_sha256": "b" * 64,
                "shape_slat_support_sample_sha256": "c" * 64,
                "shape_flow_noise_sample_sha256": "d" * 64,
                "shape_flow_block_injection_trace_sha256": "a" * 64,
                "shape_flow_block_injection_step_index": 0,
                "shape_flow_block_injection_block_index": 0,
                "shape_flow_block_injection_branch": branch,
                "shape_flow_block_injection_stage": stage,
                "shape_flow_block_injection_scale": 1.0,
            },
        },
    }
    report_path = tmp_path / f"{stage}.json"
    report_path.write_text(json.dumps(report))
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "report": report_path,
        "report_sha256": _sha256(report_path),
        "coords": coords,
        "noise": noise,
        "pred_pos": pred_pos,
        "pred_neg": pred_neg,
    }


def _load_kwargs(fixture, *, stage="norm1"):
    return {
        "checkpoint_path": fixture["checkpoint"],
        "report_path": fixture["report"],
        "expected_checkpoint_sha256": fixture["checkpoint_sha256"],
        "expected_report_sha256": fixture["report_sha256"],
        "expected_stage": stage,
        "expected_trace_sha256": "a" * 64,
        "expected_coords": fixture["coords"],
        "expected_noise": fixture["noise"],
        "expected_mlx_pred_pos": fixture["pred_pos"],
        "expected_conditioning_sha256": "b" * 64,
        "expected_support_sha256": "c" * 64,
        "expected_noise_sample_sha256": "d" * 64,
    }


def test_intervention_specs_are_complete_ordered_and_quotient_distinct():
    from scripts.source_cuda_shape_flow_negative_intervention_suffix import (
        intervention_candidate_specs,
    )

    specs = intervention_candidate_specs()
    assert [spec["name"] for spec in specs] == [
        "source-native-control",
        "source-pos-neg-block0-norm1",
        "source-pos-neg-block0-attention-raw",
        "source-pos-neg-block0-after-mlp",
    ]
    assert [spec.get("intervention_stage") for spec in specs] == [
        None,
        "norm1",
        "attention_raw",
        "after_mlp",
    ]


def test_load_negative_intervention_binds_route_and_unchanged_witness(tmp_path):
    from scripts.source_cuda_shape_flow_negative_intervention_suffix import (
        load_negative_intervention,
    )

    fixture = _write_intervention(tmp_path)
    pred_neg, identity = load_negative_intervention(**_load_kwargs(fixture))
    assert np.array_equal(pred_neg, fixture["pred_neg"])
    assert identity["stage"] == "norm1"
    assert identity["checkpoint_sha256"] == fixture["checkpoint_sha256"]

    report = json.loads(fixture["report"].read_text())
    report["route_identity"]["route"][
        "shape_flow_block_injection_branch"
    ] = "pos"
    fixture["report"].write_text(json.dumps(report))
    kwargs = _load_kwargs(fixture)
    kwargs["expected_report_sha256"] = _sha256(fixture["report"])
    with pytest.raises(ValueError, match="branch"):
        load_negative_intervention(**kwargs)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("stage", "stage"),
        ("checkpoint_digest", "checkpoint SHA256"),
        ("report_digest", "report SHA256"),
        ("artifact_digest", "artifact digest"),
        ("route_backend", "backend"),
        ("trace", "trace"),
    ],
)
def test_load_negative_intervention_rejects_false_custody(
    tmp_path, mutation, match
):
    from scripts.source_cuda_shape_flow_negative_intervention_suffix import (
        load_negative_intervention,
    )

    fixture = _write_intervention(tmp_path)
    kwargs = _load_kwargs(fixture)
    if mutation == "stage":
        kwargs["expected_stage"] = "after_mlp"
    elif mutation == "checkpoint_digest":
        kwargs["expected_checkpoint_sha256"] = "0" * 64
    elif mutation == "report_digest":
        kwargs["expected_report_sha256"] = "0" * 64
    else:
        report = json.loads(fixture["report"].read_text())
        route = report["route_identity"]["route"]
        if mutation == "artifact_digest":
            report["artifacts"]["shape_flow_step.npz"]["sha256"] = "0" * 64
        elif mutation == "route_backend":
            route["backend"] = "cpu"
        elif mutation == "trace":
            route["shape_flow_block_injection_trace_sha256"] = "0" * 64
        fixture["report"].write_text(json.dumps(report))
        kwargs["expected_report_sha256"] = _sha256(fixture["report"])
    with pytest.raises(ValueError, match=match):
        load_negative_intervention(**kwargs)


@pytest.mark.parametrize(
    ("array_name", "match"),
    [
        ("coords", "coordinates"),
        ("noise", "noise"),
        ("sample_feats", "sample"),
        ("pred_pos", "positive"),
        ("pred_neg", "finite"),
    ],
)
def test_load_negative_intervention_rejects_changed_checkpoint_arrays(
    tmp_path, array_name, match
):
    from scripts.source_cuda_shape_flow_negative_intervention_suffix import (
        load_negative_intervention,
    )

    fixture = _write_intervention(tmp_path)
    with np.load(fixture["checkpoint"], allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if array_name == "pred_neg":
        arrays[array_name] = arrays[array_name].copy()
        arrays[array_name][0, 0] = np.nan
    else:
        arrays[array_name] = arrays[array_name] + 1
    np.savez(fixture["checkpoint"], **arrays)
    fixture["checkpoint_sha256"] = _sha256(fixture["checkpoint"])
    report = json.loads(fixture["report"].read_text())
    artifact = report["artifacts"]["shape_flow_step.npz"]
    artifact["sha256"] = fixture["checkpoint_sha256"]
    artifact["size_bytes"] = fixture["checkpoint"].stat().st_size
    fixture["report"].write_text(json.dumps(report))
    fixture["report_sha256"] = _sha256(fixture["report"])
    with pytest.raises(ValueError, match=match):
        load_negative_intervention(**_load_kwargs(fixture))


def test_control_gate_prevents_intervention_execution():
    from scripts.source_cuda_shape_flow_negative_intervention_suffix import (
        intervention_candidate_specs,
        run_control_gated_interventions,
    )

    events = []

    def execute(index, spec):
        events.append((index, spec["name"]))
        return {
            "name": spec["name"],
            "vs_source_anchor": {"exact": False, "nonzero": 1},
        }

    with pytest.raises(ValueError, match="source-native control"):
        run_control_gated_interventions(
            specs=intervention_candidate_specs(),
            execute_candidate=execute,
        )
    assert events == [(0, "source-native-control")]


def _result_fixture():
    from scripts.source_cuda_shape_flow_negative_intervention_suffix import (
        intervention_candidate_specs,
    )

    candidates = []
    for index, spec in enumerate(intervention_candidate_specs()):
        candidates.append(
            {
                **spec,
                "output_key": f"candidate_{index}_shape_slat",
                "shape": [2, 3],
                "sha256": "0" * 64,
                "source_step_indices": list(range(1, 8)),
                "source_step_count": 7,
                "step_elapsed_seconds": [1.0] * 7,
                "vs_source_anchor": {
                    "exact": index == 0,
                    "mean_abs": 0.0 if index == 0 else 1.0,
                    "max_abs": 0.0 if index == 0 else 1.0,
                    "nonzero": 0 if index == 0 else 1,
                },
                "vs_mlx_anchor": {
                    "exact": False,
                    "mean_abs": 1.0,
                    "max_abs": 1.0,
                    "nonzero": 1,
                },
            }
        )
    return {
        "status": "done",
        "effective_route": {
            "device_type": "cuda",
            "attention_backend": "sdpa",
            "conv_backend": "none",
            "steps": 8,
            "one_model_load": True,
            "candidate_names": [candidate["name"] for candidate in candidates],
        },
        "candidate_specs": intervention_candidate_specs(),
        "candidates": candidates,
        "inputs": {},
        "anchors": {},
        "timing": {
            "source_steps_completed": 28,
            "source_steps_requested": 28,
            "candidates_completed": 4,
            "candidates_requested": 4,
        },
    }


def _artifact_arrays(report):
    from scripts.source_cuda_shape_flow_negative_intervention_suffix import (
        INTERVENTION_STAGES,
    )

    shape = (2, 3)
    arrays = {
        "coords": np.zeros((2, 4), dtype=np.int32),
        "source_anchor_shape_slat": np.ones(shape, dtype=np.float32),
        "mlx_anchor_shape_slat": np.full(shape, 2.0, dtype=np.float32),
        "source_transition0_pred_pos": np.full(shape, 3.0, dtype=np.float32),
        "source_transition0_pred_neg": np.full(shape, 4.0, dtype=np.float32),
    }
    for stage in INTERVENTION_STAGES:
        arrays[f"intervention_{stage}_pred_neg"] = np.full(
            shape, 5.0, dtype=np.float32
        )
    for index, candidate in enumerate(report["candidates"]):
        arrays[f"candidate_{index}_transition0_sample_next"] = np.full(
            shape, 6.0 + index, dtype=np.float32
        )
        output = np.full(shape, 10.0 + index, dtype=np.float32)
        arrays[candidate["output_key"]] = output
        candidate["sha256"] = hashlib.sha256(output.tobytes()).hexdigest()
    return arrays


def test_result_and_saved_artifact_bind_all_four_cuda_continuations(tmp_path):
    from scripts.source_cuda_shape_flow_negative_intervention_suffix import (
        _artifact_metadata,
        validate_result_manifest,
        validate_saved_artifact,
    )

    report = _result_fixture()
    arrays = _artifact_arrays(report)
    validate_result_manifest(report)
    arrays["metadata_json"] = np.asarray(
        json.dumps(_artifact_metadata(report, arrays), sort_keys=True)
    )
    output = tmp_path / "result.npz"
    np.savez(output, **arrays)

    validation = validate_saved_artifact(
        output, candidates=report["candidates"]
    )
    assert validation == {
        "schema": (
            "trellis2mlx.source_cuda_shape_flow_negative_intervention_suffix"
            ".v1.saved_artifact"
        ),
        "candidate_count": 4,
        "all_arrays_bound": True,
        "array_count": 16,
    }

    with np.load(output, allow_pickle=False) as archive:
        corrupted = {key: np.asarray(archive[key]) for key in archive.files}
    corrupted.pop("intervention_attention_raw_pred_neg")
    np.savez(output, **corrupted)
    with pytest.raises(ValueError, match="attention_raw"):
        validate_saved_artifact(output, candidates=report["candidates"])


def test_failed_artifact_validation_removes_partial_output_and_writes_report(
    tmp_path,
):
    from scripts.source_cuda_shape_flow_negative_intervention_suffix import (
        _artifact_metadata,
        _write_artifact,
    )

    report = _result_fixture()
    arrays = _artifact_arrays(report)
    arrays.pop("intervention_after_mlp_pred_neg")
    arrays["metadata_json"] = np.asarray(
        json.dumps(_artifact_metadata(report, arrays), sort_keys=True)
    )
    output = tmp_path / "result.npz"
    report_path = tmp_path / "result.json"

    assert _write_artifact(output, report_path, arrays, report) == 1
    assert not output.exists()
    saved = json.loads(report_path.read_text())
    assert saved["status"] == "failed"
    assert saved["failure_phase"] == "write_outputs"
    assert saved["primary_output_status"] == "invalid_removed"

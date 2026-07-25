import hashlib
import json

import numpy as np
import pytest


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_transition_capture(tmp_path, *, omit=None):
    shape = (8, 2, 3)
    sample_in = np.zeros(shape, dtype=np.float32)
    arrays = {
        "noise": sample_in[0].copy(),
        "sample_in": sample_in,
        "pred_pos": np.full(shape, 1.0, dtype=np.float32),
        "pred_neg": np.full(shape, 2.0, dtype=np.float32),
        "pred_cfg": np.full(shape, 3.0, dtype=np.float32),
        "x0_pos": np.full(shape, 4.0, dtype=np.float32),
        "x0_cfg": np.full(shape, 5.0, dtype=np.float32),
        "x0_rescaled": np.full(shape, 6.0, dtype=np.float32),
        "x0_after_rescale": np.full(shape, 7.0, dtype=np.float32),
        "pred_final": np.full(shape, 8.0, dtype=np.float32),
        "sample_next": np.full(shape, 9.0, dtype=np.float32),
        "t": np.linspace(1.0, 0.2, 8, dtype=np.float32),
        "t_prev": np.linspace(0.9, 0.1, 8, dtype=np.float32),
        "std_pos": np.ones(8, dtype=np.float32),
        "std_cfg": np.ones(8, dtype=np.float32),
        "ratio_raw": np.ones(8, dtype=np.float32),
        "std_ratio": np.ones(8, dtype=np.float32),
        "ratio_effective": np.ones(8, dtype=np.float32),
    }
    if omit:
        arrays.pop(omit)
    path = tmp_path / "shape_flow_steps.npz"
    np.savez(path, **arrays)
    return path


def _matrix_artifact_fixture():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        MLX_COMPONENT_NAMES,
        transition0_candidate_specs,
    )

    shape = (2, 3)
    arrays = {
        "coords": np.zeros((shape[0], 4), dtype=np.int32),
        "source_anchor_shape_slat": np.full(shape, 1.0, dtype=np.float32),
        "mlx_anchor_shape_slat": np.full(shape, 2.0, dtype=np.float32),
        "accepted_switch_1_shape_slat": np.full(shape, 3.0, dtype=np.float32),
        "source_transition0_pred_pos": np.full(shape, 4.0, dtype=np.float32),
        "source_transition0_pred_neg": np.full(shape, 5.0, dtype=np.float32),
    }
    for index, name in enumerate(MLX_COMPONENT_NAMES):
        arrays[f"mlx_transition0_{name}"] = np.full(
            shape, 10.0 + index, dtype=np.float32
        )
    candidates = []
    for index, spec in enumerate(transition0_candidate_specs()):
        start_key = f"candidate_{index}_transition0_sample_next"
        output_key = f"candidate_{index}_shape_slat"
        arrays[start_key] = np.full(shape, 30.0 + index, dtype=np.float32)
        value = np.full(shape, 40.0 + index, dtype=np.float32)
        arrays[output_key] = value
        candidates.append(
            {
                **spec,
                "output_key": output_key,
                "shape": list(shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        )
    return arrays, candidates


def test_transition0_candidate_specs_are_complete_and_nonredundant():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        transition0_candidate_specs,
    )

    specs = transition0_candidate_specs()
    assert [spec["name"] for spec in specs] == [
        "source-native-control",
        "mlx-pos-source-neg",
        "source-pos-mlx-neg",
        "mlx-both-source-post",
        "mlx-final-source-euler",
    ]
    assert len({json.dumps(spec, sort_keys=True) for spec in specs}) == len(specs)
    assert specs[0] == {
        "name": "source-native-control",
        "positive": "source",
        "negative": "source",
        "post": "source-guidance-rescale-euler",
    }
    assert specs[-1]["post"] == "mlx-final-source-euler"


def test_load_transition0_components_requires_complete_float32_intermediates(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        load_mlx_transition0_components,
    )

    capture = _write_transition_capture(tmp_path)
    components = load_mlx_transition0_components(
        capture,
        expected_sha256=_sha256(capture),
        expected_shape=(2, 3),
    )
    assert set(components) >= {
        "noise",
        "pred_pos",
        "pred_neg",
        "pred_cfg",
        "x0_pos",
        "x0_cfg",
        "x0_rescaled",
        "x0_after_rescale",
        "pred_final",
        "sample_next",
    }
    assert components["pred_pos"].shape == (2, 3)

    incomplete = _write_transition_capture(tmp_path, omit="pred_neg")
    with pytest.raises(ValueError, match="pred_neg"):
        load_mlx_transition0_components(
            incomplete,
            expected_sha256=_sha256(incomplete),
            expected_shape=(2, 3),
        )


def test_compose_candidate_pairs_maps_only_the_declared_prediction_branch():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        compose_candidate_pairs,
    )

    source_pos = object()
    source_neg = object()
    mlx_pos = object()
    mlx_neg = object()
    pairs = compose_candidate_pairs(
        source_pos=source_pos,
        source_neg=source_neg,
        mlx_pos=mlx_pos,
        mlx_neg=mlx_neg,
    )

    assert pairs["source-native-control"] == (source_pos, source_neg)
    assert pairs["mlx-pos-source-neg"] == (mlx_pos, source_neg)
    assert pairs["source-pos-mlx-neg"] == (source_pos, mlx_neg)
    assert pairs["mlx-both-source-post"] == (mlx_pos, mlx_neg)
    assert "mlx-final-source-euler" not in pairs


def test_validate_result_requires_cuda_route_all_candidates_and_exact_control():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        transition0_candidate_specs,
        validate_result_manifest,
    )

    candidates = []
    for index, spec in enumerate(transition0_candidate_specs()):
        candidates.append(
            {
                "name": spec["name"],
                "output_key": f"candidate_{index}_shape_slat",
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
    payload = {
        "status": "done",
        "effective_route": {
            "device_type": "cuda",
            "cuda_device": "Tesla T4",
            "attention_backend": "sdpa",
            "conv_backend": "none",
            "steps": 8,
            "one_model_load": True,
            "candidate_names": [candidate["name"] for candidate in candidates],
        },
        "candidates": candidates,
        "timing": {
            "source_steps_completed": 35,
            "source_steps_requested": 35,
            "candidates_completed": 5,
            "candidates_requested": 5,
        },
    }

    validate_result_manifest(payload)
    payload["candidates"][0]["vs_source_anchor"]["exact"] = False
    with pytest.raises(ValueError, match="source-native control"):
        validate_result_manifest(payload)


def test_saved_artifact_binds_every_candidate_array_and_metadata(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        build_artifact_metadata,
        validate_saved_artifact,
    )

    arrays, candidates = _matrix_artifact_fixture()
    report = {
        "effective_route": {"device_type": "cuda"},
        "inputs": {"expected_digests": {}},
        "candidate_specs": [
            {key: candidate[key] for key in ("name", "positive", "negative", "post")}
            for candidate in candidates
        ],
        "candidates": candidates,
        "anchors": {},
    }
    metadata = build_artifact_metadata(report, arrays)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output = tmp_path / "matrix.npz"
    np.savez(output, **arrays)

    validation = validate_saved_artifact(output, candidates=candidates)
    assert validation["candidate_count"] == 5
    assert validation["all_matrix_arrays_bound"] is True

    with np.load(output, allow_pickle=False) as archive:
        corrupted = {key: np.asarray(archive[key]) for key in archive.files}
    corrupted.pop("mlx_transition0_pred_neg")
    np.savez(output, **corrupted)
    with pytest.raises(ValueError, match="mlx_transition0_pred_neg"):
        validate_saved_artifact(output, candidates=candidates)

    arrays, candidates = _matrix_artifact_fixture()
    metadata = build_artifact_metadata(report, arrays)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    arrays["candidate_3_transition0_sample_next"] = np.full(
        (2, 3), -999.0, dtype=np.float32
    )
    np.savez(output, **arrays)
    with pytest.raises(ValueError, match="candidate_3_transition0_sample_next digest"):
        validate_saved_artifact(output, candidates=candidates)


def test_accepted_suffix_is_cross_bound_to_current_admitted_inputs(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        _load_accepted_suffix,
    )

    coords = np.zeros((2, 4), dtype=np.int32)
    source_anchor = np.ones((2, 3), dtype=np.float32)
    result_path = tmp_path / "suffix.npz"
    np.savez(
        result_path,
        coords=coords,
        accepted_source_anchor_shape_slat=source_anchor,
        mlx_anchor_shape_slat=np.full((2, 3), 2.0, dtype=np.float32),
        switch_0_shape_slat=source_anchor,
        switch_1_shape_slat=np.full((2, 3), 3.0, dtype=np.float32),
    )
    mlx_identity = {
        "capture_sha256": "1" * 64,
        "run_report_sha256": "2" * 64,
        "conditioning_sha256": "5" * 64,
        "shape_flow_noise_sample_sha256": "8" * 64,
        "shape_slat_support_sample_sha256": "9" * 64,
    }
    source_identity = {
        "baseline_sha256": "3" * 64,
        "report_sha256": "4" * 64,
    }
    report = {
        "status": "done",
        "effective_route": {
            "device_type": "cuda",
            "attention_backend": "sdpa",
            "conv_backend": "none",
            "steps": 8,
            "one_model_load": True,
        },
        "inputs": {
            "mlx": dict(mlx_identity),
            "conditioning_sha256": "5" * 64,
            "source_tar_sha256": "6" * 64,
            "accepted_source": dict(source_identity),
        },
        "points": [{"switch_step": index} for index in range(9)],
    }
    report_path = tmp_path / "suffix.json"
    report_path.write_text(json.dumps(report))
    kwargs = {
        "expected_result_sha256": _sha256(result_path),
        "expected_report_sha256": _sha256(report_path),
        "source_anchor": source_anchor,
        "coords": coords,
        "expected_mlx_identity": mlx_identity,
        "expected_source_identity": source_identity,
        "expected_conditioning_sha256": "5" * 64,
        "expected_source_tar_sha256": "6" * 64,
    }

    _load_accepted_suffix(result_path, report_path, **kwargs)
    report["inputs"]["mlx"]["capture_sha256"] = "7" * 64
    report_path.write_text(json.dumps(report))
    kwargs["expected_report_sha256"] = _sha256(report_path)
    with pytest.raises(ValueError, match="MLX capture"):
        _load_accepted_suffix(result_path, report_path, **kwargs)


def test_source_control_guard_rejects_before_later_candidates():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        require_exact_source_control,
    )

    require_exact_source_control(
        candidate_index=0, metrics={"exact": True, "nonzero": 0}
    )
    with pytest.raises(ValueError, match="source-native control"):
        require_exact_source_control(
            candidate_index=0, metrics={"exact": False, "nonzero": 1}
        )
    require_exact_source_control(
        candidate_index=1, metrics={"exact": False, "nonzero": 1}
    )


def test_control_gated_coordinator_defers_all_intervention_work():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        run_control_gated_candidates,
        transition0_candidate_specs,
    )

    specs = transition0_candidate_specs()
    events = []

    def build_starts(names):
        events.append(("build", tuple(names)))
        return (
            {name: f"start:{name}" for name in names},
            {name: f"first:{name}" for name in names},
        )

    def execute_mismatch(index, spec, start, first_step):
        events.append(("execute", index, spec["name"], start, first_step))
        return {
            "name": spec["name"],
            "vs_source_anchor": {"exact": False, "nonzero": 1},
        }

    with pytest.raises(ValueError, match="source-native control"):
        run_control_gated_candidates(
            specs=specs,
            build_starts=build_starts,
            execute_candidate=execute_mismatch,
        )
    assert events == [
        ("build", ("source-native-control",)),
        (
            "execute",
            0,
            "source-native-control",
            "start:source-native-control",
            "first:source-native-control",
        ),
    ]

    events.clear()

    def execute_exact(index, spec, start, first_step):
        events.append(("execute", index, spec["name"], start, first_step))
        return {
            "name": spec["name"],
            "vs_source_anchor": {
                "exact": index == 0,
                "nonzero": 0 if index == 0 else 1,
            },
        }

    results = run_control_gated_candidates(
        specs=specs,
        build_starts=build_starts,
        execute_candidate=execute_exact,
    )
    assert [result["name"] for result in results] == [
        spec["name"] for spec in specs
    ]
    assert events[0] == ("build", ("source-native-control",))
    assert events[1][0:3] == ("execute", 0, "source-native-control")
    assert events[2] == ("build", tuple(spec["name"] for spec in specs[1:]))
    assert [event[2] for event in events[3:]] == [
        spec["name"] for spec in specs[1:]
    ]


def _matrix_cli_args(tmp_path, *, expected_overrides=None):
    expected_overrides = expected_overrides or {}
    paths = {
        "mlx_steps": tmp_path / "steps.npz",
        "mlx_report": tmp_path / "mlx-report.json",
        "conditioning": tmp_path / "conditioning.npz",
        "source_baseline": tmp_path / "source-baseline.npz",
        "source_report": tmp_path / "source-report.json",
        "suffix_result": tmp_path / "suffix-result.npz",
        "suffix_report": tmp_path / "suffix-report.json",
        "source_tar": tmp_path / "source.tar",
    }
    for path in paths.values():
        path.write_bytes(b"input:" + path.name.encode())
    expected = {name: _sha256(path) for name, path in paths.items()}
    expected.update(expected_overrides)
    args = [
        "--mlx-shape-flow-steps", str(paths["mlx_steps"]),
        "--mlx-shape-flow-steps-sha256", expected["mlx_steps"],
        "--mlx-run-report", str(paths["mlx_report"]),
        "--mlx-run-report-sha256", expected["mlx_report"],
        "--conditioning", str(paths["conditioning"]),
        "--conditioning-sha256", expected["conditioning"],
        "--accepted-source-baseline", str(paths["source_baseline"]),
        "--accepted-source-baseline-sha256", expected["source_baseline"],
        "--accepted-source-report", str(paths["source_report"]),
        "--accepted-source-report-sha256", expected["source_report"],
        "--accepted-suffix-result", str(paths["suffix_result"]),
        "--accepted-suffix-result-sha256", expected["suffix_result"],
        "--accepted-suffix-report", str(paths["suffix_report"]),
        "--accepted-suffix-report-sha256", expected["suffix_report"],
        "--source-tar", str(paths["source_tar"]),
        "--source-tar-sha256", expected["source_tar"],
        "--no-download",
    ]
    return args, paths


def test_cli_missing_inputs_preserves_stale_primary_and_writes_failure_report(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import main

    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")
    status = main(
        [
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
            "--mlx-shape-flow-steps",
            str(tmp_path / "missing-steps.npz"),
            "--mlx-shape-flow-steps-sha256",
            "0" * 64,
            "--mlx-run-report",
            str(tmp_path / "missing-run.json"),
            "--mlx-run-report-sha256",
            "0" * 64,
            "--conditioning",
            str(tmp_path / "missing-conditioning.npz"),
            "--conditioning-sha256",
            "0" * 64,
            "--accepted-source-baseline",
            str(tmp_path / "missing-baseline.npz"),
            "--accepted-source-baseline-sha256",
            "0" * 64,
            "--accepted-source-report",
            str(tmp_path / "missing-source.json"),
            "--accepted-source-report-sha256",
            "0" * 64,
            "--accepted-suffix-result",
            str(tmp_path / "missing-suffix.npz"),
            "--accepted-suffix-result-sha256",
            "0" * 64,
            "--accepted-suffix-report",
            str(tmp_path / "missing-suffix.json"),
            "--accepted-suffix-report-sha256",
            "0" * 64,
            "--source-tar",
            str(tmp_path / "missing-source.tar"),
            "--source-tar-sha256",
            "0" * 64,
            "--no-download",
        ]
    )

    assert status == 1
    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "arguments_parsed"
    assert report["primary_output_status"] == "preexisting_untrusted_preserved"
    assert output_npz.read_bytes() == b"stale"


def test_cli_rejects_substituted_input_digest_before_stale_output_mutation(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import main

    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")
    args, _ = _matrix_cli_args(
        tmp_path,
        expected_overrides={"suffix_result": "0" * 64},
    )
    status = main(
        [
            "--output-json", str(output_json),
            "--output-npz", str(output_npz),
            *args,
        ]
    )

    assert status == 1
    report = json.loads(output_json.read_text())
    assert report["failure_phase"] == "request_validation"
    assert "accepted suffix result SHA256 mismatch" in report["error"]
    assert output_npz.read_bytes() == b"stale"


def test_cli_report_collision_preserves_input_and_writes_safe_failure_report(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import main

    output_npz = tmp_path / "result.npz"
    args, paths = _matrix_cli_args(tmp_path)
    source_report_bytes = paths["source_report"].read_bytes()
    status = main(
        [
            "--output-json", str(paths["source_report"]),
            "--output-npz", str(output_npz),
            *args,
        ]
    )

    fallback = paths["source_report"].with_name(
        paths["source_report"].name
        + ".transition0-recoverability.failure.json"
    )
    assert status == 1
    assert paths["source_report"].read_bytes() == source_report_bytes
    report = json.loads(fallback.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["requested_output_json"] == str(paths["source_report"])
    assert report["effective_failure_report"] == str(fallback)
    assert "collides" in report["error"]

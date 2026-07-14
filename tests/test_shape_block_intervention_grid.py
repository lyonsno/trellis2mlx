import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_trace(path: Path, *, alpha: float, beta: float, offset: float = 0.0) -> Path:
    base = np.arange(12, dtype=np.float32).reshape(1, 3, 4) + offset
    injection = {
        "route_identity_evidence": True,
        "manifest_identity": {
            "schema": "trellis2mlx.shape_block_injection_manifest.v1",
            "comparison_class": "block29_after_self_cross_attention_raw_delta_grid",
            "grid_coordinate": {"alpha": alpha, "beta": beta},
        },
        "manifest_sha256": "0" * 64,
        "sites": [
            {
                "block_index": 28,
                "step_index": 0,
                "stage": "after_mlp",
                "branch": "both",
                "source_delta_scale": 1.0,
            },
            {
                "block_index": 29,
                "step_index": 0,
                "stage": "after_self",
                "branch": "both",
                "source_delta_scale": alpha,
            },
            {
                "block_index": 29,
                "step_index": 0,
                "stage": "cross_attention_raw",
                "branch": "both",
                "source_delta_scale": beta,
            },
        ],
    }
    arrays = {"shape_flow_block_injection_json": np.asarray(json.dumps(injection))}
    for branch_index, branch in enumerate(("pos", "neg")):
        branch_base = base + branch_index
        for stage_index, stage in enumerate(
            (
                "block29_after_self",
                "block29_cross_attention_raw",
                "block29_cross_attn",
                "block29_after_cross",
                "block29_after_mlp",
                "final_output",
            )
        ):
            arrays[f"{branch}_{stage}"] = branch_base + stage_index
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def _write_run_evidence(output_dir: Path, manifest_path: Path, trace_path: Path) -> None:
    manifest_sha = _sha256(manifest_path)
    with np.load(trace_path, allow_pickle=False) as trace:
        injection = json.loads(str(np.asarray(trace["shape_flow_block_injection_json"]).item()))
        injection["manifest_sha256"] = manifest_sha
        arrays = {name: np.asarray(trace[name]) for name in trace.files}
        arrays["shape_flow_block_injection_json"] = np.asarray(json.dumps(injection))
    np.savez(trace_path, **arrays)
    trace_sha = _sha256(trace_path)
    route = {
        "family": "trellis2mlx/mlx",
        "backend": "mlx-metal",
        "attention_backend": "fast",
        "repo_root": "/worktree",
        "conditioning_sample_sha256": "1" * 64,
        "shape_flow_noise_sample_sha256": "2" * 64,
        "shape_slat_support_sample_sha256": "3" * 64,
        "shared_noise_sha256": "4" * 64,
        "shape_flow_block_injection_manifest_sha256": manifest_sha,
        "shape_flow_trace_block_index": 29,
        "shape_flow_trace_step_index": 0,
        "shape_flow_trace_key_selection": "explicit",
        "shape_flow_trace_keys": [
            "pos_block29_after_self",
            "pos_block29_cross_attention_raw",
            "pos_block29_cross_attn",
            "pos_block29_after_cross",
            "pos_block29_after_mlp",
            "pos_final_output",
            "neg_block29_after_self",
            "neg_block29_cross_attention_raw",
            "neg_block29_cross_attn",
            "neg_block29_after_cross",
            "neg_block29_after_mlp",
            "neg_final_output",
        ],
        "steps": 8,
    }
    (output_dir / "route_identity.json").write_text(
        json.dumps({"route": route}), encoding="utf-8"
    )
    (output_dir / "run_report.json").write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
                "status": "done",
                "exit_code": 0,
                "failure_phase": None,
                "primary_output_status": "written",
                "artifacts": {
                    "shape_flow_block_trace.npz": {
                        "path": str(trace_path),
                        "sha256": trace_sha,
                        "size_bytes": trace_path.stat().st_size,
                    }
                },
                "route_identity": {"route": route},
            }
        ),
        encoding="utf-8",
    )


def test_grid_plan_writes_full_cartesian_manifests_and_semantic_corners(tmp_path: Path) -> None:
    from scripts.build_shape_block_intervention_grid import build_grid_plan

    manifest_dir = tmp_path / "manifests"
    run_root = tmp_path / "runs"
    controls = {
        (0.0, 0.0): tmp_path / "natural.npz",
        (1.0, 0.0): tmp_path / "after-self.npz",
        (0.0, 1.0): tmp_path / "cross-raw.npz",
        (1.0, 1.0): tmp_path / "join.npz",
    }
    index = build_grid_plan(
        manifest_dir=manifest_dir,
        run_root=run_root,
        index_path=tmp_path / "index.json",
        prefix_trace=tmp_path / "prefix.npz",
        block29_trace=tmp_path / "block29.npz",
        alphas=(0.0, 0.5, 1.0),
        betas=(0.0, 0.5, 1.0),
        control_references=controls,
    )

    assert index["schema"] == "trellis2mlx.shape_block_intervention_grid_plan.v1"
    assert index["axes"] == {"alpha": [0.0, 0.5, 1.0], "beta": [0.0, 0.5, 1.0]}
    assert len(index["points"]) == 9
    assert [point["coordinate"] for point in index["points"][:3]] == [
        {"alpha": 0.0, "beta": 0.0},
        {"alpha": 0.0, "beta": 0.5},
        {"alpha": 0.0, "beta": 1.0},
    ]
    corner_roles = {
        (point["coordinate"]["alpha"], point["coordinate"]["beta"]): point.get("control_role")
        for point in index["points"]
        if point.get("control_role")
    }
    assert corner_roles == {
        (0.0, 0.0): "zero_correction_instrumentation_control",
        (1.0, 0.0): "after_self_exact_control",
        (0.0, 1.0): "cross_attention_raw_exact_control",
        (1.0, 1.0): "exact_join_control",
    }
    for point in index["points"]:
        manifest_path = Path(point["manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        assert point["manifest_sha256"] == _sha256(manifest_path)
        assert manifest["grid_coordinate"] == point["coordinate"]
        assert [site["stage"] for site in manifest["sites"]] == [
            "after_mlp",
            "after_self",
            "cross_attention_raw",
        ]
        assert manifest["sites"][1]["source_delta_scale"] == point["coordinate"]["alpha"]
        assert manifest["sites"][2]["source_delta_scale"] == point["coordinate"]["beta"]
        assert point["expected_trace_path"].endswith("/checkpoints/shape_flow_block_trace.npz")


def test_grid_plan_cli_rejects_duplicate_control_coordinates(tmp_path: Path) -> None:
    from scripts.build_shape_block_intervention_grid import main

    common = [
        "--manifest-dir",
        str(tmp_path / "manifests"),
        "--run-root",
        str(tmp_path / "runs"),
        "--index-json",
        str(tmp_path / "index.json"),
        "--prefix-trace",
        str(tmp_path / "prefix.npz"),
        "--block29-trace",
        str(tmp_path / "block29.npz"),
        "--alphas",
        "0,1",
        "--betas",
        "0,1",
    ]

    with pytest.raises(ValueError, match="duplicate control coordinate.*1.0.*1.0"):
        main(
            common
            + [
                "--control",
                f"1,1={tmp_path / 'join-a.npz'}",
                "--control",
                f"1,1={tmp_path / 'join-b.npz'}",
            ]
        )

    assert not (tmp_path / "index.json").exists()


@pytest.mark.parametrize("axis", [(0.0, 0.0), (0.0, float("nan")), (float("inf"),)])
def test_grid_plan_rejects_duplicate_or_nonfinite_axis_values(tmp_path: Path, axis) -> None:
    from scripts.build_shape_block_intervention_grid import build_grid_plan

    with pytest.raises(ValueError, match="duplicate|finite"):
        build_grid_plan(
            manifest_dir=tmp_path / "manifests",
            run_root=tmp_path / "runs",
            index_path=tmp_path / "index.json",
            prefix_trace=tmp_path / "prefix.npz",
            block29_trace=tmp_path / "block29.npz",
            alphas=axis,
            betas=(0.0, 1.0),
            control_references={},
        )


def test_grid_plan_is_idempotent_and_does_not_remove_unrelated_files(tmp_path: Path) -> None:
    from scripts.build_shape_block_intervention_grid import build_grid_plan

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    unrelated = manifest_dir / "keep-me.txt"
    unrelated.write_text("owned elsewhere", encoding="utf-8")
    kwargs = dict(
        manifest_dir=manifest_dir,
        run_root=tmp_path / "runs",
        index_path=tmp_path / "index.json",
        prefix_trace=tmp_path / "prefix.npz",
        block29_trace=tmp_path / "block29.npz",
        alphas=(0.0, 1.0),
        betas=(0.0, 1.0),
        control_references={},
    )

    first = build_grid_plan(**kwargs)
    first_bytes = (tmp_path / "index.json").read_bytes()
    second = build_grid_plan(**kwargs)

    assert first == second
    assert (tmp_path / "index.json").read_bytes() == first_bytes
    assert unrelated.read_text() == "owned elsewhere"


def test_grid_plan_validation_rejects_duplicate_point_names() -> None:
    from scripts.summarize_shape_block_intervention_grid import _validate_plan

    index = {
        "schema": "trellis2mlx.shape_block_intervention_grid_plan.v1",
        "axes": {"alpha": [0.0, 1.0], "beta": [0.0]},
        "points": [
            {"name": "same-name", "coordinate": {"alpha": 0.0, "beta": 0.0}},
            {"name": "same-name", "coordinate": {"alpha": 1.0, "beta": 0.0}},
        ],
    }

    with pytest.raises(ValueError, match="duplicate point names"):
        _validate_plan(index)


def test_grid_summary_requires_complete_route_evidence_and_exact_controls(tmp_path: Path) -> None:
    from scripts.build_shape_block_intervention_grid import build_grid_plan
    from scripts.summarize_shape_block_intervention_grid import summarize_grid

    controls = {}
    source_trace = _write_trace(tmp_path / "block29.npz", alpha=1.0, beta=1.0)
    with np.load(source_trace, allow_pickle=False) as source:
        source_arrays = {name: np.asarray(source[name]) for name in source.files}
    for branch in ("pos", "neg"):
        raw_name = f"{branch}_block29_cross_attention_raw"
        source_arrays[raw_name] = source_arrays[raw_name].reshape(1, 3, 2, 2)
        final_name = f"{branch}_final_output"
        source_arrays[final_name] = source_arrays[final_name][0]
    np.savez(source_trace, **source_arrays)
    for alpha, beta, name in (
        (0.0, 0.0, "natural"),
        (1.0, 0.0, "after-self"),
        (0.0, 1.0, "cross-raw"),
        (1.0, 1.0, "join"),
    ):
        controls[(alpha, beta)] = _write_trace(
            tmp_path / f"control-{name}.npz", alpha=alpha, beta=beta, offset=alpha + beta
        )
    index = build_grid_plan(
        manifest_dir=tmp_path / "manifests",
        run_root=tmp_path / "runs",
        index_path=tmp_path / "index.json",
        prefix_trace=tmp_path / "prefix.npz",
        block29_trace=source_trace,
        alphas=(0.0, 1.0),
        betas=(0.0, 1.0),
        control_references=controls,
    )
    for point in index["points"]:
        coordinate = point["coordinate"]
        output_dir = Path(point["output_dir"])
        trace_path = _write_trace(
            Path(point["expected_trace_path"]),
            alpha=coordinate["alpha"],
            beta=coordinate["beta"],
            offset=coordinate["alpha"] + coordinate["beta"],
        )
        _write_run_evidence(output_dir, Path(point["manifest_path"]), trace_path)

    summary = summarize_grid(tmp_path / "index.json")

    assert summary["schema"] == "trellis2mlx.shape_block_intervention_grid_summary.v2"
    assert summary["status"] == "done"
    assert summary["point_count"] == 4
    assert all(point["control_exact"] for point in summary["points"])
    assert all(point["route"]["backend"] == "mlx-metal" for point in summary["points"])
    assert all(point["route"]["attention_backend"] == "fast" for point in summary["points"])
    assert summary["route_vector"]["conditioning_sample_sha256"] == "1" * 64
    assert all(set(point["state_digests"]) == set(summary["compared_arrays"]) for point in summary["points"])

    geometry = summary["coordinate_geometry"]
    assert geometry["coordinate_system"] == {
        "alpha": "source_delta_scale at block29 after_self",
        "beta": "source_delta_scale at block29 cross_attention_raw",
        "projection": "none",
    }
    assert len(geometry["cells"]) == 1
    cell = geometry["cells"][0]
    witness = cell["arrays"]["pos_block29_after_cross"]
    assert witness["lower_corner_tangents"]["cosine"] == pytest.approx(1.0)
    assert witness["opposite_edge_transport"]["alpha"]["difference"]["l2_norm"] == 0.0
    assert witness["opposite_edge_transport"]["beta"]["difference"]["l2_norm"] == 0.0
    assert witness["mixed_second_difference"]["l2_norm"] == 0.0
    collapsed = [
        group
        for group in geometry["quotient_classes"]["pos_block29_after_cross"]
        if group["point_count"] == 2
    ]
    assert len(collapsed) == 1
    assert collapsed[0]["coordinates"] == [
        {"alpha": 0.0, "beta": 1.0},
        {"alpha": 1.0, "beta": 0.0},
    ]

    first = index["points"][0]
    (Path(first["output_dir"]) / "run_report.json").unlink()
    with pytest.raises(ValueError, match="run report"):
        summarize_grid(tmp_path / "index.json")


def test_coordinate_geometry_exposes_mixed_interaction_without_projection() -> None:
    from scripts.summarize_shape_block_intervention_grid import _summarize_array_geometry

    states = {
        (alpha, beta): (
            f"a{alpha}-b{beta}",
            np.full((2, 2), alpha * beta, dtype=np.float32),
        )
        for alpha in (0.0, 1.0)
        for beta in (0.0, 1.0)
    }

    geometry = _summarize_array_geometry(
        "interaction", alpha_values=(0.0, 1.0), beta_values=(0.0, 1.0), states=states
    )

    cell = geometry["cells"][0]
    assert cell["lower_corner_tangents"]["alpha"]["l2_norm"] == 0.0
    assert cell["lower_corner_tangents"]["beta"]["l2_norm"] == 0.0
    assert cell["lower_corner_tangents"]["cosine"] is None
    assert cell["mixed_second_difference"]["mean_abs"] == pytest.approx(1.0)
    assert cell["mixed_second_difference"]["l2_norm"] == pytest.approx(2.0)
    assert cell["opposite_edge_transport"]["alpha"]["difference"]["l2_norm"] == pytest.approx(2.0)
    assert cell["opposite_edge_transport"]["beta"]["difference"]["l2_norm"] == pytest.approx(2.0)


@pytest.mark.parametrize("fault", ["missing", "shape", "nonfinite"])
def test_coordinate_geometry_rejects_incomplete_or_malformed_states(fault: str) -> None:
    from scripts.summarize_shape_block_intervention_grid import _summarize_array_geometry

    states = {
        (alpha, beta): (
            f"a{alpha}-b{beta}",
            np.full((2, 2), alpha + beta, dtype=np.float32),
        )
        for alpha in (0.0, 1.0)
        for beta in (0.0, 1.0)
    }
    if fault == "missing":
        del states[(1.0, 1.0)]
    elif fault == "shape":
        states[(1.0, 1.0)] = ("bad-shape", np.zeros((2, 3), dtype=np.float32))
    else:
        states[(1.0, 1.0)][1][0, 0] = np.nan

    with pytest.raises(ValueError, match="Cartesian|shape|non-finite"):
        _summarize_array_geometry(
            "malformed", alpha_values=(0.0, 1.0), beta_values=(0.0, 1.0), states=states
        )


def test_state_digest_binds_dtype_shape_and_bytes() -> None:
    from scripts.summarize_shape_block_intervention_grid import _state_digest

    base = np.asarray([[1, 2]], dtype=np.float32)

    assert _state_digest(base) == _state_digest(base.copy())
    assert _state_digest(base) != _state_digest(base.astype(np.int32))
    assert _state_digest(base) != _state_digest(base.reshape(2, 1))


def test_grid_summary_rejects_corner_that_only_matches_at_endpoint(tmp_path: Path) -> None:
    from scripts.build_shape_block_intervention_grid import build_grid_plan
    from scripts.summarize_shape_block_intervention_grid import summarize_grid

    control = _write_trace(tmp_path / "control.npz", alpha=0.0, beta=0.0)
    source_trace = _write_trace(tmp_path / "block29.npz", alpha=1.0, beta=1.0)
    index = build_grid_plan(
        manifest_dir=tmp_path / "manifests",
        run_root=tmp_path / "runs",
        index_path=tmp_path / "index.json",
        prefix_trace=tmp_path / "prefix.npz",
        block29_trace=source_trace,
        alphas=(0.0,),
        betas=(0.0,),
        control_references={(0.0, 0.0): control},
    )
    point = index["points"][0]
    trace_path = _write_trace(Path(point["expected_trace_path"]), alpha=0.0, beta=0.0)
    with np.load(trace_path, allow_pickle=False) as trace:
        arrays = {name: np.asarray(trace[name]) for name in trace.files}
    arrays["pos_block29_after_cross"] = arrays["pos_block29_after_cross"] + 1
    np.savez(trace_path, **arrays)
    _write_run_evidence(Path(point["output_dir"]), Path(point["manifest_path"]), trace_path)

    with pytest.raises(ValueError, match="control.*pos_block29_after_cross"):
        summarize_grid(tmp_path / "index.json")


def test_grid_summary_cli_writes_failure_report_before_primary_summary(tmp_path: Path) -> None:
    from scripts.summarize_shape_block_intervention_grid import main

    index = tmp_path / "missing-runs.json"
    index.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.shape_block_intervention_grid_plan.v1",
                "axes": {"alpha": [0.0], "beta": [0.0]},
                "points": [
                    {
                        "coordinate": {"alpha": 0.0, "beta": 0.0},
                        "output_dir": str(tmp_path / "missing"),
                        "manifest_path": str(tmp_path / "missing.json"),
                        "manifest_sha256": "0" * 64,
                        "expected_trace_path": str(tmp_path / "missing" / "trace.npz"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"

    assert main(["--grid-index", str(index), "--output-json", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "admit_grid_runs"
    assert report["last_trustworthy_evidence"]["grid_index"] == str(index)


def test_grid_summary_cli_distinguishes_coordinate_geometry_failure(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.summarize_shape_block_intervention_grid as module
    from scripts.summarize_shape_block_intervention_grid import CoordinateGeometryError

    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")
    output = tmp_path / "summary.json"

    def fail_geometry(_index_path):
        raise CoordinateGeometryError("mixed tensor shape")

    monkeypatch.setattr(module, "summarize_grid", fail_geometry)

    assert module.main(["--grid-index", str(index), "--output-json", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["failure_phase"] == "coordinate_geometry"
    assert report["error_message"] == "mixed tensor shape"


def test_grid_summary_route_rejects_incomplete_effective_trace_key_selection() -> None:
    from scripts.summarize_shape_block_intervention_grid import _route_vector

    route = {
        "family": "trellis2mlx/mlx",
        "backend": "mlx-metal",
        "attention_backend": "fast",
        "repo_root": "/worktree",
        "conditioning_sample_sha256": "1" * 64,
        "shape_flow_noise_sample_sha256": "2" * 64,
        "shape_slat_support_sample_sha256": "3" * 64,
        "shared_noise_sha256": "4" * 64,
        "shape_flow_block_injection_manifest_sha256": "5" * 64,
        "shape_flow_trace_block_index": 29,
        "shape_flow_trace_step_index": 0,
        "shape_flow_trace_key_selection": "explicit",
        "shape_flow_trace_keys": ["pos_final_output", "neg_final_output"],
        "steps": 8,
    }

    with pytest.raises(ValueError, match="trace key selection"):
        _route_vector(route, name="incomplete")

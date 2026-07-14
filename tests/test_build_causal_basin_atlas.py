import hashlib
import json
import math
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _prefix_summary(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "prefix.json",
        {
            "schema": "trellis2mlx.shape_flow.source_prefix_curve.v3",
            "source_path": "/evidence/source.npz",
            "baseline_path": "/evidence/baseline.npz",
            "rows": [
                {
                    "boundary": "mlx_baseline",
                    "pred_final_source_mean_abs": 0.03,
                    "pred_final_remaining_norm_ratio": 1.0,
                    "sample_next_source_mean_abs": 0.001,
                },
                {
                    "boundary": "after_block0",
                    "block": 0,
                    "path": "/runs/block0/checkpoints/shape_flow_step.npz",
                    "pred_final_source_mean_abs": 0.02,
                    "pred_final_remaining_norm_ratio": 0.8,
                    "sample_next_source_mean_abs": 0.0008,
                },
                {
                    "boundary": "after_block2",
                    "block": 2,
                    "path": "/runs/block2/checkpoints/shape_flow_step.npz",
                    "pred_final_source_mean_abs": 0.01,
                    "pred_final_remaining_norm_ratio": 0.4,
                    "sample_next_source_mean_abs": 0.0004,
                },
            ],
        },
    )


def _alpha_summary(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    return _write_json(
        tmp_path / name,
        {
            "schema": "trellis2mlx.shape_flow.attention_alpha_curve.v1",
            "baseline": "/evidence/baseline.npz",
            "source": "/evidence/source.npz",
            "rows": rows,
        },
    )


def _composition_summary(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "composition.json",
        {
            "schema": "trellis2mlx.shape_flow.source_island_composition.v1",
            "baseline": "/evidence/baseline.npz",
            "source": "/evidence/source.npz",
            "rows": [
                {
                    "name": "combo_alpha0p25",
                    "path": "/runs/combo/checkpoints/shape_flow_step.npz",
                    "move_to_source_norm_ratio": 1.01,
                    "cosine_to_source_direction": 0.51,
                    "projection_fraction": 0.52,
                    "pred_final_source_mean_abs": 0.029,
                    "pred_final_changed_values": 246304,
                }
            ],
        },
    )


def _operation_summary(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "operation.json",
        {
            "schema": "trellis2mlx.shape_flow.block29_source_prefix28_operation_compare.v1",
            "comparison_class": "exact_source_prefix_then_mlx_block29",
            "rows": [
                {
                    "branch": "pos",
                    "stage": "input",
                    "source_mean_abs": 0.0,
                    "source_relative_norm": 0.0,
                },
                {
                    "branch": "pos",
                    "stage": "after_self",
                    "source_mean_abs": 0.0003,
                    "source_relative_norm": 0.00015,
                },
                {
                    "branch": "neg",
                    "stage": "input",
                    "source_mean_abs": 0.0,
                    "source_relative_norm": 0.0,
                },
            ],
        },
    )


def _operation_replays_summary(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "operation-replays.json",
        {
            "schema": "trellis2mlx.shape_block_operation_replays.v1",
            "status": "done",
            "replay_rows": [
                {
                    "name": "natural",
                    "artifact": "/runs/natural.npz",
                    "intervention_depth": 0,
                    "intervention_topology": "main_chain",
                    "causal_parent": None,
                    "pred_final_source_mean_abs": 0.0015,
                    "sample_next_source_mean_abs": 0.00007,
                },
                {
                    "name": "after_self",
                    "artifact": "/runs/after-self.npz",
                    "intervention_depth": 1,
                    "intervention_topology": "main_chain",
                    "causal_parent": "natural",
                    "pred_final_source_mean_abs": 0.00013,
                    "sample_next_source_mean_abs": 0.000006,
                },
                {
                    "name": "cross_attention_raw",
                    "artifact": "/runs/cross-raw.npz",
                    "intervention_depth": 2,
                    "intervention_topology": "side_branch",
                    "causal_parent": "natural",
                    "pred_final_source_mean_abs": 0.001,
                    "sample_next_source_mean_abs": 0.00005,
                },
                {
                    "name": "source",
                    "artifact": "/runs/source.npz",
                    "intervention_depth": 3,
                    "intervention_topology": "main_chain",
                    "causal_parent": "after_self",
                    "pred_final_source_mean_abs": 0.0,
                    "sample_next_source_mean_abs": 0.0,
                },
            ],
        },
    )


def test_build_atlas_preserves_semantic_coordinates_and_all_nodes(tmp_path: Path) -> None:
    from scripts.build_causal_basin_atlas import build_atlas

    coarse = _alpha_summary(
        tmp_path,
        "coarse.json",
        [
            {
                "alpha": 0.0,
                "artifact": "/runs/a0/checkpoints/shape_flow_step.npz",
                "pred_final_source_mean_abs": 0.03,
                "move_to_source_norm_ratio": 0.0,
                "cosine_to_source_direction": None,
                "pred_final_changed_values": 0,
            },
            {
                "alpha": 0.25,
                "artifact": "/runs/a25/checkpoints/shape_flow_step.npz",
                "pred_final_source_mean_abs": 0.028,
                "move_to_source_norm_ratio": 0.96,
                "cosine_to_source_direction": 0.49,
                "pred_final_changed_values": 246303,
            },
        ],
    )
    fine = _alpha_summary(
        tmp_path,
        "fine.json",
        [
            {
                "alpha": 0.1875,
                "artifact": "/runs/a1875/checkpoints/shape_flow_step.npz",
                "pred_final_source_mean_abs": 0.029,
                "move_to_source_norm_ratio": 0.95,
                "cosine_to_source_direction": 0.48,
                "pred_final_changed_values": 246304,
            },
            {
                "alpha": 0.25,
                "artifact": "/runs/a25/checkpoints/shape_flow_step.npz",
                "pred_final_source_mean_abs": 0.028,
                "move_to_source_norm_ratio": 0.96,
                "cosine_to_source_direction": 0.49,
                "pred_final_changed_values": 246303,
            },
        ],
    )

    atlas = build_atlas(
        prefix_path=_prefix_summary(tmp_path),
        alpha_paths=[coarse, fine],
        composition_path=_composition_summary(tmp_path),
        operation_path=_operation_summary(tmp_path),
        operation_replays_path=_operation_replays_summary(tmp_path),
    )

    assert atlas["schema"] == "trellis2mlx.causal_basin_atlas.v1"
    prefix = atlas["charts"]["source_prefix"]
    assert [node["coordinate"]["block"] for node in prefix["nodes"]] == [-1, 0, 2]
    assert len(prefix["nodes"]) == 3
    assert [edge["intervention"]["added_blocks"] for edge in prefix["edges"]] == [1, 2]

    alpha = atlas["charts"]["attention_alpha"]
    assert [node["coordinate"]["alpha"] for node in alpha["nodes"]] == [0.0, 0.1875, 0.25]
    assert len(alpha["edges"]) == 2
    assert all(edge["kind"] == "attention_delta_scale" for edge in alpha["edges"])

    composition = atlas["charts"]["composition"]
    assert composition["nodes"][0]["placement"] == "off_chart"
    assert composition["nodes"][0]["metrics"]["cosine_to_source_direction"] == 0.51
    assert composition["x_axis"]["field"] == "projection_fraction"

    operation = atlas["charts"]["block_operation"]
    assert [node["coordinate"] for node in operation["nodes"]] == [
        {"branch": "pos", "stage_index": 0, "stage": "input"},
        {"branch": "pos", "stage_index": 1, "stage": "after_self"},
        {"branch": "neg", "stage_index": 0, "stage": "input"},
    ]
    replay = atlas["charts"]["operation_replay"]
    assert [node["coordinate"]["intervention_depth"] for node in replay["nodes"]] == [0, 1, 2, 3]
    assert [(edge["from"], edge["to"], edge["kind"]) for edge in replay["edges"]] == [
        (
            "operation-replay:natural",
            "operation-replay:after_self",
            "causal_boundary_replay",
        ),
        (
            "operation-replay:natural",
            "operation-replay:cross_attention_raw",
            "causal_intervention_branch",
        ),
        (
            "operation-replay:after_self",
            "operation-replay:source",
            "causal_boundary_replay",
        ),
    ]
    assert replay["nodes"][2]["placement"] == "side_branch"
    assert "side branches" in replay["x_axis"]["semantic"]


def test_operation_replay_chart_rejects_orphan_branch_and_duplicate_names() -> None:
    from scripts.build_causal_basin_atlas import AtlasContractError, _build_operation_replay_chart

    natural = {
        "name": "natural",
        "artifact": "/runs/natural.npz",
        "intervention_depth": 0,
        "intervention_topology": "main_chain",
        "causal_parent": None,
        "pred_final_source_mean_abs": 0.0015,
    }
    orphan = {
        "name": "cross_attention_raw",
        "artifact": "/runs/cross.npz",
        "intervention_depth": 3,
        "intervention_topology": "side_branch",
        "causal_parent": None,
        "pred_final_source_mean_abs": 0.001,
    }
    payload = {
        "status": "done",
        "replay_rows": [natural, orphan],
    }
    with pytest.raises(AtlasContractError, match="side branch.*causal parent"):
        _build_operation_replay_chart(payload)

    duplicate = {
        **natural,
        "intervention_depth": 2,
        "pred_final_source_mean_abs": 0.0001,
    }
    payload["replay_rows"] = [natural, duplicate]
    with pytest.raises(AtlasContractError, match="duplicate operation replay names"):
        _build_operation_replay_chart(payload)


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        (
            [("natural", 0, "natural"), ("source", 1, None)],
            "cannot parent itself",
        ),
        (
            [("natural", 0, None), ("after_self", 2, "source"), ("source", 6, None)],
            "must precede child",
        ),
        (
            [("natural", 0, None), ("branch_a", 2, None), ("branch_b", 2, "branch_a")],
            "must precede child",
        ),
        (
            [("natural", 0, "after_self"), ("after_self", 1, "natural")],
            "causal parent cycle",
        ),
    ],
    ids=("self-parent", "forward-parent", "equal-depth-parent", "cycle"),
)
def test_operation_replay_chart_rejects_impossible_parent_graphs(
    rows: list[tuple[str, int, str | None]], error: str
) -> None:
    from scripts.build_causal_basin_atlas import AtlasContractError, _build_operation_replay_chart

    def row(name: str, depth: int, parent: str | None) -> dict[str, object]:
        return {
            "name": name,
            "artifact": f"/runs/{name}.npz",
            "intervention_depth": depth,
            "intervention_topology": "main_chain",
            "causal_parent": parent,
            "pred_final_source_mean_abs": 0.001 / (depth + 1),
        }

    with pytest.raises(AtlasContractError, match=error):
        _build_operation_replay_chart(
            {
                "status": "done",
                "replay_rows": [row(name, depth, parent) for name, depth, parent in rows],
            }
        )


def test_operation_replay_chart_preserves_join_and_state_equivalence() -> None:
    from scripts.build_causal_basin_atlas import _build_operation_replay_chart

    def row(
        name: str,
        depth: int,
        topology: str,
        parents: list[str],
        equivalents: list[str] | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "name": name,
            "artifact": f"/runs/{name}.npz",
            "intervention_depth": depth,
            "intervention_topology": topology,
            "causal_parents": parents,
            "state_equivalents": equivalents or [],
            "pred_final_source_mean_abs": 0.001 / (depth + 1),
        }
        if equivalents:
            result["equivalence_evidence"] = {
                "comparison_class": "exact_block29_after_cross_through_final_output",
                "target": equivalents[0],
                "all_exact": True,
                "compared_arrays": [
                    "pos_block29_after_cross",
                    "neg_block29_after_cross",
                    "pos_block29_after_mlp",
                    "neg_block29_after_mlp",
                    "pos_final_output",
                    "neg_final_output",
                ],
            }
        return result

    chart = _build_operation_replay_chart(
        {
            "schema": "trellis2mlx.shape_block_operation_replays.v2",
            "status": "done",
            "replay_rows": [
                row("natural", 0, "main_chain", []),
                row("after_self", 2, "main_chain", ["natural"]),
                row("cross_attention_raw", 3, "side_branch", ["natural"]),
                row("after_cross", 4, "main_chain", ["after_self"]),
                row(
                    "after_self_cross_raw_join",
                    4,
                    "join",
                    ["after_self", "cross_attention_raw"],
                    ["after_cross"],
                ),
            ],
        }
    )

    joined = next(node for node in chart["nodes"] if node["label"].endswith("join"))
    assert joined["placement"] == "join"
    assert joined["intervention"]["causal_parents"] == [
        "after_self",
        "cross_attention_raw",
    ]
    assert {(edge["from"], edge["to"], edge["kind"]) for edge in chart["edges"]} >= {
        (
            "operation-replay:after_self",
            "operation-replay:after_self_cross_raw_join",
            "causal_intervention_join",
        ),
        (
            "operation-replay:cross_attention_raw",
            "operation-replay:after_self_cross_raw_join",
            "causal_intervention_join",
        ),
        (
            "operation-replay:after_self_cross_raw_join",
            "operation-replay:after_cross",
            "continuation_equivalence",
        ),
    }


def test_operation_replay_chart_rejects_unproven_state_equivalence() -> None:
    from scripts.build_causal_basin_atlas import AtlasContractError, _build_operation_replay_chart

    payload = {
        "schema": "trellis2mlx.shape_block_operation_replays.v2",
        "status": "done",
        "replay_rows": [
            {
                "name": "after_cross",
                "artifact": "/runs/after-cross.npz",
                "intervention_depth": 4,
                "intervention_topology": "main_chain",
                "causal_parents": [],
                "state_equivalents": [],
                "pred_final_source_mean_abs": 0.00002,
            },
            {
                "name": "claimed_join",
                "artifact": "/runs/claimed-join.npz",
                "intervention_depth": 5,
                "intervention_topology": "join",
                "causal_parents": ["after_cross", "natural"],
                "state_equivalents": ["after_cross"],
                "pred_final_source_mean_abs": 0.00002,
            },
            {
                "name": "natural",
                "artifact": "/runs/natural.npz",
                "intervention_depth": 0,
                "intervention_topology": "main_chain",
                "causal_parents": [],
                "state_equivalents": [],
                "pred_final_source_mean_abs": 0.0015,
            },
        ],
    }
    with pytest.raises(AtlasContractError, match="equivalence evidence"):
        _build_operation_replay_chart(payload)


def test_atlas_records_input_hashes_and_route_identity_visibility(tmp_path: Path) -> None:
    from scripts.build_causal_basin_atlas import build_atlas

    atlas = build_atlas(
        prefix_path=_prefix_summary(tmp_path),
        alpha_paths=[
            _alpha_summary(
                tmp_path,
                "alpha.json",
                [
                    {
                        "alpha": 0.0,
                        "artifact": "/missing/run.npz",
                        "pred_final_source_mean_abs": 0.03,
                        "move_to_source_norm_ratio": 0.0,
                        "cosine_to_source_direction": None,
                        "pred_final_changed_values": 0,
                    }
                ],
            )
        ],
        composition_path=_composition_summary(tmp_path),
        operation_path=None,
        operation_replays_path=None,
    )

    assert len(atlas["sources"]) == 3
    assert all(source["size_bytes"] > 0 for source in atlas["sources"])
    assert all(len(source["sha256"]) == 64 for source in atlas["sources"])
    node = atlas["charts"]["attention_alpha"]["nodes"][0]
    assert node["route_identity"]["status"] == "missing"
    assert node["route_identity"]["authority"] == "summary_only"


def test_route_identity_reads_effective_nested_mlx_route(tmp_path: Path) -> None:
    from scripts.build_causal_basin_atlas import _route_identity

    artifact = tmp_path / "run" / "checkpoints" / "shape_flow_step.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    _write_json(
        artifact.parent.parent / "route_identity.json",
        {
            "schema": "trellis2mlx.mlx_stage_capture_route.v1",
            "requested_stop": "shape_flow_step",
            "route": {
                "family": "trellis2mlx/mlx",
                "backend": "mlx-metal",
                "attention_backend": "fast",
                "shape_flow_block_injection_manifest_path": "/evidence/manifest.json",
            },
        },
    )
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    route_payload = json.loads((artifact.parent.parent / "route_identity.json").read_text())
    _write_json(
        artifact.parent.parent / "run_report.json",
        {
            "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
            "status": "done",
            "primary_output_status": "written",
            "route_identity": route_payload,
            "artifacts": {
                artifact.name: {
                    "path": str(artifact),
                    "size_bytes": artifact.stat().st_size,
                    "sha256": artifact_sha,
                }
            },
        },
    )

    route = _route_identity(str(artifact))

    assert route["status"] == "visible"
    assert route["effective_route"] == "trellis2mlx/mlx"
    assert route["effective_device"] == "mlx-metal"
    assert route["effective_attention_backend"] == "fast"
    assert route["intervention_manifest"] == "/evidence/manifest.json"
    assert route["artifact_sha256"] == artifact_sha


def test_route_identity_rejects_stale_artifact_beside_failed_or_unbound_run(tmp_path: Path) -> None:
    from scripts.build_causal_basin_atlas import _route_identity

    artifact = tmp_path / "run" / "checkpoints" / "shape_flow_step.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"stale artifact")
    route = {
        "schema": "trellis2mlx.mlx_stage_capture_route.v1",
        "requested_stop": "shape_flow_step",
        "route": {"family": "trellis2mlx/mlx", "backend": "mlx-metal"},
    }
    _write_json(artifact.parent.parent / "route_identity.json", route)
    report_path = artifact.parent.parent / "run_report.json"
    _write_json(
        report_path,
        {
            "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
            "status": "failed",
            "primary_output_status": "missing",
            "failure_phase": "generate_subprocess",
            "route_identity": route,
            "artifacts": {},
        },
    )
    assert _route_identity(str(artifact))["status"] == "failed"

    report = json.loads(report_path.read_text())
    report.update(status="done", primary_output_status="written")
    report["artifacts"] = {artifact.name: str(artifact)}
    _write_json(report_path, report)
    identity = _route_identity(str(artifact))
    assert identity["status"] == "unverified"
    assert "digest" in identity["reason"]


def test_build_atlas_rejects_missing_or_duplicate_semantic_coordinates(tmp_path: Path) -> None:
    from scripts.build_causal_basin_atlas import AtlasContractError, build_atlas

    bad_prefix = _write_json(
        tmp_path / "bad-prefix.json",
        {
            "schema": "trellis2mlx.shape_flow.source_prefix_curve.v3",
            "source_path": "/source.npz",
            "baseline_path": "/baseline.npz",
            "rows": [{"boundary": "after_block0", "block": 0}],
        },
    )
    alpha = _alpha_summary(
        tmp_path,
        "alpha.json",
        [
            {
                "alpha": 0.0,
                "artifact": "/run.npz",
                "pred_final_source_mean_abs": 0.03,
                "move_to_source_norm_ratio": 0.0,
                "cosine_to_source_direction": None,
                "pred_final_changed_values": 0,
            }
        ],
    )
    with pytest.raises(AtlasContractError, match="mlx_baseline"):
        build_atlas(
            prefix_path=bad_prefix,
            alpha_paths=[alpha],
            composition_path=_composition_summary(tmp_path),
            operation_path=None,
            operation_replays_path=None,
        )


def test_render_html_is_self_contained_and_names_evidence_limits(tmp_path: Path) -> None:
    from scripts.build_causal_basin_atlas import build_atlas, render_html

    atlas = build_atlas(
        prefix_path=_prefix_summary(tmp_path),
        alpha_paths=[
            _alpha_summary(
                tmp_path,
                "alpha.json",
                [
                    {
                        "alpha": 0.0,
                        "artifact": "/run.npz",
                        "pred_final_source_mean_abs": 0.03,
                        "move_to_source_norm_ratio": 0.0,
                        "cosine_to_source_direction": None,
                        "pred_final_changed_values": 0,
                    }
                ],
            )
        ],
        composition_path=_composition_summary(tmp_path),
        operation_path=_operation_summary(tmp_path),
        operation_replays_path=_operation_replays_summary(tmp_path),
    )
    html = render_html(atlas)

    assert "TRELLIS.2 Causal Basin Atlas" in html
    assert "source-prefix depth" in html
    assert "attention delta scale" in html
    assert "Operation chart" in html
    assert "Block29 causal replay endpoints" in html
    assert "No PCA or learned embedding defines these coordinates." in html
    assert 'type="application/json" id="atlas-data"' in html
    assert "https://" not in html
    assert "http://" not in html
    assert "evidence-legend" in html
    assert "node-unverified" in html
    assert "node-missing" in html
    assert "edge-join" in html
    assert "edge-equivalence" in html
    assert "pointY" in html
    assert "aria-label" in html
    assert "log10(source mean absolute delta + 1e-9)" in html
    assert "branch==='neg'?.08" not in html


@pytest.mark.parametrize("bad_value", [None, "N/A", math.nan, math.inf, -math.inf])
def test_atlas_rejects_malformed_or_nonfinite_plotted_metrics(tmp_path: Path, bad_value: object) -> None:
    from scripts.build_causal_basin_atlas import AtlasContractError, build_atlas

    composition = json.loads(_composition_summary(tmp_path).read_text())
    composition["rows"][0]["pred_final_source_mean_abs"] = bad_value
    composition_path = _write_json(tmp_path / "bad-composition.json", composition)
    with pytest.raises(AtlasContractError, match="finite numeric"):
        build_atlas(
            prefix_path=_prefix_summary(tmp_path),
            alpha_paths=[_alpha_summary(tmp_path, "alpha-valid.json", [{
                "alpha": 0.0,
                "artifact": "/run.npz",
                "pred_final_source_mean_abs": 0.03,
                "move_to_source_norm_ratio": 0.0,
            }])],
            composition_path=composition_path,
            operation_path=None,
        )


def test_atlas_rejects_missing_operation_metric_and_wrong_source_schema(tmp_path: Path) -> None:
    from scripts.build_causal_basin_atlas import AtlasContractError, build_atlas

    operation = json.loads(_operation_summary(tmp_path).read_text())
    del operation["rows"][0]["source_mean_abs"]
    alpha = _alpha_summary(tmp_path, "alpha-valid.json", [{
        "alpha": 0.0,
        "artifact": "/run.npz",
        "pred_final_source_mean_abs": 0.03,
        "move_to_source_norm_ratio": 0.0,
    }])
    with pytest.raises(AtlasContractError, match="source_mean_abs"):
        build_atlas(
            prefix_path=_prefix_summary(tmp_path), alpha_paths=[alpha],
            composition_path=_composition_summary(tmp_path),
            operation_path=_write_json(tmp_path / "bad-operation.json", operation),
        )

    wrong = json.loads(_composition_summary(tmp_path).read_text())
    wrong["schema"] = "wrong.schema"
    with pytest.raises(AtlasContractError, match="composition.*schema"):
        build_atlas(
            prefix_path=_prefix_summary(tmp_path), alpha_paths=[alpha],
            composition_path=_write_json(tmp_path / "wrong-schema.json", wrong),
            operation_path=None,
        )


def test_alpha_duplicates_require_exact_causal_metrics_and_preserve_provenance(tmp_path: Path) -> None:
    from scripts.build_causal_basin_atlas import AtlasContractError, build_atlas

    row = {
        "alpha": 0.25,
        "artifact": "/runs/a25/checkpoints/shape_flow_step.npz",
        "pred_final_source_mean_abs": 0.028,
        "pred_final_source_max_abs": 0.2,
        "sample_next_source_mean_abs": 0.001,
        "move_to_source_norm_ratio": 0.96,
    }
    first = _alpha_summary(tmp_path, "first.json", [row])
    second_row = dict(
        row,
        artifact="/runs/a25-repeat/checkpoints/shape_flow_step.npz",
        source_displacement_norm=18.0,
    )
    second = _alpha_summary(tmp_path, "second.json", [second_row])
    atlas = build_atlas(
        prefix_path=_prefix_summary(tmp_path), alpha_paths=[first, second],
        composition_path=_composition_summary(tmp_path), operation_path=None,
    )
    observations = atlas["charts"]["attention_alpha"]["nodes"][0]["source_observations"]
    assert [Path(item["path"]).name for item in observations] == ["first.json", "second.json"]
    assert [item["artifact"] for item in observations] == [row["artifact"], second_row["artifact"]]
    assert observations[0]["metrics"]["pred_final_source_max_abs"] == 0.2
    assert observations[1]["metrics"]["source_displacement_norm"] == 18.0
    assert all("route_identity" in item for item in observations)
    node = atlas["charts"]["attention_alpha"]["nodes"][0]
    assert node["artifact"] is None
    assert node["route_identity"]["status"] == "missing"
    assert node["metrics"]["source_displacement_norm"] == 18.0

    for metric, value in (
        ("pred_final_source_mean_abs", 0.029),
        ("pred_final_source_max_abs", 0.3),
        ("sample_next_source_mean_abs", 0.002),
    ):
        conflicting = dict(row, **{metric: value})
        with pytest.raises(AtlasContractError, match=f"conflicting.*{metric}"):
            build_atlas(
                prefix_path=_prefix_summary(tmp_path),
                alpha_paths=[first, _alpha_summary(tmp_path, f"conflict-{metric}.json", [conflicting])],
                composition_path=_composition_summary(tmp_path), operation_path=None,
            )


def test_alpha_duplicate_normalization_preserves_fraction_and_count_metrics_separately(tmp_path: Path) -> None:
    from scripts.build_causal_basin_atlas import build_atlas

    improved_fraction = 0.5054850915941276
    worsened_fraction = 0.49451084838248666
    coarse = _write_json(
        tmp_path / "coarse-points.json",
        {
            "schema": "trellis2mlx.shape_attention_alpha_curve.v1",
            "points": {
                "0.25": {
                    "path": "/runs/coarse.npz",
                    "arrays": {
                        "pred_final": {
                            "source_mean_abs_after": 0.028,
                            "source_max_abs_after": 0.2,
                            "move_norm_over_source_displacement_norm": 0.96,
                            "move_vs_source_displacement_cosine": 0.49,
                            "projection_fraction_of_source_displacement": 0.47,
                            "move_nonzero": 246303,
                            "cells_improved_fraction": improved_fraction,
                            "cells_worsened_fraction": worsened_fraction,
                        }
                    },
                }
            },
        },
    )
    fine = _alpha_summary(
        tmp_path,
        "fine-rows.json",
        [{
            "alpha": 0.25,
            "artifact": "/runs/fine.npz",
            "pred_final_source_mean_abs": 0.028,
            "pred_final_source_max_abs": 0.2,
            "move_to_source_norm_ratio": 0.96,
            "cosine_to_source_direction": 0.49,
            "projection_fraction": 0.47,
            "pred_final_changed_values": 246303,
            "pred_final_improved_values": 124503,
            "pred_final_worsened_values": 121800,
        }],
    )

    atlas = build_atlas(
        prefix_path=_prefix_summary(tmp_path), alpha_paths=[coarse, fine],
        composition_path=_composition_summary(tmp_path), operation_path=None,
    )

    metrics = atlas["charts"]["attention_alpha"]["nodes"][0]["metrics"]
    assert metrics["pred_final_cells_improved_fraction"] == improved_fraction
    assert metrics["pred_final_cells_worsened_fraction"] == worsened_fraction
    assert metrics["pred_final_improved_values"] == 124503
    assert metrics["pred_final_worsened_values"] == 121800

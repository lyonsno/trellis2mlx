import json
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
            "baseline": "/evidence/baseline.npz",
            "source": "/evidence/source.npz",
            "rows": [
                {
                    "name": "combo_alpha0p25",
                    "path": "/runs/combo/checkpoints/shape_flow_step.npz",
                    "move_to_source_norm_ratio": 1.01,
                    "cosine_to_source_direction": 0.51,
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
            "schema": "trellis2mlx.shape_flow.block_operation_chart.v1",
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
                    "pred_final_source_mean_abs": 0.0015,
                    "sample_next_source_mean_abs": 0.00007,
                },
                {
                    "name": "after_self",
                    "artifact": "/runs/after-self.npz",
                    "intervention_depth": 1,
                    "pred_final_source_mean_abs": 0.00013,
                    "sample_next_source_mean_abs": 0.000006,
                },
                {
                    "name": "source",
                    "artifact": "/runs/source.npz",
                    "intervention_depth": 2,
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

    operation = atlas["charts"]["block_operation"]
    assert [node["coordinate"] for node in operation["nodes"]] == [
        {"branch": "pos", "stage_index": 0, "stage": "input"},
        {"branch": "pos", "stage_index": 1, "stage": "after_self"},
        {"branch": "neg", "stage_index": 0, "stage": "input"},
    ]
    replay = atlas["charts"]["operation_replay"]
    assert [node["coordinate"]["intervention_depth"] for node in replay["nodes"]] == [0, 1, 2]
    assert [edge["kind"] for edge in replay["edges"]] == [
        "causal_boundary_replay",
        "causal_boundary_replay",
    ]


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

    route = _route_identity(str(artifact))

    assert route["status"] == "visible"
    assert route["effective_route"] == "trellis2mlx/mlx"
    assert route["effective_device"] == "mlx-metal"
    assert route["effective_attention_backend"] == "fast"
    assert route["intervention_manifest"] == "/evidence/manifest.json"


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

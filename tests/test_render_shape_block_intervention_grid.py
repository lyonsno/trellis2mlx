import copy
import hashlib
import json
import math
from pathlib import Path

import pytest


def _metrics(value: float) -> dict:
    return {
        "mean_abs": value,
        "max_abs": value * 2.0,
        "l2_norm": value * 3.0,
        "nonzero": 4,
    }


def _summary() -> dict:
    from scripts.summarize_shape_block_intervention_grid import COMPARED_ARRAYS

    coordinates = [
        {"alpha": 0.0, "beta": 0.0},
        {"alpha": 0.0, "beta": 1.0},
        {"alpha": 1.0, "beta": 0.0},
        {"alpha": 1.0, "beta": 1.0},
    ]
    points = []
    for index, coordinate in enumerate(coordinates):
        points.append(
            {
                "name": f"point-{index}",
                "coordinate": coordinate,
                "artifact": f"/evidence/point-{index}.npz",
                "artifact_sha256": f"{index + 1:064x}",
                "manifest_sha256": f"{index + 11:064x}",
                "control_exact": index in {0, 3},
                "source_metrics": {
                    name: {
                        "mean_abs": float(index + 1) / 1000.0,
                        "max_abs": float(index + 1) / 100.0,
                        "nonzero": index + 1,
                        "relative_norm": float(index + 1) / 10000.0,
                    }
                    for name in COMPARED_ARRAYS
                },
                "state_digests": {
                    name: hashlib.sha256(f"{name}-{index // 2}".encode()).hexdigest()
                    for name in COMPARED_ARRAYS
                },
            }
        )

    quotient_classes = {}
    for name in COMPARED_ARRAYS:
        quotient_classes[name] = [
            {
                "state_digest": points[0]["state_digests"][name],
                "point_count": 2,
                "point_names": [points[0]["name"], points[1]["name"]],
                "coordinates": coordinates[:2],
            },
            {
                "state_digest": points[2]["state_digests"][name],
                "point_count": 2,
                "point_names": [points[2]["name"], points[3]["name"]],
                "coordinates": coordinates[2:],
            },
        ]

    cell_arrays = {}
    for index, name in enumerate(COMPARED_ARRAYS, start=1):
        value = float(index) / 10.0
        cell_arrays[name] = {
            "lower_corner_tangents": {
                "alpha": _metrics(value),
                "beta": _metrics(value + 0.1),
                "cosine": 0.25,
            },
            "opposite_edge_transport": {
                "alpha": {"difference": _metrics(value + 0.2), "cosine": 0.5},
                "beta": {"difference": _metrics(value + 0.3), "cosine": -0.5},
            },
            "mixed_second_difference": _metrics(value + 0.4),
        }

    return {
        "schema": "trellis2mlx.shape_block_intervention_grid_summary.v2",
        "status": "done",
        "comparison_class": "block29_after_self_cross_attention_raw_delta_grid",
        "grid_index": "/evidence/grid-index.json",
        "grid_index_sha256": "a" * 64,
        "axes": {"alpha": [0.0, 1.0], "beta": [0.0, 1.0]},
        "point_count": 4,
        "route_vector": {
            "family": "trellis2mlx/mlx",
            "backend": "mlx-metal",
            "attention_backend": "fast",
            "repo_root": "/worktree",
            "conditioning_sample_sha256": "1" * 64,
            "shape_flow_noise_sample_sha256": "2" * 64,
            "shape_slat_support_sample_sha256": "3" * 64,
            "shared_noise_sha256": "4" * 64,
            "shape_flow_trace_block_index": 29,
            "shape_flow_trace_step_index": 0,
            "shape_flow_trace_key_selection": "explicit",
            "shape_flow_trace_keys": list(COMPARED_ARRAYS),
            "steps": 8,
        },
        "source_trace": "/evidence/source.npz",
        "source_trace_sha256": "b" * 64,
        "compared_arrays": list(COMPARED_ARRAYS),
        "points": points,
        "coordinate_geometry": {
            "coordinate_system": {
                "alpha": "source_delta_scale at block29 after_self",
                "beta": "source_delta_scale at block29 cross_attention_raw",
                "projection": "none",
            },
            "sorted_axes": {"alpha": [0.0, 1.0], "beta": [0.0, 1.0]},
            "quotient_classes": quotient_classes,
            "cells": [
                {
                    "bounds": {"alpha": [0.0, 1.0], "beta": [0.0, 1.0]},
                    "delta": {"alpha": 1.0, "beta": 1.0},
                    "arrays": cell_arrays,
                }
            ],
        },
    }


def test_rendered_grid_is_self_contained_and_preserves_causal_coordinates() -> None:
    from scripts.render_shape_block_intervention_grid import render_html, validate_summary
    from scripts.summarize_shape_block_intervention_grid import COMPARED_ARRAYS

    summary = _summary()
    validate_summary(summary)
    html = render_html(summary, summary_sha256="c" * 64)

    assert "TRELLIS.2 Block29 Intervention Surface" in html
    assert "source_delta_scale at block29 after_self" in html
    assert "source_delta_scale at block29 cross_attention_raw" in html
    assert "No PCA or learned embedding defines these coordinates." in html
    assert 'type="application/json" id="grid-data"' in html
    assert 'data-summary-sha256="' + "c" * 64 + '"' in html
    assert "quotient classes" in html.lower()
    assert "mixed second difference" in html.lower()
    assert "opposite-edge transport" in html.lower()
    assert "lower-corner tangent cosine" in html.lower()
    assert "point source mean absolute" in html.lower()
    assert '<option value="pos_final_output" selected>' in html
    assert "http://" not in html
    assert "https://" not in html
    assert all(f'value="{name}"' in html for name in COMPARED_ARRAYS)


@pytest.mark.parametrize("fault", ["status", "schema", "arrays", "cells", "nonfinite"])
def test_renderer_rejects_untrusted_or_incomplete_summary(fault: str) -> None:
    from scripts.render_shape_block_intervention_grid import GridRenderContractError, validate_summary

    summary = copy.deepcopy(_summary())
    if fault == "status":
        summary["status"] = "failed"
    elif fault == "schema":
        summary["schema"] = "trellis2mlx.shape_block_intervention_grid_summary.v1"
    elif fault == "arrays":
        summary["compared_arrays"].pop()
    elif fault == "cells":
        summary["coordinate_geometry"]["cells"] = []
    else:
        array_name = summary["compared_arrays"][0]
        summary["coordinate_geometry"]["cells"][0]["arrays"][array_name][
            "mixed_second_difference"
        ]["l2_norm"] = math.inf

    with pytest.raises(GridRenderContractError):
        validate_summary(summary)


@pytest.mark.parametrize(
    ("fault", "value"),
    [
        ("partial", None),
        ("shape_flow_trace_block_index", 28),
        ("shape_flow_trace_step_index", 1),
        ("shape_flow_trace_key_selection", "full"),
        ("shape_flow_trace_keys", None),
        ("shape_flow_trace_keys", ["pos_final_output"]),
    ],
)
def test_renderer_rejects_partial_or_wrong_effective_route(fault: str, value: object) -> None:
    from scripts.render_shape_block_intervention_grid import (
        GridRenderContractError,
        validate_summary,
    )

    summary = copy.deepcopy(_summary())
    if fault == "partial":
        summary["route_vector"] = {
            "family": "trellis2mlx/mlx",
            "backend": "mlx-metal",
            "attention_backend": "fast",
        }
    elif value is None:
        summary["route_vector"].pop(fault)
    else:
        summary["route_vector"][fault] = value

    with pytest.raises(GridRenderContractError):
        validate_summary(summary)


def test_renderer_cli_writes_route_bound_report_and_failure_before_html(tmp_path: Path) -> None:
    from scripts.render_shape_block_intervention_grid import main

    summary_path = tmp_path / "summary.json"
    html_path = tmp_path / "grid.html"
    report_path = tmp_path / "render-report.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")

    assert main(
        [
            "--summary-json",
            str(summary_path),
            "--output-html",
            str(html_path),
            "--output-report",
            str(report_path),
        ]
    ) == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "done"
    assert report["summary_json"] == str(summary_path)
    assert report["summary_sha256"] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    assert report["output_html"] == str(html_path)
    assert report["output_html_sha256"] == hashlib.sha256(html_path.read_bytes()).hexdigest()

    broken = _summary()
    broken["status"] = "failed"
    summary_path.write_text(json.dumps(broken), encoding="utf-8")
    html_path.write_text("stale authoritative-looking chart", encoding="utf-8")
    assert main(
        [
            "--summary-json",
            str(summary_path),
            "--output-html",
            str(html_path),
            "--output-report",
            str(report_path),
        ]
    ) == 1
    failure = json.loads(report_path.read_text())
    assert failure["status"] == "failed"
    assert failure["failure_phase"] == "validate_summary"
    assert failure["summary_sha256"] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    assert not html_path.exists()

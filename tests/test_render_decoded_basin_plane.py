import copy
import hashlib
import json
import math
from pathlib import Path

import pytest


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _name(alpha: float, beta: float) -> str:
    def part(value: float) -> str:
        return str(int(value)) if value.is_integer() else str(value).replace(".", "p")

    return f"alpha-{part(alpha)}_beta-{part(beta)}"


def _fixture() -> tuple[dict, dict, list[dict]]:
    coordinates = [
        {"alpha": alpha, "beta": beta}
        for alpha in (0.0, 1.0)
        for beta in (0.0, 1.0)
    ]
    names = [_name(point["alpha"], point["beta"]) for point in coordinates]
    latent_pairs = []
    for left_index, left in enumerate(coordinates):
        for right_index in range(left_index + 1, len(coordinates)):
            right = coordinates[right_index]
            value = 0.1 + left_index * 0.03 + right_index * 0.01
            latent_pairs.append(
                {
                    "a": left,
                    "b": right,
                    "metrics": {
                        "exact": False,
                        "mean_abs": value,
                        "max_abs": value * 2,
                        "l2": value * 3,
                        "rmse": value * 4,
                        "nonzero": 32,
                    },
                }
            )
    latent = {
        "schema": "trellis2mlx.source_cuda_block29_basin_plane_summary.v1",
        "status": "done",
        "primary_sha256": _sha("primary"),
        "report_sha256": _sha("source-report"),
        "receipt_sha256": _sha("source-receipt"),
        "axes": {"alpha": [0.0, 1.0], "beta": [0.0, 1.0]},
        "points": [{"coordinate": coordinate} for coordinate in coordinates],
        "pairwise": latent_pairs,
        "effective_route": {
            "route": "official-source-cuda-full-eight-step-shape-flow-with-fixed-block29-endpoints",
            "device_type": "cuda",
            "cuda_device": "Tesla T4",
            "attention_backend": "sdpa",
            "conv_backend": "none",
            "block_index": 29,
            "step_index": 0,
            "steps": 8,
            "endpoint_semantics": "current + scale * (source - current)",
            "model_ref": "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
            "one_model_load": True,
        },
    }
    source_hashes = {name: _sha(name) for name in names}
    atlas = {
        "schema": "trellis2mlx.mesh_surface_support_atlas.v1",
        "status": "done",
        "route": "shared_grid_vertex_surface_support",
        "embedding_authority": "none",
        "grid_sizes": [32, 64],
        "names": names,
        "reference": names[-1],
        "sources": [
            {"name": name, "sha256": source_hashes[name], "vertices": 100 + index}
            for index, name in enumerate(names)
        ],
        "scales": {
            "32": {
                "occupied_cells": {name: 80 + index for index, name in enumerate(names)},
                "pairwise_jaccard_distance": [
                    [0.0 if row == col else 0.01 + abs(row - col) * 0.001 for col in range(4)]
                    for row in range(4)
                ],
            },
            "64": {
                "occupied_cells": {name: 160 + index for index, name in enumerate(names)},
                "pairwise_jaccard_distance": [
                    [0.0 if row == col else 0.02 + abs(row - col) * 0.001 for col in range(4)]
                    for row in range(4)
                ],
            },
        },
        "forbidden_inferences": [
            "vertex support occupancy is not watertight volume occupancy",
            "Jaccard distance is not global learned-manifold distance",
            "projected support deltas are not topology or winding evidence",
        ],
    }
    reports = []
    for chunk in (names[:2], names[2:]):
        artifacts = []
        for name in chunk:
            for variant in ("raw", "filled"):
                artifacts.append(
                    {
                        "coordinate_key": name,
                        "variant": variant,
                        "status": "written",
                        "sha256": source_hashes[name],
                        **(
                            {"fill_holes_effective_change": False}
                            if variant == "filled"
                            else {}
                        ),
                    }
                )
        reports.append(
            {
                "schema": "trellis2mlx.source_cuda_shape_slat_grid_decode.v1",
                "status": "done",
                "failure_phase": None,
                "selected_point_names": chunk,
                "expected_artifact_count": len(artifacts),
                "written_artifact_count": len(artifacts),
                "mesh_artifacts": artifacts,
                "source_basin_primary": {"sha256": latent["primary_sha256"]},
                "source_basin_report": {"sha256": latent["report_sha256"]},
                "source_basin_route": {
                    "route": latent["effective_route"]["route"],
                    "device_type": "cuda",
                    "cuda_device": "Tesla T4",
                    "attention_backend": "sdpa",
                    "conv_backend": "none",
                    "block_index": 29,
                    "step_index": 0,
                    "steps": 8,
                    "endpoint_semantics": "current + scale * (source - current)",
                    "one_model_load": True,
                },
                "effective_route": {
                    "route": "official-source-cuda-shape-slat-decoder",
                    "device_type": "cuda",
                    "cuda_device": "Tesla T4",
                    "model_ref": "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
                    "model_training": False,
                    "one_model_load": True,
                    "sparse_attention_backend": "sdpa",
                    "sparse_conv_backend": "none",
                    "resolution": 512,
                    "raw_meshes": True,
                    "post_fill_holes_snapshots": True,
                    "fill_holes_effective_change_count": 0,
                },
                "model_load": {
                    "model_ref": "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
                    "training_before_eval": True,
                    "training": False,
                },
                "point_results": [
                    {
                        "coordinate_key": name,
                        "fill_holes_effective_change": False,
                    }
                    for name in chunk
                ],
            }
        )
    return latent, atlas, reports


def test_direct_plane_preserves_coordinates_and_binds_both_routes() -> None:
    from scripts.render_decoded_basin_plane import build_payload, render_html

    latent, atlas, reports = _fixture()
    payload = build_payload(
        latent=latent,
        atlas=atlas,
        decode_reports=reports,
        input_sha256={"latent": _sha("latent-file"), "atlas": _sha("atlas-file")},
        decode_report_sha256=[_sha("decode-a"), _sha("decode-b")],
    )
    html = render_html(payload)

    assert "TRELLIS.2 Latent-to-Geometry Basin Plane" in html
    assert "No PCA or learned embedding" in html
    assert "latent mean absolute" in html
    assert "support Jaccard distance" in html
    assert "alpha-0p5" not in html
    assert 'data-latent-sha256="' + _sha("latent-file") + '"' in html
    assert payload["axes"] == {"alpha": [0.0, 1.0], "beta": [0.0, 1.0]}
    assert payload["source_shape_route"]["block_index"] == 29
    assert payload["geometry_decode_route"]["model_training"] is False
    assert payload["geometry_decode_route"]["resolution"] == 512
    assert payload["geometry_decode_route"]["raw_meshes"] is True
    assert payload["geometry_decode_route"]["post_fill_holes_snapshots"] is True
    assert payload["geometry_decode_route"]["fill_holes_effective_change_count"] == 0
    assert payload["scales"] == [32, 64]
    assert len(payload["pairs"]) == 6


def test_direct_plane_precomputes_every_asymmetric_nearest_relation() -> None:
    from scripts.render_decoded_basin_plane import build_payload

    latent, atlas, reports = _fixture()
    payload = build_payload(
        latent=latent,
        atlas=atlas,
        decode_reports=reports,
        input_sha256={"latent": _sha("latent-file"), "atlas": _sha("atlas-file")},
        decode_report_sha256=[_sha("decode-a"), _sha("decode-b")],
    )

    expected = set()
    for point in payload["points"]:
        options = [
            pair
            for pair in payload["pairs"]
            if point["name"] in (pair["a_name"], pair["b_name"])
        ]
        best = min(options, key=lambda pair: pair["support_jaccard_distance"]["64"])
        expected.add(tuple(sorted((best["a_name"], best["b_name"]))))

    actual = {
        tuple((chord["a_name"], chord["b_name"]))
        for chord in payload["nearest_chords"]["support"]["64"]
    }
    assert actual == expected


def test_direct_plane_normalizes_legacy_paired_mesh_route_from_artifacts() -> None:
    from scripts.render_decoded_basin_plane import build_payload

    latent, atlas, reports = copy.deepcopy(_fixture())
    route = reports[0]["effective_route"]
    route.pop("raw_meshes")
    route.pop("post_fill_holes_snapshots")
    route.pop("fill_holes_effective_change_count")
    route["raw_and_filled_meshes"] = True
    for result in reports[0]["point_results"]:
        result.pop("fill_holes_effective_change")
    for artifact in reports[0]["mesh_artifacts"]:
        artifact.pop("fill_holes_effective_change", None)

    payload = build_payload(
        latent=latent,
        atlas=atlas,
        decode_reports=reports,
        input_sha256={"latent": _sha("latent-file"), "atlas": _sha("atlas-file")},
        decode_report_sha256=[_sha("decode-a"), _sha("decode-b")],
    )

    assert payload["geometry_decode_route"]["raw_meshes"] is True
    assert payload["geometry_decode_route"]["post_fill_holes_snapshots"] is True
    assert payload["geometry_decode_route"]["fill_holes_effective_change_count"] == 0
    assert payload["geometry_decode_route"]["identity_sources"] == [
        "current-explicit",
        "legacy-derived-from-written-pairs",
    ]


def test_direct_plane_first_render_contains_static_evidence() -> None:
    from scripts.render_decoded_basin_plane import build_payload, render_html

    latent, atlas, reports = _fixture()
    payload = build_payload(
        latent=latent,
        atlas=atlas,
        decode_reports=reports,
        input_sha256={"latent": _sha("latent-file"), "atlas": _sha("atlas-file")},
        decode_report_sha256=[_sha("decode-a"), _sha("decode-b")],
    )

    html = render_html(payload)

    assert 'data-static-chart="support-jaccard-64"' in html
    assert html.count('class="edge static"') == 4
    assert html.count('class="node static"') == 4
    assert 'data-static-detail="true"' in html
    assert "latent vs support Pearson" in html
    assert "support distance range" in html


@pytest.mark.parametrize(
    "fault",
    [
        "latent_status",
        "latent_route",
        "atlas_embedding",
        "atlas_matrix_shape",
        "atlas_source_hash",
        "decode_training",
        "missing_decode_point",
        "nonfinite",
        "decode_resolution",
        "decode_fill_count",
    ],
)
def test_direct_plane_rejects_false_closure_inputs(fault: str) -> None:
    from scripts.render_decoded_basin_plane import BasinPlaneContractError, build_payload

    latent, atlas, reports = copy.deepcopy(_fixture())
    if fault == "latent_status":
        latent["status"] = "failed"
    elif fault == "latent_route":
        latent["effective_route"]["device_type"] = "mps"
    elif fault == "atlas_embedding":
        atlas["embedding_authority"] = "pca"
    elif fault == "atlas_matrix_shape":
        atlas["scales"]["64"]["pairwise_jaccard_distance"].pop()
    elif fault == "atlas_source_hash":
        atlas["sources"][0]["sha256"] = _sha("wrong")
    elif fault == "decode_training":
        reports[0]["model_load"]["training"] = True
    elif fault == "missing_decode_point":
        reports[1]["mesh_artifacts"] = reports[1]["mesh_artifacts"][2:]
    elif fault == "nonfinite":
        latent["pairwise"][0]["metrics"]["mean_abs"] = math.inf
    elif fault == "decode_resolution":
        reports[0]["effective_route"]["resolution"] = 256
    else:
        reports[0]["effective_route"]["fill_holes_effective_change_count"] = 1

    with pytest.raises(BasinPlaneContractError):
        build_payload(
            latent=latent,
            atlas=atlas,
            decode_reports=reports,
            input_sha256={"latent": _sha("latent-file"), "atlas": _sha("atlas-file")},
            decode_report_sha256=[_sha("decode-a"), _sha("decode-b")],
        )


@pytest.mark.parametrize("fault", ["counter", "missing_filled"])
def test_direct_plane_rejects_incomplete_decode_accounting(fault: str) -> None:
    from scripts.render_decoded_basin_plane import BasinPlaneContractError, build_payload

    latent, atlas, reports = copy.deepcopy(_fixture())
    if fault == "counter":
        reports[0]["written_artifact_count"] -= 1
    else:
        reports[0]["mesh_artifacts"] = [
            artifact
            for artifact in reports[0]["mesh_artifacts"]
            if not (
                artifact["coordinate_key"] == reports[0]["selected_point_names"][0]
                and artifact["variant"] == "filled"
            )
        ]

    with pytest.raises(BasinPlaneContractError):
        build_payload(
            latent=latent,
            atlas=atlas,
            decode_reports=reports,
            input_sha256={"latent": _sha("latent-file"), "atlas": _sha("atlas-file")},
            decode_report_sha256=[_sha("decode-a"), _sha("decode-b")],
        )


def test_direct_plane_cli_removes_stale_html_and_writes_failure_report(tmp_path: Path) -> None:
    from scripts.render_decoded_basin_plane import main

    latent, atlas, reports = _fixture()
    latent_path = tmp_path / "latent.json"
    atlas_path = tmp_path / "atlas.json"
    report_paths = [tmp_path / "decode-a.json", tmp_path / "decode-b.json"]
    output_html = tmp_path / "plane.html"
    output_report = tmp_path / "plane-report.json"
    latent_path.write_text(json.dumps(latent))
    atlas_path.write_text(json.dumps(atlas))
    for path, report in zip(report_paths, reports):
        path.write_text(json.dumps(report))

    args = [
        "--latent-summary",
        str(latent_path),
        "--mesh-atlas",
        str(atlas_path),
        "--decode-report",
        str(report_paths[0]),
        "--decode-report",
        str(report_paths[1]),
        "--output-html",
        str(output_html),
        "--output-report",
        str(output_report),
    ]
    assert main(args) == 0
    done = json.loads(output_report.read_text())
    assert done["status"] == "done"
    assert done["output_html_sha256"] == hashlib.sha256(output_html.read_bytes()).hexdigest()

    atlas["status"] = "failed"
    atlas_path.write_text(json.dumps(atlas))
    output_html.write_text("stale authoritative chart")
    assert main(args) == 1
    failure = json.loads(output_report.read_text())
    assert failure["status"] == "failed"
    assert failure["failure_phase"] == "validate_inputs"
    assert not output_html.exists()

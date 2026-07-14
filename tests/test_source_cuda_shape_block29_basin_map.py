import json

import numpy as np
import pytest


def test_parse_axis_values_and_cartesian_coordinates_are_uncapped_and_ordered():
    from scripts.source_cuda_shape_block29_basin_map import cartesian_coordinates, parse_axis_values

    alpha = parse_axis_values("0,0.25,0.5,0.75,1", name="alpha")
    beta = parse_axis_values("0,0.5,1", name="beta")
    coordinates = cartesian_coordinates(alpha, beta)

    assert len(coordinates) == 15
    assert coordinates[0] == (0.0, 0.0)
    assert coordinates[-1] == (1.0, 1.0)
    with pytest.raises(ValueError, match="duplicate alpha"):
        parse_axis_values("0,0.5,0.5,1", name="alpha")
    with pytest.raises(ValueError, match="finite beta"):
        parse_axis_values("0,nan,1", name="beta")


def test_interpolate_endpoint_recovers_exact_endpoints_and_midpoint():
    from scripts.source_cuda_shape_block29_basin_map import interpolate_endpoint

    current = np.asarray([[0.0, 2.0, 4.0]], dtype=np.float32)
    source = np.asarray([[4.0, 6.0, 8.0]], dtype=np.float32)

    assert np.array_equal(interpolate_endpoint(current, source, 0.0), current)
    assert np.array_equal(interpolate_endpoint(current, source, 1.0), source)
    assert np.array_equal(
        interpolate_endpoint(current, source, 0.5),
        np.asarray([[2.0, 4.0, 6.0]], dtype=np.float32),
    )


def test_endpoint_target_shape_adapts_flat_raw_attention_to_live_heads():
    from scripts.source_cuda_shape_block29_basin_map import endpoint_target_shape

    assert endpoint_target_shape((7697, 1536), (7697, 12, 128)) == (7697, 12, 128)
    assert endpoint_target_shape((7697, 1536), (7697, 1536)) == (7697, 1536)
    with pytest.raises(ValueError, match="element count"):
        endpoint_target_shape((7697, 1536), (7697, 12, 64))


def test_output_key_is_unique_for_semantic_coordinates():
    from scripts.source_cuda_shape_block29_basin_map import coordinate_key

    keys = {
        coordinate_key(alpha, beta)
        for alpha in (0.0, 0.5, 1.0)
        for beta in (0.0, 0.5, 1.0)
    }
    assert len(keys) == 9
    assert coordinate_key(0.5, 1.0) == "alpha-0p5_beta-1"


def test_execution_order_checks_source_then_current_before_middle_points():
    from scripts.source_cuda_shape_block29_basin_map import prioritized_execution_coordinates

    requested = [
        (alpha, beta)
        for alpha in (0.0, 0.5, 1.0)
        for beta in (0.0, 0.5, 1.0)
    ]
    ordered = prioritized_execution_coordinates(requested)

    assert ordered[:2] == [(1.0, 1.0), (0.0, 0.0)]
    assert len(ordered) == len(requested)
    assert set(ordered) == set(requested)


def test_validate_result_requires_exact_source_control_and_all_requested_points():
    from scripts.source_cuda_shape_block29_basin_map import validate_result_manifest

    coordinates = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    points = [
        {
            "coordinate": {"alpha": alpha, "beta": beta},
            "output_key": f"point_{index}",
            "shape": [2, 4],
            "sha256": str(index) * 64,
        }
        for index, (alpha, beta) in enumerate(coordinates)
    ]
    payload = {
        "status": "done",
        "effective_route": {
            "device_type": "cuda",
            "attention_backend": "sdpa",
            "steps": 8,
            "block_index": 29,
            "step_index": 0,
        },
        "points": points,
        "source_control": {
            "coordinate": {"alpha": 1.0, "beta": 1.0},
            "exact": True,
            "max_abs": 0.0,
            "nonzero": 0,
        },
    }

    validate_result_manifest(payload, coordinates=coordinates)
    payload["source_control"]["exact"] = False
    with pytest.raises(ValueError, match="source control"):
        validate_result_manifest(payload, coordinates=coordinates)


def test_cli_no_download_failure_records_route_and_last_trustworthy_phase(tmp_path):
    from scripts.source_cuda_shape_block29_basin_map import main

    endpoints = tmp_path / "endpoints.npz"
    metadata = {
        "schema": "trellis2mlx.shape_block29_cuda_basin_endpoints.v1",
        "status": "done",
        "comparison_class": "fixed_block29_endpoint_affine_plane",
        "endpoint_semantics": "current + scale * (source - current)",
        "steps": 8,
        "block_index": 29,
        "step_index": 0,
    }
    np.savez(
        endpoints,
        coords=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = tmp_path / "result.json"
    output = tmp_path / "result.npz"
    status = main(
        [
            "--output-json",
            str(report),
            "--output-npz",
            str(output),
            "--endpoints",
            str(endpoints),
            "--conditioning",
            str(tmp_path / "missing-conditioning.npz"),
            "--shape-flow-noise-sample",
            str(tmp_path / "missing-noise.npz"),
            "--source-tar",
            str(tmp_path / "missing-source.tar.gz"),
            "--no-download",
        ]
    )

    assert status == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["failure_phase"] == "input_validation"
    assert payload["last_trustworthy_phase"] == "arguments_parsed"
    assert payload["requested_route"]["alphas"] == [0.0, 0.5, 1.0]
    assert payload["primary_output_status"] == "missing"

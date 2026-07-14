import json
import hashlib

import numpy as np
import pytest


BRANCH_STAGE_KEYS = (
    "pos_block29_after_self",
    "pos_block29_cross_attention_raw",
    "neg_block29_after_self",
    "neg_block29_cross_attention_raw",
)


def _write_endpoint_packet(path, *, corrupt_key=None):
    coords = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    arrays = {"coords": coords}
    endpoint_digests = {}
    for index, key in enumerate(BRANCH_STAGE_KEYS):
        endpoint_digests[key] = {}
        for endpoint_index, endpoint in enumerate(("current", "source")):
            values = np.full((1, 1, 4), index + endpoint_index + 1, dtype=np.float32)
            words = (values.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
            endpoint_digests[key][f"{endpoint}_float32_sha256"] = hashlib.sha256(
                values.tobytes()
            ).hexdigest()
            packed_key = f"{key}_{endpoint}_bf16_words"
            if packed_key == corrupt_key:
                words = words.copy()
                words.reshape(-1)[0] ^= np.uint16(1)
            arrays[packed_key] = words
    metadata = {
        "schema": "trellis2mlx.shape_block29_cuda_basin_endpoints.v1",
        "status": "done",
        "comparison_class": "fixed_block29_endpoint_affine_plane",
        "endpoint_semantics": "current + scale * (source - current)",
        "steps": 8,
        "block_index": 29,
        "step_index": 0,
        "endpoint_digests": endpoint_digests,
        "current_route": {
            "conditioning_sample_sha256": "1" * 64,
            "shape_flow_noise_sample_sha256": "2" * 64,
        },
        "source_route": {"source_tar_sha256": "3" * 64},
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(path, **arrays)


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
    assert payload["last_trustworthy_phase"] == "request_validation"
    assert payload["requested_route"]["alphas"] == [0.0, 0.5, 1.0]
    assert payload["primary_output_status"] == "missing"


def test_cli_rejects_same_shaped_corrupt_endpoint_before_primary_output(tmp_path):
    from scripts.source_cuda_shape_block29_basin_map import main

    endpoints = tmp_path / "corrupt-endpoints.npz"
    _write_endpoint_packet(
        endpoints,
        corrupt_key="pos_block29_after_self_current_bf16_words",
    )
    conditioning = tmp_path / "conditioning.npz"
    noise = tmp_path / "noise.npz"
    source_tar = tmp_path / "source.tar.gz"
    np.savez(conditioning, cond=np.zeros((1, 1, 1)), neg_cond=np.zeros((1, 1, 1)))
    np.savez(noise, noise=np.zeros((1, 1)), coords=np.asarray([[0, 1, 2, 3]]))
    source_tar.write_bytes(b"not blank")
    report = tmp_path / "failure.json"
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
            str(conditioning),
            "--shape-flow-noise-sample",
            str(noise),
            "--source-tar",
            str(source_tar),
            "--no-download",
        ]
    )

    assert status == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["failure_phase"] == "input_validation"
    assert payload["primary_output_status"] == "missing"
    assert "endpoint digest mismatch" in payload["error"]
    assert not output.exists()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--alphas", "0,0.5,0.5,1", "duplicate alpha"),
        ("--betas", "0,nan,1", "finite beta"),
    ],
)
def test_cli_malformed_axes_write_request_validation_report(tmp_path, flag, value, message):
    from scripts.source_cuda_shape_block29_basin_map import main

    report = tmp_path / f"{flag[2:]}-failure.json"
    output = tmp_path / f"{flag[2:]}-result.npz"
    status = main(
        [
            "--output-json",
            str(report),
            "--output-npz",
            str(output),
            "--endpoints",
            str(tmp_path / "missing-endpoints.npz"),
            "--conditioning",
            str(tmp_path / "missing-conditioning.npz"),
            "--shape-flow-noise-sample",
            str(tmp_path / "missing-noise.npz"),
            "--source-tar",
            str(tmp_path / "missing-source.tar.gz"),
            flag,
            value,
        ]
    )

    assert status == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["failure_phase"] == "request_validation"
    assert payload["last_trustworthy_phase"] == "arguments_parsed"
    assert payload["primary_output_status"] == "missing"
    assert message in payload["error"]
    assert not output.exists()

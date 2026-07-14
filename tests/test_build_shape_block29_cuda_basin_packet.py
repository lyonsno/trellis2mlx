import json
from pathlib import Path

import numpy as np
import pytest


BRANCH_STAGE_KEYS = (
    "pos_block29_after_self",
    "pos_block29_cross_attention_raw",
    "neg_block29_after_self",
    "neg_block29_cross_attention_raw",
)


def _write_trace(path: Path, *, offset: float = 0.0) -> None:
    coords = np.asarray([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    arrays = {
        "coords": coords,
        "trace_block_index": np.asarray(29, dtype=np.int32),
        "shape_flow_trace_step_index": np.asarray(0, dtype=np.int32),
        "steps": np.asarray(8, dtype=np.int32),
    }
    for index, key in enumerate(BRANCH_STAGE_KEYS):
        values = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
        arrays[key] = (values + np.float32(index + offset)).astype(np.float32)
    np.savez(path, **arrays)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_grid_summary(path: Path, current_trace: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.shape_block_intervention_grid_summary.v2",
                "status": "done",
                "comparison_class": "block29_after_self_cross_attention_raw_delta_grid",
                "axes": {"alpha": [0.0, 0.5, 1.0], "beta": [0.0, 0.5, 1.0]},
                "points": [
                    {
                        "coordinate": {"alpha": 0.0, "beta": 0.0},
                        "artifact": str(current_trace),
                        "artifact_sha256": _sha256(current_trace),
                        "route": {
                            "backend": "mlx-metal",
                            "family": "trellis2mlx/mlx",
                            "attention_backend": "fast",
                            "shape_flow_trace_block_index": 29,
                            "shape_flow_trace_step_index": 0,
                            "shape_flow_trace_key_selection": "explicit",
                            "shape_flow_trace_keys": list(BRANCH_STAGE_KEYS),
                            "steps": 8,
                            "conditioning_sample_sha256": "1" * 64,
                            "shape_slat_support_sample_sha256": "2" * 64,
                            "shape_flow_noise_sample_sha256": "3" * 64,
                        },
                    }
                ],
            }
        )
        + "\n"
    )


def _write_source_report(path: Path, source_trace: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.source_cuda_shape_block_trace.v1",
                "status": "done",
                "primary_output_status": "written",
                "primary_output": {"path": str(source_trace), "sha256": _sha256(source_trace)},
                "route_identity": {
                    "backend": "source-trellis",
                    "device": "cuda",
                    "effective_device_type": "cuda",
                    "effective_route": "official-trellis2-source-cuda-shape-flow-block-trace",
                    "branch": "both",
                    "block_indices": [29],
                    "shape_flow_trace_step_index": 0,
                    "steps": 8,
                    "conditioning_sha256": "1" * 64,
                    "shape_slat_support_sample_sha256": "2" * 64,
                    "shape_flow_noise_sample_sha256": "3" * 64,
                    "source_tar_sha256": "4" * 64,
                },
            }
        )
        + "\n"
    )


def test_build_packet_packs_exact_bf16_endpoints_and_binds_routes(tmp_path):
    from scripts.build_shape_block29_cuda_basin_packet import build_packet, decode_bf16_words

    current = tmp_path / "current.npz"
    source = tmp_path / "source.npz"
    summary = tmp_path / "grid-summary.json"
    source_report = tmp_path / "source-report.json"
    output = tmp_path / "endpoints.npz"
    report = tmp_path / "endpoints.json"
    _write_trace(current)
    _write_trace(source, offset=8.0)
    _write_grid_summary(summary, current)
    _write_source_report(source_report, source)

    result = build_packet(
        grid_summary_path=summary,
        source_report_path=source_report,
        output_npz=output,
        output_json=report,
    )

    assert result["status"] == "done"
    assert result["comparison_class"] == "fixed_block29_endpoint_affine_plane"
    assert result["endpoint_semantics"] == "current + scale * (source - current)"
    assert result["source_route"]["effective_device_type"] == "cuda"
    assert result["current_route"]["attention_backend"] == "fast"
    with np.load(output, allow_pickle=False) as packed:
        assert packed["coords"].tolist() == [[0, 1, 2, 3], [0, 4, 5, 6]]
        for key in BRANCH_STAGE_KEYS:
            current_key = f"{key}_current_bf16_words"
            source_key = f"{key}_source_bf16_words"
            assert packed[current_key].dtype == np.uint16
            assert packed[source_key].dtype == np.uint16
            with np.load(current, allow_pickle=False) as original:
                assert np.array_equal(decode_bf16_words(packed[current_key]), original[key])
            with np.load(source, allow_pickle=False) as original:
                assert np.array_equal(decode_bf16_words(packed[source_key]), original[key])


def test_build_packet_rejects_source_route_and_digest_lies(tmp_path):
    from scripts.build_shape_block29_cuda_basin_packet import build_packet

    current = tmp_path / "current.npz"
    source = tmp_path / "source.npz"
    summary = tmp_path / "grid-summary.json"
    source_report = tmp_path / "source-report.json"
    _write_trace(current)
    _write_trace(source, offset=8.0)
    _write_grid_summary(summary, current)
    _write_source_report(source_report, source)

    payload = json.loads(source_report.read_text())
    payload["route_identity"]["effective_device_type"] = "cpu"
    source_report.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="effective_device_type='cuda'"):
        build_packet(
            grid_summary_path=summary,
            source_report_path=source_report,
            output_npz=tmp_path / "bad-route.npz",
            output_json=tmp_path / "bad-route.json",
        )

    _write_source_report(source_report, source)
    payload = json.loads(source_report.read_text())
    payload["primary_output"]["sha256"] = "0" * 64
    source_report.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="source trace digest"):
        build_packet(
            grid_summary_path=summary,
            source_report_path=source_report,
            output_npz=tmp_path / "bad-digest.npz",
            output_json=tmp_path / "bad-digest.json",
        )


def test_encode_bf16_words_rejects_non_bf16_aligned_values():
    from scripts.build_shape_block29_cuda_basin_packet import encode_bf16_words

    with pytest.raises(ValueError, match="not exactly representable as BF16"):
        encode_bf16_words(np.asarray([1.0001], dtype=np.float32), name="not-bf16")


def test_cli_failure_before_primary_output_still_writes_report(tmp_path):
    from scripts.build_shape_block29_cuda_basin_packet import main

    output = tmp_path / "missing-primary.npz"
    report = tmp_path / "failure.json"
    status = main(
        [
            "--grid-summary",
            str(tmp_path / "missing-summary.json"),
            "--source-report",
            str(tmp_path / "missing-source.json"),
            "--output-npz",
            str(output),
            "--output-json",
            str(report),
        ]
    )

    assert status == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["failure_phase"] == "input_validation"
    assert payload["primary_output_status"] == "missing"
    assert not output.exists()

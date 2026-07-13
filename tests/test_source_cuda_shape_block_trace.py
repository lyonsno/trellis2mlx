import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import source_cuda_shape_block_trace as trace


def test_route_identity_rejects_non_cuda_device(tmp_path):
    with pytest.raises(RuntimeError, match="CUDA"):
        trace.build_route_identity(
            device_type="cpu",
            output_npz=tmp_path / "trace.npz",
            conditioning_path=tmp_path / "conditioning.npz",
            support_sample_path=tmp_path / "support.npz",
            noise_sample_path=tmp_path / "noise.npz",
            block_indices=[0],
            trace_names=["after_self"],
            steps=8,
            seed=42,
            branch="both",
        )


def test_route_identity_records_effective_inputs_and_trace_scope(tmp_path):
    route = trace.build_route_identity(
        device_type="cuda",
        output_npz=tmp_path / "trace.npz",
        conditioning_path=tmp_path / "conditioning.npz",
        support_sample_path=tmp_path / "support.npz",
        noise_sample_path=tmp_path / "noise.npz",
        block_indices=[0, 7],
        trace_names=["after_self", "after_mlp"],
        steps=8,
        seed=42,
        branch="both",
    )

    assert route["backend"] == "source-trellis"
    assert route["device"] == "cuda"
    assert route["primary_output"] == str(tmp_path / "trace.npz")
    assert route["conditioning_sample"] == str(tmp_path / "conditioning.npz")
    assert route["shape_slat_support_sample"] == str(tmp_path / "support.npz")
    assert route["shape_flow_noise_sample"] == str(tmp_path / "noise.npz")
    assert route["shape_flow_trace_block_indices"] == [0, 7]
    assert route["trace_names"] == ["after_self", "after_mlp"]
    assert route["steps"] == 8
    assert route["seed"] == 42
    assert route["branch"] == "both"
    assert route["block_input_replay_sample"] is None
    assert route["block_input_replay_scope"] == []


def test_route_identity_records_block_input_replay_scope(tmp_path):
    replay = tmp_path / "mlx-block-inputs.npz"
    route = trace.build_route_identity(
        device_type="cuda",
        output_npz=tmp_path / "trace.npz",
        conditioning_path=tmp_path / "conditioning.npz",
        support_sample_path=tmp_path / "support.npz",
        noise_sample_path=tmp_path / "noise.npz",
        block_indices=[1],
        trace_names=["input", "after_mlp"],
        steps=8,
        seed=42,
        branch="both",
        block_input_replay_sample=replay,
        block_input_replay_scope=["pos_block1_input", "neg_block1_input"],
    )

    assert route["effective_route"].endswith("-with-captured-block-input-replay")
    assert route["block_input_replay_sample"] == str(replay)
    assert route["block_input_replay_scope"] == ["pos_block1_input", "neg_block1_input"]
    assert "not a captured-MLX-block-input replay" not in route["forbidden_inferences"]


def test_route_identity_records_block_stage_replay_scope(tmp_path):
    replay = tmp_path / "mlx-block-stages.npz"
    route = trace.build_route_identity(
        device_type="cuda",
        output_npz=tmp_path / "trace.npz",
        conditioning_path=tmp_path / "conditioning.npz",
        support_sample_path=tmp_path / "support.npz",
        noise_sample_path=tmp_path / "noise.npz",
        block_indices=[1],
        trace_names=["after_cross", "after_mlp"],
        steps=8,
        seed=42,
        branch="both",
        block_stage_replay_sample=replay,
        block_stage_replay_scope=["pos_block1_after_cross", "neg_block1_after_cross"],
    )

    assert route["effective_route"].endswith("-with-captured-block-stage-replay")
    assert route["block_stage_replay_sample"] == str(replay)
    assert route["block_stage_replay_scope"] == [
        "pos_block1_after_cross",
        "neg_block1_after_cross",
    ]
    assert "not a captured-MLX-block-stage replay" not in route["forbidden_inferences"]


def test_load_support_and_noise_rejects_coordinate_mismatch(tmp_path):
    support = tmp_path / "support.npz"
    noise = tmp_path / "noise.npz"
    np.savez(
        support,
        coords=np.array([[0, 1, 2, 3], [0, 2, 3, 4]], dtype=np.int32),
        coords_3d=np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int32),
    )
    np.savez(
        noise,
        coords=np.array([[0, 1, 2, 3], [0, 9, 9, 9]], dtype=np.int32),
        noise=np.zeros((2, 8), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="coordinate"):
        trace.load_support_and_noise(support, noise)


def test_load_support_and_noise_accepts_legacy_latents_key(tmp_path):
    support = tmp_path / "support.npz"
    noise = tmp_path / "noise.npz"
    coords = np.array([[0, 1, 2, 3], [0, 2, 3, 4]], dtype=np.int32)
    coords_3d = coords[:, 1:]
    latents = np.arange(16, dtype=np.float32).reshape(2, 8)
    np.savez(support, coords=coords, coords_3d=coords_3d)
    np.savez(noise, coords=coords, latents=latents)

    loaded = trace.load_support_and_noise(support, noise)

    assert np.array_equal(loaded.coords, coords)
    assert np.array_equal(loaded.coords_3d, coords_3d)
    assert np.array_equal(loaded.noise, latents)


def test_load_block_input_replay_accepts_batched_requested_inputs(tmp_path):
    replay = tmp_path / "replay.npz"
    pos = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
    neg = np.arange(12, 24, dtype=np.float32).reshape(2, 6)
    np.savez(replay, pos_block1_input=pos, neg_block1_input=neg)

    loaded = trace.load_block_input_replay(
        replay,
        branches=["pos", "neg"],
        block_indices=[1],
        token_count=2,
    )

    assert set(loaded.arrays) == {("pos", 1), ("neg", 1)}
    assert np.array_equal(loaded.arrays[("pos", 1)], pos[0])
    assert np.array_equal(loaded.arrays[("neg", 1)], neg)
    assert loaded.scope == ["pos_block1_input", "neg_block1_input"]


def test_load_block_input_replay_rejects_missing_requested_key(tmp_path):
    replay = tmp_path / "replay.npz"
    np.savez(replay, pos_block1_input=np.zeros((1, 2, 6), dtype=np.float32))

    with pytest.raises(ValueError, match="missing neg_block1_input"):
        trace.load_block_input_replay(
            replay,
            branches=["pos", "neg"],
            block_indices=[1],
            token_count=2,
        )


def test_load_block_input_replay_rejects_wrong_token_count(tmp_path):
    replay = tmp_path / "replay.npz"
    np.savez(replay, pos_block1_input=np.zeros((1, 3, 6), dtype=np.float32))

    with pytest.raises(ValueError, match="token count"):
        trace.load_block_input_replay(
            replay,
            branches=["pos"],
            block_indices=[1],
            token_count=2,
        )


def test_load_block_stage_replay_accepts_requested_stages(tmp_path):
    replay = tmp_path / "replay.npz"
    pos = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
    neg = np.arange(12, 24, dtype=np.float32).reshape(2, 6)
    np.savez(replay, pos_block1_after_cross=pos, neg_block1_after_cross=neg)

    loaded = trace.load_block_stage_replay(
        replay,
        branches=["pos", "neg"],
        block_indices=[1],
        stages=["after_cross"],
        token_count=2,
    )

    assert set(loaded.arrays) == {("pos", 1, "after_cross"), ("neg", 1, "after_cross")}
    assert np.array_equal(loaded.arrays[("pos", 1, "after_cross")], pos[0])
    assert np.array_equal(loaded.arrays[("neg", 1, "after_cross")], neg)
    assert loaded.scope == ["pos_block1_after_cross", "neg_block1_after_cross"]


def test_load_block_stage_replay_rejects_unknown_stage(tmp_path):
    replay = tmp_path / "replay.npz"
    np.savez(replay, pos_block1_after_cross=np.zeros((1, 2, 6), dtype=np.float32))

    with pytest.raises(ValueError, match="unsupported block stage replay"):
        trace.load_block_stage_replay(
            replay,
            branches=["pos"],
            block_indices=[1],
            stages=["attention_raw"],
            token_count=2,
        )


def test_load_block_stage_replay_rejects_missing_requested_key(tmp_path):
    replay = tmp_path / "replay.npz"
    np.savez(replay, pos_block1_after_cross=np.zeros((1, 2, 6), dtype=np.float32))

    with pytest.raises(ValueError, match="missing neg_block1_after_cross"):
        trace.load_block_stage_replay(
            replay,
            branches=["pos", "neg"],
            block_indices=[1],
            stages=["after_cross"],
            token_count=2,
        )


def test_trace_names_parser_preserves_order_and_all_expansion():
    assert trace.parse_trace_names("after_self,after_mlp") == ["after_self", "after_mlp"]
    assert trace.parse_trace_names(["after_self", "after_mlp"]) == ["after_self", "after_mlp"]
    assert "attention_raw" in trace.parse_trace_names("all")


def test_apply_sparse_backend_env_sets_source_import_contract(monkeypatch):
    for key in ("SPARSE_CONV_BACKEND", "SPARSE_ATTN_BACKEND", "ATTN_BACKEND"):
        monkeypatch.delenv(key, raising=False)

    forced = trace.apply_sparse_backend_env("none", "sdpa")

    assert forced == {
        "SPARSE_CONV_BACKEND": "none",
        "SPARSE_ATTN_BACKEND": "sdpa",
        "ATTN_BACKEND": "sdpa",
    }
    assert trace.os.environ["SPARSE_CONV_BACKEND"] == "none"
    assert trace.os.environ["SPARSE_ATTN_BACKEND"] == "sdpa"
    assert trace.os.environ["ATTN_BACKEND"] == "sdpa"


def test_cli_failure_writes_durable_report(tmp_path):
    script = Path(trace.__file__)
    report = tmp_path / "report.json"
    output_npz = tmp_path / "trace.npz"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-json",
            str(report),
            "--output-npz",
            str(output_npz),
            "--conditioning",
            str(tmp_path / "missing-conditioning.npz"),
            "--shape-slat-support-sample",
            str(tmp_path / "missing-support.npz"),
            "--shape-flow-noise-sample",
            str(tmp_path / "missing-noise.npz"),
            "--source-tar",
            str(tmp_path / "missing-source.tar.gz"),
            "--no-download",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert report.exists()
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["primary_output_status"] == "missing"
    assert payload["failure_phase"]
    assert payload["last_trustworthy_phase"] in {
        "arguments_parsed",
        "route_identity_written",
        "input_validation",
    }

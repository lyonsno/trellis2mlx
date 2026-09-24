import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from trellmlx.checkpoint import has_checkpoint, load_checkpoint
from trellmlx.hr_recovery import run_hr_flow, resume_hr_flow, replay_hr_input


class ConstantModel:
    def __call__(self, sample, timestep, cond, **kwargs):
        return mx.ones_like(sample)


class FailOnceModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, sample, timestep, cond, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("synthetic Metal failure")
        return mx.ones_like(sample)


class FailingCacheModel(ConstantModel):
    def build_cross_kv_cache(self, cond):
        raise RuntimeError("synthetic cache failure")


class DoubleModel:
    def __call__(self, sample, timestep, cond, **kwargs):
        return mx.ones_like(sample) * 2


class MustNotRunModel:
    def __call__(self, sample, timestep, cond, **kwargs):
        raise AssertionError("HR Euler must not start")


def _inputs():
    return (
        mx.array(np.full((3, 32), 2, dtype=np.float32)),
        mx.zeros((1, 1, 4)),
        mx.zeros((3, 3), dtype=mx.int32),
    )


def test_hr_failure_preserves_exact_input_and_replays_completed_steps(tmp_path):
    noise, cond, coords = _inputs()
    directory = str(tmp_path / "checkpoint")
    params = dict(steps=3, guidance_strength=1.0, guidance_rescale=0.0,
                  guidance_interval=(0.6, 1.0), rescale_t=3.0)
    with pytest.raises(RuntimeError, match="synthetic Metal failure"):
        run_hr_flow(FailOnceModel(), noise, cond, cond, coords,
                    checkpoint_dir=directory, sampler=params,
                    runtime={"mlx_version": "test"})
    assert has_checkpoint(directory, "hr_flow_input")
    assert has_checkpoint(directory, "hr_flow_step_000")
    assert not has_checkpoint(directory, "hr_flow_step_001")
    np.testing.assert_array_equal(load_checkpoint(directory, "hr_flow_input")["noise"], noise)
    with open(tmp_path / "checkpoint" / "_control" / "failure.json") as stream:
        failure = json.load(stream)
    assert failure["stage"] == "hr_flow"
    assert failure["last_complete_step"] == 0
    assert failure["active_step"] == 1
    assert "synthetic Metal failure" in failure["error"]
    resumed = resume_hr_flow(ConstantModel(), directory,
                             expected_runtime={"mlx_version": "test"})
    from trellmlx.samplers import flow_euler_sample
    full = flow_euler_sample(ConstantModel(), noise, cond, cond,
                             coords=coords, verbose=False, **params)
    np.testing.assert_array_equal(np.array(resumed), np.array(full))


def test_hr_failure_receipt_names_pre_step_cache_build(tmp_path):
    noise, cond, coords = _inputs()
    directory = str(tmp_path / "checkpoint")
    params = dict(steps=2, guidance_strength=1.0, guidance_rescale=0.0,
                  guidance_interval=(0.6, 1.0), rescale_t=3.0)
    with pytest.raises(RuntimeError, match="synthetic cache failure"):
        run_hr_flow(FailingCacheModel(), noise, cond, cond, coords,
                    checkpoint_dir=directory, sampler=params,
                    runtime={"mlx_version": "test"})
    failure = json.loads((tmp_path / "checkpoint" / "_control" / "failure.json").read_text())
    assert failure["active_phase"] == "positive_cache_build"
    assert failure["active_step"] is None
    assert has_checkpoint(directory, "hr_flow_input")


def test_hr_resume_rejects_runtime_mismatch(tmp_path):
    noise, cond, coords = _inputs()
    directory = str(tmp_path / "checkpoint")
    params = dict(steps=1, guidance_strength=1.0, guidance_rescale=0.0,
                  guidance_interval=(0.6, 1.0), rescale_t=3.0)
    run_hr_flow(ConstantModel(), noise, cond, cond, coords,
                checkpoint_dir=directory, sampler=params,
                runtime={"mlx_version": "0.32.2"})
    with pytest.raises(ValueError, match="runtime mismatch"):
        resume_hr_flow(ConstantModel(), directory,
                       expected_runtime={"mlx_version": "0.31.2"})


def test_hr_resume_rejects_noncontiguous_completed_steps(tmp_path):
    noise, cond, coords = _inputs()
    directory = str(tmp_path / "checkpoint")
    params = dict(steps=3, guidance_strength=1.0, guidance_rescale=0.0,
                  guidance_interval=(0.6, 1.0), rescale_t=3.0)
    run_hr_flow(ConstantModel(), noise, cond, cond, coords,
                checkpoint_dir=directory, sampler=params,
                runtime={"mlx_version": "test"})
    for suffix in (".npz", ".json", ".complete.json"):
        (tmp_path / "checkpoint" / f"hr_flow_step_001{suffix}").unlink()
    with pytest.raises(ValueError, match="noncontiguous"):
        resume_hr_flow(ConstantModel(), directory,
                       expected_runtime={"mlx_version": "test"})


def test_hr_resume_rejects_step_copied_from_same_shaped_other_run(tmp_path):
    noise, cond, coords = _inputs()
    params = dict(steps=2, guidance_strength=1.0, guidance_rescale=0.0,
                  guidance_interval=(0.6, 1.0), rescale_t=3.0)
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_hr_flow(ConstantModel(), noise, cond, cond, coords,
                checkpoint_dir=str(first), sampler=params,
                runtime={"mlx_version": "test"})
    run_hr_flow(ConstantModel(), noise + 1, cond, cond, coords,
                checkpoint_dir=str(second), sampler=params,
                runtime={"mlx_version": "test"})
    for suffix in (".npz", ".json", ".complete.json"):
        (first / f"hr_flow_step_000{suffix}").write_bytes(
            (second / f"hr_flow_step_000{suffix}").read_bytes()
        )
    with pytest.raises(ValueError, match="different HR input"):
        resume_hr_flow(ConstantModel(), str(first),
                       expected_runtime={"mlx_version": "test"})


def test_runtime_identity_tracks_env_only_attention_route(tmp_path, monkeypatch):
    from generate import _hr_runtime_identity
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"test")
    args = SimpleNamespace(quantize=None, compile=False)
    monkeypatch.setenv("TRELLIS2MLX_ATTENTION_BACKEND", "fast")
    first = _hr_runtime_identity(args, str(weights), None)
    monkeypatch.setenv("TRELLIS2MLX_ATTENTION_BACKEND", "manual")
    second = _hr_runtime_identity(args, str(weights), None)
    assert first["shape_attention_route"] != second["shape_attention_route"]


def test_runtime_identity_tracks_weight_bytes_not_just_stat(tmp_path):
    from generate import _hr_runtime_identity
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"aaaa")
    args = SimpleNamespace(quantize=None, compile=False)
    first = _hr_runtime_identity(args, str(weights), None)
    stat = weights.stat()
    weights.write_bytes(b"bbbb")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = _hr_runtime_identity(args, str(weights), None)
    assert first["weights_sha256"] != second["weights_sha256"]


def test_changed_route_replays_saved_input_as_isolated_new_trajectory(tmp_path):
    from trellmlx.checkpoint import prepare_new_checkpoint_dir
    noise, cond, coords = _inputs()
    source = str(tmp_path / "source")
    destination = str(tmp_path / "destination")
    params = dict(steps=2, guidance_strength=1.0, guidance_rescale=0.0,
                  guidance_interval=(0.6, 1.0), rescale_t=3.0)
    original = run_hr_flow(ConstantModel(), noise, cond, cond, coords,
                           checkpoint_dir=source, sampler=params,
                           runtime={"route": "old"})
    source_step = load_checkpoint(source, "hr_flow_step_000")["sample_next"].copy()
    prepare_new_checkpoint_dir(destination)
    replayed = replay_hr_input(DoubleModel(), source, destination,
                               runtime={"route": "repaired"})
    assert not np.array_equal(np.array(replayed), np.array(original))
    np.testing.assert_array_equal(load_checkpoint(source, "hr_flow_step_000")["sample_next"], source_step)
    input_data = load_checkpoint(destination, "hr_flow_input")
    np.testing.assert_array_equal(input_data["noise"], np.array(noise))
    provenance = json.loads(input_data["replay_provenance_json"])
    assert provenance["kind"] == "new_trajectory_from_saved_input"
    assert provenance["source_runtime"] == {"route": "old"}
    assert provenance["new_runtime"] == {"route": "repaired"}
    assert not np.array_equal(load_checkpoint(destination, "hr_flow_step_000")["sample_next"], source_step)


def test_stop_after_hr_input_saves_real_replay_boundary_before_euler(tmp_path):
    noise, cond, coords = _inputs()
    directory = str(tmp_path / "checkpoint")
    result = run_hr_flow(
        MustNotRunModel(), noise, cond, cond, coords,
        checkpoint_dir=directory,
        sampler={"steps": 2, "guidance_strength": 1.0},
        runtime={"route": "test"}, stop_after_input=True,
    )
    assert result is None
    assert has_checkpoint(directory, "hr_flow_input")
    assert not has_checkpoint(directory, "hr_flow_step_000")


def test_generate_cli_refuses_partial_resume_without_starting_inference(tmp_path):
    directory = str(tmp_path / "checkpoint")
    from trellmlx.checkpoint import save_checkpoint
    save_checkpoint(directory, "conditioning", cond=np.ones((1, 2, 3)))
    save_checkpoint(directory, "sparse_coords", coords=np.zeros((2, 4)))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "generate.py"),
         "--resume", directory],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "no supported resume boundary" in result.stderr
    assert "No image — random conditioning" not in result.stdout


def test_generate_cli_does_not_call_hr_only_recovery_full_resume(tmp_path):
    directory = str(tmp_path / "checkpoint")
    from trellmlx.checkpoint import save_checkpoint
    save_checkpoint(directory, "hr_flow_input", noise=np.zeros((2, 32)),
                    cond=np.zeros((1, 1, 4)), neg_cond=np.zeros((1, 1, 4)),
                    coords=np.zeros((2, 3)), quant_coords=np.zeros((2, 4)),
                    mesh_grid_size=768,
                    sampler_json=json.dumps({"steps": 1}),
                    runtime_json=json.dumps({"mlx_version": "test"}))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "generate.py"),
         "--resume", directory], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "--recover-hr-only" in result.stderr
    assert "No image — random conditioning" not in result.stdout


def test_generate_failure_before_mesh_writes_pipeline_report(tmp_path):
    directory = str(tmp_path / "checkpoint")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "generate.py"),
         "--image", str(tmp_path / "source.png"),
         "--edit-target", str(tmp_path / "missing.png"),
         "--save-checkpoints", directory],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    report_path = tmp_path / "checkpoint" / "_control" / "pipeline_failure.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["last_trustworthy_stages"] == []
    assert "edit-target path does not exist" in report["error"]

"""CPU checks for saved natural LR input and opt-in synchronization."""

import argparse
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import generate
from trellmlx import hr_recovery, samplers
from trellmlx.checkpoint import has_checkpoint, load_checkpoint


@pytest.fixture(autouse=True)
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_lr_input_is_saved_before_first_model_call_and_each_step_is_bound(tmp_path, monkeypatch):
    noise = mx.full((3, 32), 2.0)
    cond = mx.zeros((1, 1, 4))
    coords = mx.zeros((3, 3), mx.int32)
    seen = []

    class Model:
        def __call__(self, sample, timestep, conditioning, **kwargs):
            data = load_checkpoint(str(tmp_path), "lr_flow_input")
            np.testing.assert_array_equal(data["noise"], np.array(noise))
            np.testing.assert_array_equal(data["coords"], np.array(coords))
            seen.append(np.array(sample))
            return mx.ones_like(sample)

    monkeypatch.setattr(mx, "synchronize", lambda: None)
    result = hr_recovery.run_lr_flow(
        Model(), noise, cond, cond, coords, checkpoint_dir=str(tmp_path),
        sampler={"steps": 2, "guidance_strength": 1.0}, runtime={"route": "fixture"},
        synchronize_gpu=True,
    )
    data = load_checkpoint(str(tmp_path), "lr_flow_input")
    assert json.loads(data["runtime_json"])["gpu_synchronization"] == "per-model-block-and-sampler-calculation"
    np.testing.assert_array_equal(seen[0], data["noise"])
    for index in range(2):
        step = load_checkpoint(str(tmp_path), f"lr_flow_step_{index:03d}")
        assert step["input_identity"] == hr_recovery._input_identity(str(tmp_path), stage="lr_flow")
        assert step["step_index"] == index
    np.testing.assert_array_equal(result, step["sample_next"])
    assert not has_checkpoint(str(tmp_path), "hr_flow_input")


def test_lr_failure_keeps_exact_input_and_names_current_block(tmp_path, monkeypatch):
    class Model:
        def __call__(self, sample, timestep, cond, **kwargs):
            kwargs["execution_barrier"]("block.0", sample)
            kwargs["on_execution_phase"]("block.1")
            raise RuntimeError("synthetic internal block failure")

    monkeypatch.setattr(mx, "synchronize", lambda: None)
    with pytest.raises(RuntimeError, match="synthetic internal block failure"):
        hr_recovery.run_lr_flow(
            Model(), mx.ones((3, 32)), mx.zeros((1, 1, 4)), mx.zeros((1, 1, 4)),
            mx.zeros((3, 3), mx.int32), checkpoint_dir=str(tmp_path),
            sampler={"steps": 2, "guidance_strength": 1.0}, runtime={"route": "fixture"},
            synchronize_gpu=True,
        )
    failure = json.loads((tmp_path / "_control/failure.json").read_text())
    assert failure["stage"] == "lr_flow"
    assert failure["active_step"] == 0
    assert failure["active_phase"] == "positive_forward.block.1"
    assert failure["last_completed_gpu_boundary"] == {"step": 0, "phase": "positive_forward.block.0"}
    assert failure["last_complete_step"] == -1
    assert has_checkpoint(str(tmp_path), "lr_flow_input")
    assert not has_checkpoint(str(tmp_path), "lr_flow_step_000")


@pytest.mark.parametrize("enabled", [False, True])
def test_natural_lr_call_selects_diagnostic_only_when_requested(tmp_path, monkeypatch, enabled):
    calls = []
    marker = object()
    monkeypatch.setattr(samplers, "flow_euler_sample", lambda *a, **kw: calls.append(("ordinary", kw)) or marker)
    monkeypatch.setattr(hr_recovery, "run_lr_flow", lambda *a, **kw: calls.append(("diagnostic", kw)) or marker, raising=False)
    monkeypatch.setattr(generate, "_hr_runtime_identity", lambda *a: {"weights": "observed-fixture"})
    args = SimpleNamespace(synchronize_gpu=enabled, save_checkpoints=str(tmp_path))
    capture = {"capture_first_step": None, "shape_block_injection": None} if enabled else {"capture_first_step": {}, "shape_block_injection": object()}
    result = generate._sample_lr_shape(
        args, object(), object(), object(), object(), object(),
        sampler={"steps": 12}, weights_path="weights", shape_attention_route=None,
        quant_coords=np.zeros((3, 4), np.int32), mesh_grid_size=512, **capture,
    )
    assert result is marker
    route, kwargs = calls[0]
    assert len(calls) == 1
    if enabled:
        assert route == "diagnostic"
        assert kwargs["synchronize_gpu"] is True
        assert kwargs["runtime"] == {"weights": "observed-fixture"}
        assert kwargs["checkpoint_dir"] == str(tmp_path)
        assert kwargs["sampler"] == {"steps": 12}
    else:
        assert route == "ordinary"
        assert "synchronize_gpu" not in kwargs
        assert kwargs["steps"] == 12
        assert all(kwargs[key] is value for key, value in capture.items())


def route_args(**overrides):
    values = dict(synchronize_gpu=True, compile=False, replay_hr_input=False,
                  resume=None, image="warrior.webp", save_checkpoints="fresh",
                  stop_after_stage=None, edit_target=None, shape_slat_sample=None,
                  shape_slat_support_sample=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cli_validation_admits_checkpointed_natural_shape_and_existing_hr_replay():
    parser = argparse.ArgumentParser()
    generate._validate_gpu_synchronization_route(parser, route_args())
    generate._validate_gpu_synchronization_route(parser, route_args(replay_hr_input=True, resume="saved", image=None))


@pytest.mark.parametrize("changes", [
    {"compile": True}, {"save_checkpoints": None}, {"image": None},
    {"resume": "old-final"}, {"stop_after_stage": "shape_flow_block_trace"},
    {"edit_target": "target.webp"}, {"shape_slat_sample": "derived"},
    {"shape_slat_support_sample": "derived"},
])
def test_cli_validation_rejects_routes_that_could_skip_the_diagnostic(changes, capsys):
    with pytest.raises(SystemExit) as failure:
        generate._validate_gpu_synchronization_route(argparse.ArgumentParser(), route_args(**changes))
    assert failure.value.code == 2
    assert "--synchronize-gpu requires" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["shape_flow_block_injection_trace", "shape_flow_block_injection_manifest", "shape_timestep_modulation_lut"])
def test_sync_does_not_silently_discard_shape_interventions(field):
    with pytest.raises(SystemExit):
        generate._validate_gpu_synchronization_route(
            argparse.ArgumentParser(), route_args(**{field: "intervention"}),
        )


def test_failed_lr_input_save_reports_stage_without_claiming_completed_input(tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("synthetic input publication failure")

    monkeypatch.setattr(hr_recovery, "save_checkpoint", fail_save)
    with pytest.raises(OSError, match="synthetic input publication failure"):
        hr_recovery.run_lr_flow(
            object(), mx.ones((3, 32)), mx.zeros((1, 1, 4)), mx.zeros((1, 1, 4)),
            mx.zeros((3, 3), mx.int32), checkpoint_dir=str(tmp_path),
            sampler={"steps": 2}, runtime={"route": "fixture"}, synchronize_gpu=True,
        )
    failure = json.loads((tmp_path / "_control/failure.json").read_text())
    assert failure["stage"] == "lr_flow"
    assert failure["active_phase"] == "saving_input"
    assert failure["last_complete_step"] == -1
    assert not has_checkpoint(str(tmp_path), "lr_flow_input")

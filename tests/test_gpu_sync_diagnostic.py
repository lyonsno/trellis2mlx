"""Small CPU fixtures for opt-in execution boundaries, not Metal conformance."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from trellmlx.checkpoint import has_checkpoint, load_checkpoint
from trellmlx.hr_recovery import replay_hr_input, run_hr_flow
from trellmlx.models import slat_flow
from trellmlx.samplers import flow_euler_sample
from trellmlx import samplers


@pytest.fixture(autouse=True)
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class BoundaryModel:
    def __call__(self, sample, timestep, cond, execution_barrier=None, **kwargs):
        if execution_barrier is not None:
            execution_barrier("block.0", sample)
            execution_barrier("block.1", sample)
        return mx.ones_like(sample)


def inputs():
    return mx.ones((3, 32)), mx.zeros((1, 1, 4)), mx.zeros((3, 3), mx.int32)


def test_opt_in_waits_for_both_branches_and_each_sampler_boundary(monkeypatch):
    noise, cond, coords = inputs()
    active = []
    waited = []
    completed = []
    monkeypatch.setattr(mx, "synchronize", lambda: waited.append(active[-1]))
    flow_euler_sample(
        BoundaryModel(), noise, cond, cond, coords=coords, steps=1,
        guidance_strength=2.0, guidance_rescale=0.0, verbose=False,
        synchronize_gpu=True, on_phase=lambda i, p: active.append((i, p)),
        on_execution_complete=lambda i, p: completed.append((i, p)),
    )
    assert waited == completed
    assert waited == [
        (None, "initial_drain"),
        (0, "positive_forward.block.0"), (0, "positive_forward.block.1"),
        (0, "positive_forward.output"),
        (0, "negative_forward.block.0"), (0, "negative_forward.block.1"),
        (0, "negative_forward.output"), (0, "guidance.combine"),
        (0, "euler.delta"), (0, "euler.update"),
    ]


def test_default_sampler_never_installs_barrier_or_waits(monkeypatch):
    noise, cond, coords = inputs()

    def forbidden(*args):
        raise AssertionError("default path gained diagnostic waits")

    class OrdinaryModel(BoundaryModel):
        def __call__(self, *args, **kwargs):
            assert "execution_barrier" not in kwargs
            return super().__call__(*args, **kwargs)

    monkeypatch.setattr(mx, "synchronize", forbidden)
    flow_euler_sample(OrdinaryModel(), noise, cond, cond, coords=coords,
                      steps=1, guidance_strength=1.0, verbose=False)


def test_guidance_rescale_waits_before_euler(monkeypatch):
    noise, cond, coords = inputs()
    phases = []
    monkeypatch.setattr(mx, "synchronize", lambda: None)
    monkeypatch.setattr(samplers, "_cfg_rescale_std",
                        lambda value, **kwargs: mx.array(1.0))
    result = flow_euler_sample(
        BoundaryModel(), noise, cond, cond, coords=coords, steps=1,
        guidance_strength=2.0, guidance_rescale=0.5, verbose=False,
        synchronize_gpu=True,
        on_execution_complete=lambda i, p: phases.append(p),
    )
    assert phases[-7:] == ["guidance.combine", "guidance.xstart",
                           "guidance.std_positive", "guidance.std_cfg",
                           "guidance.rescale", "euler.delta", "euler.update"]
    assert np.isfinite(np.array(result)).all()


@pytest.mark.parametrize("args", [[], ["--replay-hr-input", "--recover-hr-only",
                                      "--resume", "missing", "--compile"]])
def test_cli_refuses_unsupported_sync_route_before_loading_weights(args):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "generate.py"),
         "--synchronize-gpu", *args], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--synchronize-gpu requires --replay-hr-input and an uncompiled model" in result.stderr
    assert "Resuming from" not in result.stdout


def test_sync_failure_names_failed_and_last_completed_boundary(tmp_path, monkeypatch):
    noise, cond, coords = inputs()
    calls = []

    def fail_third_wait():
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError("synthetic Metal wait failure")

    monkeypatch.setattr(mx, "synchronize", fail_third_wait)
    with pytest.raises(RuntimeError, match="synthetic Metal wait failure"):
        run_hr_flow(
            BoundaryModel(), noise, cond, cond, coords,
            checkpoint_dir=str(tmp_path), sampler={"steps": 1, "guidance_strength": 1.0},
            runtime={"route": "fixture"}, synchronize_gpu=True,
        )
    failure = json.loads((tmp_path / "_control/failure.json").read_text())
    assert failure["active_step"] == 0
    assert failure["active_phase"] == "positive_forward.block.1"
    assert failure["last_completed_gpu_boundary"] == {
        "step": 0, "phase": "positive_forward.block.0"
    }
    assert failure["last_complete_step"] == -1
    assert failure["synchronize_gpu"] is True
    assert has_checkpoint(str(tmp_path), "hr_flow_input")
    assert not has_checkpoint(str(tmp_path), "hr_flow_step_000")


def test_replay_records_actual_sync_route_and_preserves_source(tmp_path, monkeypatch):
    noise, cond, coords = inputs()
    source, destination = tmp_path / "source", tmp_path / "destination"
    sampler = {"steps": 1, "guidance_strength": 1.0}
    run_hr_flow(BoundaryModel(), noise, cond, cond, coords,
                checkpoint_dir=str(source), sampler=sampler, runtime={"route": "fixture"})
    original = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    monkeypatch.setattr(mx, "synchronize", lambda: None)
    replay_hr_input(BoundaryModel(), str(source), str(destination),
                    runtime={"route": "fixture"}, synchronize_gpu=True)
    assert {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()} == original
    saved = load_checkpoint(str(destination), "hr_flow_input")
    runtime = json.loads(saved["runtime_json"])
    assert runtime["gpu_synchronization"] == "per-model-block-and-sampler-calculation"
    provenance = json.loads(saved["replay_provenance_json"])
    assert provenance["new_runtime"] == runtime
    assert "gpu_synchronization" not in provenance["source_runtime"]
    np.testing.assert_array_equal(saved["noise"], np.array(noise))


def test_real_slat_forward_exposes_each_block_not_grouped_six(monkeypatch):
    boundaries = []
    model = SimpleNamespace(
        _compiled=False, shape_flow_layernorm=True,
        input_layer=lambda x: x, t_embedder=None, adaLN_modulation=None,
        blocks=[lambda x, *a, **kw: x + 1 for _ in range(2)],
        _final_projection=lambda x, dtype: (x, x),
    )
    monkeypatch.setattr(slat_flow, "_infer_compute_dtype", lambda model: np.float32)
    monkeypatch.setattr(slat_flow, "_shape_shared_modulation",
                        lambda *args: np.zeros((1, 2), np.float32))
    slat_flow.SLatFlowModel.__call__(
        model, np.zeros((3, 2), np.float32), np.array([1], np.float32),
        np.zeros((1, 1, 2), np.float32),
        execution_barrier=lambda phase, *values: boundaries.append(phase),
    )
    assert boundaries == ["prepare", "block.0", "block.1"]


def test_slat_reports_current_block_before_an_internal_evaluation_can_fail(monkeypatch):
    phases = []
    completed = []

    def failing_block(*args, **kwargs):
        raise RuntimeError("synthetic internal block evaluation failure")

    model = SimpleNamespace(
        _compiled=False, shape_flow_layernorm=True,
        input_layer=lambda x: x, t_embedder=None, adaLN_modulation=None,
        blocks=[lambda x, *a, **kw: x + 1, failing_block],
    )
    monkeypatch.setattr(slat_flow, "_infer_compute_dtype", lambda model: np.float32)
    monkeypatch.setattr(slat_flow, "_shape_shared_modulation",
                        lambda *args: np.zeros((1, 2), np.float32))
    with pytest.raises(RuntimeError, match="synthetic internal block"):
        slat_flow.SLatFlowModel.__call__(
            model, np.zeros((3, 2), np.float32), np.array([1], np.float32),
            np.zeros((1, 1, 2), np.float32),
            execution_barrier=lambda p, *v: completed.append(p),
            on_execution_phase=lambda p: phases.append(p),
        )
    assert phases[-1] == "block.1"
    assert completed[-1] == "block.0"


def test_diagnostic_refuses_compiled_forward_instead_of_silent_bypass():
    with pytest.raises(ValueError, match="compiled"):
        slat_flow.SLatFlowModel.__call__(
            SimpleNamespace(_compiled=True), np.zeros((3, 2), np.float32),
            np.array([1], np.float32), np.zeros((1, 1, 2), np.float32),
            execution_barrier=lambda *args: None,
        )

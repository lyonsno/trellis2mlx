"""Ordinary CLI callers inherit the observed Greenroom managed-job contract.

The context envelope matches canonical command.v2 job 8ac362a06592 (installed
Greenroom 598c7da). Temporary process identities are supplied by the real SDK.
All tensor checks in this module run on CPU.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import weakref

import mlx.core as mx
import pytest

pytest.importorskip("gpu_queue", reason="managed checkpoint conformance uses the installed Greenroom SDK")


@pytest.fixture(autouse=True)
def cpu():
    before = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(before)


@pytest.fixture
def managed(tmp_path, monkeypatch):
    from gpu_queue.models import JobRequest, JobState, JobStatus
    from gpu_queue.queue import GPUQueue

    queue = GPUQueue(tmp_path / "queue")
    job = queue.queue_dir / "running" / "testjob"
    job.mkdir()
    request = JobRequest(
        job_type="command", input_path="", job_id=job.name,
        agent_id="test-owner", repo_root=str(Path(__file__).resolve().parents[1]),
        command_argv=[sys.executable, str(Path(__file__).resolve().parents[1] / "generate.py")],
        command_cwd=str(tmp_path),
        route_identity="test/ordinary-trellis-cli", cooperative_checkpoint=True,
    )
    raw = request.to_json().encode()
    (job / "request.json").write_bytes(raw)
    state = JobState(
        job_id=job.name, status=JobStatus.RUNNING, job_type="command", input_path="",
        output_dir=str(tmp_path), child_pid=os.getpid(), child_process_group=os.getpgrp(),
        child_start_identity=queue._process_start_identity(os.getpid()),
        dispatch={"service_class": "normal"},
    )
    (job / "status.json").write_text(state.to_json())
    context = job / "context.json"
    context.write_text(json.dumps({
        "schema": "gpu-greenroom.job-context.v1", "queue_dir": str(queue.queue_dir),
        "job_id": job.name, "request_sha256": hashlib.sha256(raw).hexdigest(),
    }))
    monkeypatch.setenv("GPU_GREENROOM_CONTEXT", str(context))
    monkeypatch.setenv("GPU_GREENROOM_CANCEL_REQUEST", str(job / "cancel-request.json"))
    monkeypatch.delenv("GPU_GREENROOM_READY_FD", raising=False)
    monkeypatch.delenv("GPU_GREENROOM_RESUME_CHECKPOINT", raising=False)
    monkeypatch.delenv("TRELLIS2MLX_GREENROOM_QUEUE_DIR", raising=False)
    monkeypatch.delenv("TRELLIS2MLX_GREENROOM_PAUSE_REPORT", raising=False)
    return queue, job, request


def test_generate_automatically_binds_managed_checkpoint_without_caller_flags(managed, tmp_path, monkeypatch):
    import generate

    observed = []
    monkeypatch.setattr(sys, "argv", ["generate.py", "--output", str(tmp_path / "mesh.glb")])
    monkeypatch.setattr(generate, "_generate", lambda args, parser, control: observed.append(control))
    generate.main()
    assert observed[0] is not None, "ordinary managed TRELLIS discarded its checkpoint context"
    assert observed[0].state["mode"] == "checkpoint-exit-continuation"


def test_stale_managed_context_fails_before_generation(managed, tmp_path, monkeypatch):
    import generate

    _, job, _ = managed
    (job / "request.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["generate.py", "--output", str(tmp_path / "mesh.glb")])
    monkeypatch.setattr(generate, "_generate", lambda *args: pytest.fail("unverified context reached inference"))
    with pytest.raises((ValueError, TypeError), match="identity|request|job_type"):
        generate.main()


def test_nested_controller_is_not_misrepresented_as_an_independent_continuation(managed, tmp_path, monkeypatch):
    import generate

    _, job, request = managed
    wrapper = tmp_path / "parent.py"
    wrapper.write_text("pass\n")
    request.command_argv = [sys.executable, str(wrapper)]
    raw = request.to_json().encode()
    (job / "request.json").write_bytes(raw)
    context_path = job / "context.json"
    context = json.loads(context_path.read_text())
    context["request_sha256"] = hashlib.sha256(raw).hexdigest()
    context_path.write_text(json.dumps(context))
    monkeypatch.setattr(sys, "argv", ["generate.py", "--output", str(tmp_path / "mesh.glb")])
    monkeypatch.setattr(generate, "_generate", lambda *args: pytest.fail("nested owner reached inference"))
    with pytest.raises(ValueError, match="nested controllers"):
        generate.main()
    reports = list((tmp_path / "_control").glob("*.pause.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["phase"] == "managed_preflight"


def continue_job(queue, job, monkeypatch):
    """Replay the real SDK handoff; no alternate tensor/descriptor contract."""
    import shutil
    from gpu_queue.models import JobState, JobStatus

    handoff = json.loads((job / "checkpoint-handoff.json").read_text())
    assert handoff["status"] == "committed"
    successor = queue.queue_dir / "running" / handoff["next_job_id"]
    shutil.move(queue.queue_dir / "pending" / successor.name, successor)
    raw = (successor / "request.json").read_bytes()
    state = JobState.from_json((successor / "status.json").read_text())
    state.status = JobStatus.RUNNING
    state.child_pid = os.getpid()
    state.child_process_group = os.getpgrp()
    state.child_start_identity = queue._process_start_identity(os.getpid())
    state.dispatch = {"service_class": "normal"}
    (successor / "status.json").write_text(state.to_json())
    context = successor / "context.json"
    context.write_text(json.dumps({
        "schema": "gpu-greenroom.job-context.v1", "queue_dir": str(queue.queue_dir),
        "job_id": successor.name, "request_sha256": hashlib.sha256(raw).hexdigest(),
    }))
    monkeypatch.setenv("GPU_GREENROOM_CONTEXT", str(context))
    monkeypatch.setenv("GPU_GREENROOM_RESUME_CHECKPOINT", handoff["checkpoint"])
    return successor, Path(handoff["checkpoint"])


def test_checkpoint_unwinds_models_replays_values_and_preserves_rng(managed, tmp_path, monkeypatch):
    from trellmlx.greenroom_checkpoint import ManagedCheckpoint, checkpointed

    queue, job, _ = managed
    calls, owners, outputs = [], [], []

    class Model:
        pass

    @checkpointed
    def stage(model):
        calls.append("computed")
        value = {"bf16": mx.array([1.25, -2.5], dtype=mx.bfloat16),
                 "numpy": __import__("numpy").array([1, 2], dtype="int32"),
                 "tuple": (mx.random.normal((3,)), None)}
        outputs.append(value)
        queue.pause(owner="test", epoch="test-epoch")
        return value

    def generation():
        mx.random.seed(19)
        model = Model()
        owners.append(weakref.ref(model))
        output = stage(model)
        outputs.append(output)
        return mx.random.normal((2,))

    control = ManagedCheckpoint(tmp_path / "first.json", arguments={"seed": 19})
    with pytest.raises(SystemExit) as stopped:
        control.run(generation)
    assert stopped.value.code == 0
    assert all(owner() is None for owner in owners), "checkpoint retained a model frame"
    assert calls == ["computed"]
    assert json.loads((tmp_path / "first.json").read_text())["status"] == "checkpointed"
    _, checkpoint = continue_job(queue, job, monkeypatch)
    assert json.loads(checkpoint.read_text())["payload"]["records"]
    queue.resume(owner="test", epoch="test-epoch")
    resumed = ManagedCheckpoint(tmp_path / "second.json", arguments={"seed": 19})
    after = resumed.run(generation)
    assert calls == ["computed"], "completed stage was recomputed"
    import numpy as np
    np.testing.assert_array_equal(np.array(outputs[0]["tuple"][0]), np.array(outputs[1]["tuple"][0]))
    assert outputs[1]["bf16"].dtype == mx.bfloat16
    np.testing.assert_array_equal(outputs[1]["numpy"], [1, 2])
    mx.random.seed(19)
    mx.random.normal((3,))
    np.testing.assert_array_equal(np.array(after), np.array(mx.random.normal((2,))))


def test_partial_euler_resumes_next_step_and_preserves_captures(managed, tmp_path, monkeypatch):
    from trellmlx.greenroom_checkpoint import ManagedCheckpoint
    from trellmlx.greenroom_pause import _active_control
    from trellmlx.samplers import flow_euler_sample
    import numpy as np

    queue, job, _ = managed
    calls, captures = [], []
    pause = None
    control = ManagedCheckpoint(tmp_path / "flow-first.json", arguments={"steps": 4})

    def model(sample, timestep, conditioning):
        calls.append(float(timestep.item()))
        return sample * 0.1 + 0.25

    def boundary(index):
        nonlocal pause
        if len(calls) == 2 and pause is None:
            pause = queue.pause(owner="test", epoch="step-two")
        control.boundary("sampling", index)

    def generation():
        captures.clear()
        return flow_euler_sample(
            model, mx.zeros((1, 2)), mx.zeros((1, 1, 2)), mx.zeros((1, 1, 2)),
            steps=4, guidance_strength=1, verbose=False, capture_steps=captures,
            on_step_boundary=boundary,
        )

    token = _active_control.set(control)
    try:
        with pytest.raises(SystemExit):
            control.run(generation)
        assert len(calls) == 2
        _, checkpoint = continue_job(queue, job, monkeypatch)
        saved = json.loads(checkpoint.read_text())["payload"]
        assert saved["partial"] is not None
        queue.resume(owner="test", epoch=pause["epoch"])
        control = ManagedCheckpoint(tmp_path / "flow-second.json", arguments={"steps": 4})
        actual = control.run(generation)
    finally:
        _active_control.reset(token)
    assert len(calls) == 4, "completed Euler steps were recomputed"
    assert len(captures) == 4, "diagnostic captures were dropped at restart"
    expected = flow_euler_sample(model, mx.zeros((1, 2)), mx.zeros((1, 1, 2)), mx.zeros((1, 1, 2)),
                                 steps=4, guidance_strength=1, verbose=False)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_changed_input_cannot_resume_and_leaves_preflight_failure(managed, tmp_path, monkeypatch):
    from trellmlx.greenroom_checkpoint import managed_checkpoint, checkpointed

    queue, job, _ = managed
    image = tmp_path / "image.bin"
    image.write_bytes(b"original")

    @checkpointed
    def stage():
        queue.pause(owner="test", epoch="input-test")
        return mx.ones((2,))

    control = managed_checkpoint(tmp_path / "before.json", {"image": [str(image)]})
    with pytest.raises(SystemExit):
        control.run(stage)
    continue_job(queue, job, monkeypatch)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source/input/config identity"):
        managed_checkpoint(tmp_path / "after.json", {"image": [str(image)]})
    failure = json.loads((tmp_path / "after.json").read_text())
    assert failure["status"] == "failed" and failure["phase"] == "managed_preflight"


def test_corrupt_saved_tensor_is_not_a_completed_result(managed, tmp_path, monkeypatch):
    from trellmlx.greenroom_checkpoint import ManagedCheckpoint, checkpointed

    queue, job, _ = managed

    @checkpointed
    def stage():
        queue.pause(owner="test", epoch="tensor-test")
        return mx.ones((2,))

    control = ManagedCheckpoint(tmp_path / "before.json", arguments={})
    with pytest.raises(SystemExit):
        control.run(stage)
    _, checkpoint = continue_job(queue, job, monkeypatch)
    payload = json.loads(checkpoint.read_text())["payload"]
    Path(payload["records"][0]["value"]["files"]["mlx"]["path"]).write_bytes(b"bad")
    queue.resume(owner="test", epoch="tensor-test")
    resumed = ManagedCheckpoint(tmp_path / "after.json", arguments={})
    with pytest.raises(ValueError, match="tensor bytes changed"):
        resumed.run(stage)
    assert json.loads((tmp_path / "after.json").read_text())["status"] == "failed"


def test_cancellation_does_not_publish_a_continuation(managed, tmp_path):
    import signal
    from trellmlx.greenroom_checkpoint import ManagedCheckpoint, checkpointed

    queue, job, _ = managed

    @checkpointed
    def stage():
        queue.pause(owner="test", epoch="cancel-test")
        os.kill(os.getpid(), signal.SIGTERM)
        return mx.ones((2,))

    control = ManagedCheckpoint(tmp_path / "cancel.json", arguments={})
    with pytest.raises(SystemExit) as stopped:
        control.run(stage)
    assert stopped.value.code == 130
    assert not (job / "checkpoint-handoff.json").exists()
    assert not list((queue.queue_dir / "pending").iterdir())
    assert json.loads((tmp_path / "cancel.json").read_text())["status"] == "cancelled"

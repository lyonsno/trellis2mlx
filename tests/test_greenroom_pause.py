"""Resident Pause/Play must preserve sampling, not signal or restart it."""

import json
import threading
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from trellmlx.samplers import flow_euler_sample


def test_sampler_calls_pause_boundary_without_forwarding_it_to_model():
    events = []

    def model(sample, timestep, conditioning):
        events.append("model")
        return mx.ones_like(sample) * 0.25

    result = flow_euler_sample(
        model, mx.zeros((1, 2)), mx.zeros((1, 1, 2)), mx.zeros((1, 1, 2)),
        steps=3, guidance_strength=1, verbose=False,
        on_step_complete=lambda index, value: events.append(("saved", index)),
        on_step_boundary=lambda index: events.append(("boundary", index)),
    )
    assert events == [
        ("boundary", -1), "model", ("saved", 0), ("boundary", 0),
        "model", ("saved", 1), ("boundary", 1),
        "model", ("saved", 2), ("boundary", 2),
    ]
    np.testing.assert_allclose(np.array(result), -0.25, atol=1e-7)


def queue_root(tmp_path):
    root = tmp_path / "queue"
    for name in ("pending", "running", "done", "failed", "cancelled"):
        (root / name).mkdir(parents=True)
    return root


def pause(root, epoch="one", **overrides):
    payload = {"schema": "gpu-greenroom.pause-state.v1", "status": "effective",
               "epoch": epoch, "queue_dir": str(root.resolve()), **overrides}
    (root / "paused").write_text(json.dumps(payload))


def test_pause_drains_then_blocks_until_play_preserving_epoch_and_memory_semantics(tmp_path, monkeypatch):
    from trellmlx.greenroom_pause import GreenroomPause

    root = queue_root(tmp_path)
    report = tmp_path / "state.json"
    control = GreenroomPause(root, report)
    pause(root)
    order = []

    def drain():
        assert json.loads(report.read_text())["status"] == "draining"
        order.append("drain")

    def wait(seconds):
        state = json.loads(report.read_text())
        assert state["status"] == "paused"
        assert state["submitted_work_drained"] is True
        assert state["memory_released"] is False
        assert state["worker_lock_released"] is False
        assert state["stage"] == "lr_shape" and state["completed_step"] == 2
        order.append(state["pause"]["epoch"])
        if len(order) == 2:
            pause(root, "two")  # A replaced pause is still a pause, not Play.
        else:
            (root / "paused").unlink()

    monkeypatch.setattr(mx, "synchronize", drain)
    monkeypatch.setattr("trellmlx.greenroom_pause.time.sleep", wait)
    control.boundary("lr_shape", 2)
    assert order == ["drain", "one", "two"]
    state = json.loads(report.read_text())
    assert state["status"] == "running"
    assert state["submitted_work_drained"] is False
    assert state["last_pause"]["epoch"] == "two"


def test_play_already_set_does_not_wait_or_synchronize(tmp_path, monkeypatch):
    from trellmlx.greenroom_pause import GreenroomPause

    control = GreenroomPause(queue_root(tmp_path), tmp_path / "state.json")
    monkeypatch.setattr(mx, "synchronize", lambda: pytest.fail("unexpected GPU barrier"))
    monkeypatch.setattr("trellmlx.greenroom_pause.time.sleep", lambda _: pytest.fail("unexpected wait"))
    control.boundary("texture", 0)


@pytest.mark.parametrize("override", [{"schema": "future-version"}, {"queue_dir": "/wrong/queue"}, {"status": "projected"}])
def test_bad_control_identity_is_not_play(tmp_path, override):
    from trellmlx.greenroom_pause import pause_session

    root = queue_root(tmp_path)
    pause(root, **override)
    report = tmp_path / "state.json"
    with pytest.raises(ValueError):
        with pause_session(root, report):
            pytest.fail("invalid pause must not continue generation")
    state = json.loads(report.read_text())
    assert state["status"] == "failed" and not state["submitted_work_drained"]


def test_barrier_failure_never_acknowledges_paused(tmp_path, monkeypatch):
    from trellmlx.greenroom_pause import pause_session

    root = queue_root(tmp_path)
    pause(root)
    report = tmp_path / "state.json"

    def fail():
        raise RuntimeError("Metal drain failed")

    monkeypatch.setattr(mx, "synchronize", fail)
    with pytest.raises(RuntimeError, match="Metal drain failed"):
        with pause_session(root, report):
            pass
    state = json.loads(report.read_text())
    assert state["status"] == "failed" and not state["submitted_work_drained"]
    assert state["stage"] == "startup"


def test_missing_queue_and_stale_report_fail_visibly(tmp_path):
    from trellmlx.greenroom_pause import GreenroomPause

    with pytest.raises(ValueError, match="queue"):
        GreenroomPause(tmp_path / "absent", tmp_path / "state.json")
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "failed" and state["queue_verified"] is False
    assert state["stage"] == "control_preflight"
    report = tmp_path / "stale.json"
    report.write_text('{"status":"paused"}')
    with pytest.raises(FileExistsError):
        GreenroomPause(queue_root(tmp_path), report)
    assert report.read_text() == '{"status":"paused"}'


def test_report_cannot_create_a_pause_marker_or_job_state(tmp_path):
    from trellmlx.greenroom_pause import GreenroomPause

    root = queue_root(tmp_path)
    for report in (root / "paused", root / "running" / "fake.json"):
        with pytest.raises(ValueError, match="control/job state"):
            GreenroomPause(root, report)
        assert not report.exists()


def test_disappearing_queue_does_not_resume_a_parked_run(tmp_path, monkeypatch):
    from trellmlx.greenroom_pause import pause_session

    root = queue_root(tmp_path)
    pause(root)
    report = tmp_path / "state.json"
    monkeypatch.setattr("trellmlx.greenroom_pause.time.sleep", lambda _: root.rename(tmp_path / "moved-queue"))
    with pytest.raises(FileNotFoundError):
        with pause_session(root, report):
            pytest.fail("a missing queue is not Play")
    assert json.loads(report.read_text())["status"] == "failed"


def test_disabled_session_has_no_filesystem_or_gpu_side_effect(tmp_path):
    from trellmlx.greenroom_pause import pause_session

    with pause_session(None, tmp_path / "unused.json") as control:
        assert control is None
    assert not list(tmp_path.iterdir())


def test_legacy_pause_and_sampling_resume_are_numerically_identical(tmp_path, monkeypatch):
    from trellmlx.greenroom_pause import pause_session

    root = queue_root(tmp_path)
    report = tmp_path / "state.json"
    calls = []

    def model(sample, timestep, conditioning):
        calls.append(float(np.array(timestep)[0]))
        if len(calls) == 2:
            (root / "paused").touch()
        return sample * 0.1 + 0.25

    monkeypatch.setattr("trellmlx.greenroom_pause.time.sleep", lambda _: (root / "paused").unlink())
    kwargs = dict(steps=4, guidance_strength=1, verbose=False)
    noise, cond = mx.ones((1, 2)), mx.zeros((1, 1, 2))
    baseline = flow_euler_sample(lambda s, t, c: s * 0.1 + 0.25, noise, cond, cond, **kwargs)
    with pause_session(root, report) as control:
        output = flow_euler_sample(model, noise, cond, cond,
                                   on_step_boundary=lambda i: control.boundary("sparse_flow", i), **kwargs)
    np.testing.assert_array_equal(np.array(baseline), np.array(output))
    assert len(calls) == 4
    state = json.loads(report.read_text())
    assert state["status"] == "finished" and state["last_pause"]["epoch"] is None


@pytest.mark.parametrize("synchronized", [False, True])
def test_generate_lr_hook_composes_with_exact_step_checkpoints(tmp_path, synchronized):
    import generate
    from trellmlx.checkpoint import has_checkpoint
    from trellmlx.greenroom_pause import GreenroomPause

    control = GreenroomPause(queue_root(tmp_path), tmp_path / "pause.json")
    seen = []
    control.boundary = lambda stage, index: seen.append((stage, index))
    args = SimpleNamespace(synchronize_gpu=synchronized, save_checkpoints=str(tmp_path / "checkpoints"))
    noise, cond, coords = mx.ones((2, 32)), mx.zeros((1, 1, 4)), mx.zeros((2, 3), mx.int32)
    from unittest.mock import patch
    with patch.object(generate, "_hr_runtime_identity", return_value={"route": "cpu-fixture"}):
        result = generate._sample_lr_shape(
            args, lambda s, t, c, **kw: mx.ones_like(s), noise, cond, cond, coords,
            sampler={"steps": 2, "guidance_strength": 1}, weights_path="fixture",
            shape_attention_route=None, quant_coords=np.zeros((2, 4)), mesh_grid_size=512,
            pause_control=control,
        )
    assert seen == [("lr_shape", -1), ("lr_shape", 0), ("lr_shape", 1)]
    assert has_checkpoint(args.save_checkpoints, "lr_flow_step_001") == synchronized
    np.testing.assert_allclose(np.array(result), 0, atol=1e-7)


def test_generate_cli_scopes_control_and_closes_status(tmp_path, monkeypatch):
    import generate

    root = queue_root(tmp_path)
    report = tmp_path / "pause.json"
    monkeypatch.setattr("sys.argv", ["generate.py", "--greenroom-queue-dir", str(root),
                                    "--greenroom-pause-report", str(report)])
    observed = []
    monkeypatch.setattr(generate, "_generate", lambda args, parser, control: observed.append(control.queue_dir))
    generate.main()
    assert observed == [root.resolve()]
    assert json.loads(report.read_text())["status"] == "finished"


def test_actual_greenroom_pause_play_preserves_all_sampler_steps(tmp_path):
    # Optional conformance run: PYTHONPATH names the observed Greenroom checkout.
    queue_module = pytest.importorskip("gpu_queue.queue")
    from trellmlx.greenroom_pause import pause_session

    queue = queue_module.GPUQueue(tmp_path / "queue")
    report = tmp_path / "pause.json"
    paused = threading.Event()
    errors, pauses, calls = [], [], []

    def operator_play():
        try:
            assert paused.wait(5), "Trellis never acknowledged the real Pause"
            state = json.loads(report.read_text())
            assert state["status"] == "paused" and state["completed_step"] == 1
            assert state["submitted_work_drained"] is True
            assert len(calls) == 2
            pauses.append(state)
            queue.resume(owner="isolated-test-operator", epoch=state["pause"]["epoch"])
        except BaseException as error:
            errors.append(error)
            queue.resume(owner="isolated-test-cleanup")

    def model(sample, timestep, conditioning):
        calls.append(float(np.array(timestep)[0]))
        if len(calls) == 2:
            queue.pause(owner="isolated-test-operator")
        return sample * 0.1 + 0.25

    thread = threading.Thread(target=operator_play, daemon=True)
    thread.start()
    noise, cond = mx.ones((1, 2)), mx.zeros((1, 1, 2))
    kwargs = dict(steps=4, guidance_strength=1, verbose=False)
    baseline = flow_euler_sample(lambda s, t, c: s * 0.1 + 0.25, noise, cond, cond, **kwargs)
    try:
        with pause_session(queue.queue_dir, report) as control:
            write = control._write

            def observed_write(**changes):
                write(**changes)
                if changes.get("status") == "paused":
                    paused.set()

            control._write = observed_write
            output = flow_euler_sample(model, noise, cond, cond,
                                      on_step_boundary=lambda i: control.boundary("sparse_flow", i), **kwargs)
    finally:
        thread.join(6)
    assert not thread.is_alive() and not errors and len(pauses) == 1
    np.testing.assert_array_equal(np.array(baseline), np.array(output))
    assert len(calls) == 4 and queue.pause_state() is None
    state = json.loads(report.read_text())
    assert state["status"] == "finished" and state["mode"] == "resident-cooperative-wait"

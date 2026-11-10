"""Ordinary CLI callers inherit the observed Greenroom managed-job contract.

The context envelope matches observed command.v2 output from Greenroom 598c7da.
Temporary process identities are supplied by the real SDK.
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
    calls, captures, first = [], [], {}
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
        first.clear()
        return flow_euler_sample(
            model, mx.zeros((1, 2)), mx.zeros((1, 1, 2)), mx.zeros((1, 1, 2)),
            steps=4, guidance_strength=1, verbose=False, capture_steps=captures, capture_first_step=first,
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
    assert first["t"].item() == 1.0, "resumed step overwrote the first-step diagnostic capture"
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


@pytest.mark.parametrize("changed", ["sidecar", "new-sidecar", "resume-input", "attention-env"])
def test_implicit_inputs_and_effective_backend_are_bound(managed, tmp_path, monkeypatch, changed):
    from trellmlx.greenroom_checkpoint import ManagedCheckpoint, checkpointed

    queue, job, _ = managed
    sample = tmp_path / "sample.npz"
    sample.write_bytes(b"fixture")
    if changed == "sidecar":
        sample.with_suffix(".json").write_text('{"mesh_grid_size":512}')
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "mesh_raw.npz").write_bytes(b"mesh")
    (resume / "texture.npz").write_bytes(b"texture")
    arguments = {"conditioning_sample": str(sample), "resume": str(resume)}

    @checkpointed
    def stage():
        queue.pause(owner="test", epoch="implicit-input")
        return mx.ones((2,))

    first = ManagedCheckpoint(tmp_path / "before.json", arguments=arguments)
    with pytest.raises(SystemExit):
        first.run(stage)
    continue_job(queue, job, monkeypatch)
    if changed in ("sidecar", "new-sidecar"):
        sample.with_suffix(".json").write_text('{"mesh_grid_size":1024}')
    elif changed == "resume-input":
        (resume / "texture.npz").write_bytes(b"different texture")
    else:
        monkeypatch.setenv("TRELLIS2MLX_ATTENTION_CHUNK_SIZE", "17")
    with pytest.raises(ValueError, match="source/input/config identity"):
        ManagedCheckpoint(tmp_path / "after.json", arguments=arguments)


def test_ordinary_direct_cli_automatically_queues_without_pause_or_identity_flags(tmp_path, monkeypatch):
    import generate
    import gpu_queue.queue
    from types import SimpleNamespace

    submitted = []

    class Queue:
        def __init__(self, path):
            self.queue_dir = Path(path)

        def submit(self, request):
            submitted.append(request)
            path = self.queue_dir / "done" / request.job_id
            path.mkdir(parents=True)
            (path / "request.json").write_text(request.to_json())
            return path

        def get_job(self, job):
            return SimpleNamespace(status=SimpleNamespace(value="done"), exit_code=0)

    monkeypatch.setattr(gpu_queue.queue, "GPUQueue", Queue)
    for key in ("GPU_GREENROOM_CONTEXT", "GPU_GREENROOM_JOB_ID", "GREENROOM_JOB_ID", "TRELLIS2MLX_GREENROOM_QUEUE_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GPU_GREENROOM_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setattr(sys, "argv", [str(Path(generate.__file__)), "--output", str(tmp_path / "mesh.glb")])
    monkeypatch.setattr(generate, "_generate", lambda *args: pytest.fail("direct unadmitted CLI reached model execution"))
    assert generate.main() == 0
    assert len(submitted) == 1
    assert submitted[0].cooperative_checkpoint is True
    assert submitted[0].agent_id is None
    assert "--cooperative-checkpoint" not in submitted[0].command_argv
    assert Path(submitted[0].output_dir).parent == tmp_path / "_control"
    assert Path(submitted[0].output_dir) != tmp_path, "caller directory became job-owned storage"


def test_foreground_cli_follows_real_sdk_successor_and_accepts_its_incoming_edge(tmp_path, monkeypatch):
    import threading
    import time
    from types import SimpleNamespace
    from gpu_queue.queue import GPUQueue
    from trellmlx import greenroom_entry

    root = tmp_path / "producer"
    root.mkdir()
    (root / ".gpu-greenroom-producer.json").write_text(json.dumps({
        "schema": "gpu-greenroom.producer-checkpoints.v1", "entrypoints": ["generate.py"]}))
    (root / "generate.py").write_text('''import json,pathlib,time,uuid
from dataclasses import replace
from gpu_queue.cooperative import _context,load_checkpoint,yield_if_requested
q,job,r,s=_context()
state=load_checkpoint() or {'step':0}
if state['step']==0:
 q.pause(owner='isolated-entry-fixture')
 yield_if_requested(save_checkpoint=lambda p:{'step':1},quiesce=lambda:True,
  continuation=lambda p:replace(r,job_id=uuid.uuid4().hex[:12],submitted_at=time.time()))
pathlib.Path(r.output_dir,'fixture-result.json').write_text(json.dumps(state))
''')
    monkeypatch.setattr(greenroom_entry, "__file__", str(root / "trellmlx" / "greenroom_entry.py"))
    queue = GPUQueue(tmp_path / "queue")
    for key in ("GPU_GREENROOM_CONTEXT", "GPU_GREENROOM_JOB_ID", "GREENROOM_JOB_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GPU_GREENROOM_QUEUE_DIR", str(queue.queue_dir))
    monkeypatch.setattr(sys, "argv", [str(root / "generate.py"), "--output", str(tmp_path / "mesh.glb")])
    stop = threading.Event()
    errors = []

    def service():
        try:
            while not stop.is_set():
                pending = list((queue.queue_dir / "pending").iterdir())
                if queue.is_paused() and pending and list((queue.queue_dir / "done").iterdir()):
                    queue.resume(owner="isolated-entry-fixture")
                if not queue.run_one({}):
                    stop.wait(.01)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=service)
    worker.start()
    try:
        assert greenroom_entry.run_unmanaged(SimpleNamespace(greenroom_queue_dir=None, output=str(tmp_path / "mesh.glb"))) == 0
    finally:
        stop.set()
        worker.join()
    assert errors == []
    receipts = list((tmp_path / "_control").glob("*.admission.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert json.loads((Path(receipt["job_output_dir"]) / "fixture-result.json").read_text()) == {"step": 1}
    assert receipt["status"] == "done" and len(receipt["chain"]) == 1
    assert len(list((queue.queue_dir / "done").iterdir())) == 2


def test_foreground_cancel_follows_a_parent_that_already_handed_off(tmp_path, monkeypatch):
    import gpu_queue.queue
    from trellmlx import greenroom_entry
    from types import SimpleNamespace
    from dataclasses import replace

    cancelled = []

    class Queue:
        def __init__(self, path):
            self.queue_dir = Path(path)
            self.first_read = True

        def submit(self, request):
            self.parent = request.job_id
            self.child = "fixture-successor"
            parent = self.queue_dir / "done" / self.parent
            child = self.queue_dir / "pending" / self.child
            parent.mkdir(parents=True); child.mkdir(parents=True)
            (parent / "request.json").write_text(request.to_json())
            (parent / "checkpoint-handoff.json").write_text(json.dumps({
                "status": "committed", "job_id": self.parent, "next_job_id": self.child}))
            successor = replace(request, job_id=self.child, params={**request.params, "continuation_of": self.parent})
            (child / "request.json").write_text(successor.to_json())
            return parent

        def get_job(self, job):
            if self.first_read:
                self.first_read = False
                raise KeyboardInterrupt()
            return SimpleNamespace(status=SimpleNamespace(value="done" if job == self.parent else "pending"), exit_code=0)

        def cancel(self, job):
            cancelled.append(job)
            return True

    monkeypatch.setattr(gpu_queue.queue, "GPUQueue", Queue)
    for key in ("GPU_GREENROOM_CONTEXT", "GPU_GREENROOM_JOB_ID", "GREENROOM_JOB_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GPU_GREENROOM_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setattr(sys, "argv", [str(Path(greenroom_entry.__file__).resolve().parents[1] / "generate.py")])
    code = greenroom_entry.run_unmanaged(SimpleNamespace(greenroom_queue_dir=None, output=str(tmp_path / "mesh.glb")))
    assert code == 130
    assert cancelled == ["fixture-successor"], "foreground exit left its resumed job queued"


@pytest.mark.parametrize('preserve', ['none', 'unclassified', 'research', 'witness', 'failure', 'stale-output', 'changed-scratch'])
def test_only_completed_final_export_releases_owned_replay_scratch(managed, tmp_path, preserve):
    from trellmlx.greenroom_checkpoint import ManagedCheckpoint, checkpointed

    output = tmp_path / 'mesh.glb'
    output.write_bytes(b'old asset')
    arguments = {'output': str(output)}
    if preserve == 'research':
        arguments['save_checkpoints'] = str(tmp_path / 'research')
    control = ManagedCheckpoint(tmp_path / 'scratch-report.json', arguments=arguments)
    control.request.output_class = 'witness' if preserve == 'witness' else 'final'
    if preserve == 'unclassified':
        control.request.output_class = None
    unknown = control.directory / 'caller-note.txt'
    unknown.write_text('retain unknown content')

    @checkpointed
    def stage():
        return mx.ones((2,))

    def generation():
        stage()
        if preserve == 'failure':
            raise RuntimeError('export failed')
        if preserve != 'stale-output':
            output.write_bytes(b'new final asset')
            control.output_published(output)
        if preserve == 'changed-scratch':
            next(control.directory.glob('*.safetensors')).write_bytes(b'external change')

    if preserve == 'failure':
        with pytest.raises(RuntimeError, match='export failed'):
            control.run(generation)
    else:
        control.run(generation)
    tensors = list(control.directory.glob('*.safetensors'))
    assert bool(tensors) is (preserve not in ('none', 'unclassified'))
    assert unknown.read_text() == 'retain unknown content'
    assert output.read_bytes() == (b'old asset' if preserve in ('failure', 'stale-output') else b'new final asset')
    report = json.loads((tmp_path / 'scratch-report.json').read_text())
    if preserve in ('none', 'unclassified'):
        assert report['scratch_cleanup']['removed_files'] == 1


@pytest.mark.parametrize('transition', ['running', 'done', 'handoff', 'running-done', 'submit', 'rejected'])
def test_foreground_cancel_reconciles_a_pending_transition(tmp_path, monkeypatch, transition):
    import gpu_queue.queue
    from trellmlx import greenroom_entry
    from types import SimpleNamespace
    from dataclasses import replace

    accepted = []

    class Queue:
        def __init__(self, path):
            self.queue_dir = Path(path)
            self.first = True
            self.status = 'pending'
        def submit(self, request):
            self.request = request
            self.current = request.job_id
            path = self.queue_dir / 'pending' / self.current
            path.mkdir(parents=True)
            (path / 'request.json').write_text(request.to_json())
            if transition == 'submit':
                self.first = False
                raise KeyboardInterrupt()
            return path
        def get_job(self, job):
            if self.first:
                self.first = False
                raise KeyboardInterrupt()
            return SimpleNamespace(status=SimpleNamespace(value=self.status if job == self.current else 'pending'),
                                   started_at=42 if self.status == 'running' else None, exit_code=0)
        def cancel(self, job):
            if job != self.current:
                accepted.append(job)
                return True
            self.status = 'running' if transition in ('running', 'running-done', 'submit', 'rejected') else 'done'
            before = self.queue_dir / 'pending' / job
            after = self.queue_dir / self.status / job
            after.parent.mkdir(exist_ok=True)
            before.rename(after)
            if transition == 'handoff':
                child = replace(self.request, job_id='successor', params={**self.request.params, 'continuation_of': job})
                path = self.queue_dir / 'pending' / child.job_id
                path.mkdir()
                (path / 'request.json').write_text(child.to_json())
                (after / 'checkpoint-handoff.json').write_text(json.dumps({
                    'status': 'committed', 'job_id': job, 'next_job_id': child.job_id}))
            return False
        def request_cancel(self, **kwargs):
            assert kwargs['expected_status'] == 'running' and kwargs['expected_started_at'] == 42
            if transition == 'rejected':
                raise ValueError('running worker does not support confirmed cancellation')
            if transition == 'running-done':
                self.status = 'done'
                before = self.queue_dir / 'running' / self.current
                after = self.queue_dir / 'done' / self.current
                after.parent.mkdir(exist_ok=True)
                before.rename(after)
                raise ValueError('job changed; refresh before cancelling')
            accepted.append(kwargs['job_id'])
            return {'status': 'cancelling', 'job_id': kwargs['job_id'], 'receipt_path': 'actual-sdk-receipt'}

    monkeypatch.setattr(gpu_queue.queue, 'GPUQueue', Queue)
    for key in ('GPU_GREENROOM_CONTEXT', 'GPU_GREENROOM_JOB_ID', 'GREENROOM_JOB_ID'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('GPU_GREENROOM_QUEUE_DIR', str(tmp_path / 'queue'))
    monkeypatch.setattr(sys, 'argv', [str(Path(greenroom_entry.__file__).resolve().parents[1] / 'generate.py')])
    args = SimpleNamespace(greenroom_queue_dir=None, output=str(tmp_path / 'mesh.glb'))
    if transition == 'rejected':
        with pytest.raises(ValueError, match='does not support'):
            greenroom_entry.run_unmanaged(args)
    else:
        assert greenroom_entry.run_unmanaged(args) == 130
    receipt = json.loads(next((tmp_path / '_control').glob('*.admission.json')).read_text())
    if transition == 'rejected':
        assert accepted == []
        assert receipt['status'] == 'cancellation-failed'
    elif transition in ('done', 'running-done'):
        assert accepted == []
        assert receipt['status'] == 'done', 'already completed work was labelled cancellation-requested'
    else:
        assert len(accepted) == 1, 'cancellation was lost at the pending transition'
        assert receipt['job_id'] == accepted[0]


def test_foreground_cancel_pending_claim_uses_actual_sdk_disposition(tmp_path, monkeypatch):
    import threading
    import time
    import gpu_queue.queue
    from gpu_queue.queue import GPUQueue
    from trellmlx import greenroom_entry
    from types import SimpleNamespace

    root = tmp_path / 'producer'
    root.mkdir()
    started, release = tmp_path / 'started', tmp_path / 'release'
    (root / '.gpu-greenroom-producer.json').write_text(json.dumps({
        'schema': 'gpu-greenroom.producer-checkpoints.v1', 'entrypoints': ['generate.py']}))
    (root / 'generate.py').write_text(
        f"import pathlib,time\npathlib.Path({str(started)!r}).touch()\n"
        f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(.01)\n")
    threads, errors = [], []

    class Queue(GPUQueue):
        first = True
        def get_job(self, job):
            if self.first:
                self.first = False
                raise KeyboardInterrupt()
            return super().get_job(job)
        def cancel(self, job):
            def service():
                try:
                    self.run_one({})
                except BaseException as error:
                    errors.append(repr(error))
            thread = threading.Thread(target=service)
            threads.append(thread)
            thread.start()
            while not started.exists() and thread.is_alive():
                time.sleep(.01)
            return super().cancel(job)

    monkeypatch.setattr(gpu_queue.queue, 'GPUQueue', Queue)
    monkeypatch.setattr(greenroom_entry, '__file__', str(root / 'trellmlx/greenroom_entry.py'))
    for key in ('GPU_GREENROOM_CONTEXT', 'GPU_GREENROOM_JOB_ID', 'GREENROOM_JOB_ID'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('GPU_GREENROOM_QUEUE_DIR', str(tmp_path / 'queue'))
    monkeypatch.setattr(sys, 'argv', [str(root / 'generate.py')])
    try:
        assert greenroom_entry.run_unmanaged(SimpleNamespace(greenroom_queue_dir=None, output=str(tmp_path / 'mesh.glb'))) == 130
        receipt = json.loads(next((tmp_path / '_control').glob('*.admission.json')).read_text())
        assert receipt['cancellation']['status'] == 'cancelling'
        accepted = json.loads(Path(receipt['cancellation']['receipt_path']).read_text())
        assert accepted['phase'] == 'completed'
        assert accepted['request']['job_id'] == receipt['job_id']
        assert accepted['request']['expected_status'] == 'running'
    finally:
        release.touch()
        for thread in threads:
            thread.join()
    assert errors == []

"""Make an ordinary local CLI use installed Greenroom automatically.

Machines without Greenroom keep standalone generation. On a shared machine,
the foreground client waits for the exact queued command and its successors;
only the admitted child loads models. Caller identity remains unrecorded when
none was supplied. Durable admission state is separate from asset success.
"""

import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import uuid

from .greenroom_checkpoint import _sdk, _write, _hash


def _successor(queue, expected, current, observed, seen):
    if observed is None or observed.status.value != "done":
        return None
    # Terminal status is written before the worker moves the directory. The
    # committed edge already exists at that point; follow its actual record.
    for bucket in ("running", "done", "failed", "cancelled"):
        path = queue.queue_dir / bucket / current
        try:
            raw = json.loads((path / "request.json").read_text())
        except FileNotFoundError:
            continue
        try:
            edge = json.loads((path / "checkpoint-handoff.json").read_text())
        except FileNotFoundError:
            if path.is_dir():
                return None
            continue  # The record moved during the read; try its next bucket.
        break
    else:
        raise RuntimeError(f"cannot locate TRELLIS terminal record: {current}")
    if raw.get("job_id") != current:
        raise ValueError("TRELLIS parent record identity changed")
    if edge.get("status") != "committed":
        raise ValueError("incomplete TRELLIS checkpoint handoff")
    if edge.get("job_id") != current:
        if edge.get("next_job_id") != current or edge.get("job_id") != raw.get("params", {}).get("continuation_of"):
            raise ValueError("unrelated incoming TRELLIS checkpoint handoff")
        return None
    successor = edge["next_job_id"]
    if successor in seen or successor == current:
        raise ValueError("cyclic TRELLIS checkpoint handoff")
    paths = [queue.queue_dir / bucket / successor for bucket in ("pending", "running", "done", "failed", "cancelled", "deferred")
             if (queue.queue_dir / bucket / successor / "request.json").exists()]
    if len(paths) != 1:
        raise ValueError("missing or ambiguous TRELLIS successor")
    candidate = json.loads((paths[0] / "request.json").read_text())
    for key in ("agent_id", "repo_root", "route_identity", "command_cwd", "command_argv"):
        if candidate.get(key) != getattr(expected, key):
            raise ValueError(f"TRELLIS successor changed {key}")
    if candidate.get("cooperative_checkpoint") is not True or candidate.get("params", {}).get("continuation_of") != current:
        raise ValueError("TRELLIS successor lost its checkpoint ownership")
    if edge.get("next_request_sha256") and _hash(paths[0] / "request.json") != edge["next_request_sha256"]:
        raise ValueError("TRELLIS successor request bytes changed")
    return successor


def _terminal_published(queue, current, observed):
    # The worker writes its terminal receipt before this directory transition.
    return (queue.queue_dir / observed.status.value / current).is_dir()


def _cancel_current(queue, expected, current, state):
    """Reconcile transitions until cancellation is accepted or work ended."""
    while True:
        observed = queue.get_job(current)
        if observed is None:
            raise RuntimeError(f"cannot locate TRELLIS job for cancellation: {current}")
        successor = _successor(queue, expected, current, observed, state["chain"])
        if successor is not None:
            state["chain"].append(current)
            current = successor
            state["job_id"] = current
            continue
        status = observed.status.value
        if status in ("done", "failed", "cancelled"):
            if not _terminal_published(queue, current, observed):
                time.sleep(.25)
                continue
            return current, {"status": status, "action": "already-terminal"}
        if status in ("pending", "deferred"):
            if queue.cancel(current):
                return current, {"status": "cancelled", "action": "cancelled-before-start"}
            continue  # Worker/control moved it; the old observation is spent.
        if status != "running":
            raise RuntimeError(f"unsupported TRELLIS cancellation state: {status}")
        path = queue.queue_dir / "running" / current
        try:
            result = queue.request_cancel(job_id=current, request_digest=_hash(path / "request.json"),
                expected_status="running", expected_started_at=observed.started_at,
                request_id=uuid.uuid4().hex, requested_by="ordinary TRELLIS foreground client", force=False)
        except (ValueError, FileNotFoundError):
            refreshed = queue.get_job(current)
            if refreshed is None or (refreshed.status.value, refreshed.started_at) == (status, observed.started_at):
                raise
            continue  # Completion/handoff won; reconcile its actual disposition.
        return current, result


def run_unmanaged(args):
    if os.environ.get("GPU_GREENROOM_CONTEXT"):
        return None
    if os.environ.get("GPU_GREENROOM_JOB_ID") or os.environ.get("GREENROOM_JOB_ID"):
        raise RuntimeError("this managed job did not admit TRELLIS checkpoint support; update its Greenroom producer route")
    try:
        _sdk()
    except RuntimeError:
        if shutil.which("gpu-greenroom") is None:
            return None
        raise
    try:
        producers = importlib.import_module("gpu_queue.producer_checkpoints")
    except ModuleNotFoundError as error:
        raise RuntimeError("installed Greenroom lacks automatic producer admission; update Greenroom before shared TRELLIS execution") from error
    if not hasattr(producers, "checkpoint_owner"):
        raise RuntimeError("installed Greenroom lacks job-owned checkpoints for ordinary callers")
    queue_type = importlib.import_module("gpu_queue.queue").GPUQueue
    admission = importlib.import_module("gpu_queue.admission")
    root = Path(__file__).resolve().parents[1]
    queue_root = args.greenroom_queue_dir or os.environ.get("GPU_GREENROOM_QUEUE_DIR") or str(Path.home() / ".local/state/gpu-greenroom")
    queue = queue_type(Path(queue_root).expanduser().resolve())
    output = Path(args.output).resolve()
    run_id = f"{output.stem}.{uuid.uuid4().hex}"
    receipt = output.parent / "_control" / f"{run_id}.admission.json"
    job_output = output.parent / "_control" / f"{run_id}.job"
    env = {k:v for k,v in os.environ.items() if k.startswith("TRELLIS2MLX_") or k in
           {"PATH", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE"}}
    env["PYTHONPATH"] = str(root) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    request = admission.command_request(admission.normalize_manifest({
        "schema": "gpu-greenroom.command.v1", "repo_root": str(root), "cwd": os.getcwd(),
        "argv": [sys.executable, "-u", str(root / "generate.py"), *sys.argv[1:]],
        "output_dir": str(job_output), "output_class": "final",
        "route_identity": "trellis2mlx/ordinary-cli/automatic-checkpoint", "env": env,
    }))
    state = {"schema": "trellis2mlx.automatic-admission.v1", "status": "submitting",
             "repo_root": str(root), "queue_dir": str(queue.queue_dir),
             "argv": request.command_argv, "caller_agent_id": None,
             "claim_ceiling": "queued-command terminality; inspect produced artifacts separately",
             "chain": [], "receipt": str(receipt)}
    state.update(job_output_dir=str(job_output), requested_output=str(output))
    _write(receipt, state)
    current = None
    previous_term = None
    try:
        current = request.job_id
        path = queue.submit(request)
        state.update(status="waiting", job_id=current, request_path=str(path / "request.json"))
        _write(receipt, state)
        print(f"Greenroom job {current}; automatic checkpoint support. Admission: {receipt}", flush=True)
        previous_term = signal.getsignal(signal.SIGTERM)
        def interrupted(*_):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupted)
        offsets = {}
        while True:
            observed = queue.get_job(current)
            if observed is None:
                raise RuntimeError(f"queued TRELLIS job disappeared: {current}")
            for filename, destination in (("stdout.log", sys.stdout), ("stderr.log", sys.stderr)):
                log = queue.queue_dir / observed.status.value / current / filename
                if log.is_file():
                    key = (current, filename)
                    with log.open("rb") as stream:
                        stream.seek(offsets.get(key, 0))
                        data = stream.read()
                        offsets[key] = stream.tell()
                    if data:
                        destination.write(data.decode("utf-8", errors="replace"))
                        destination.flush()
            if observed.status.value not in ("done", "failed", "cancelled"):
                time.sleep(.25)
                continue
            if not _terminal_published(queue, current, observed):
                time.sleep(.25)
                continue
            path = queue.queue_dir / observed.status.value / current
            successor = _successor(queue, request, current, observed, state["chain"])
            if successor is not None:
                state["chain"].append(current)
                current = successor
                state["job_id"] = current
                _write(receipt, state)
                continue
            code = observed.exit_code if observed.exit_code is not None else 1
            state.update(status=observed.status.value, exit_code=code, terminal_job=current)
            _write(receipt, state)
            return code
    except KeyboardInterrupt:
        try:
            current, result = _cancel_current(queue, request, current, state)
        except BaseException as error:
            state.update(status="cancellation-failed", job_id=state.get("job_id", current),
                         error_type=type(error).__name__, error=str(error))
            _write(receipt, state)
            raise
        state.update(status="cancellation-requested" if result["status"] == "cancelling" else result["status"],
                     job_id=current, cancellation=result)
        _write(receipt, state)
        return 130
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        _write(receipt, state)
        raise
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)

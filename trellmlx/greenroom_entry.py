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
    path = queue.queue_dir / "done" / current
    edge_path = path / "checkpoint-handoff.json"
    if not edge_path.exists():
        return None
    edge = json.loads(edge_path.read_text())
    raw = json.loads((path / "request.json").read_text())
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
        path = queue.submit(request)
        current = request.job_id
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
        if current is not None:
            observed = queue.get_job(current)
            while observed is not None:
                successor = _successor(queue, request, current, observed, state["chain"])
                if successor is None:
                    break
                state["chain"].append(current)
                current = successor
                observed = queue.get_job(current)
            if observed is not None and observed.status.value == "pending":
                queue.cancel(current)
            elif observed is not None and observed.status.value == "running":
                path = queue.queue_dir / "running" / current
                queue.request_cancel(job_id=current, request_digest=_hash(path / "request.json"),
                    expected_status="running", expected_started_at=observed.started_at,
                    request_id=uuid.uuid4().hex, requested_by="ordinary TRELLIS foreground client", force=False)
        state.update(status="cancellation-requested", job_id=current)
        _write(receipt, state)
        return 130
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        _write(receipt, state)
        raise
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)

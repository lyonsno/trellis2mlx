"""Managed generation replay: save values, unwind models, then hand off.

The scheduler owns admission and process-group quiescence. This producer owns
the exact source/configuration and completed calls, including partial Euler
progress. No pickle, signal-based suspension, or parked GPU model is used.
"""

from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import mlx.core as mx
import numpy as np


_current = ContextVar("trellis_checkpoint_replay", default=None)


class _CheckpointRequested(BaseException):
    pass


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _sdk():
    """Use the active SDK, or the installed CLI's SDK from a private venv.

    The global Greenroom command is already required for managed admission.
    Discover its actual package, rather than hardcoding another checkout or
    requiring each Trellis caller to alter PYTHONPATH.
    """
    try:
        return importlib.import_module("gpu_queue.cooperative")
    except ModuleNotFoundError as error:
        if error.name != "gpu_queue":
            raise
    executable = shutil.which("gpu-greenroom")
    if not executable:
        raise RuntimeError("managed Trellis requires the installed gpu-greenroom SDK")
    first = Path(executable).read_text().splitlines()[0]
    interpreter = first.removeprefix("#!").strip()
    if not first.startswith("#!/") or not Path(interpreter).is_file():
        raise RuntimeError("cannot resolve the installed Greenroom interpreter")
    result = subprocess.run(
        [interpreter, "-c", "import pathlib,gpu_queue;print(pathlib.Path(gpu_queue.__file__).resolve().parent.parent)"],
        check=True, text=True, capture_output=True,
    )
    package_root = Path(result.stdout.strip())
    if not (package_root / "gpu_queue" / "cooperative.py").is_file():
        raise RuntimeError("installed Greenroom did not resolve a checkpoint SDK")
    sys.path.insert(0, str(package_root))
    return importlib.import_module("gpu_queue.cooperative")


def checkpointed(function):
    """Persist a completed expensive call only inside managed generation."""
    @wraps(function)
    def invoke(*args, **kwargs):
        control = _current.get()
        if control is None:
            return function(*args, **kwargs)
        return control.call(function, args, kwargs)
    return invoke


def checkpoint_call(label, function, *args, **kwargs):
    control = _current.get()
    if control is None:
        return function(*args, **kwargs)
    return control.call(function, args, kwargs, label=label)


def sampler_resume():
    control = _current.get()
    return control.partial if control is not None and control.in_sampler else None


def sampler_progress(index, sample, captures):
    control = _current.get()
    if control is not None and control.in_sampler:
        control.partial = {"next_step": index + 1, "sample": sample,
                           "rng": list(mx.random.state), "captures": captures}
        control._write(stage=f"sampler_{control.active_index}", completed_step=index)


def restore_rng(state):
    """Restore MLX's observed two-uint32 key via its public seed API.

    MLX 0.32 exposes a read-only _RandomState. Replacing that module attribute
    only replaces its Python view, not the native generator. seed(uint64)
    reconstructs the two key words; check the actual state before proceeding.
    The CPU conformance witness exercises the real 0.32.4 generator.
    """
    if len(state) != 1:
        raise ValueError("unsupported MLX random-state layout")
    words = np.asarray(state[0])
    if words.shape != (2,) or words.dtype != np.uint32:
        raise ValueError("unsupported MLX random key layout")
    mx.random.seed((int(words[0]) << 32) | int(words[1]))
    if not np.array_equal(np.asarray(mx.random.state[0]), words):
        raise ValueError("MLX random-state restore did not preserve the native key")


def bind_weight_file(path):
    control = _current.get()
    if control is not None:
        path = str(Path(path).resolve())
        value = _hash(path)
        if path in control.weights and control.weights[path] != value:
            raise ValueError(f"checkpoint weight identity changed: {path}")
        control.weights[path] = value


class ManagedCheckpoint:
    def __init__(self, report_path, *, arguments, sdk=None):
        self.sdk = sdk or _sdk()
        context = self.sdk._context()  # Verify request, group, PID/start and owner.
        if context is None:
            raise ValueError("managed checkpoint context is missing")
        self.queue, self.job, self.request, self.process = context
        self.report_path = Path(report_path).resolve()
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(__file__).resolve().parents[1]
        sources = [self.root / "generate.py", *sorted((self.root / "trellmlx").rglob("*.py"))]
        # Args supply actual inputs, not cwd-based guesses; output/control paths
        # are intentionally excluded because generation writes them.
        inputs = {}
        excluded = {"output", "resume", "save_checkpoints", "greenroom_pause_report",
                    "greenroom_queue_dir", "checkpoint_stop_file"}
        for key, value in arguments.items():
            if key in excluded:
                continue
            for item in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(item, str) and Path(item).is_file():
                    path = str(Path(item).resolve())
                    inputs[path] = _hash(path)
        self.identity = {"schema": "trellis2mlx.managed-replay.v1", "arguments": arguments,
                         "source_root": str(self.root),
                         "sources": {str(p.relative_to(self.root)): _hash(p) for p in sources},
                         "inputs": inputs, "mlx_version": mx.__version__}
        self.identity = json.loads(json.dumps(self.identity))
        saved = self.sdk.load_checkpoint()
        self.records, self.weights, self.partial = [], {}, None
        if saved is not None:
            if saved.get("identity") != self.identity:
                raise ValueError("Trellis checkpoint source/input/config identity changed")
            self.records, self.weights = saved["records"], saved["weights"]
            self.partial = self._load(saved["partial"]) if saved.get("partial") else None
            self.directory = Path(saved["directory"])
            for path, digest in self.weights.items():
                if _hash(path) != digest:
                    raise ValueError(f"checkpoint weight identity changed: {path}")
        else:
            self.directory = self.report_path.parent / f"replay-{uuid.uuid4().hex}"
            self.directory.mkdir()
        self.cursor, self.in_sampler, self.active_index = 0, False, None
        self.state = {"schema": "trellis2mlx.managed-pause.v1", "mode": "checkpoint-exit-continuation",
                      "status": "running", "pid": os.getpid(), "queue_dir": str(self.queue.queue_dir),
                      "job_id": self.request.job_id, "source_root": str(self.root),
                      "sdk": str(Path(self.sdk.__file__).resolve()),
                      "sdk_sha256": _hash(self.sdk.__file__),
                      "memory_released": False, "worker_lock_released": False,
                      "report_path": str(self.report_path), "directory": str(self.directory)}
        self._write()

    def _write(self, **changes):
        self.state.update(changes, updated_at=time.time())
        _write(self.report_path, self.state)

    def _save(self, value, name):
        arrays, numpy_arrays = {}, {}

        def encode(item):
            if isinstance(item, mx.array):
                key = str(len(arrays))
                arrays[key] = item
                return {"mlx": key}
            if isinstance(item, np.ndarray):
                key = str(len(numpy_arrays))
                numpy_arrays[key] = item
                return {"numpy": key}
            if isinstance(item, np.generic):
                return encode(np.asarray(item))
            if isinstance(item, (tuple, list)):
                return {"tuple" if isinstance(item, tuple) else "list": [encode(x) for x in item]}
            if isinstance(item, dict):
                if not all(isinstance(k, str) for k in item):
                    raise TypeError("checkpoint dictionaries require string keys")
                return {"dict": {k: encode(v) for k, v in item.items()}}
            if item is None or type(item) in (str, int, float, bool):
                return {"scalar": item}
            raise TypeError(f"unsupported checkpoint value: {type(item).__name__}")

        tree = encode(value)
        files = {}
        if arrays:
            path = self.directory / f"{name}.safetensors"
            mx.save_safetensors(str(path), arrays)
            files["mlx"] = {"path": str(path), "sha256": _hash(path)}
        if numpy_arrays:
            path = self.directory / f"{name}.npz"
            np.savez(path, **numpy_arrays)
            files["numpy"] = {"path": str(path), "sha256": _hash(path)}
        return {"tree": tree, "files": files}

    def _load(self, record):
        sources = {}
        for kind, artifact in record["files"].items():
            path = Path(artifact["path"])
            if _hash(path) != artifact["sha256"]:
                raise ValueError(f"checkpoint tensor bytes changed: {path}")
            sources[kind] = mx.load(str(path)) if kind == "mlx" else np.load(path, allow_pickle=False)

        def decode(node):
            kind, value = next(iter(node.items()))
            if kind in ("mlx", "numpy"):
                return sources[kind][value]
            if kind == "dict":
                return {k: decode(v) for k, v in value.items()}
            if kind == "tuple":
                return tuple(decode(v) for v in value)
            if kind == "list":
                return [decode(v) for v in value]
            if kind == "scalar":
                return value
            raise ValueError("unsupported checkpoint tensor tree")
        try:
            return decode(record["tree"])
        finally:
            for source in sources.values():
                if hasattr(source, "close"):
                    source.close()

    def call(self, function, args, kwargs, *, label=None):
        # Nested implementation details belong to their outer checkpoint.
        if self.active_index is not None:
            return function(*args, **kwargs)
        index = self.cursor
        self.cursor += 1
        key = label or f"{function.__module__}.{function.__qualname__}"
        if index < len(self.records):
            record = self.records[index]
            if record["key"] != key:
                raise ValueError("checkpoint call order changed")
            value = self._load(record["value"])
            restore_rng(value["rng"])
            self._restore_captures(kwargs, value.get("captures", {}))
            return value["result"]
        if index != len(self.records):
            raise ValueError("checkpoint completed-call prefix is incomplete")
        self.active_index = index
        self.in_sampler = function.__name__ == "flow_euler_sample"
        try:
            result = function(*args, **kwargs)
            payload = {"result": result, "rng": list(mx.random.state),
                       "captures": self._captures(kwargs)}
            record = {"key": key, "value": self._save(payload, f"call-{index}")}
            self.records.append(record)
            self.partial = None
        finally:
            self.active_index, self.in_sampler = None, False
        self.boundary(key, None)
        return result

    @staticmethod
    def _captures(kwargs):
        return {k: kwargs[k] for k in ("capture_first_step", "capture_steps") if kwargs.get(k) is not None}

    @staticmethod
    def _restore_captures(kwargs, captures):
        for key, value in captures.items():
            target = kwargs[key]
            target.clear()
            target.update(value) if isinstance(target, dict) else target.extend(value)

    def boundary(self, stage, completed_step=None):
        self.state.update(stage=stage, completed_step=completed_step)
        self.stop.checkpoint()
        dispatch = importlib.import_module("gpu_queue.dispatch")
        policy = dispatch.policy(self.queue.queue_dir)
        hold = policy.get("job_holds", {}).get(self.request.job_id)
        if hold and hold["request_digest"] != _hash(self.job / "request.json"):
            raise ValueError("pause request identity changed")
        pending = dispatch.pending_order(self.queue.queue_dir, policy)
        quick = False
        if pending and self.process.get("dispatch", {}).get("service_class") == "normal":
            request, digest = dispatch.read_request(pending[0] / "request.json")
            quick = dispatch.classify(request, policy, request_digest=digest, queue_dir=self.queue.queue_dir)[0] == "quick"
        if hold or self.queue.is_paused() or quick:
            raise _CheckpointRequested()

    def run(self, operation):
        from .cleanup import _clear_mlx_cache
        cancellation = importlib.import_module("gpu_queue.cancellation").cancellation
        with cancellation() as self.stop:
            while True:
                self.cursor = 0
                token = _current.set(self)
                try:
                    self.boundary("startup")
                    result = operation()
                except _CheckpointRequested:
                    pass  # Drop traceback/model frames before quiescence.
                except BaseException as error:
                    self._write(status="cancelled" if isinstance(error, SystemExit) and error.code == 130 else "failed",
                                error_type=type(error).__name__, error=str(error))
                    raise
                else:
                    self._write(status="finished", completed_calls=len(self.records))
                    return result
                finally:
                    _current.reset(token)
                # Save live partial tensors before unloading the default stream.
                partial = self._save(self.partial, f"partial-{uuid.uuid4().hex}") if self.partial else None
                self.partial = None
                gc.collect()
                mx.synchronize()
                _clear_mlx_cache()
                self._write(status="handoff-pending", completed_calls=len(self.records),
                            submitted_work_drained=True)
                payload = {"identity": self.identity, "records": self.records, "weights": self.weights,
                           "directory": str(self.directory), "partial": partial}
                self.sdk.yield_if_requested(
                    save_checkpoint=lambda directory: payload,
                    quiesce=lambda: True,
                    continuation=lambda checkpoint: replace(self.request, job_id=uuid.uuid4().hex[:12],
                                                           submitted_at=time.time()),
                    on_submitted=lambda request: self._write(status="checkpointed", next_job_id=request.job_id),
                )
                # Pause may have been withdrawn or cancellation won the race.
                try:
                    self.stop.checkpoint()
                except BaseException as error:
                    self._write(status="cancelled", error_type=type(error).__name__, error=str(error))
                    raise
                self.partial = self._load(partial) if partial else None
                self._write(status="running")


def managed_checkpoint(report_path, arguments):
    if not os.environ.get("GPU_GREENROOM_CONTEXT"):
        return None
    try:
        control = ManagedCheckpoint(report_path, arguments=arguments)
        admission = importlib.import_module("gpu_queue.admission")
        scripts = admission.entry_script(control.request.command_argv, control.request.command_cwd)
        if [Path(p).resolve() for p in scripts] != [control.root / "generate.py"]:
            raise ValueError("managed Trellis CLI must be the admitted entrypoint; nested controllers own their continuation")
        binding = json.loads((control.job / "request.json").read_text()).get("params", {}).get("_producer_checkpoint")
        if binding is not None:
            if binding.get("schema") != "gpu-greenroom.producer-checkpoint-binding.v1":
                raise ValueError("unsupported admitted producer checkpoint binding")
            for key in ("declaration", "entrypoint"):
                item = binding[key]
                if _hash(item["path"]) != item["sha256"]:
                    raise ValueError(f"admitted producer checkpoint {key} changed")
        return control
    except BaseException as error:
        _write(report_path, {"schema": "trellis2mlx.managed-pause.v1", "status": "failed",
                             "phase": "managed_preflight", "pid": os.getpid(),
                             "error_type": type(error).__name__, "error": str(error)})
        raise

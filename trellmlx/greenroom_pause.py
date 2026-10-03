"""Resident cooperative Pause/Play at safe Trellis boundaries.

Consume Greenroom's existing queue/paused file; never remove it or release the
worker's lock. Model memory stays resident. This is not checkpoint-and-exit.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import uuid

import mlx.core as mx


class GreenroomPause:
    def __init__(self, queue_dir, report_path):
        self.queue_dir = Path(queue_dir).resolve()
        self.pause_path = self.queue_dir / "paused"
        self.report_path = Path(report_path).resolve()
        if self.report_path.is_relative_to(self.queue_dir):
            relative = self.report_path.relative_to(self.queue_dir)
            if not relative.parts or relative.parts[0] != "outputs":
                raise ValueError("pause report must not write into Greenroom control/job state")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        # A prior invocation's paused acknowledgment must never look current.
        with self.report_path.open("x"):
            pass
        self.state = {
            "schema": "trellis2mlx.greenroom-pause.v1",
            "run_id": uuid.uuid4().hex, "pid": os.getpid(),
            "source_root": str(Path(__file__).resolve().parents[1]),
            "queue_dir": str(self.queue_dir), "pause_path": str(self.pause_path),
            "mode": "resident-cooperative-wait", "device": str(mx.default_device()),
            "status": "starting", "stage": "control_preflight", "completed_step": None,
            "queue_verified": False,
            "submitted_work_drained": False, "drained_stream": "mlx-default",
            "memory_released": False, "worker_lock_released": False,
        }
        self._write()
        try:
            if not all((self.queue_dir / name).is_dir() for name in
                       ("pending", "running", "done", "failed", "cancelled")):
                raise ValueError(f"not an initialized Greenroom queue: {self.queue_dir}")
            info = self.queue_dir.stat()
            self._queue_identity = (info.st_dev, info.st_ino)
        except Exception as error:
            self._write(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        self._write(status="running", stage="startup", queue_verified=True)

    def _write(self, **changes):
        self.state.update(changes, updated_at=time.time())
        temporary = self.report_path.with_name(f".{self.report_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w") as stream:
                json.dump(self.state, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.report_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_pause(self):
        info = self.queue_dir.stat()  # Losing the queue is not Play.
        if (info.st_dev, info.st_ino) != self._queue_identity:
            raise ValueError("Greenroom queue was replaced during this run")
        try:
            text = self.pause_path.read_text().strip()
        except FileNotFoundError:
            return None
        if not text:
            return {"schema": "gpu-greenroom.pause-state.legacy-marker", "epoch": None}
        payload = json.loads(text)
        if not isinstance(payload, dict) or payload.get("schema") != "gpu-greenroom.pause-state.v1":
            raise ValueError("unsupported Greenroom pause state")
        if payload.get("status") != "effective" or payload.get("queue_dir") != str(self.queue_dir):
            raise ValueError("Greenroom pause state does not name this effective queue")
        return payload

    def boundary(self, stage, completed_step=None):
        self.state.update(stage=stage, completed_step=completed_step)
        pause = self._read_pause()
        if pause is None:
            return
        self._write(status="draining", pause=pause, submitted_work_drained=False)
        mx.synchronize()  # Never acknowledge idle GPU work before this returns.
        self._write(status="paused", submitted_work_drained=True)
        print(f"  Greenroom paused at {stage} (completed step {completed_step}); model memory retained", flush=True)
        while True:
            pause = self._read_pause()
            if pause is None:
                break
            if pause != self.state["pause"]:
                self._write(pause=pause)
            time.sleep(0.25)
        self._write(status="running", last_pause=self.state["pause"], pause=None,
                    submitted_work_drained=False)
        print(f"  Greenroom Play: continuing {stage}", flush=True)


@contextmanager
def pause_session(queue_dir, report_path):
    if queue_dir is None:
        yield None
        return
    control = GreenroomPause(queue_dir, report_path)
    try:
        control.boundary("startup")
        yield control
    except BaseException as error:
        try:
            control._write(status="stopped" if isinstance(error, SystemExit) else "failed",
                           submitted_work_drained=False,
                           error_type=type(error).__name__, error=str(error))
        except Exception as report_error:
            error.add_note(f"Could not update Greenroom pause report: {report_error}")
        raise
    else:
        control._write(status="finished", submitted_work_drained=False)

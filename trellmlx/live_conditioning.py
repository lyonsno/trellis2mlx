"""One-shot loopback ingress for Kaminos's live DINOv3 F32 conditioning tensor.

The HTTP reply remains pending until the real generate.py sampler reports use of
the MLX arrays. This module does not claim that the producer's WebGPU work ran.
"""

import base64
import binascii
from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
from threading import Event, Lock, Thread
from uuid import uuid4

import numpy as np


SHAPE = (1, 1029, 1024)
BYTE_LENGTH = int(np.prod(SHAPE)) * 4
ENVELOPE_SCHEMA = "kaminos.trellis-dinov3-live-conditioning.v1"
RECEIPT_SCHEMA = "trellis2mlx.live-conditioning-consumer.v1"


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def write_live_preflight_failure(report_path, error):
    _write_json(report_path, {"schema": RECEIPT_SCHEMA, "ok": False,
                              "failurePhase": "local-preflight", "error": str(error),
                              "lastTrustworthyEvidence": "parsed command arguments only"})


@dataclass(frozen=True)
class ReceivedConditioning:
    body: bytes
    envelope: dict
    sha256: str


class LiveConditioningReceiver:
    """Receive exactly one POST on an ephemeral loopback port."""

    def __init__(self, start_receipt_path, report_path, *, source_revision,
                 requested_stage="sparse_flow_step"):
        self.start_receipt_path = Path(start_receipt_path)
        self.report_path = Path(report_path)
        self.source_revision = source_revision
        self.requested_stage = requested_stage
        self.session_id = str(uuid4())
        self._lock = Lock()
        self._received_event = Event()
        self._completed_event = Event()
        self._claimed = False
        self._received = None
        self._error = None
        self._http_status = None
        self._receipt = None
        self._server = None
        self._thread = None
        self.url = None

    def start(self):
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def _respond(self, status, payload):
                body = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path != "/conditioning":
                    self._respond(404, {"ok": False, "failurePhase": "wrong-path"})
                    return
                with receiver._lock:
                    if receiver._claimed:
                        if self.headers.get("Content-Length") == str(BYTE_LENGTH):
                            self.rfile.read(BYTE_LENGTH)
                        self._respond(409, {"ok": False, "failurePhase": "duplicate-conditioning-request"})
                        return
                try:
                    received = receiver._validate_and_read(self)
                except (ValueError, UnicodeError, json.JSONDecodeError) as error:
                    self._respond(400, receiver._reject(str(error)))
                    return
                with receiver._lock:
                    if receiver._claimed:
                        self._respond(409, {"ok": False, "failurePhase": "duplicate-conditioning-request"})
                        return
                    receiver._claimed = True
                receiver._received = received
                receiver._received_event.set()
                receiver._completed_event.wait()
                try:
                    self._respond(receiver._http_status, receiver._receipt)
                except (BrokenPipeError, ConnectionResetError):
                    # The durable terminal report, not the socket, owns the result.
                    pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_port}/conditioning"
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        _write_json(self.start_receipt_path, {
            "schema": "trellis2mlx.live-conditioning-start.v1",
            "pid": os.getpid(), "sessionId": self.session_id,
            "sourceRevision": self.source_revision, "requestedStage": self.requested_stage,
            "url": self.url,
        })
        return self

    def _validate_and_read(self, handler):
        headers = handler.headers
        if headers.get("Content-Type") != "application/octet-stream":
            raise ValueError("Content-Type must be application/octet-stream")
        raw_length = headers.get("Content-Length", "")
        if not raw_length.isdecimal() or int(raw_length) > BYTE_LENGTH:
            raise ValueError(f"Content-Length must be {BYTE_LENGTH}")
        body = handler.rfile.read(int(raw_length))
        if raw_length != str(BYTE_LENGTH):
            raise ValueError(f"Content-Length must be {BYTE_LENGTH}")
        encoded = headers.get("X-Kaminos-Conditioning-Envelope")
        if not encoded:
            raise ValueError("missing Kaminos conditioning envelope")
        try:
            envelope = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        except (ValueError, UnicodeError, binascii.Error) as error:
            raise ValueError("invalid Kaminos conditioning envelope") from error
        if not isinstance(envelope, dict):
            raise ValueError("conditioning envelope must be a JSON object")
        tensor = envelope.get("tensor")
        producer = envelope.get("producer")
        if envelope.get("schema") != ENVELOPE_SCHEMA or not isinstance(tensor, dict) or not isinstance(producer, dict):
            raise ValueError("wrong conditioning envelope schema")
        request_id = envelope.get("requestId")
        if not isinstance(request_id, str) or not request_id or headers.get("X-Request-Id") != request_id:
            raise ValueError("request identity mismatch")
        if not isinstance(producer.get("sessionId"), str) or not producer["sessionId"]:
            raise ValueError("missing producer session identity")
        if (tensor.get("name"), tensor.get("dtype"), tensor.get("byteOrder"), tensor.get("layout")) != (
            "cond", "float32", "little-endian", "BSH"
        ):
            raise ValueError("wrong conditioning tensor format")
        if tensor.get("shape") != list(SHAPE) or tensor.get("byteLength") != BYTE_LENGTH:
            raise ValueError("wrong conditioning tensor shape or byte length")
        if (headers.get("X-Tensor-Name"), headers.get("X-Tensor-Dtype"), headers.get("X-Tensor-Shape")) != (
            "cond", "float32", "1,1029,1024"
        ):
            raise ValueError("tensor headers differ from the pinned F32 BSH format")
        sha = tensor.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("invalid envelope tensor SHA-256")
        if headers.get("X-Tensor-Sha256") != sha:
            raise ValueError("tensor header and envelope SHA-256 differ")
        if len(body) != BYTE_LENGTH:
            raise ValueError("partial conditioning body")
        actual_sha = hashlib.sha256(body).hexdigest()
        if actual_sha != sha:
            raise ValueError("conditioning body SHA-256 mismatch")
        values = np.frombuffer(body, dtype="<f4")
        if not np.isfinite(values).all() or not np.any(values):
            raise ValueError("conditioning body contains non-finite values or is blank")
        return ReceivedConditioning(body=body, envelope=envelope, sha256=sha)

    def _reject(self, error):
        with self._lock:
            if self._claimed:
                return {"schema": RECEIPT_SCHEMA, "ok": False,
                        "failurePhase": "duplicate-conditioning-request", "error": error}
            self._claimed = True
        self._error = error
        self._receipt = {"schema": RECEIPT_SCHEMA, "ok": False,
                         "failurePhase": "validate-conditioning-request", "error": error,
                         "receiver": self._receiver_identity()}
        self._http_status = 400
        try:
            _write_json(self.report_path, self._receipt)
        except OSError as report_error:
            self._receipt = {**self._receipt, "reportWriteError": str(report_error)}
        finally:
            self._received_event.set()
            self._completed_event.set()
        return self._receipt

    def _receiver_identity(self):
        return {"pid": os.getpid(), "sessionId": self.session_id,
                "sourceRevision": self.source_revision, "url": self.url}

    def wait(self):
        self._received_event.wait()
        if self._error:
            raise ValueError(self._error)
        return self._received

    def complete(self, *, mlx_device, cond_object_id, output_finite):
        if not output_finite:
            self.fail("sparse-flow-step", "sampler output was not finite")
            return
        if self._received is None:
            raise RuntimeError("no live conditioning tensor was accepted")
        envelope = self._received.envelope
        self._receipt = {
            "schema": RECEIPT_SCHEMA, "ok": True,
            "requestId": envelope["requestId"],
            "producerSessionId": envelope["producer"]["sessionId"],
            "receiver": self._receiver_identity(),
            "tensor": {"sha256": self._received.sha256, "shape": list(SHAPE),
                       "dtype": "float32", "byteOrder": "little-endian", "byteLength": BYTE_LENGTH},
            "transfer": {"receiverObservedHttpBodyBytes": BYTE_LENGTH,
                         "mlxUpload": "mx.array from receiver-owned F32 body; evaluated"},
            "mlx": {"device": str(mlx_device), "condArrayObjectId": int(cond_object_id),
                    "condEvaluated": True, "negCondPolicy": "mx.zeros_like(cond)",
                    "negCondEvaluated": True},
            "sampler": {"stage": "sparse_flow_step", "consumedCond": True,
                        "consumedNegCond": True, "outputFinite": True},
        }
        try:
            _write_json(self.report_path, self._receipt)
        except OSError as report_error:
            self._error = f"terminal report write failed: {report_error}"
            self._http_status = 500
            self._receipt = {"schema": RECEIPT_SCHEMA, "ok": False,
                             "failurePhase": "write-terminal-report", "error": self._error,
                             "receiver": self._receiver_identity(),
                             "lastTrustworthyEvidence": {"receivedTensorSha256": self._received.sha256}}
            raise
        else:
            self._http_status = 200
        finally:
            self._completed_event.set()

    def fail(self, phase, error):
        if self._completed_event.is_set():
            return
        self._error = str(error)
        self._receipt = {"schema": RECEIPT_SCHEMA, "ok": False,
                         "failurePhase": phase, "error": str(error),
                         "receiver": self._receiver_identity(),
                         "lastTrustworthyEvidence": {"receivedTensorSha256": self._received.sha256}
                         if self._received else None}
        self._http_status = 500
        try:
            _write_json(self.report_path, self._receipt)
        except OSError as report_error:
            self._receipt = {**self._receipt, "reportWriteError": str(report_error)}
        finally:
            self._received_event.set()
            self._completed_event.set()

    def close(self):
        if self._server is None:
            return
        if not self._completed_event.is_set():
            self.fail("receiver-closed", "receiver closed before sparse sampler completion")
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
        self._server = None

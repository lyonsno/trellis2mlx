"""The real HTTP wire into generate.py, without requiring a GPU in these tests."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from trellmlx.live_conditioning import BYTE_LENGTH, LiveConditioningReceiver


GENERATE = Path(__file__).resolve().parents[1] / "generate.py"


def test_generate_live_route_requires_real_sampler_stage(tmp_path):
    command = [__import__("sys").executable, str(GENERATE),
               "--live-conditioning-listen", "127.0.0.1:0",
               "--live-conditioning-start-receipt", str(tmp_path / "start.json"),
               "--live-conditioning-report", str(tmp_path / "terminal.json"),
               "--save-checkpoints", str(tmp_path / "checkpoints"),
               "--stop-after-stage", "conditioning"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "--live-conditioning-listen requires --stop-after-stage sparse_flow_step" in result.stderr
    assert not (tmp_path / "start.json").exists()
    failure = json.loads((tmp_path / "terminal.json").read_text())
    assert failure["failurePhase"] == "local-preflight"
    assert failure["ok"] is False


def _body():
    return struct.pack("<f", 1.25) + bytes(BYTE_LENGTH - 4)


def _request(url, body, *, digest=None, shape="1,1029,1024"):
    sha = digest or hashlib.sha256(body).hexdigest()
    envelope = {
        "schema": "kaminos.trellis-dinov3-live-conditioning.v1",
        "requestId": "request-7",
        "producer": {"process": "Chrome", "pid": 123, "sessionId": "session-7", "sourceRevision": "a" * 40},
        "tensor": {"name": "cond", "dtype": "float32", "byteOrder": "little-endian", "layout": "BSH",
                   "shape": [1, 1029, 1024], "byteLength": BYTE_LENGTH, "sha256": sha},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(envelope).encode()).decode().rstrip("=")
    return Request(url, data=body, method="POST", headers={
        "Content-Type": "application/octet-stream", "X-Request-Id": "request-7",
        "X-Tensor-Name": "cond", "X-Tensor-Dtype": "float32", "X-Tensor-Shape": shape,
        "X-Tensor-Sha256": sha, "X-Kaminos-Conditioning-Envelope": encoded,
    })


def test_one_post_waits_for_sampler_and_returns_consumer_receipt(tmp_path):
    start = tmp_path / "start.json"
    report = tmp_path / "terminal.json"
    receiver = LiveConditioningReceiver(start, report, source_revision="b" * 40,
                                         requested_stage="sparse_flow_step").start()
    try:
        start_data = json.loads(start.read_text())
        assert start_data["url"] == receiver.url
        assert start_data["pid"] > 0
        body = _body()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(urlopen, _request(receiver.url, body))
            received = receiver.wait()
            assert received.body == body
            assert received.envelope["producer"]["sessionId"] == "session-7"
            assert not pending.done(), "HTTP success must wait for actual sampler completion"
            receiver.complete(mlx_device="gpu", cond_object_id=88, output_finite=True)
            response = pending.result(timeout=5)
            payload = json.load(response)
        assert response.status == 200
        assert payload == json.loads(report.read_text())
        assert payload["schema"] == "trellis2mlx.live-conditioning-consumer.v1"
        assert payload["ok"] is True
        assert payload["tensor"]["sha256"] == hashlib.sha256(body).hexdigest()
        assert payload["sampler"] == {"stage": "sparse_flow_step", "consumedCond": True,
                                       "consumedNegCond": True, "outputFinite": True}
        assert payload["mlx"]["condArrayObjectId"] == 88
    finally:
        receiver.close()


@pytest.mark.parametrize("truncate,shape", [(True, "1,1029,1024"), (False, "1,1,1024")])
def test_partial_or_wrong_shape_fails_with_durable_report(tmp_path, truncate, shape):
    report = tmp_path / "terminal.json"
    receiver = LiveConditioningReceiver(tmp_path / "start.json", report,
                                         source_revision="b" * 40,
                                         requested_stage="sparse_flow_step").start()
    try:
        body = _body()[:-4] if truncate else _body()
        with pytest.raises(HTTPError) as error:
            urlopen(_request(receiver.url, body, shape=shape))
        assert error.value.code == 400
        with pytest.raises(ValueError):
            receiver.wait()
        payload = json.loads(report.read_text())
        assert payload["ok"] is False
        assert payload["failurePhase"] == "validate-conditioning-request"
    finally:
        receiver.close()


def test_digest_mismatch_cannot_be_accepted(tmp_path):
    report = tmp_path / "terminal.json"
    receiver = LiveConditioningReceiver(tmp_path / "start.json", report,
                                         source_revision="b" * 40,
                                         requested_stage="sparse_flow_step").start()
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(_request(receiver.url, _body(), digest="0" * 64))
        assert error.value.code == 400
        assert json.loads(report.read_text())["failurePhase"] == "validate-conditioning-request"
    finally:
        receiver.close()


def test_sampler_failure_cannot_become_http_success(tmp_path):
    report = tmp_path / "terminal.json"
    receiver = LiveConditioningReceiver(tmp_path / "start.json", report,
                                         source_revision="b" * 40,
                                         requested_stage="sparse_flow_step").start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(urlopen, _request(receiver.url, _body()))
            assert receiver.wait().sha256 == hashlib.sha256(_body()).hexdigest()
            receiver.fail("sparse-flow-step", "Metal sampler failed")
            with pytest.raises(HTTPError) as error:
                pending.result(timeout=5)
        assert error.value.code == 500
        payload = json.loads(report.read_text())
        assert payload["ok"] is False
        assert payload["failurePhase"] == "sparse-flow-step"
        assert payload["lastTrustworthyEvidence"]["receivedTensorSha256"] == hashlib.sha256(_body()).hexdigest()
    finally:
        receiver.close()


def test_second_post_rejected_while_first_is_waiting(tmp_path):
    receiver = LiveConditioningReceiver(tmp_path / "start.json", tmp_path / "terminal.json",
                                         source_revision="b" * 40,
                                         requested_stage="sparse_flow_step").start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(urlopen, _request(receiver.url, _body()))
            receiver.wait()
            with pytest.raises(HTTPError) as error:
                urlopen(_request(receiver.url, _body()))
            assert error.value.code == 409
            receiver.complete(mlx_device="gpu", cond_object_id=88, output_finite=True)
            assert pending.result(timeout=5).status == 200
    finally:
        receiver.close()

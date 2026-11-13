"""Tests for false closure in the installed-route consumer observer (no MLX)."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "installed_paws_asset", Path(__file__).parents[1] / "scripts/check_installed_paws_asset.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_terminal_waits_for_published_directory_and_receipt():
    # Observed worker contract: done status is written while still in running;
    # immutable receipt and terminal rename are later publication milestones.
    pending_move = {"bucket": "running", "state": {"status": "done"}}
    published = {"bucket": "done", "state": {"status": "done"}, "receipt": {"status": "done"}}
    reads = iter([pending_move, published])
    result = m.wait_terminal(lambda _: next(reads), "parent", poll=lambda _: None)
    assert result is published, "status-field terminality preceded actual terminal publication"


def test_terminal_publication_without_receipt_fails_visibly():
    with pytest.raises(ValueError, match="no execution receipt"):
        m.wait_terminal(lambda _: {"bucket": "done", "state": {"status": "done"}}, "parent")


def plan(tmp_path):
    return {"run_root": str(tmp_path), "output": str(tmp_path / "asset/output.glb"),
            "producer_root": "/product", "producer_sha": "product-sha", "sdk_root": "/sdk", "sdk_sha": "sdk-sha",
            "cwd": "/caller", "argv": ["/python", "-u", "/product/generate.py", "--image", "/input.png"]}


@pytest.mark.parametrize("change", [
    {"cooperative_checkpoint": False}, {"route_identity": "fallback"}, {"agent_id": "substituted"},
    {"repo_root": "/other"}, {"command_argv": ["/other"]}, {"params": {}},
])
def test_wrong_route_or_admission_rejected(tmp_path, change):
    p = plan(tmp_path)
    request = {"repo_root": p["producer_root"], "command_argv": p["argv"], "command_cwd": p["cwd"],
               "agent_id": None, "route_identity": m.ROUTE, "cooperative_checkpoint": True,
               "params": {"_producer_checkpoint": {"source": "declared"}}}
    m.validate_request(request, p)
    with pytest.raises(ValueError, match="wrong ordinary"):
        m.validate_request({**request, **change}, p)


def test_actual_worker_source_and_effective_argv_required(tmp_path):
    p = plan(tmp_path)
    receipt = {"status": "done", "exit_code": 0, "effective_argv": p["argv"], "effective_cwd": p["cwd"],
               "requested_route": m.ROUTE, "worker": {"source_root": p["sdk_root"], "commit": p["sdk_sha"], "git_dirty": False}}
    snapshot = {"state": {"status": "done", "exit_code": 0}, "receipt": receipt}
    m.validate_receipt(snapshot, p)
    with pytest.raises(ValueError, match="route/source"):
        m.validate_receipt({**snapshot, "receipt": {**receipt, "worker": {}}}, p)
    with pytest.raises(ValueError, match="route/source"):
        m.validate_receipt({**snapshot, "receipt": {**receipt, "effective_argv": ["fallback"]}}, p)


def test_resident_pause_or_other_job_cannot_supply_handoff_evidence(tmp_path):
    p = {**plan(tmp_path), "queue": "/queue"}
    state = {"mode": "checkpoint-exit-continuation", "queue_dir": "/queue", "job_id": "ours",
             "source_root": p["producer_root"], "sdk": "/sdk/gpu_queue/cooperative.py"}
    m.validate_pause(state, p, "ours")
    for change in [{"mode": "resident"}, {"job_id": "other"}, {"sdk": "/old/sdk"}]:
        with pytest.raises(ValueError, match="wrong route/source/job"):
            m.validate_pause({**state, **change}, p, "ours")


def tiny_glb(path, mutation=lambda scene: None):
    xyz = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    indices = struct.pack("<3I", 0, 1, 2)
    # Structural fixture: this observer checks the container and image signature,
    # not a decoded image, rendered appearance or an external GLB exporter.
    pixels = b"\x89PNG\r\n\x1a\nfixture"
    binary = xyz + indices + pixels
    binary += b"\0" * (-len(binary) % 4)
    scene = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
             "buffers": [{"byteLength": len(binary)}],
             "bufferViews": [{"byteLength": 36}, {"byteOffset": 36, "byteLength": 12}, {"byteOffset": 48, "byteLength": len(pixels)}],
             "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                           {"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"}],
             "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
             "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
             "textures": [{"source": 0}], "images": [{"bufferView": 2, "mimeType": "image/png"}]}
    mutation(scene)
    header = json.dumps(scene).encode(); header += b" " * (-len(header) % 4)
    payload = (struct.pack("<4sII", b"glTF", 2, 28 + len(header) + len(binary))
               + struct.pack("<I4s", len(header), b"JSON") + header
               + struct.pack("<I4s", len(binary), b"BIN\0") + binary)
    path.write_bytes(payload)


def test_glb_rejects_blank_partial_empty_and_out_of_bounds_output(tmp_path):
    path = tmp_path / "output.glb"
    for payload in [b"", b"glTF", b"an old error page"]:
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="container"):
            m.validate_glb(path)
    tiny_glb(path)
    assert m.validate_glb(path)["primitives"] == 1
    for mutation in [lambda s: s.update(meshes=[]), lambda s: s["accessors"][0].update(count=0),
                     lambda s: s["accessors"][0].update(count=999)]:
        tiny_glb(path, mutation)
        with pytest.raises(ValueError):
            m.validate_glb(path)
    tiny_glb(path, lambda s: s["materials"][0].update(pbrMetallicRoughness={}))
    with pytest.raises(KeyError):
        m.validate_glb(path)


@pytest.mark.parametrize("reason", ["stale", "source"])
def test_prelaunch_failure_reports_phase_and_does_not_start_model(tmp_path, monkeypatch, reason):
    p = plan(tmp_path)
    if reason == "stale":
        Path(p["output"]).parent.mkdir()
        Path(p["output"]).write_bytes(b"stale")
    else:
        def wrong_source(*_):
            raise ValueError("source mismatch")
        monkeypatch.setattr(m, "source", wrong_source)
    def forbidden(*a, **k):
        pytest.fail("model started before validated fresh source/output")
    monkeypatch.setattr(m.subprocess, "Popen", forbidden)
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match=reason):
        m.run(p, report)
    result = json.loads(report.read_text())
    assert result["status"] == "failed" and result["failure_phase"] == "preflight"


def test_missing_plan_still_writes_failure_report(tmp_path):
    report = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(Path(m.__file__)), "--plan", str(tmp_path / "absent"),
                             "--report", str(report)], capture_output=True)
    assert result.returncode != 0
    assert json.loads(report.read_text())["failure_phase"] == "read-plan"

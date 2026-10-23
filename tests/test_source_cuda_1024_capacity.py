"""Capacity evidence must distinguish admission, partial work and completion."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/source_cuda_1024_capacity.py"
spec = importlib.util.spec_from_file_location("cuda1024", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def completed():
    return {"effective_route": {"cuda_device": "Tesla T4", "capability": [7, 5],
        "source_commit": runner.SOURCE_COMMIT, "resolution": 1024,
        "attention": "sdpa", "sparse_attention": "sdpa", "sparse_conv": "flex_gemm"},
        "pipeline_run_calls": 1, "sampler_calls": {"sparse": 1, "shape": 2, "texture": 1},
        "artifacts": {name: {"sha256": "a" * 64} for name in ("raw_mesh.npz", "shape_terminal.npz", "texture_terminal.npz", "output.glb")},
        "glb": {"faces": 100}}


def test_complete_source_route():
    runner.validate_completion(completed())


@pytest.mark.parametrize("mutate", [
    lambda r: r["effective_route"].update(resolution=512),
    lambda r: r["effective_route"].update(cuda_device="CPU"),
    lambda r: r["effective_route"].update(source_commit="wrong"),
    lambda r: r["effective_route"].update(attention="fallback"),
    lambda r: r["sampler_calls"].update(shape=1),
    lambda r: r["artifacts"].pop("raw_mesh.npz"),
    lambda r: r["glb"].update(faces=0),
])
def test_partial_or_changed_route_is_not_completion(mutate):
    report = copy.deepcopy(completed())
    mutate(report)
    with pytest.raises(ValueError):
        runner.validate_completion(report)


def test_failure_before_primary_mesh_still_reports_and_retains_partial_bundle(tmp_path):
    report = tmp_path / "report.json"
    code = runner.main(["--request", str(tmp_path / "missing.json"), "--output-json", str(report),
        "--output-dir", str(tmp_path / "output"), "--runtime", str(tmp_path / "runtime")])
    assert code == 1
    state = json.loads(report.read_text())
    assert state["status"] == "failed"
    assert state["failure_phase"] == "input_admission"
    assert Path(state["bundle"]["path"]).is_file()
    assert not (tmp_path / "output" / "output.glb").exists()


def test_existing_output_is_preserved_and_refusal_is_separate(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    old = output / "output.glb"
    old.write_bytes(b"previous-evidence")
    report = tmp_path / "report.json"
    report.write_text('{"status":"completed","old":true}')
    code = runner.main(["--request", str(tmp_path / "missing.json"), "--output-json", str(report),
        "--output-dir", str(output), "--runtime", str(tmp_path / "runtime")])
    assert code == 1
    assert json.loads(report.read_text())["old"] is True
    assert old.read_bytes() == b"previous-evidence"
    refusal, = tmp_path.glob("report.json.refusal-*.json")
    assert json.loads(refusal.read_text())["status"] == "failed"


def test_changed_request_refuses_before_loading_models(tmp_path):
    with pytest.raises(ValueError, match="1024 source settings"):
        runner.validate_request({"settings": {"pipeline_type": "512"}}, tmp_path)

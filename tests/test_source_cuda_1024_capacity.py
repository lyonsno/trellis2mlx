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
        "attention": "sdpa", "sparse_attention": "xformers", "sparse_conv": "flex_gemm"},
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


@pytest.mark.parametrize("phase,preceding", [("conditioning", "official_preprocess"), ("shape_upsample", "shape-sampler-1")])
def test_native_failure_is_not_reported_as_preceding_completed_stage(phase, preceding):
    report = {"phase": "pipeline_run", "last_completed_stage": preceding}
    def fail():
        raise RuntimeError("native failure")
    wrapped = runner.phase_observer(fail, phase, report, lambda _: None)
    with pytest.raises(RuntimeError, match="native failure"):
        wrapped()
    assert report["phase"] == phase
    assert report["last_completed_stage"] == preceding


def test_sampler_restores_outer_phase_and_preserves_exact_texture_conditioning_on_failure():
    from types import SimpleNamespace
    report = {"phase": "texture_slat", "sampler_calls": {}, "last_completed_stage": "shape_cascade"}
    saved = {}
    def capture(name, **arrays):
        saved[name] = arrays
    def convert(value, prefix):
        if value is None:
            return {}
        return {prefix + "_feats": value.feats, prefix + "_coords": value.coords} if hasattr(value, "feats") else {prefix: value}
    def fail(*args, **kwargs):
        raise RuntimeError("texture failure")
    shape = SimpleNamespace(feats=[[1., 2.]], coords=[[0, 1, 2, 3]])
    wrapped = runner.sampler_observer(fail, "texture", report, lambda _: None, capture, convert)
    with pytest.raises(RuntimeError, match="texture failure"):
        wrapped(None, [9.], cond=[3.], neg_cond=[0.], concat_cond=shape)
    assert report["phase"] == "texture-sampler-1"
    assert report["last_completed_stage"] == "shape_cascade"
    arrays = saved["texture-sampler-1-input.npz"]
    assert arrays["concat_cond_feats"] == [[1., 2.]]
    assert arrays["concat_cond_coords"] == [[0, 1, 2, 3]]


def test_completed_sampler_returns_to_native_outer_phase():
    from types import SimpleNamespace
    report = {"phase": "shape_cascade", "sampler_calls": {}}
    wrapped = runner.sampler_observer(lambda *a, **k: SimpleNamespace(samples=2), "shape", report,
        lambda _: None, lambda *a, **k: None, lambda value, prefix: {})
    wrapped(None, 1)
    assert report["phase"] == "shape_cascade"
    assert report["last_completed_stage"] == "shape-sampler-1"

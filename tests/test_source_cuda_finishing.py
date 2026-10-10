import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "cuda_finishing", Path(__file__).parents[1] / "scripts/source_cuda_finishing.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixtures(tmp_path):
    mesh = tmp_path / "mesh_raw.npz"
    tex = tmp_path / "texture.npz"
    np.savez(mesh, vertices=np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0]], np.float32),
             faces=np.array([[0, 1, 2]], np.int64))
    np.savez(tex, tex_np=np.ones((2, 6), np.float32),
             tex_coords_spatial=np.array([[1, 2, 3], [4, 5, 6]], np.int32))
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (mesh, tex)]
    return mesh, tex, hashes


def test_exact_raw_inputs_are_admitted_without_cleanup(tmp_path):
    mesh, tex, hashes = fixtures(tmp_path)
    arrays = m.load_inputs(mesh, tex, *hashes, 768)
    assert arrays["faces"].tolist() == [[0, 1, 2]]
    assert arrays["vertices"].shape == (3, 3)
    assert arrays["attrs"].shape == (2, 6)
    assert arrays["coords"].tolist() == [[1, 2, 3], [4, 5, 6]]


def test_wrong_checkpoint_digest_rejected(tmp_path):
    mesh, tex, hashes = fixtures(tmp_path)
    with pytest.raises(ValueError, match="digest"):
        m.load_inputs(mesh, tex, "0" * 64, hashes[1], 768)


def test_out_of_grid_coordinates_rejected(tmp_path):
    mesh, tex, hashes = fixtures(tmp_path)
    with pytest.raises(ValueError, match="grid"):
        m.load_inputs(mesh, tex, *hashes, 4)


def test_substituted_source_file_rejected(tmp_path):
    source = tmp_path / "postprocess.py"
    source.write_text("def to_glb(): pass\n")
    with pytest.raises(ValueError, match="source digest"):
        m.verify_source(source, "0" * 64)


def test_blank_export_is_not_success(tmp_path):
    path = tmp_path / "output.glb"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="GLB"):
        m.validate_glb(path, 4096)


def test_glb_without_textured_geometry_is_not_success(tmp_path):
    document = json.dumps({"asset": {"version": "2.0"}, "meshes": []}).encode()
    document += b" " * (-len(document) % 4)
    path = tmp_path / "output.glb"
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 20 + len(document))
                     + struct.pack("<I4s", len(document), b"JSON") + document)
    with pytest.raises(ValueError, match="geometry"):
        m.validate_glb(path, 4096)


def test_observer_preserves_native_arguments_and_return():
    token = object()
    class Native:
        def __init__(self):
            self.calls = []
        def fill_holes(self, **kwargs):
            self.calls.append(kwargs)
            return token
        def read(self):
            return np.zeros((3, 3)), np.array([[0, 1, 2]])
    captures = []
    observed = m.observe_mesh_class(Native, lambda label, arrays: captures.append((label, arrays)))()
    assert observed.fill_holes(max_hole_perimeter=.03) is token
    assert observed.calls == [{"max_hole_perimeter": .03}]
    assert captures[0][0] == "fill_holes"
    assert captures[0][1]["faces"].tolist() == [[0, 1, 2]]


def test_failure_before_output_still_has_report(tmp_path):
    def fail(report):
        report["phase"] = "bootstrap"
        raise RuntimeError("compiler unavailable")
    path = tmp_path / "report.json"
    assert m.run_reported(path, {"status": "running", "phase": "start"}, fail) == 1
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "bootstrap"
    assert "compiler unavailable" in report["error"]


def test_both_arms_run_even_when_first_fails(tmp_path, monkeypatch):
    args = SimpleNamespace(mesh=tmp_path / "mesh", appearance=tmp_path / "tex",
                           mesh_sha="mesh", appearance_sha="tex", grid_size=768,
                           target_faces=1000000, texture_size=4096,
                           remesh_project=0, arms=["non-remesh", "remesh"],
                           runtime=tmp_path / "runtime", output_dir=tmp_path / "finishing",
                           output_json=tmp_path / "report.json")
    monkeypatch.setattr(m, "load_inputs", lambda *a: {})
    monkeypatch.setattr(m, "bootstrap", lambda *a: None)
    seen = []
    def run(argv, **kwargs):
        arm = argv[-1]
        seen.append(arm)
        m.write_json(args.output_dir / f"{arm}-report.json",
                     {"status": "failed" if arm == "non-remesh" else "completed"})
        return SimpleNamespace(returncode=1 if arm == "non-remesh" else 0)
    monkeypatch.setattr(m.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="one or more"):
        m.run_both(args, {})
    assert seen == ["non-remesh", "remesh"]
    assert (tmp_path / "finishing-bundle.tar").is_file()


def test_projection_control_reaches_actual_finishing_settings():
    args = m.parser().parse_args([
        "--mesh", "mesh_raw.npz", "--appearance", "texture.npz",
        "--mesh-sha", "mesh", "--appearance-sha", "texture",
        "--grid-size", "768", "--arms", "remesh",
        "--arm", "remesh", "--remesh-project", "0.9",
    ])
    assert args.arms == ["remesh"]
    settings = m.finishing_settings(args)
    assert settings["remesh"] is True
    assert settings["remesh_project"] == 0.9
    assert settings["remesh_band"] == 1
    assert settings["grid_size"] == 768
    assert settings["decimation_target"] == 1000000
    assert settings["texture_size"] == 4096


def test_selected_arm_and_projection_reach_child(tmp_path, monkeypatch):
    args = SimpleNamespace(mesh=tmp_path / "mesh", appearance=tmp_path / "tex",
                           mesh_sha="mesh", appearance_sha="tex", grid_size=768,
                           target_faces=1000000, texture_size=4096,
                           remesh_project=0.9, arms=["remesh"],
                           runtime=tmp_path / "runtime", output_dir=tmp_path / "finishing",
                           output_json=tmp_path / "report.json")
    monkeypatch.setattr(m, "load_inputs", lambda *a: {})
    monkeypatch.setattr(m, "bootstrap", lambda *a: None)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        m.write_json(args.output_dir / f"{argv[-1]}-report.json", {"status": "completed"})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess, "run", run)
    report = {}
    m.run_both(args, report)
    assert len(calls) == 1
    assert calls[0][-1] == "remesh"
    assert calls[0][calls[0].index("--remesh-project") + 1] == "0.9"
    assert set(report["arms"]) == {"remesh"}
    assert (tmp_path / "finishing-bundle.tar").is_file()


def test_original_two_arm_zero_projection_defaults_are_preserved():
    args = m.parser().parse_args([
        "--mesh", "mesh_raw.npz", "--appearance", "texture.npz",
        "--mesh-sha", "mesh", "--appearance-sha", "texture", "--grid-size", "768",
    ])
    assert args.arms == ["non-remesh", "remesh"]
    assert args.remesh_project == 0


def test_reconstruction_override_preserves_appearance_grid():
    args = m.parser().parse_args([
        "--mesh", "mesh_raw.npz", "--appearance", "texture.npz",
        "--mesh-sha", "mesh", "--appearance-sha", "texture",
        "--grid-size", "768", "--arms", "remesh", "--arm", "remesh",
        "--remesh-project", "0.9", "--remesh-resolution", "1024",
    ])
    assert args.remesh_resolution == 1024
    settings = m.finishing_settings(args)
    assert settings["grid_size"] == 768
    assert "remesh_resolution" not in settings
    assert settings["remesh_project"] == 0.9


def test_observed_reconstruction_override_reaches_native_call_only():
    seen, receipts = [], []
    token = object()
    def native(*args, **kwargs):
        seen.append((args, kwargs))
        return token
    original = dict(center="center", scale=771 / 768,
                    resolution=768, band=1, project_back=0.9, bvh="bvh")
    observed = m.observe_remesher(native, 1024, receipts.append)
    assert observed("vertices", "faces", **original) is token
    assert original["resolution"] == 768
    args, actual = seen[0]
    assert args == ("vertices", "faces")
    assert actual == dict(original, resolution=1024, scale=1027 / 1024)
    assert receipts == [{"source_resolution": 768, "effective_resolution": 1024,
                         "source_scale": 771 / 768, "effective_scale": 1027 / 1024,
                         "band": 1, "project_back": 0.9,
                         "resolution_override": True}]


def test_remesher_observer_default_forwards_without_reinterpretation():
    seen, receipts = [], []
    def native(*args, **kwargs):
        seen.append((args, kwargs))
        return "original"
    original = dict(resolution=512, scale=515 / 512, band=1, project_back=0)
    assert m.observe_remesher(native, None, receipts.append)(1, 2, **original) == "original"
    assert seen == [((1, 2), original)]
    assert receipts[0]["effective_resolution"] == 512
    assert receipts[0]["resolution_override"] is False


def test_remesher_receipt_precedes_native_failure():
    receipts = []
    def fail(*args, **kwargs):
        assert receipts[0]["effective_resolution"] == 1024
        raise RuntimeError("native allocation failed")
    with pytest.raises(RuntimeError, match="allocation"):
        m.observe_remesher(fail, 1024, receipts.append)(
            1, 2, resolution=768, scale=771 / 768, band=1, project_back=0.9)
    assert len(receipts) == 1


def test_resolution_control_reaches_child_without_changing_grid(tmp_path, monkeypatch):
    args = m.parser().parse_args([
        "--mesh", str(tmp_path / "mesh"), "--appearance", str(tmp_path / "tex"),
        "--mesh-sha", "mesh", "--appearance-sha", "tex", "--grid-size", "768",
        "--arms", "remesh", "--remesh-project", "0.9", "--remesh-resolution", "1024",
        "--runtime", str(tmp_path / "runtime"),
        "--output-dir", str(tmp_path / "finishing"),
        "--output-json", str(tmp_path / "report.json"),
    ])
    monkeypatch.setattr(m, "load_inputs", lambda *a: {})
    monkeypatch.setattr(m, "bootstrap", lambda *a: None)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        m.write_json(args.output_dir / f"{argv[-1]}-report.json", {"status": "completed"})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess, "run", run)
    m.run_both(args, {})
    assert len(calls) == 1
    assert calls[0][calls[0].index("--grid-size") + 1] == "768"
    assert calls[0][calls[0].index("--remesh-resolution") + 1] == "1024"


def test_invalid_reconstruction_resolution_rejected_before_native_call():
    with pytest.raises(ValueError, match="positive"):
        m.observe_remesher(lambda *a, **kw: pytest.fail("native called"), 0, lambda r: None)

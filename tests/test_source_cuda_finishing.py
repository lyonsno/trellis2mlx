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

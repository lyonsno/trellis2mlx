"""Local branch semantics, using CPU tensors and deliberately fake GPU primitives.

These tests do not establish the external Metal kernel contract.
"""
import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class UVBoundary(Exception):
    pass


def geometry_fixture():
    events = []
    raw_v = torch.tensor([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.]])
    raw_f = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    filled_v = raw_v + .01
    rebuilt_v = raw_v + .1

    class Mesh:
        num_vertices = 3
        num_faces = 1

        def init(self, v, f):
            events.append("init")
            self.v, self.f = v, f

        def read(self):
            return self.v, self.f

        def fill_holes(self, **kwargs):
            events.append("fill")
            assert kwargs == {"max_hole_perimeter": .03}
            self.v = filled_v

        def simplify(self, target, **kwargs):
            events.append("simplify")
            assert target == 17

        def uv_unwrap(self, **kwargs):
            raise UVBoundary

        def remove_duplicate_faces(self):
            events.append("duplicates")

        def repair_non_manifold_edges(self):
            events.append("split")

        def remove_small_connected_components(self, *args):
            events.append("delete_components")

    def bvh(v, f):
        events.append("bvh")
        assert torch.equal(v, filled_v)
        return SimpleNamespace(original_vertices=v, original_faces=f)

    def remesh(v, f, **kwargs):
        events.append("remesh")
        assert v is kwargs["bvh"].original_vertices
        assert kwargs["resolution"] == 768
        assert kwargs["project_back"] == .9
        assert kwargs["band"] == 1
        assert kwargs["scale"] == pytest.approx(771 / 768)
        return rebuilt_v, raw_f

    backend = SimpleNamespace(mesh=Mesh, bvh=bvh, remesh=remesh)
    return raw_v, raw_f, backend, events, filled_v


def test_original_remesh_geometry_order_and_original_surface():
    v, f, backend, events, filled = geometry_fixture()
    # The pre-fix experiment targets the actual native reference exporter.
    # Preserve that replay without making its checkout a test dependency.
    reference = os.environ.get("TRELLIS_REMESH_PREFIX_REFERENCE")
    if reference:
        spec = importlib.util.spec_from_file_location("reference_postprocess", reference)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._HAS_GPU_DEPS = True
        module._BACKEND = "metal"
        module._MeshBackend = backend.mesh
        module._BVH = backend.bvh
        module._remesh_narrow_band_dc = backend.remesh
        with pytest.raises(UVBoundary):
            module.to_glb(v, f, torch.ones((1, 6)), torch.zeros((1, 3), dtype=torch.int32),
                          {}, [[-.5] * 3, [.5] * 3], grid_size=768,
                          decimation_target=17, remesh=True)
    else:
        from trellmlx.source_metal_remesh import rebuild_surface
        mesh, original_v, original_f, bvh = rebuild_surface(
            v, f, backend=backend, grid_size=768, target_faces=17,
            band=1, project_back=.9)
        assert torch.equal(original_v, filled)
        assert original_v is bvh.original_vertices
        assert original_f is bvh.original_faces
        assert not torch.equal(mesh.read()[0], original_v)
    assert events == ["init", "fill", "bvh", "remesh", "init", "simplify"]


def texture_backend(blank=False, fail_sample=False):
    v, f, backend, events, filled = geometry_fixture()
    parent = backend.mesh

    class Mesh(parent):
        def uv_unwrap(self, **kwargs):
            assert kwargs["compute_charts_kwargs"] == {
                "threshold_cone_half_angle_rad": np.radians(90.),
                "refine_iterations": 0, "global_iterations": 1, "smooth_strength": 1}
            # A deliberate UV seam duplicates source vertex zero.
            return self.v[[0, 1, 2, 0]], torch.tensor([[3, 1, 2]], dtype=torch.int32), torch.tensor(
                [[0., 0.], [1., 0.], [0., 1.], [0., 0.]]), torch.tensor([0, 1, 2, 0])

        def compute_vertex_normals(self):
            events.append("normals")

        def read_vertex_normals(self):
            return torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    class BVH:
        def __init__(self, v, f):
            events.append("bvh")
            self.original_vertices, self.original_faces = v, f

        def unsigned_distance(self, query, **kwargs):
            assert kwargs == {"return_uvw": True}
            return torch.zeros(len(query)), torch.zeros(len(query), dtype=torch.long), torch.tensor([[1., 0., 0.]]).expand(len(query), -1)

    def rasterize(ctx, uv, faces, resolution):
        rast = torch.zeros((1, *resolution, 4))
        if not blank:
            rast[..., 3] = 1
        return rast, None

    def interpolate(vertices, rast, faces):
        return torch.ones((1, *rast.shape[1:3], 3)) * .4, None

    def sample(attrs, coords, *, shape, grid, mode):
        if fail_sample:
            raise RuntimeError("sample failed")
        assert shape == torch.Size([1, 6, 768, 768, 768])
        assert mode == "trilinear"
        # Must use the prefilled ORIGINAL mesh, not interpolated/rebuilt surface.
        assert torch.allclose(grid, ((filled[0] + .5) * 768).expand_as(grid))
        return attrs[0].expand(1, grid.shape[1], -1)

    backend.mesh, backend.bvh = Mesh, BVH
    backend.torch = torch
    backend.dr = SimpleNamespace(MtlRasterizeContext=lambda: None, rasterize=rasterize, interpolate=interpolate)
    backend.sample = sample
    return v, f, backend


def test_complete_texture_transfer_normal_mapping_and_material_policy():
    from trellmlx.source_metal_remesh import to_glb
    v, f, backend = texture_backend()
    captured = []
    mesh = to_glb(v, f, torch.tensor([[.2, .3, .4, .5, .6, 1.]]),
                  torch.zeros((1, 3), dtype=torch.int32), backend=backend,
                  grid_size=768, target_faces=17, texture_size=8,
                  capture=lambda name, data: captured.append(name))
    assert captured == ["raw", "filled_original", "rebuilt", "reduced", "uv", "normals"]
    assert np.array_equal(mesh.vertex_normals[0], mesh.vertex_normals[3])
    assert np.array_equal(mesh.vertex_normals[1], [0., 0., -1.])
    material = mesh.visual.material
    assert material.alphaMode == "OPAQUE"
    assert material.doubleSided is False
    assert material.baseColorTexture.size == (8, 8)
    assert np.array_equal(np.asarray(material.metallicRoughnessTexture)[0, 0], [0, 153, 127])


@pytest.mark.parametrize("blank,fail,reason", [(True, False, "no covered texels"), (False, True, "sample failed")])
def test_no_blank_texture_or_sampling_fallback(blank, fail, reason):
    from trellmlx.source_metal_remesh import to_glb
    v, f, backend = texture_backend(blank, fail)
    with pytest.raises(RuntimeError, match=reason):
        to_glb(v, f, torch.ones((1, 6)), torch.zeros((1, 3), dtype=torch.int32),
               backend=backend, grid_size=768, target_faces=17, texture_size=8)


def test_input_identity_and_grid_mismatch_fail_before_backend(tmp_path):
    from scripts.source_metal_remesh_finishing import digest, load_inputs
    v, f, *_ = geometry_fixture()
    np.savez(tmp_path / "mesh_raw.npz", vertices=v.numpy(), faces=f.numpy())
    np.savez(tmp_path / "texture.npz", tex_np=np.ones((1, 6)),
             tex_coords_spatial=np.zeros((1, 3), dtype=np.int32), mesh_grid_size=768)
    mesh_sha, texture_sha = [digest(tmp_path / p) for p in ("mesh_raw.npz", "texture.npz")]
    with pytest.raises(ValueError, match="input digest"):
        load_inputs(tmp_path, "wrong", texture_sha, 768)
    with pytest.raises(ValueError, match="saved mesh_grid_size"):
        load_inputs(tmp_path, mesh_sha, texture_sha, 512)
    assert len(load_inputs(tmp_path, mesh_sha, texture_sha, 768)[0]) == 3


def test_early_failure_has_durable_report_without_overwriting_previous_output(tmp_path):
    from argparse import Namespace
    import json
    from scripts.source_metal_remesh_finishing import run
    output = tmp_path / "existing"
    output.mkdir()
    previous = output / "report.json"
    previous.write_text("previous evidence")
    args = Namespace(output_dir=output)
    result = run(args, backend_loader=lambda: pytest.fail("backend must not run"))
    assert result["status"] == "failed"
    assert result["failure_phase"] == "output_admission"
    assert previous.read_text() == "previous evidence"
    reports = list(tmp_path.glob("existing-failure-*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["last_trustworthy_phase"] == "request"


def test_non_macos_route_fails_loud(monkeypatch):
    from trellmlx.source_metal_remesh import load_backend
    monkeypatch.setattr("platform.system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="requires macOS"):
        load_backend()


def test_missing_actual_native_library_cannot_validate(tmp_path):
    import json
    from scripts.source_metal_remesh_finishing import digest, verify_backend
    module = tmp_path / "implementation.py"
    module.write_text("observed implementation")
    actual = tmp_path / "actual.so"
    actual.write_bytes(b"actually imported library")
    unrelated = tmp_path / "other.so"
    unrelated.write_bytes(b"not imported")
    manifest = tmp_path / "backend.json"
    manifest.write_text(json.dumps({"modules": {"implementation": {"path": str(module), "sha256": digest(module)}},
                                   "binaries": [{"path": str(unrelated), "sha256": digest(unrelated)}]}))
    backend = SimpleNamespace(identity={"implementation": str(module)},
                              native_libraries=[str(actual)], declared_metallibs=[])
    with pytest.raises(ValueError, match="native library set"):
        verify_backend(backend, manifest)


def test_smoke_preparation_refusal_is_reported_without_dispatch(tmp_path, monkeypatch):
    import json
    from scripts import source_metal_remesh_smoke as smoke
    (tmp_path / "inputs").mkdir()
    marker = tmp_path / "inputs" / "previous.npz"
    marker.write_bytes(b"previous evidence")
    monkeypatch.setattr("sys.argv", ["smoke", "--root", str(tmp_path),
                                    "--backend-manifest", str(tmp_path / "backend.json"),
                                    "--expected-revision", "expected"])
    monkeypatch.setattr(smoke, "run", lambda *a, **kw: pytest.fail("must not dispatch"))
    assert smoke.main() == 1
    reports = list(tmp_path.glob("smoke-report*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "input_preparation"
    assert report["last_trustworthy_phase"] == "request"
    assert marker.read_bytes() == b"previous evidence"


def test_implementation_mutation_and_imported_extension_path_mismatch(tmp_path):
    import json
    from scripts.source_metal_remesh_finishing import digest, verify_backend
    implementation = tmp_path / "sampler.py"
    implementation.write_text("original sampling implementation")
    extension = tmp_path / "imported.so"
    extension.write_bytes(b"imported extension")
    record = {"path": str(implementation), "sha256": digest(implementation)}
    backend = SimpleNamespace(identity={"sampler": str(implementation)},
                              native_libraries=[str(extension)], declared_metallibs=[])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"modules": {"sampler": record},
                                   "binaries": [{"path": str(extension), "sha256": digest(extension)}]}))
    verify_backend(backend, manifest)
    implementation.write_text("changed sampling implementation")
    with pytest.raises(ValueError, match="source identity"):
        verify_backend(backend, manifest)
    implementation.write_text("original sampling implementation")
    backend.native_libraries = [str(tmp_path / "other-imported.so")]
    with pytest.raises(ValueError, match="native library set"):
        verify_backend(backend, manifest)


def test_smoke_preparation_write_failure_has_report(tmp_path, monkeypatch):
    import json
    from scripts import source_metal_remesh_smoke as smoke
    monkeypatch.setattr("sys.argv", ["smoke", "--root", str(tmp_path),
                                    "--backend-manifest", str(tmp_path / "backend.json"),
                                    "--expected-revision", "expected"])
    monkeypatch.setattr("numpy.savez_compressed", lambda *a, **kw: (_ for _ in ()).throw(OSError("write failed")))
    monkeypatch.setattr(smoke, "run", lambda *a, **kw: pytest.fail("must not dispatch"))
    assert smoke.main() == 1
    report = json.loads((tmp_path / "smoke-report.json").read_text())
    assert report["failure_phase"] == "input_preparation"
    assert report["last_trustworthy_phase"] == "request"
    assert "write failed" in report["error"]


def test_dependent_run_requires_composition_before_backend(tmp_path, monkeypatch):
    from argparse import Namespace
    from scripts import source_metal_remesh_finishing as runner
    monkeypatch.setattr(runner.subprocess, "check_output", lambda argv, **kw: "expected" if argv[-1] == "HEAD" else "")
    monkeypatch.setattr(runner, "load_inputs", lambda *a: [np.zeros((1, 3))] * 4)
    args = Namespace(output_dir=tmp_path / "new", expected_revision="expected", checkpoints=tmp_path,
                     mesh_sha256="mesh", texture_sha256="texture", grid_size=768,
                     require_composition_report=tmp_path / "missing-smoke.json", backend_manifest=tmp_path / "backend.json")
    result = runner.run(args, backend_loader=lambda: (_ for _ in ()).throw(RuntimeError("backend must not run")))
    assert result["status"] == "failed"
    assert result["failure_phase"] == "composition_admission"
    assert "backend must not run" not in result["error"]


@pytest.mark.parametrize("defect", [None, "revision", "manifest", "status", "grid", "backend", "partial", "asset"])
def test_composition_admission_rejects_false_closure(tmp_path, defect):
    import json
    from scripts.source_metal_remesh_finishing import digest, validate_composition
    manifest = tmp_path / "backend.json"
    manifest.write_text("bound backend manifest")
    asset = tmp_path / "output.glb"
    asset.write_bytes(b"structural output admitted by its producer")
    report = {"schema": "trellis2mlx.original-metal-remesh.v1", "status": "complete",
              "source": {"revision": "expected"}, "backend_manifest_sha256": digest(manifest),
              "effective": {"accelerator": "Metal custom kernels", "host_tensors": "torch-cpu",
                            "grid_size": 32, "target_faces": 2000, "texture_size": 64,
                            "band": 1., "project_back": .9, "alpha_mode": "OPAQUE", "double_sided": False},
              "stages": [{"name": name} for name in ("raw", "filled_original", "rebuilt", "reduced", "uv", "normals")],
              "output": {"path": str(asset), "sha256": digest(asset)}}
    if defect == "revision":
        report["source"]["revision"] = "old"
    elif defect == "manifest":
        report["backend_manifest_sha256"] = "old"
    elif defect == "status":
        report["status"] = "failed"
    elif defect == "grid":
        report["effective"]["grid_size"] = 16
    elif defect == "backend":
        report["effective"]["accelerator"] = "CPU fallback"
    elif defect == "partial":
        report["stages"].pop()
    elif defect == "asset":
        asset.write_bytes(b"changed asset")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    if defect:
        with pytest.raises(ValueError):
            validate_composition(path, "expected", manifest)
    else:
        assert validate_composition(path, "expected", manifest)["output_sha256"] == digest(asset)

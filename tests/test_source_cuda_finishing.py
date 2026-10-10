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


class TensorLike(np.ndarray):
    """CPU arithmetic fixture; actual CUDA conformance is a separate live gate."""
    def long(self):
        return self.astype(np.int64)

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)

    def sum(self, axis=None, dim=None, **kwargs):
        return super().sum(axis=dim if dim is not None else axis, **kwargs)

    def clone(self):
        return self.copy()


def tensor(value, dtype=np.float32):
    return np.array(value, dtype=dtype).view(TensorLike)


def test_batched_projection_keeps_every_row_and_original_arithmetic():
    original = tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = tensor([[0, 1, 2]], np.int32)
    points = tensor([[i / 10, i / 20, .1] for i in range(11)])
    baseline = points.copy()
    weights = tensor([[.2, .3, .5]] * len(points))
    expected = baseline - .9 * (baseline - (original[faces[tensor([0] * 11, np.int64)]] * weights.unsqueeze(-1)).sum(axis=1))
    seen = []
    class BVH:
        def unsigned_distance(self, batch, return_uvw):
            assert return_uvw is True
            seen.append(batch.copy())
            return tensor([0] * len(batch)), tensor([0] * len(batch), np.int64), tensor([[.2, .3, .5]] * len(batch))
    receipt = m.project_vertices_batched(points, original, faces, BVH(), .9, 4)
    np.testing.assert_array_equal(points, expected)
    np.testing.assert_array_equal(np.concatenate(seen), baseline)
    assert receipt == {"vertices": 11, "processed_vertices": 11, "batch_vertices": 4, "batches": 3, "project_back": .9}


def test_deferred_projection_releases_native_temporaries_before_query():
    import weakref
    refs, calls, records = [], [], []
    points = tensor([[0, 0, .1], [1, 0, .1]])
    faces = tensor([[0, 1, 2]], np.int32)
    vertices = tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    class BVH:
        def unsigned_distance(self, batch, return_uvw):
            assert refs[0]() is None
            return tensor([0] * len(batch)), tensor([0] * len(batch), np.int64), tensor([[1, 0, 0]] * len(batch))
    bvh = BVH()
    def native(vertices, faces, *, project_back, bvh, resolution):
        temporary = np.zeros(100)
        refs.append(weakref.ref(temporary))
        calls.append((project_back, bvh, resolution))
        return points, faces
    out, triangles = m.defer_remesh_projection(native, 1, records.append)(vertices, faces, project_back=.9, bvh=bvh, resolution=1024)
    assert triangles is faces and out is points
    assert calls == [(0, bvh, 1024)]
    assert records[0]["processed_vertices"] == 2
    assert records[0]["native_project_back"] == 0
    assert records[0]["project_back"] == .9


def test_memory_batch_size_invalid_and_missing_bvh_fail_loud():
    with pytest.raises(ValueError, match="positive"):
        m.defer_remesh_projection(lambda: None, 0, lambda record: None)
    def native(vertices, faces, *, project_back, bvh=None):
        pytest.fail("native must not run without a reusable BVH")
    with pytest.raises(ValueError, match="BVH"):
        m.defer_remesh_projection(native, 10, lambda record: None)(None, None, project_back=.9)


def test_zero_projection_retains_native_route():
    token = object()
    def native(vertices, faces, *, project_back=0, bvh=None):
        assert project_back == 0
        return token
    assert m.defer_remesh_projection(native, 10, lambda record: None)(None, None) is token


def test_memory_control_reaches_child_without_mutating_scientific_settings(tmp_path, monkeypatch):
    args = m.parser().parse_args([
        "--mesh", str(tmp_path / "mesh"), "--appearance", str(tmp_path / "tex"),
        "--mesh-sha", "mesh", "--appearance-sha", "tex", "--grid-size", "768",
        "--arms", "remesh", "--remesh-project", "0.9", "--remesh-resolution", "1024",
        "--projection-batch-vertices", "262144", "--runtime", str(tmp_path / "runtime"),
        "--output-dir", str(tmp_path / "finishing"), "--output-json", str(tmp_path / "report.json"),
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
    assert calls[0][calls[0].index("--projection-batch-vertices") + 1] == "262144"
    assert m.finishing_settings(args)["grid_size"] == 768
    assert m.finishing_settings(args)["remesh_project"] == .9


def test_projection_conformance_detects_batch_dependent_backend():
    vertices = tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = tensor([[0, 1, 2]], np.int32)
    queries = tensor([[.2, .3, .1]] * 11)
    class BVH:
        def unsigned_distance(self, batch, return_uvw):
            weights = [.2, .3, .5] if len(batch) == 11 else [.3, .2, .5]
            return tensor([0] * len(batch)), tensor([0] * len(batch), np.int64), tensor([weights] * len(batch))
    with pytest.raises(ValueError, match="equivalence"):
        m.projection_conformance(queries, vertices, faces, BVH(), .9, 4,
                                 lambda x, y: np.array_equal(x, y))


def test_projection_conformance_preserves_input_and_exact_outputs():
    vertices = tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = tensor([[0, 1, 2]], np.int32)
    queries = tensor([[.2, .3, .1]] * 11)
    before = queries.copy()
    class BVH:
        def unsigned_distance(self, batch, return_uvw):
            return tensor([0] * len(batch)), tensor([0] * len(batch), np.int64), tensor([[.2, .3, .5]] * len(batch))
    expected, actual, receipt = m.projection_conformance(
        queries, vertices, faces, BVH(), .9, 4, lambda x, y: np.array_equal(x, y))
    np.testing.assert_array_equal(queries, before)
    np.testing.assert_array_equal(expected, actual)
    assert receipt["exact_equal"] is True and receipt["processed_vertices"] == 11


def checkpoint_fixture(tmp_path):
    checkpoint = tmp_path / "rebuilt.npz"
    filled = tmp_path / "filled.npz"
    arrays = {"vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
              "faces": np.array([[0, 1, 2]], np.int32)}
    np.savez_compressed(checkpoint, **arrays)
    np.savez_compressed(filled, **arrays)
    args = m.parser().parse_args([
        "--mesh", "raw", "--appearance", "appearance", "--mesh-sha", "raw-sha",
        "--appearance-sha", "appearance-sha", "--grid-size", "768",
        "--remesh-project", ".9", "--remesh-resolution", "1024", "--arm", "remesh"])
    report = {"source_pins": m.SOURCES, "requested": vars(args).copy(),
        "effective_route": {"postprocess_sha256": m.POSTPROCESS_SHA,
            "model_inference": False, "local_cleanup": False, "device": "Tesla T4",
            "capability": [7, 5], "settings": m.finishing_settings(args),
            "reconstruction": {"appearance_grid_size": 768, "calls": [{
                "source_resolution": 768, "effective_resolution": 1024,
                "source_scale": 771 / 768, "effective_scale": 1027 / 1024,
                "band": 1, "project_back": .9, "resolution_override": True}]}},
        "stages": [{"label": "init"}, {"label": "fill_holes", "sha256": m.digest(filled)},
                   {"label": "init", "sha256": m.digest(checkpoint), "arrays": {
                       k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}}]}
    report["requested"] = {k: str(v) if isinstance(v, Path) else v
                           for k, v in report["requested"].items()}
    path = tmp_path / "origin.json"
    m.write_json(path, report)
    args.resume_rebuilt, args.resume_filled = checkpoint, filled
    args.resume_report, args.resume_report_sha = path, m.digest(path)
    return args, report, arrays


def test_checkpoint_admission_binds_origin_and_complete_geometry(tmp_path):
    args, _, arrays = checkpoint_fixture(tmp_path)
    loaded, filled, receipt = m.load_reconstruction_checkpoint(args)
    np.testing.assert_array_equal(loaded["faces"], arrays["faces"])
    np.testing.assert_array_equal(filled["vertices"], arrays["vertices"])
    assert receipt["rebuilt_sha256"] == m.digest(args.resume_rebuilt)
    assert receipt["origin_report_sha256"] == args.resume_report_sha
    assert receipt["reconstruction_executed_this_run"] is False


@pytest.mark.parametrize("change", ["wrong-origin", "wrong-appearance", "wrong-grid", "wrong-stage",
                                   "wrong-reconstruction", "missing-calibration", "partial-projection"])
def test_checkpoint_rejects_wrong_or_partial_source(tmp_path, change):
    args, origin, _ = checkpoint_fixture(tmp_path)
    if change == "wrong-origin":
        args.resume_report_sha = "0" * 64
    else:
        if change == "wrong-appearance": origin["requested"]["appearance_sha"] = "other"
        if change == "wrong-grid": origin["effective_route"]["settings"]["grid_size"] = 1024
        if change == "wrong-stage": origin["stages"][2]["label"] = "simplify"
        if change == "wrong-reconstruction": origin["effective_route"]["reconstruction"]["calls"][0]["effective_resolution"] = 768
        if change in {"missing-calibration", "partial-projection"}:
            origin["effective_route"]["projection_memory"] = {
                "conformance": {"exact_equal": change != "missing-calibration"},
                "calls": [{"vertices": 3, "processed_vertices": 2, "project_back": .9}]}
        m.write_json(args.resume_report, origin)
        args.resume_report_sha = m.digest(args.resume_report)
    with pytest.raises(ValueError): m.load_reconstruction_checkpoint(args)


def test_resume_remesher_checks_original_surface_and_never_reconstructs(tmp_path):
    args, origin, arrays = checkpoint_fixture(tmp_path)
    _, _, receipt = m.load_reconstruction_checkpoint(args)
    records = []
    call = m.resume_remesher(arrays, arrays, receipt, lambda value: value, np.array_equal, records.append)
    output = call(arrays["vertices"], arrays["faces"], resolution=1024,
                  scale=1027 / 1024, band=1, project_back=.9, bvh=object())
    assert output[0] is arrays["vertices"] and output[1] is arrays["faces"]
    assert records[0]["original_surface_exact"] is True
    with pytest.raises(ValueError, match="original surface"):
        call(arrays["vertices"] + 1, arrays["faces"], resolution=1024,
             scale=1027 / 1024, band=1, project_back=.9, bvh=object())


def test_native_failure_keeps_memory_snapshots_and_releases_only_unused_cache(tmp_path):
    events = []
    class CUDA:
        def synchronize(self): events.append("sync")
        def mem_get_info(self): return (100, 1000)
        def memory_allocated(self): return 200
        def memory_reserved(self): return 400
        def max_memory_allocated(self): return 300
        def empty_cache(self): events.append("empty-cache")
    torch = SimpleNamespace(cuda=CUDA())
    report = {"stages": []}
    hook = m.memory_boundary(torch, report, tmp_path / "memory.json", True)
    class Native:
        def simplify(self, target):
            events.append("native")
            raise RuntimeError("native CUDA allocation failure")
    mesh = m.observe_mesh_class(Native, lambda *a: pytest.fail("no success capture"), hook)()
    with pytest.raises(RuntimeError, match="allocation failure"): mesh.simplify(1000000)
    saved = json.loads((tmp_path / "memory.json").read_text())
    assert [row["boundary"] for row in saved["cuda_memory"]] == [
        "before-simplify", "after-empty-cache-before-simplify", "failed-simplify"]
    assert events.index("empty-cache") < events.index("native")
    assert saved["cuda_memory"][-1]["torch_reserved_unused_bytes"] == 200
    assert saved["phase"] == "native_simplify"


def test_checkpoint_flags_and_cache_release_reach_independent_child(tmp_path, monkeypatch):
    args, _, _ = checkpoint_fixture(tmp_path)
    args.arm = None
    args.arms = ["remesh"]
    args.output_dir, args.output_json = tmp_path / "finish", tmp_path / "finish.json"
    args.runtime = tmp_path / "runtime"
    args.release_torch_cache_before_simplify = True
    monkeypatch.setattr(m, "load_inputs", lambda *a: {})
    monkeypatch.setattr(m, "bootstrap", lambda *a: None)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        m.write_json(args.output_dir / "remesh-report.json", {"status": "completed"})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess, "run", run)
    m.run_both(args, {})
    for flag in ("--resume-rebuilt", "--resume-filled", "--resume-report", "--resume-report-sha",
                 "--release-torch-cache-before-simplify"):
        assert flag in calls[0]


def test_checkpoint_can_replay_observed_legacy_unmodified_source_route(tmp_path):
    args, report, _ = checkpoint_fixture(tmp_path)
    args.remesh_resolution = None
    report["requested"]["remesh_resolution"] = None
    report["effective_route"].pop("reconstruction")
    report["effective_route"]["finishing"] = "unmodified-official-to_glb"
    m.write_json(args.resume_report, report)
    args.resume_report_sha = m.digest(args.resume_report)
    _, _, receipt = m.load_reconstruction_checkpoint(args)
    assert receipt["origin_reconstruction"]["effective_resolution"] == 768
    assert receipt["origin_reconstruction"]["effective_scale"] == 771 / 768
    assert receipt["origin_call_evidence"] == "pinned-unmodified-to_glb-derived-settings"


def test_replay_gate_requires_live_success_original_surface_and_exact_reduction(tmp_path):
    reference = tmp_path / "reference.npz"
    actual = tmp_path / "actual.npz"
    np.savez_compressed(reference, vertices=np.ones((3, 3), np.float32), faces=np.array([[0, 1, 2]], np.int32))
    np.savez_compressed(actual, vertices=np.ones((3, 3), np.float32), faces=np.array([[0, 1, 2]], np.int32))
    report = {"status": "completed", "effective_route": {
        "device": "Tesla T4", "capability": [7, 5],
        "finishing": "official-to_glb-with-saved-reconstruction-continuation",
        "checkpoint_continuation": {"original_surface_exact": True}},
        "stages": [{"label": "init"}, {"label": "fill_holes"}, {"label": "init"},
                   {"label": "simplify", "path": str(actual), "sha256": m.digest(actual)},
                   {"label": "uv_unwrap"}, {"label": "compute_vertex_normals"}],
        "glb": {"image_sizes": [[4096, 4096], [4096, 4096]]}}
    record = m.check_replay_gate(report, reference, m.digest(reference))
    assert record["reduced_arrays_exact"] is True
    for field, value in (("status", "failed"), ("status", "running")):
        broken = dict(report, **{field: value})
        with pytest.raises(ValueError): m.check_replay_gate(broken, reference, m.digest(reference))
    report["effective_route"]["device"] = "CPU"
    with pytest.raises(ValueError): m.check_replay_gate(report, reference, m.digest(reference))
    report["effective_route"]["device"] = "Tesla T4"
    np.savez_compressed(actual, vertices=np.zeros((3, 3), np.float32), faces=np.array([[0, 1, 2]], np.int32))
    report["stages"][3]["sha256"] = m.digest(actual)
    with pytest.raises(ValueError, match="reduced"): m.check_replay_gate(report, reference, m.digest(reference))


def test_checkpoint_rejects_missing_projection_account(tmp_path):
    args, origin, _ = checkpoint_fixture(tmp_path)
    origin["effective_route"]["finishing"] = "official-to_glb-with-deferred-batched-projection"
    m.write_json(args.resume_report, origin)
    args.resume_report_sha = m.digest(args.resume_report)
    with pytest.raises(ValueError, match="projection"): m.load_reconstruction_checkpoint(args)


def test_control_failure_prevents_target_and_keeps_failed_bundle(tmp_path, monkeypatch):
    args, origin, arrays = checkpoint_fixture(tmp_path)
    args.arm = None
    args.output_dir, args.output_json = tmp_path / "finish", tmp_path / "report.json"
    args.control_rebuilt, args.control_filled = args.resume_rebuilt, args.resume_filled
    args.control_report, args.control_report_sha = args.resume_report, args.resume_report_sha
    args.control_reduced = tmp_path / "reduced.npz"
    np.savez_compressed(args.control_reduced, **arrays)
    origin["status"] = "completed"
    origin["stages"].append({"label": "simplify", "sha256": m.digest(args.control_reduced)})
    m.write_json(args.resume_report, origin)
    args.resume_report_sha = args.control_report_sha = m.digest(args.resume_report)
    monkeypatch.setattr(m, "load_inputs", lambda *a: {})
    monkeypatch.setattr(m, "load_reconstruction_checkpoint", lambda *a: None)
    monkeypatch.setattr(m, "bootstrap", lambda *a: None)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        directory = Path(argv[argv.index("--output-dir") + 1])
        m.write_json(directory / "remesh-report.json", {"status": "failed", "failure_phase": "native_simplify"})
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(m.subprocess, "run", run)
    report = {"phase": "start"}
    assert m.run_reported(args.output_json, report, lambda r: m.run_checkpoint_suite(args, r)) == 1
    assert len(calls) == 1
    assert not (args.output_dir / "target-1024").exists()
    assert (tmp_path / "finishing-bundle.tar").is_file()
    saved = json.loads(args.output_json.read_text())
    assert saved["failure_phase"] == "control-768"
    assert saved["continuations"]["control-768"]["report"]["failure_phase"] == "native_simplify"

"""Contracts for the authenticated source-native Metal step witness."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import trimesh


SCRIPT = Path("scripts/run_mtlmesh_step_repeatability_assay.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("mtlmesh_step_assay_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_grid(path: Path, side: int = 8) -> None:
    xs, ys = np.meshgrid(
        np.linspace(-1.0, 1.0, side),
        np.linspace(-1.0, 1.0, side),
        indexing="xy",
    )
    vertices = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)]).astype(np.float32)
    faces: list[list[int]] = []
    for y in range(side - 1):
        for x in range(side - 1):
            a = y * side + x
            b = a + 1
            c = a + side
            d = c + 1
            faces.extend(([a, b, d], [a, d, c]))
    trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False).export(
        path, file_type="glb"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _argv(source: Path, output_dir: Path, report: Path, source_root: Path) -> list[str]:
    return [
        "--input",
        str(source),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report),
        "--source-root",
        str(source_root),
        "--expected-source-commit",
        "c" * 40,
        "--expected-backend-sha256",
        "b" * 64,
        "--expected-extension-sha256",
        "e" * 64,
        "--expected-metallib-sha256",
        "m" * 64,
        "--repeats",
        "3",
        "--max-steps",
        "2",
    ]


class _FakeMesh:
    run_index = 0
    divergent = False
    action_log: list[tuple] = []

    def __init__(self) -> None:
        type(self).run_index += 1
        self.repeat = type(self).run_index

    def init(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float32).copy()
        self.faces = np.asarray(faces, dtype=np.int32).copy()

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def simplify_step(
        self,
        _lambda_edge_length: float,
        _lambda_skinny: float,
        _threshold: float,
        _timing: bool,
        _reuse_vertex_face_adjacency: bool = False,
    ) -> tuple[int, int]:
        type(self).action_log.append(("ordinary", _reuse_vertex_face_adjacency))
        remove = 8
        if self.divergent and self.repeat == 3:
            remove = 10
        self.faces = self.faces[:-remove]
        return len(self.vertices), len(self.faces)

    def get_vertex_face_adjacency(self) -> None:
        type(self).action_log.append(("get_adjacency",))

    def sort_vertex_face_adjacency(self) -> None:
        type(self).action_log.append(("sort_adjacency",))

    def simplify_step_turing(
        self,
        rsqrt_delta,
        _lambda_edge_length: float,
        _lambda_skinny: float,
        _threshold: float,
        _timing: bool,
        reuse_vertex_face_adjacency: bool = False,
    ) -> tuple[int, int]:
        type(self).action_log.append(
            (
                "turing",
                str(rsqrt_delta.dtype),
                tuple(rsqrt_delta.shape),
                reuse_vertex_face_adjacency,
            )
        )
        self.faces = self.faces[:-8]
        return len(self.vertices), len(self.faces)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.copy(), self.faces.copy()


def _route(source_root: Path) -> dict:
    return {
        "id": "source-native-mtlmesh-metal-step-v1",
        "source_root": str(source_root.resolve()),
        "source_commit": "c" * 40,
        "backend_sha256": "b" * 64,
        "extension_sha256": "e" * 64,
        "metallib_sha256": "m" * 64,
        "python": "test-python",
        "torch": "test-torch",
        "mps_available": True,
    }


def test_repeated_fresh_mesh_steps_record_exact_trajectory(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "out"
    report = tmp_path / "report.json"
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    _write_grid(source)
    _FakeMesh.run_index = 0
    _FakeMesh.divergent = False
    _FakeMesh.action_log = []
    monkeypatch.setattr(runner, "_probe_source_route", lambda _args: _route(source_root))
    monkeypatch.setattr(runner, "_load_mesh_class", lambda _route: _FakeMesh)

    returncode = runner.main(_argv(source, output_dir, report, source_root))

    assert returncode == 0
    data = json.loads(report.read_text())
    assert data["status"] == "completed"
    assert data["route"]["source_commit"] == "c" * 40
    assert data["source"]["sha256"] == _sha256(source)
    assert data["effective_config"]["fresh_mesh_per_repeat"] is True
    assert data["effective_config"]["target_face_count"] is None
    assert data["repeat_stability"]["all_steps_exact"] is True
    assert len(data["runs"]) == 3
    for run in data["runs"]:
        assert len(run["steps"]) == 2
        assert run["steps"][0]["input_faces"] == 98
        assert run["steps"][0]["output_faces"] == 90
        assert run["steps"][1]["input_faces"] == 90
        assert run["steps"][1]["output_faces"] == 82
        assert Path(run["steps"][0]["mesh_path"]).is_file()
    assert data["step_stability"][0]["exact"] is True
    assert data["step_stability"][1]["exact"] is True


def test_canonical_turing_route_sorts_consumed_cache_and_cannot_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "out"
    report = tmp_path / "report.json"
    source_root = tmp_path / "mtlmesh"
    lut = tmp_path / "turing-rsqrt.npz"
    source_root.mkdir()
    _write_grid(source)
    np.savez(lut, normalized_delta=np.zeros(1 << 24, dtype=np.int8))
    _FakeMesh.run_index = 0
    _FakeMesh.divergent = False
    _FakeMesh.action_log = []
    monkeypatch.setattr(runner, "_probe_source_route", lambda _args: _route(source_root))
    monkeypatch.setattr(runner, "_load_mesh_class", lambda _route: _FakeMesh)
    monkeypatch.setattr(runner, "_turing_tensor", lambda array: array)

    returncode = runner.main(
        [
            *_argv(source, output_dir, report, source_root),
            "--canonical-adjacency",
            "--turing-rsqrt-lut",
            str(lut),
            "--expected-turing-rsqrt-lut-sha256",
            _sha256(lut),
        ]
    )

    assert returncode == 0
    data = json.loads(report.read_text())
    assert data["route"]["simplifier"] == {
        "id": "canonical-adjacency-turing-rsqrt-step-v1",
        "adjacency_order": "ascending-face-id-per-vertex",
        "reuse_vertex_face_adjacency": True,
        "rsqrt_lut_npz": str(lut.resolve()),
        "rsqrt_lut_sha256": _sha256(lut),
        "normalized_delta_sha256": hashlib.sha256(bytes(1 << 24)).hexdigest(),
    }
    assert data["effective_config"]["canonical_adjacency"] is True
    assert data["effective_config"]["turing_rsqrt"] is True
    assert data["runs"][0]["steps"][0]["simplify_method"] == "simplify_step_turing"
    assert _FakeMesh.action_log == [
        ("get_adjacency",),
        ("sort_adjacency",),
        ("turing", "int8", (1 << 24,), True),
        ("get_adjacency",),
        ("sort_adjacency",),
        ("turing", "int8", (1 << 24,), True),
    ] * 3
    assert not any(action[0] == "ordinary" for action in _FakeMesh.action_log)


def test_turing_lut_hash_mismatch_fails_before_route_or_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "out"
    report = tmp_path / "report.json"
    source_root = tmp_path / "mtlmesh"
    lut = tmp_path / "turing-rsqrt.npz"
    source_root.mkdir()
    _write_grid(source)
    np.savez(lut, normalized_delta=np.zeros(1 << 24, dtype=np.int8))
    calls = 0

    def should_not_probe(_args):
        nonlocal calls
        calls += 1
        raise AssertionError("route probe should not run after LUT digest mismatch")

    monkeypatch.setattr(runner, "_probe_source_route", should_not_probe)

    returncode = runner.main(
        [
            *_argv(source, output_dir, report, source_root),
            "--canonical-adjacency",
            "--turing-rsqrt-lut",
            str(lut),
            "--expected-turing-rsqrt-lut-sha256",
            "0" * 64,
        ]
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "authenticate_turing_lut"
    assert data["primary_output_status"] == "not_started"
    assert calls == 0
    assert not output_dir.exists()


def test_report_lut_collision_preserves_lut_and_reroutes_failure(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "out"
    source_root = tmp_path / "mtlmesh"
    lut = tmp_path / "turing-rsqrt.npz"
    source_root.mkdir()
    _write_grid(source)
    np.savez(lut, normalized_delta=np.zeros(1 << 24, dtype=np.int8))
    lut_hash = _sha256(lut)
    calls = 0

    def should_not_probe(_args):
        nonlocal calls
        calls += 1
        raise AssertionError("route probe should not run after invalid LUT custody")

    monkeypatch.setattr(runner, "_probe_source_route", should_not_probe)

    returncode = runner.main(
        [
            *_argv(source, output_dir, lut, source_root),
            "--canonical-adjacency",
            "--turing-rsqrt-lut",
            str(lut),
            "--expected-turing-rsqrt-lut-sha256",
            lut_hash,
        ]
    )

    safe_report = source.with_name(source.name + ".assay-error.json")
    assert returncode != 0
    assert _sha256(lut) == lut_hash
    assert safe_report.is_file()
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["report"] == {
        "requested_path": str(lut),
        "effective_path": str(safe_report),
        "rerouted": True,
    }
    assert calls == 0
    assert not output_dir.exists()


def test_output_containing_lut_fails_before_probe_or_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "out"
    report = tmp_path / "report.json"
    source_root = tmp_path / "mtlmesh"
    lut = output_dir / "turing-rsqrt.npz"
    source_root.mkdir()
    output_dir.mkdir()
    _write_grid(source)
    np.savez(lut, normalized_delta=np.zeros(1 << 24, dtype=np.int8))
    lut_hash = _sha256(lut)
    calls = 0

    def should_not_probe(_args):
        nonlocal calls
        calls += 1
        raise AssertionError("route probe should not run after invalid LUT custody")

    monkeypatch.setattr(runner, "_probe_source_route", should_not_probe)

    returncode = runner.main(
        [
            *_argv(source, output_dir, report, source_root),
            "--canonical-adjacency",
            "--turing-rsqrt-lut",
            str(lut),
            "--expected-turing-rsqrt-lut-sha256",
            lut_hash,
        ]
    )

    assert returncode != 0
    assert _sha256(lut) == lut_hash
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["primary_output_status"] == "not_started"
    assert calls == 0


def test_turing_route_requires_canonical_adjacency_and_expected_digest(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    _write_grid(source)
    base = _argv(source, tmp_path / "out", tmp_path / "report.json", source_root)

    for suffix in (
        ["--turing-rsqrt-lut", str(tmp_path / "lut.npz")],
        [
            "--canonical-adjacency",
            "--turing-rsqrt-lut",
            str(tmp_path / "lut.npz"),
        ],
    ):
        returncode = runner.main([*base, *suffix])
        assert returncode != 0
        data = json.loads((tmp_path / "report.json").read_text())
        assert data["failure_phase"] == "validate_config"
        assert data["primary_output_status"] == "not_started"


def test_repeat_divergence_is_reported_without_false_exact_claim(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "out"
    report = tmp_path / "report.json"
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    _write_grid(source)
    _FakeMesh.run_index = 0
    _FakeMesh.divergent = True
    monkeypatch.setattr(runner, "_probe_source_route", lambda _args: _route(source_root))
    monkeypatch.setattr(runner, "_load_mesh_class", lambda _route: _FakeMesh)

    returncode = runner.main(_argv(source, output_dir, report, source_root))

    assert returncode == 0
    data = json.loads(report.read_text())
    assert data["status"] == "completed"
    assert data["repeat_stability"]["all_steps_exact"] is False
    assert data["step_stability"][0]["exact"] is False
    assert data["step_stability"][0]["distinct_output_face_counts"] == [88, 90]


def test_wrong_source_route_replaces_stale_success_before_outputs(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "out"
    report = tmp_path / "report.json"
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    _write_grid(source)
    report.write_text(json.dumps({"status": "completed", "route": {"id": "stale"}}))

    def reject(_args):
        raise runner.AssayError("authenticate_route", "source commit mismatch")

    monkeypatch.setattr(runner, "_probe_source_route", reject)

    returncode = runner.main(_argv(source, output_dir, report, source_root))

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["failure_phase"] == "authenticate_route"
    assert data["primary_output_status"] == "not_started"
    assert not output_dir.exists()


def test_expected_input_hash_mismatch_fails_before_route_or_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "out"
    report = tmp_path / "report.json"
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    _write_grid(source)
    calls = 0

    def should_not_probe(_args):
        nonlocal calls
        calls += 1
        raise AssertionError("route probe should not run after source digest mismatch")

    monkeypatch.setattr(runner, "_probe_source_route", should_not_probe)

    returncode = runner.main(
        [*_argv(source, output_dir, report, source_root), "--expected-input-sha256", "0" * 64]
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["failure_phase"] == "source_identity"
    assert data["primary_output_status"] == "not_started"
    assert data["source"]["sha256"] == _sha256(source)
    assert calls == 0
    assert not output_dir.exists()


def test_report_source_collision_preserves_source_and_reroutes_failure(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "out"
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    _write_grid(source)
    source_hash = _sha256(source)
    monkeypatch.setattr(runner, "_probe_source_route", lambda _args: _route(source_root))

    returncode = runner.main(_argv(source, output_dir, source, source_root))

    safe_report = tmp_path / "grid.glb.assay-error.json"
    assert returncode != 0
    assert _sha256(source) == source_hash
    assert safe_report.is_file()
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["report"]["requested_path"] == str(source)
    assert data["report"]["effective_path"] == str(safe_report)


def test_report_temporary_source_collision_preserves_source(tmp_path: Path) -> None:
    runner = _load_runner()
    report = tmp_path / "assay.report.json"
    source = tmp_path / "assay.report.json.tmp"
    output_dir = tmp_path / "out"
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    _write_grid(source)
    source_hash = _sha256(source)

    returncode = runner.main(_argv(source, output_dir, report, source_root))

    safe_report = tmp_path / "assay.report.json.assay-error.json"
    assert returncode != 0
    assert _sha256(source) == source_hash
    assert safe_report.is_file()
    assert json.loads(safe_report.read_text())["failure_phase"] == "validate_paths"


def test_output_and_report_cannot_overlap_authenticated_source_root(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    _write_grid(source)
    calls = 0

    def should_not_probe(_args):
        nonlocal calls
        calls += 1
        raise AssertionError("route probe should not run after invalid path custody")

    monkeypatch.setattr(runner, "_probe_source_route", should_not_probe)
    cases = [
        (tmp_path / "route-equal", tmp_path / "route-equal", tmp_path / "equal.json"),
        (
            tmp_path / "route-child",
            tmp_path / "route-child" / "outputs",
            tmp_path / "child.json",
        ),
        (tmp_path / "ancestor" / "route", tmp_path / "ancestor", tmp_path / "ancestor.json"),
        (
            tmp_path / "route-report",
            tmp_path / "safe-out",
            tmp_path / "route-report" / "report.json",
        ),
        (
            tmp_path / "report-ancestor" / "route",
            tmp_path / "safe-out-ancestor",
            tmp_path / "report-ancestor",
        ),
    ]
    for index, (source_root, output_dir, report) in enumerate(cases):
        source_root.mkdir(parents=True, exist_ok=True)
        output_existed = output_dir.exists()
        returncode = runner.main(_argv(source, output_dir, report, source_root))
        assert returncode != 0, index
        effective_report = report
        if report.is_relative_to(source_root) or source_root.is_relative_to(report):
            effective_report = source.with_name(source.name + ".assay-error.json")
        data = json.loads(effective_report.read_text())
        assert data["failure_phase"] == "validate_paths"
        assert output_dir.exists() is output_existed
    assert calls == 0


def test_probe_rejects_compiled_artifact_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    root = tmp_path / "mtlmesh"
    package = root / "cumesh"
    package.mkdir(parents=True)
    (package / "metal_backend.py").write_text("# backend\n")
    (package / "_C.cpython-311-darwin.so").write_bytes(b"extension")
    (package / "cumesh.metallib").write_bytes(b"metallib")
    monkeypatch.setattr(
        runner,
        "_git_output",
        lambda _root, *args: "" if args == ("status", "--porcelain") else "c" * 40,
    )
    args = runner.parse_args(
        _argv(tmp_path / "input.glb", tmp_path / "out", tmp_path / "r.json", root)
    )

    try:
        runner._probe_source_route(args)
    except runner.AssayError as exc:
        assert exc.phase == "authenticate_route"
        assert "backend SHA256 mismatch" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("wrong backend hash was accepted")


def test_failure_after_first_mesh_records_and_removes_partial_output(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "out"
    report = tmp_path / "report.json"
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    _write_grid(source)
    _FakeMesh.run_index = 0
    _FakeMesh.divergent = False
    monkeypatch.setattr(runner, "_probe_source_route", lambda _args: _route(source_root))
    monkeypatch.setattr(runner, "_load_mesh_class", lambda _route: _FakeMesh)
    real_write = runner.write_geometry_glb
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second output failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(runner, "write_geometry_glb", fail_second)

    returncode = runner.main(_argv(source, output_dir, report, source_root))

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["failure_phase"] == "run_steps"
    assert data["primary_output_status"] == "partial_removed"
    assert data["last_trustworthy_evidence"]["repeat"] == 1
    assert data["last_trustworthy_evidence"]["step"] == 1
    assert data["partial_output_cleanup"]["artifact_count"] == 1
    assert data["partial_output_cleanup"]["removed"] is True
    assert not output_dir.exists()

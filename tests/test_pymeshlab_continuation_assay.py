"""Contracts for the authenticated PyMeshLab continuation assay."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh


SCRIPT = Path("scripts/run_pymeshlab_continuation_assay.py")
EXPECTED_VERSION = "2025.7.post1"


def _load_runner():
    spec = importlib.util.spec_from_file_location("pymeshlab_assay_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_grid(path: Path, side: int = 12) -> None:
    xs, ys = np.meshgrid(
        np.linspace(-1.0, 1.0, side),
        np.linspace(-1.0, 1.0, side),
        indexing="xy",
    )
    vertices = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)]).astype(np.float64)
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class _FakeMesh:
    def __init__(self, target: int, perturbation: float = 0.0):
        self._vertices = np.asarray(
            [[0.0 + perturbation, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        self._faces = np.tile(np.asarray([[0, 1, 2]], dtype=np.int32), (target, 1))

    def vertex_matrix(self) -> np.ndarray:
        return self._vertices

    def face_matrix(self) -> np.ndarray:
        return self._faces

    def vertex_number(self) -> int:
        return len(self._vertices)

    def face_number(self) -> int:
        return len(self._faces)


class _FakeMeshSet:
    loads: list[str] = []
    filters: list[dict[str, object]] = []
    run_count = 0
    nondeterministic = False
    fail_on_run: int | None = None

    @classmethod
    def reset(cls) -> None:
        cls.loads = []
        cls.filters = []
        cls.run_count = 0
        cls.nondeterministic = False
        cls.fail_on_run = None

    def __init__(self):
        self.mesh = _FakeMesh(242)

    def load_new_mesh(self, path: str) -> None:
        type(self).loads.append(path)

    def meshing_decimation_quadric_edge_collapse(self, **kwargs) -> None:
        cls = type(self)
        cls.run_count += 1
        cls.filters.append(dict(kwargs))
        if cls.fail_on_run == cls.run_count:
            raise RuntimeError("injected simplifier failure")
        perturbation = 0.001 * cls.run_count if cls.nondeterministic else 0.0
        self.mesh = _FakeMesh(int(kwargs["targetfacenum"]), perturbation)

    def current_mesh(self) -> _FakeMesh:
        return self.mesh


def _fake_package(version: str = EXPECTED_VERSION):
    _FakeMeshSet.reset()
    return SimpleNamespace(__version__=version, __file__="/fake/pymeshlab/__init__.py", MeshSet=_FakeMeshSet)


def _invoke(
    runner,
    monkeypatch,
    tmp_path: Path,
    *,
    expected_version: str = EXPECTED_VERSION,
    expected_input_sha256: str | None = None,
    expected_lineage_sha256: str | None = None,
    report: Path | None = None,
) -> tuple[int, Path, Path, Path, Path]:
    source = tmp_path / "grid.glb"
    lineage = tmp_path / "parent-report.json"
    output_dir = tmp_path / "outputs"
    report = report or (tmp_path / "assay.report.json")
    _write_grid(source)
    lineage.write_text('{"status":"completed","faces":27134294}\n')
    package = _fake_package()
    monkeypatch.setattr(runner, "_import_pymeshlab", lambda: package)
    returncode = runner.main(
        [
            "--input",
            str(source),
            "--expected-input-sha256",
            expected_input_sha256 or _sha256(source),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report),
            "--target-faces",
            "160",
            "80",
            "--repeats",
            "2",
            "--expected-pymeshlab-version",
            expected_version,
            "--lineage-ref",
            "parent-pymeshlab-partial",
            str(lineage),
            expected_lineage_sha256 or _sha256(lineage),
        ]
    )
    return returncode, source, lineage, output_dir, report


def test_assay_binds_continuation_lineage_and_explicit_filter_contract(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()

    returncode, source, lineage, output_dir, report = _invoke(
        runner, monkeypatch, tmp_path
    )

    assert returncode == 0
    data = json.loads(report.read_text())
    assert data["status"] == "completed"
    assert data["route"]["id"] == "pymeshlab-topology-preserving-continuation-v1"
    assert data["route"]["package_version"] == EXPECTED_VERSION
    assert data["route"]["harness_path"] == str(SCRIPT.resolve())
    assert data["route"]["harness_sha256"] == _sha256(SCRIPT)
    assert data["source"]["sha256"] == _sha256(source)
    assert data["continuation_provenance"]["classification"] == "continuation-not-common-source"
    assert data["continuation_provenance"]["refs"][0]["sha256"] == _sha256(lineage)
    assert data["trajectory_semantics"]["target_affects"] == "termination-only-in-inspected-source"
    assert data["trajectory_admission"]["repeat_exact_all_targets"] is True
    assert data["trajectory_admission"]["effective_single_trajectory_admitted"] is True
    assert len(_FakeMeshSet.loads) == 4
    assert all(path == str(source) for path in _FakeMeshSet.loads)
    assert len(_FakeMeshSet.filters) == 4
    for call in _FakeMeshSet.filters:
        assert call == {
            "targetfacenum": call["targetfacenum"],
            "targetperc": 0.0,
            "qualitythr": 0.3,
            "preserveboundary": False,
            "boundaryweight": 1.0,
            "preservenormal": False,
            "preservetopology": True,
            "optimalplacement": True,
            "planarquadric": False,
            "planarweight": 0.001,
            "qualityweight": False,
            "autoclean": True,
            "selected": False,
        }
    assert len(list(output_dir.glob("*.glb"))) == 4


def test_runtime_nondeterminism_completes_but_denies_single_trajectory(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    package = _fake_package()
    _FakeMeshSet.nondeterministic = True
    monkeypatch.setattr(runner, "_import_pymeshlab", lambda: package)
    source = tmp_path / "grid.glb"
    lineage = tmp_path / "parent.json"
    report = tmp_path / "report.json"
    output_dir = tmp_path / "outputs"
    _write_grid(source)
    lineage.write_text("{}\n")

    returncode = runner.main(
        [
            "--input", str(source),
            "--expected-input-sha256", _sha256(source),
            "--output-dir", str(output_dir),
            "--report", str(report),
            "--target-faces", "160", "80",
            "--repeats", "2",
            "--expected-pymeshlab-version", EXPECTED_VERSION,
            "--lineage-ref", "parent", str(lineage), _sha256(lineage),
        ]
    )

    assert returncode == 0
    data = json.loads(report.read_text())
    assert data["repeat_stability"]["all_exact"] is False
    assert data["trajectory_admission"]["effective_single_trajectory_admitted"] is False
    assert data["trajectory_admission"]["required_interpretation"] == "per-target-replicate-distributions"


def test_wrong_runtime_version_fails_before_primary_outputs(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()

    returncode, _, _, output_dir, report = _invoke(
        runner, monkeypatch, tmp_path, expected_version="2025.8"
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "runtime_identity"
    assert data["primary_output_status"] == "not_started"
    assert not output_dir.exists()


def test_wrong_input_digest_fails_before_runtime_import(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_import_pymeshlab",
        lambda: (_ for _ in ()).throw(AssertionError("runtime imported too early")),
    )

    returncode, _, _, output_dir, report = _invoke(
        runner, monkeypatch, tmp_path, expected_input_sha256="0" * 64
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "source_identity"
    assert data["primary_output_status"] == "not_started"
    assert not output_dir.exists()


def test_wrong_lineage_digest_fails_before_runtime_import(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_import_pymeshlab",
        lambda: (_ for _ in ()).throw(AssertionError("runtime imported too early")),
    )

    returncode, _, _, output_dir, report = _invoke(
        runner, monkeypatch, tmp_path, expected_lineage_sha256="f" * 64
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "lineage_identity"
    assert data["primary_output_status"] == "not_started"
    assert not output_dir.exists()


def test_report_collision_preserves_source_and_reroutes_failure(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    source = tmp_path / "grid.glb"
    _write_grid(source)
    source_hash = _sha256(source)
    lineage = tmp_path / "parent.json"
    lineage.write_text("{}\n")
    output_dir = tmp_path / "outputs"
    package = _fake_package()
    monkeypatch.setattr(runner, "_import_pymeshlab", lambda: package)

    returncode = runner.main(
        [
            "--input", str(source),
            "--expected-input-sha256", source_hash,
            "--output-dir", str(output_dir),
            "--report", str(source),
            "--target-faces", "160", "80",
            "--repeats", "2",
            "--expected-pymeshlab-version", EXPECTED_VERSION,
            "--lineage-ref", "parent", str(lineage), _sha256(lineage),
        ]
    )

    assert returncode != 0
    assert _sha256(source) == source_hash
    safe_report = tmp_path / "grid.glb.assay-error.json"
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["report"]["rerouted"] is True


def test_failure_after_one_output_removes_partial_artifacts_and_preserves_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    package = _fake_package()
    _FakeMeshSet.fail_on_run = 2
    monkeypatch.setattr(runner, "_import_pymeshlab", lambda: package)
    source = tmp_path / "grid.glb"
    lineage = tmp_path / "parent.json"
    output_dir = tmp_path / "outputs"
    report = tmp_path / "report.json"
    _write_grid(source)
    lineage.write_text("{}\n")

    returncode = runner.main(
        [
            "--input", str(source),
            "--expected-input-sha256", _sha256(source),
            "--output-dir", str(output_dir),
            "--report", str(report),
            "--target-faces", "160", "80",
            "--repeats", "2",
            "--expected-pymeshlab-version", EXPECTED_VERSION,
            "--lineage-ref", "parent", str(lineage), _sha256(lineage),
        ]
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "simplify"
    assert data["primary_output_status"] == "partial_removed"
    assert data["last_trustworthy_evidence"]["repeat"] == 1
    assert data["partial_output_cleanup"]["artifact_count"] == 1
    assert data["partial_output_cleanup"]["removed"] is True
    assert not output_dir.exists()

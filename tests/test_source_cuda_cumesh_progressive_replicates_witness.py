import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts.source_cuda_cumesh_postprocess_witness import (
    CUMESH_COMMIT,
    EXPECTED_CUDA_CAPABILITY,
    EXPECTED_CUDA_DEVICE_NAME,
    TRELLIS_COMMIT,
    TRELLIS_POSTPROCESS_SHA256,
    WitnessError,
    sha256_file,
    write_binary_ply,
)
from scripts.source_cuda_cumesh_progressive_replicates_witness import (
    GEOMETRY_ROUTE,
    INITIAL_THRESHOLD,
    THRESHOLD_GROWTH,
    run_witness,
    validate_output_pair,
)


def _write_input(path: Path) -> None:
    write_binary_ply(
        path,
        np.arange(60, dtype=np.float32).reshape(20, 3),
        np.arange(54, dtype=np.int32).reshape(18, 3) % 20,
    )


class _Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class _Mesh:
    next_repeat = 0
    calls: list[tuple[int, float, bool]] = []

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        type(self).next_repeat += 1
        self.repeat = type(self).next_repeat
        self.vertices = vertices.copy()
        self.faces = faces.copy()
        self.cu_mesh = self

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def simplify_step(self, edge: float, skinny: float, threshold: float, timing: bool):
        type(self).calls.append((self.repeat, threshold, timing))
        remove = 1 if len(self.faces) <= 10 else 2
        if self.repeat == 3 and len(self.faces) == 18:
            remove = 3
        self.faces = self.faces[:-remove]
        return len(self.vertices), len(self.faces)

    def read(self):
        return _Tensor(self.vertices), _Tensor(self.faces)


def _runtime(*, instrumented: bool = False):
    return SimpleNamespace(
        torch=SimpleNamespace(),
        cumesh=SimpleNamespace(),
        effective_route={
            "trellis_commit": TRELLIS_COMMIT,
            "trellis_source_clean": True,
            "trellis_postprocess_sha256": TRELLIS_POSTPROCESS_SHA256,
            "cumesh_commit": CUMESH_COMMIT,
            "cumesh_source_clean_before_build": True,
            "cuda_available": True,
            "cuda_device_name": EXPECTED_CUDA_DEVICE_NAME,
            "cuda_capability": list(EXPECTED_CUDA_CAPABILITY),
            "device_type": "cuda",
            "cumesh_instrumentation": {"schema": "forbidden"} if instrumented else None,
        },
        create_mesh=lambda vertices, faces: _Mesh(vertices, faces),
        read_mesh=lambda mesh: mesh.read(),
    )


def _run_valid_witness(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    input_ply = tmp_path / "input.ply"
    archive = tmp_path / "meshes.zip"
    report_path = tmp_path / "report.json"
    _write_input(input_ply)
    input_sha256 = sha256_file(input_ply)
    _Mesh.next_repeat = 0
    _Mesh.calls = []
    run_witness(
        input_ply=input_ply,
        output_archive=archive,
        output_json=report_path,
        expected_input_sha256=input_sha256,
        work_dir=tmp_path / "runtime",
        repeats=5,
        max_steps=8,
        runtime_factory=lambda **kwargs: _runtime(),
    )
    return input_ply, archive, report_path, input_sha256


def _rewrite_archive(
    archive: Path,
    report: dict,
    *,
    replacements: dict[str, bytes] | None = None,
    retained: set[str] | None = None,
    duplicate: str | None = None,
) -> None:
    replacements = replacements or {}
    with zipfile.ZipFile(archive) as bundle:
        payloads = {name: bundle.read(name) for name in bundle.namelist()}
    if retained is not None:
        payloads = {name: payload for name, payload in payloads.items() if name in retained}
    payloads.update(replacements)
    temporary = archive.with_suffix(".rewritten.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name, payload in payloads.items():
            bundle.writestr(name, payload)
        if duplicate is not None:
            bundle.writestr(duplicate, payloads[duplicate])
    temporary.replace(archive)
    report["output_archive"]["sha256"] = sha256_file(archive)
    report["output_archive"]["size_bytes"] = archive.stat().st_size


def test_progressive_witness_uses_fresh_native_meshes_and_validates_archive(tmp_path):
    input_ply = tmp_path / "input.ply"
    archive = tmp_path / "meshes.zip"
    report_path = tmp_path / "report.json"
    _write_input(input_ply)
    _Mesh.next_repeat = 0
    _Mesh.calls = []

    report = run_witness(
        input_ply=input_ply,
        output_archive=archive,
        output_json=report_path,
        expected_input_sha256=sha256_file(input_ply),
        work_dir=tmp_path / "runtime",
        repeats=5,
        max_steps=8,
        runtime_factory=lambda **kwargs: _runtime(),
    )

    assert report["status"] == "done"
    assert report["primary_output_status"] == "validated"
    assert report["effective_route"]["geometry_route"] == GEOMETRY_ROUTE
    assert report["effective_route"]["adjacency_order"] == "native-atomic-fill"
    assert report["effective_route"]["cumesh_instrumentation"] is None
    assert len(report["runs"]) == 5
    assert all(len(run["steps"]) == 8 for run in report["runs"])
    assert report["repeat_stability"]["all_steps_exact"] is False
    assert report["step_stability"][0]["distinct_output_face_counts"] == [15, 16]
    assert len(_Mesh.calls) == 40
    assert all(timing is False for _, _, timing in _Mesh.calls)
    with zipfile.ZipFile(archive) as bundle:
        assert sorted(bundle.namelist()) == sorted(report["output_archive"]["members"])
        assert len(bundle.namelist()) == 40
    assert json.loads(report_path.read_text()) == report


def test_progressive_witness_rejects_instrumented_source_route_before_collection(tmp_path):
    input_ply = tmp_path / "input.ply"
    archive = tmp_path / "meshes.zip"
    report_path = tmp_path / "report.json"
    _write_input(input_ply)
    archive.write_bytes(b"stale success")
    _Mesh.next_repeat = 0

    with pytest.raises(WitnessError, match="unmodified official CuMesh"):
        run_witness(
            input_ply=input_ply,
            output_archive=archive,
            output_json=report_path,
            expected_input_sha256=sha256_file(input_ply),
            work_dir=tmp_path / "runtime",
            repeats=5,
            max_steps=8,
            runtime_factory=lambda **kwargs: _runtime(instrumented=True),
        )

    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "runtime_validation"
    assert report["primary_output_status"] == "not_started"
    assert not archive.exists()
    assert _Mesh.next_repeat == 0


def test_progressive_witness_removes_partial_archive_on_member_failure(tmp_path, monkeypatch):
    input_ply = tmp_path / "input.ply"
    archive = tmp_path / "meshes.zip"
    report_path = tmp_path / "report.json"
    _write_input(input_ply)
    _Mesh.next_repeat = 0

    from scripts import source_cuda_cumesh_progressive_replicates_witness as witness

    real_write = witness.write_binary_ply
    calls = 0

    def fail_second(path, vertices, faces):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected member failure")
        return real_write(path, vertices, faces)

    monkeypatch.setattr(witness, "write_binary_ply", fail_second)

    with pytest.raises(WitnessError, match="injected member failure"):
        run_witness(
            input_ply=input_ply,
            output_archive=archive,
            output_json=report_path,
            expected_input_sha256=sha256_file(input_ply),
            work_dir=tmp_path / "runtime",
            repeats=5,
            max_steps=8,
            runtime_factory=lambda **kwargs: _runtime(),
        )

    report = json.loads(report_path.read_text())
    assert report["failure_phase"] == "trajectory_collection"
    assert report["primary_output_status"] == "partial_removed"
    assert report["last_trustworthy_evidence"]["repeat"] == 1
    assert report["last_trustworthy_evidence"]["step"] == 1
    assert not archive.exists()


def test_progressive_witness_rejects_output_collisions_without_touching_input(tmp_path):
    input_ply = tmp_path / "input.ply"
    report_path = tmp_path / "report.json"
    _write_input(input_ply)
    original_sha = sha256_file(input_ply)

    with pytest.raises(WitnessError, match="paths must be distinct"):
        run_witness(
            input_ply=input_ply,
            output_archive=input_ply,
            output_json=report_path,
            expected_input_sha256=original_sha,
            work_dir=tmp_path / "runtime",
            repeats=5,
            max_steps=8,
            runtime_factory=lambda **kwargs: _runtime(),
        )

    assert sha256_file(input_ply) == original_sha
    assert json.loads(report_path.read_text())["failure_phase"] == "request_validation"


@pytest.mark.parametrize(
    "collision",
    ("archive_temporary", "trellis_runtime", "cumesh_runtime"),
)
def test_progressive_witness_rejects_destructive_path_overlap_before_runtime(
    tmp_path,
    collision,
):
    work_dir = tmp_path / "runtime"
    input_ply = tmp_path / "input.ply"
    archive = tmp_path / "meshes.zip"
    report_path = tmp_path / "report.json"
    if collision == "archive_temporary":
        input_ply = tmp_path / "meshes.zip.tmp"
    elif collision == "trellis_runtime":
        input_ply = work_dir / "TRELLIS.2" / "input.ply"
    elif collision == "cumesh_runtime":
        archive = work_dir / "CuMesh" / "meshes.zip"
    input_ply.parent.mkdir(parents=True, exist_ok=True)
    _write_input(input_ply)
    input_sha256 = sha256_file(input_ply)
    runtime_entered = False

    def forbidden_runtime(**kwargs):
        nonlocal runtime_entered
        runtime_entered = True
        raise AssertionError("runtime must not be entered for unsafe paths")

    with pytest.raises(WitnessError, match="path|overlap|protected|runtime"):
        run_witness(
            input_ply=input_ply,
            output_archive=archive,
            output_json=report_path,
            expected_input_sha256=input_sha256,
            work_dir=work_dir,
            repeats=5,
            max_steps=8,
            runtime_factory=forbidden_runtime,
        )

    assert runtime_entered is False
    assert sha256_file(input_ply) == input_sha256
    failure_reports = list(tmp_path.rglob("*.json"))
    assert failure_reports
    assert any(json.loads(path.read_text())["failure_phase"] == "request_validation" for path in failure_reports)


def test_progressive_witness_reroutes_report_when_atomic_temporary_is_input(tmp_path):
    input_ply = tmp_path / "report.json.tmp"
    report_path = tmp_path / "report.json"
    _write_input(input_ply)
    input_sha256 = sha256_file(input_ply)
    _Mesh.next_repeat = 0

    report = run_witness(
        input_ply=input_ply,
        output_archive=tmp_path / "meshes.zip",
        output_json=report_path,
        expected_input_sha256=input_sha256,
        work_dir=tmp_path / "runtime",
        repeats=5,
        max_steps=8,
        runtime_factory=lambda **kwargs: _runtime(),
    )

    assert report["status"] == "done"
    assert report["report_rerouted"] is True
    assert Path(report["effective_output_json"]) != report_path
    assert sha256_file(input_ply) == input_sha256


def test_progressive_witness_allows_zero_removal_and_grows_threshold(tmp_path):
    class NoOpThenRemoveMesh(_Mesh):
        calls: list[float] = []

        def simplify_step(self, edge, skinny, threshold, timing):
            type(self).calls.append(threshold)
            if threshold > INITIAL_THRESHOLD:
                self.faces = self.faces[:-1]
            return len(self.vertices), len(self.faces)

    runtime = _runtime()
    runtime.create_mesh = lambda vertices, faces: NoOpThenRemoveMesh(vertices, faces)
    input_ply = tmp_path / "input.ply"
    _write_input(input_ply)
    NoOpThenRemoveMesh.next_repeat = 0
    NoOpThenRemoveMesh.calls = []

    report = run_witness(
        input_ply=input_ply,
        output_archive=tmp_path / "meshes.zip",
        output_json=tmp_path / "report.json",
        expected_input_sha256=sha256_file(input_ply),
        work_dir=tmp_path / "runtime",
        repeats=2,
        max_steps=2,
        runtime_factory=lambda **kwargs: runtime,
    )

    assert report["status"] == "done"
    assert [step["removed_faces"] for step in report["runs"][0]["steps"]] == [0, 1]
    assert NoOpThenRemoveMesh.calls == [
        INITIAL_THRESHOLD,
        INITIAL_THRESHOLD * THRESHOLD_GROWTH,
        INITIAL_THRESHOLD,
        INITIAL_THRESHOLD * THRESHOLD_GROWTH,
    ]


def test_progressive_witness_default_release_runtime_accepts_target_free_route(
    tmp_path,
    monkeypatch,
):
    from scripts import source_cuda_cumesh_postprocess_witness as runtime_module

    class FakeCudaTensor(_Tensor):
        def cuda(self):
            return self

    class FakeCuMesh(_Mesh):
        def __init__(self):
            pass

        def init(self, vertices, faces):
            _Mesh.__init__(self, vertices.array, faces.array)

    fake_torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: EXPECTED_CUDA_DEVICE_NAME,
            get_device_capability=lambda _index: EXPECTED_CUDA_CAPABILITY,
        ),
        from_numpy=lambda array: FakeCudaTensor(array),
    )
    fake_cumesh = SimpleNamespace(
        __file__=str(tmp_path / "site-packages" / "cumesh.py"),
        CuMesh=FakeCuMesh,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cumesh", fake_cumesh)

    def fake_setup(command, report):
        report["setup_commands"].append({"command": command, "exit_code": 0})
        for name in ("TRELLIS.2", "CuMesh"):
            if name in command[-1]:
                (tmp_path / "runtime" / name).mkdir(parents=True, exist_ok=True)
        postprocess = tmp_path / "runtime" / "TRELLIS.2" / runtime_module.TRELLIS_POSTPROCESS_PATH
        postprocess.parent.mkdir(parents=True, exist_ok=True)
        postprocess.write_text("pinned source placeholder\n")

    def fake_command_output(command):
        if "rev-parse" in command:
            return TRELLIS_COMMIT if "TRELLIS.2" in command[2] else CUMESH_COMMIT
        return ""

    real_sha256 = runtime_module.sha256_file
    monkeypatch.setattr(runtime_module, "_run_setup_command", fake_setup)
    monkeypatch.setattr(runtime_module, "_command_output", fake_command_output)
    monkeypatch.setattr(
        runtime_module,
        "sha256_file",
        lambda path: (
            TRELLIS_POSTPROCESS_SHA256
            if Path(path).name == "postprocess.py"
            else real_sha256(path)
        ),
    )
    input_ply = tmp_path / "input.ply"
    _write_input(input_ply)
    _Mesh.next_repeat = 0
    _Mesh.calls = []

    report = run_witness(
        input_ply=input_ply,
        output_archive=tmp_path / "meshes.zip",
        output_json=tmp_path / "report.json",
        expected_input_sha256=sha256_file(input_ply),
        work_dir=tmp_path / "runtime",
        repeats=5,
        max_steps=8,
    )

    assert report["status"] == "done"
    assert "target_faces" not in report["requested_route"]
    assert "target_faces" not in report["effective_route"]
    assert report["effective_route"]["cuda_device_name"] == EXPECTED_CUDA_DEVICE_NAME
    assert report["effective_route"]["cuda_capability"] == list(EXPECTED_CUDA_CAPABILITY)
    assert len(_Mesh.calls) == 40


@pytest.mark.parametrize("runtime_root", ("TRELLIS.2", "CuMesh"))
def test_progressive_witness_report_under_destructive_root_terminates_with_safe_report(
    tmp_path,
    runtime_root,
):
    repo_root = Path(__file__).resolve().parents[1]
    input_ply = tmp_path / "input.ply"
    _write_input(input_ply)
    report_path = tmp_path / "runtime" / runtime_root / "report.json"
    code = """
from pathlib import Path
import sys
from scripts.source_cuda_cumesh_progressive_replicates_witness import run_witness
from scripts.source_cuda_cumesh_postprocess_witness import sha256_file

input_ply, archive, report_path, work_dir = map(Path, sys.argv[1:])
try:
    run_witness(
        input_ply=input_ply,
        output_archive=archive,
        output_json=report_path,
        expected_input_sha256=sha256_file(input_ply),
        work_dir=work_dir,
        repeats=5,
        max_steps=8,
        runtime_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime must not be entered")
        ),
    )
except Exception:
    pass
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(input_ply),
            str(tmp_path / "meshes.zip"),
            str(report_path),
            str(tmp_path / "runtime"),
        ],
        cwd=repo_root,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    failure_reports = [
        path
        for path in tmp_path.rglob("*.json")
        if not path.is_relative_to(tmp_path / "runtime")
    ]
    assert failure_reports
    report = json.loads(failure_reports[0].read_text())
    assert report["failure_phase"] == "request_validation"
    assert Path(report["effective_output_json"]).parent != report_path.parent
    assert sha256_file(input_ply) == report["requested_route"]["expected_input_sha256"]


def test_progressive_entrypoint_imports_from_flat_kaggle_capsule(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    for name in (
        "source_cuda_cumesh_progressive_replicates_witness.py",
        "source_cuda_cumesh_postprocess_witness.py",
    ):
        shutil.copy2(repo_root / "scripts" / name, tmp_path / name)

    completed = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "source_cuda_cumesh_progressive_replicates_witness.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--output-archive" in completed.stdout


def test_packet_declares_report_and_archive_without_npz_role(tmp_path):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        load_prepared_packet,
        prepare_packet,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    for name in ("progressive.py", "runtime.py", "input.ply"):
        (capsule / name).write_bytes(name.encode())

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/cumesh-progressive-inputs",
            kernel_id="operator/cumesh-progressive-t4",
            title="CuMesh Progressive T4",
            entrypoint="progressive.py",
            inputs=("progressive.py", "runtime.py", "input.ply"),
            entrypoint_args=("--output-archive", "meshes.zip", "--repeats", "5"),
            output_json="progressive_report.json",
            output_npz=None,
            expected_outputs=("meshes.zip",),
            enable_internet=True,
        )
    )

    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    loaded = load_prepared_packet(
        packet.output_dir,
        expected_capsule_dir=capsule,
        failure_report_dir=tmp_path / "reports",
    )
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()

    assert manifest["outputs"] == ["progressive_report.json", "meshes.zip"]
    assert manifest["output_roles"]["npz"] is None
    assert manifest["output_roles"]["expected"] == ["meshes.zip"]
    assert loaded.output_npz is None
    assert loaded.expected_outputs == ("meshes.zip",)
    assert loaded.entrypoint_args == (
        "--output-archive",
        "meshes.zip",
        "--repeats",
        "5",
    )
    assert 'command += CONFIG.get("entrypoint_args", [])' in runner


def test_downloaded_pair_rejects_self_consistent_archive_substitution(tmp_path):
    input_ply, archive, report_path, input_sha256 = _run_valid_witness(tmp_path)

    admitted = validate_output_pair(
        report_path,
        archive,
        expected_input_sha256=input_sha256,
    )
    assert admitted["archive_sha256"] == sha256_file(archive)
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("attacker/substitute.ply", b"substitute")

    with pytest.raises(WitnessError, match="archive SHA256 differs from producer report"):
        validate_output_pair(
            report_path,
            archive,
            expected_input_sha256=input_sha256,
        )


def test_downloaded_pair_rejects_internally_consistent_wrong_member_payload(tmp_path):
    _, archive, report_path, input_sha256 = _run_valid_witness(tmp_path)
    report = json.loads(report_path.read_text())
    step = report["runs"][0]["steps"][0]
    name = step["member"]["member"]
    forged_ply = tmp_path / "forged.ply"
    write_binary_ply(
        forged_ply,
        np.full((4, 3), 7, dtype=np.float32),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    forged_payload = forged_ply.read_bytes()
    step["member"].update(
        {
            "sha256": sha256_file(forged_ply),
            "size_bytes": len(forged_payload),
        }
    )
    _rewrite_archive(archive, report, replacements={name: forged_payload})
    report_path.write_text(json.dumps(report) + "\n")

    with pytest.raises(WitnessError, match="PLY|vertex|face|count|digest"):
        validate_output_pair(
            report_path,
            archive,
            expected_input_sha256=input_sha256,
        )


def test_downloaded_pair_rejects_internally_consistent_four_by_eight_pair(tmp_path):
    _, archive, report_path, input_sha256 = _run_valid_witness(tmp_path)
    report = json.loads(report_path.read_text())
    report["runs"] = report["runs"][:4]
    retained = {
        step["member"]["member"]
        for run in report["runs"]
        for step in run["steps"]
    }
    report["output_archive"]["members"] = [
        name for name in report["output_archive"]["members"] if name in retained
    ]
    report["output_archive"]["member_count"] = len(retained)
    _rewrite_archive(archive, report, retained=retained)
    report_path.write_text(json.dumps(report) + "\n")

    with pytest.raises(WitnessError, match="five|5|replicate"):
        validate_output_pair(
            report_path,
            archive,
            expected_input_sha256=input_sha256,
        )


def test_downloaded_pair_rejects_duplicate_zip_member_hidden_by_set(tmp_path):
    _, archive, report_path, input_sha256 = _run_valid_witness(tmp_path)
    report = json.loads(report_path.read_text())
    duplicate = report["output_archive"]["members"][0]
    _rewrite_archive(archive, report, duplicate=duplicate)
    report_path.write_text(json.dumps(report) + "\n")

    with pytest.raises(WitnessError, match="duplicate"):
        validate_output_pair(
            report_path,
            archive,
            expected_input_sha256=input_sha256,
        )


def test_downloaded_pair_rejects_wrong_effective_route_with_valid_archive(tmp_path):
    _, archive, report_path, input_sha256 = _run_valid_witness(tmp_path)
    report = json.loads(report_path.read_text())
    report["effective_route"]["geometry_route"] = "forged-route"
    report_path.write_text(json.dumps(report) + "\n")

    with pytest.raises(WitnessError, match="route|geometry"):
        validate_output_pair(
            report_path,
            archive,
            expected_input_sha256=input_sha256,
        )


def test_downloaded_pair_rejects_forged_stability_summary(tmp_path):
    _, archive, report_path, input_sha256 = _run_valid_witness(tmp_path)
    report = json.loads(report_path.read_text())
    report["step_stability"][0]["exact"] = not report["step_stability"][0]["exact"]
    report["repeat_stability"]["all_steps_exact"] = not report["repeat_stability"][
        "all_steps_exact"
    ]
    report_path.write_text(json.dumps(report) + "\n")

    with pytest.raises(WitnessError, match="stability"):
        validate_output_pair(
            report_path,
            archive,
            expected_input_sha256=input_sha256,
        )


@pytest.mark.parametrize(
    "missing_field",
    ("trellis_source_clean", "cumesh_source_clean_before_build", "cumesh_instrumentation"),
)
def test_downloaded_pair_rejects_omitted_route_authority_field(
    tmp_path,
    missing_field,
):
    _, archive, report_path, input_sha256 = _run_valid_witness(tmp_path)
    report = json.loads(report_path.read_text())
    del report["effective_route"][missing_field]
    report_path.write_text(json.dumps(report) + "\n")

    with pytest.raises(WitnessError, match="clean|instrumentation|route"):
        validate_output_pair(
            report_path,
            archive,
            expected_input_sha256=input_sha256,
        )


def test_downloaded_pair_rejects_done_report_with_partial_primary_status(tmp_path):
    report_path = tmp_path / "report.json"
    archive = tmp_path / "meshes.zip"
    archive.write_bytes(b"not admitted")
    report_path.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.source_cuda_cumesh_progressive_replicates.v1",
                "status": "done",
                "failure_phase": None,
                "primary_output_status": "partial",
            }
        )
    )

    with pytest.raises(WitnessError, match="primary output is not validated"):
        validate_output_pair(
            report_path,
            archive,
            expected_input_sha256="0" * 64,
        )

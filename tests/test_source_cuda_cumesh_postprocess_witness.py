from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest


class FakeCuMesh:
    def __init__(self):
        self.num_vertices = 8
        self.num_faces = 12
        self.calls = []

    def _call(self, name, *args, face_delta=0):
        self.calls.append((name, args))
        self.num_faces += face_delta

    def fill_holes(self, **kwargs):
        self._call("fill_holes", kwargs, face_delta=2)

    def simplify(self, target, **kwargs):
        self._call("simplify", target, kwargs)
        self.num_faces = target - 1

    def remove_duplicate_faces(self):
        self._call("remove_duplicate_faces")

    def repair_non_manifold_edges(self):
        self._call("repair_non_manifold_edges", face_delta=1)

    def remove_small_connected_components(self, threshold):
        self._call("remove_small_connected_components", threshold, face_delta=-2)

    def unify_face_orientations(self):
        self._call("unify_face_orientations")


def test_geometry_sequence_matches_clean_trellis_release_order():
    from scripts.source_cuda_cumesh_postprocess_witness import execute_geometry_sequence

    mesh = FakeCuMesh()
    snapshots = []

    def snapshot(operation, input_faces, output_faces, details):
        snapshots.append((operation, input_faces, output_faces, details))

    execute_geometry_sequence(mesh, 10, snapshot)

    assert [name for name, _ in mesh.calls] == [
        "fill_holes",
        "simplify",
        "remove_duplicate_faces",
        "repair_non_manifold_edges",
        "remove_small_connected_components",
        "fill_holes",
        "simplify",
        "remove_duplicate_faces",
        "repair_non_manifold_edges",
        "remove_small_connected_components",
        "fill_holes",
        "unify_face_orientations",
    ]
    assert mesh.calls[0] == ("fill_holes", ({"max_hole_perimeter": 3e-2},))
    assert mesh.calls[1] == ("simplify", (30, {"verbose": False}))
    assert mesh.calls[4] == ("remove_small_connected_components", (1e-5,))
    assert mesh.calls[6] == ("simplify", (10, {"verbose": False}))
    assert [entry[0] for entry in snapshots] == [
        "prefill_holes",
        "simplify_coarse",
        "remove_duplicate_faces_initial",
        "repair_non_manifold_edges_initial",
        "remove_small_connected_components_initial",
        "fill_holes_initial",
        "simplify_final",
        "remove_duplicate_faces_final",
        "repair_non_manifold_edges_final",
        "remove_small_connected_components_final",
        "fill_holes_final",
        "unify_face_orientations",
    ]
    assert all(entry[1] >= 0 and entry[2] >= 0 for entry in snapshots)


def test_wrong_input_hash_fails_before_runtime_and_writes_report(tmp_path):
    from scripts.source_cuda_cumesh_postprocess_witness import WitnessError, run_witness

    input_ply = tmp_path / "input.ply"
    _write_test_ply(input_ply)
    original = input_ply.read_bytes()
    report_json = tmp_path / "report.json"

    def forbidden_runtime(*args, **kwargs):
        raise AssertionError("runtime setup must not run")

    with pytest.raises(WitnessError, match="input SHA256"):
        run_witness(
            input_ply=input_ply,
            output_dir=tmp_path / "outputs",
            report_json=report_json,
            expected_input_sha256="0" * 64,
            target_faces=10,
            work_dir=tmp_path / "work",
            runtime_factory=forbidden_runtime,
        )

    report = json.loads(report_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "input_validation"
    assert report["last_trustworthy_phase"] == "request_received"
    assert report["primary_output_status"] == "not_started"
    assert report["effective_route"] is None
    assert input_ply.read_bytes() == original


def test_input_output_alias_fails_without_deleting_input(tmp_path):
    from scripts.source_cuda_cumesh_postprocess_witness import WitnessError, run_witness

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    input_ply = output_dir / "01_prefill_holes.ply"
    input_ply.write_bytes(b"protected input")
    digest = hashlib.sha256(input_ply.read_bytes()).hexdigest()
    report_json = tmp_path / "report.json"

    with pytest.raises(WitnessError, match="aliases stage output"):
        run_witness(
            input_ply=input_ply,
            output_dir=output_dir,
            report_json=report_json,
            expected_input_sha256=digest,
            target_faces=10,
            work_dir=tmp_path / "work",
        )

    assert input_ply.read_bytes() == b"protected input"
    report = json.loads(report_json.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["primary_output_status"] == "not_started"


def test_report_alias_reroutes_failure_without_overwriting_input(tmp_path):
    from scripts.source_cuda_cumesh_postprocess_witness import WitnessError, run_witness

    input_ply = tmp_path / "input.ply"
    input_ply.write_bytes(b"protected input")
    digest = hashlib.sha256(input_ply.read_bytes()).hexdigest()

    with pytest.raises(WitnessError, match="report path aliases protected input"):
        run_witness(
            input_ply=input_ply,
            output_dir=tmp_path / "outputs",
            report_json=input_ply,
            expected_input_sha256=digest,
            target_faces=10,
            work_dir=tmp_path / "work",
        )

    assert input_ply.read_bytes() == b"protected input"
    rerouted = tmp_path / "input.ply.failure.json"
    report = json.loads(rerouted.read_text())
    assert report["requested_report_json"] == str(input_ply)
    assert report["effective_report_json"] == str(rerouted)
    assert report["report_rerouted"] is True
    assert report["primary_output_status"] == "not_started"


def test_runtime_failure_removes_stale_stage_and_remains_durable(tmp_path):
    from scripts.source_cuda_cumesh_postprocess_witness import WitnessError, run_witness

    input_ply = tmp_path / "input.ply"
    _write_test_ply(input_ply)
    digest = hashlib.sha256(input_ply.read_bytes()).hexdigest()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    stale = output_dir / "01_prefill_holes.ply"
    stale.write_bytes(b"stale")
    report_json = tmp_path / "report.json"

    def failing_runtime(*args, **kwargs):
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError, match="compile failed"):
        run_witness(
            input_ply=input_ply,
            output_dir=output_dir,
            report_json=report_json,
            expected_input_sha256=digest,
            target_faces=10,
            work_dir=tmp_path / "work",
            runtime_factory=failing_runtime,
        )

    assert not stale.exists()
    report = json.loads(report_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "runtime_setup"
    assert report["last_trustworthy_phase"] == "stale_outputs_removed"
    assert report["primary_output_status"] == "not_started"
    assert report["effective_route"] is None


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeRuntime:
    def __init__(self, mesh, *, device_name="Tesla T4", capability=(7, 5)):
        self.mesh = mesh
        self.effective_route = {
            "trellis_commit": "5565d240c4a494caaf9ece7a554542b76ffa36d3",
            "trellis_postprocess_sha256": (
                "ef51a1ba0f2748ffb4c265b47d382cee956f23c6a52d0f3587e6d8beccb7e54a"
            ),
            "cumesh_commit": "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87",
            "cuda_device_name": device_name,
            "cuda_capability": list(capability),
            "device_type": "cuda",
        }

    def create_mesh(self, vertices, faces):
        return self.mesh

    def read_mesh(self, mesh):
        return (
            FakeTensor(np.zeros((mesh.num_vertices, 3), dtype=np.float32)),
            FakeTensor(np.zeros((mesh.num_faces, 3), dtype=np.int32)),
        )


def _write_test_ply(path):
    from scripts.source_cuda_cumesh_postprocess_witness import write_binary_ply

    write_binary_ply(
        path,
        np.zeros((8, 3), dtype=np.float32),
        np.zeros((12, 3), dtype=np.int32),
    )


def test_non_t4_runtime_is_rejected_before_sequence(tmp_path):
    from scripts.source_cuda_cumesh_postprocess_witness import WitnessError, run_witness

    input_ply = tmp_path / "input.ply"
    _write_test_ply(input_ply)
    digest = hashlib.sha256(input_ply.read_bytes()).hexdigest()
    runtime = FakeRuntime(FakeCuMesh(), device_name="NVIDIA P100", capability=(6, 0))

    def forbidden_sequence(*args, **kwargs):
        raise AssertionError("sequence must not run")

    with pytest.raises(WitnessError, match="Tesla T4"):
        run_witness(
            input_ply=input_ply,
            output_dir=tmp_path / "outputs",
            report_json=tmp_path / "report.json",
            expected_input_sha256=digest,
            target_faces=10,
            work_dir=tmp_path / "work",
            runtime_factory=lambda *args, **kwargs: runtime,
            sequence_runner=forbidden_sequence,
        )

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["failure_phase"] == "runtime_validation"
    assert report["primary_output_status"] == "not_started"
    assert report["effective_route"]["cuda_device_name"] == "NVIDIA P100"


def test_done_report_requires_every_reopened_stage(tmp_path):
    from scripts.source_cuda_cumesh_postprocess_witness import WitnessError, run_witness

    input_ply = tmp_path / "input.ply"
    _write_test_ply(input_ply)
    digest = hashlib.sha256(input_ply.read_bytes()).hexdigest()
    runtime = FakeRuntime(FakeCuMesh())

    def incomplete_sequence(mesh, target_faces, snapshot):
        before = mesh.num_faces
        mesh.fill_holes(max_hole_perimeter=3e-2)
        snapshot("prefill_holes", before, mesh.num_faces, {})

    with pytest.raises(WitnessError, match="stage set"):
        run_witness(
            input_ply=input_ply,
            output_dir=tmp_path / "outputs",
            report_json=tmp_path / "report.json",
            expected_input_sha256=digest,
            target_faces=10,
            work_dir=tmp_path / "work",
            runtime_factory=lambda *args, **kwargs: runtime,
            sequence_runner=incomplete_sequence,
        )

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "stage_validation"
    assert report["primary_output_status"] == "partial"
    assert len(report["stage_artifacts"]) == 1

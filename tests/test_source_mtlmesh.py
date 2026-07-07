import subprocess
from pathlib import Path
import sys
import types

import numpy as np


def test_source_native_simplify_can_run_in_reference_python_subprocess(monkeypatch, tmp_path):
    from trellmlx.source_mtlmesh import simplify_source_native

    calls = []

    def fake_run(cmd, *, capture_output, text, check, env):
        calls.append((cmd, env))
        input_npz = Path(cmd[-4])
        output_npz = Path(cmd[-3])
        target_faces = int(cmd[-2])
        data = np.load(input_npz)
        np.savez_compressed(
            output_npz,
            vertices=np.asarray(data["vertices"], dtype=np.float32)[:3],
            faces=np.asarray(data["faces"], dtype=np.int32)[:target_faces],
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="route ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    out_vertices, out_faces = simplify_source_native(
        vertices,
        faces,
        1,
        verbose=False,
        reference_python="/ref/python",
        expected_source_root="/ref/mtlmesh",
    )

    assert out_vertices.shape == (3, 3)
    assert out_faces.shape == (1, 3)
    assert calls
    cmd, env = calls[0]
    assert cmd[0] == "/ref/python"
    assert cmd[-2:] == ["1", "/ref/mtlmesh"]
    assert "/private/tmp/trellis2mlx-trellis-winding-source-successor-0706" in env["PYTHONPATH"]


def test_source_native_orient_can_run_in_reference_python_subprocess(monkeypatch):
    from trellmlx.source_mtlmesh import orient_source_native

    calls = []

    def fake_run(cmd, *, capture_output, text, check, env):
        calls.append((cmd, env))
        input_npz = Path(cmd[-3])
        output_npz = Path(cmd[-2])
        data = np.load(input_npz)
        faces = np.asarray(data["faces"], dtype=np.int32).copy()
        faces[1] = faces[1][[0, 2, 1]]
        np.savez_compressed(
            output_npz,
            vertices=np.asarray(data["vertices"], dtype=np.float32),
            faces=faces,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="route ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

    out_vertices, out_faces = orient_source_native(
        vertices,
        faces,
        verbose=False,
        reference_python="/ref/python",
        expected_source_root="/ref/mtlmesh",
    )

    np.testing.assert_array_equal(out_vertices, vertices)
    np.testing.assert_array_equal(out_faces[1], np.array([1, 3, 2], dtype=np.int32))
    cmd, env = calls[0]
    assert cmd[0] == "/ref/python"
    assert cmd[-1] == "/ref/mtlmesh"
    assert "/private/tmp/trellis2mlx-trellis-winding-source-successor-0706" in env["PYTHONPATH"]


def test_source_native_cleanup_can_run_in_reference_python_subprocess(monkeypatch):
    from trellmlx.source_mtlmesh import cleanup_source_native

    calls = []

    def fake_run(cmd, *, capture_output, text, check, env):
        calls.append((cmd, env))
        input_npz = Path(cmd[-3])
        output_npz = Path(cmd[-2])
        data = np.load(input_npz)
        np.savez_compressed(
            output_npz,
            vertices=np.asarray(data["vertices"], dtype=np.float32)[:3],
            faces=np.asarray(data["faces"], dtype=np.int32)[:1],
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="route ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

    out_vertices, out_faces = cleanup_source_native(
        vertices,
        faces,
        verbose=False,
        reference_python="/ref/python",
        expected_source_root="/ref/mtlmesh",
    )

    assert out_vertices.shape == (3, 3)
    assert out_faces.shape == (1, 3)
    cmd, env = calls[0]
    assert cmd[0] == "/ref/python"
    assert cmd[-1] == "/ref/mtlmesh"
    assert "/private/tmp/trellis2mlx-trellis-winding-source-successor-0706" in env["PYTHONPATH"]


def test_source_native_postprocess_can_run_in_reference_python_subprocess(monkeypatch):
    from trellmlx.source_mtlmesh import postprocess_source_native

    calls = []

    def fake_run(cmd, *, capture_output, text, check, env):
        calls.append((cmd, env))
        input_npz = Path(cmd[-5])
        output_npz = Path(cmd[-4])
        trace_json = Path(cmd[-3])
        target_faces = int(cmd[-2])
        data = np.load(input_npz)
        np.savez_compressed(
            output_npz,
            vertices=np.asarray(data["vertices"], dtype=np.float32)[:3],
            faces=np.asarray(data["faces"], dtype=np.int32)[:target_faces],
        )
        trace_json.write_text('[{"operation": "source_full"}]\n')
        return subprocess.CompletedProcess(cmd, 0, stdout="route ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

    out_vertices, out_faces, trace = postprocess_source_native(
        vertices,
        faces,
        1,
        verbose=False,
        reference_python="/ref/python",
        expected_source_root="/ref/mtlmesh",
    )

    assert out_vertices.shape == (3, 3)
    assert out_faces.shape == (1, 3)
    assert trace == [{"operation": "source_full"}]
    cmd, env = calls[0]
    assert cmd[0] == "/ref/python"
    assert cmd[-2:] == ["1", "/ref/mtlmesh"]
    assert "/private/tmp/trellis2mlx-trellis-winding-source-successor-0706" in env["PYTHONPATH"]


def test_source_native_subprocess_failure_names_reference_python(monkeypatch):
    from trellmlx.source_mtlmesh import simplify_source_native

    def fake_run(cmd, *, capture_output, text, check, env):
        return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="backend exploded")

    monkeypatch.setattr(subprocess, "run", fake_run)

    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    try:
        simplify_source_native(
            vertices,
            faces,
            1,
            reference_python="/ref/python",
            expected_source_root="/ref/mtlmesh",
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected source-native subprocess failure")

    assert "/ref/python" in message
    assert "backend exploded" in message


def test_source_native_uses_simplify_step_loop_when_available(monkeypatch):
    import trellmlx.source_mtlmesh as source_mtlmesh

    class FakeTensor:
        def __init__(self, array):
            self.array = array

        def contiguous(self):
            return self

    class FakeTorch(types.ModuleType):
        def from_numpy(self, array):
            return FakeTensor(array)

    class FakeMesh:
        calls = []

        def init(self, vertices, faces):
            self.vertices = vertices
            self.faces = faces
            self.num_faces = 10

        def simplify_step(self, lambda_edge_length, lambda_skinny, threshold, timing):
            self.calls.append((lambda_edge_length, lambda_skinny, threshold, timing))
            if len(self.calls) == 1:
                self.num_faces = 10
                return 8, 10
            self.num_faces = 4
            return 5, 4

        def simplify(self, *args, **kwargs):
            raise AssertionError("source-native wrapper must use simplify_step when available")

        def read(self):
            return (
                np.zeros((5, 3), dtype=np.float32),
                np.zeros((4, 3), dtype=np.int32),
            )

    fake_torch = FakeTorch("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(source_mtlmesh, "_load_source_mesh_class", lambda **kwargs: FakeMesh)

    vertices = np.zeros((6, 3), dtype=np.float32)
    faces = np.zeros((10, 3), dtype=np.int32)

    out_vertices, out_faces = source_mtlmesh.simplify_source_native(
        vertices,
        faces,
        5,
        verbose=False,
    )

    assert out_vertices.shape == (5, 3)
    assert out_faces.shape == (4, 3)
    assert FakeMesh.calls == [
        (1e-2, 1e-3, 1e-8, False),
        (1e-2, 1e-3, 10 * 1e-8, False),
    ]


def test_source_native_orient_calls_unify_face_orientations(monkeypatch):
    import trellmlx.source_mtlmesh as source_mtlmesh

    class FakeTensor:
        def __init__(self, array):
            self.array = array

        def contiguous(self):
            return self

    class FakeTorch(types.ModuleType):
        def from_numpy(self, array):
            return FakeTensor(array)

    class FakeMesh:
        calls = []

        def init(self, vertices, faces):
            self.vertices = vertices
            self.faces = faces

        def unify_face_orientations(self):
            self.calls.append("unify_face_orientations")

        def read(self):
            return (
                np.zeros((4, 3), dtype=np.float32),
                np.zeros((2, 3), dtype=np.int32),
            )

    fake_torch = FakeTorch("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(source_mtlmesh, "_load_source_mesh_class", lambda **kwargs: FakeMesh)

    out_vertices, out_faces = source_mtlmesh.orient_source_native(
        np.zeros((4, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.int32),
        verbose=False,
    )

    assert out_vertices.shape == (4, 3)
    assert out_faces.shape == (2, 3)
    assert FakeMesh.calls == ["unify_face_orientations"]


def test_source_native_cleanup_calls_source_primitives(monkeypatch):
    import trellmlx.source_mtlmesh as source_mtlmesh

    class FakeTensor:
        def __init__(self, array):
            self.array = array

        def contiguous(self):
            return self

    class FakeTorch(types.ModuleType):
        def from_numpy(self, array):
            return FakeTensor(array)

    class FakeMesh:
        calls = []

        def init(self, vertices, faces):
            self.vertices = vertices
            self.faces = faces

        def remove_duplicate_faces(self):
            self.calls.append("remove_duplicate_faces")

        def repair_non_manifold_edges(self):
            self.calls.append("repair_non_manifold_edges")

        def remove_small_connected_components(self, min_area):
            self.calls.append(("remove_small_connected_components", min_area))

        def fill_holes(self, max_hole_perimeter):
            self.calls.append(("fill_holes", max_hole_perimeter))

        def read(self):
            return (
                np.zeros((4, 3), dtype=np.float32),
                np.zeros((2, 3), dtype=np.int32),
            )

    fake_torch = FakeTorch("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(source_mtlmesh, "_load_source_mesh_class", lambda **kwargs: FakeMesh)

    out_vertices, out_faces = source_mtlmesh.cleanup_source_native(
        np.zeros((4, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.int32),
        verbose=False,
    )

    assert out_vertices.shape == (4, 3)
    assert out_faces.shape == (2, 3)
    assert FakeMesh.calls == [
        "remove_duplicate_faces",
        "repair_non_manifold_edges",
        ("remove_small_connected_components", 1e-5),
        ("fill_holes", 3e-2),
    ]


def test_source_native_postprocess_uses_one_mesh_object_and_records_trace(monkeypatch):
    import trellmlx.source_mtlmesh as source_mtlmesh

    class FakeTensor:
        def __init__(self, array):
            self.array = array

        def contiguous(self):
            return self

    class FakeTorch(types.ModuleType):
        def from_numpy(self, array):
            return FakeTensor(array)

    class FakeMesh:
        instances = []

        def __init__(self):
            self.calls = []
            self.num_faces = 10
            FakeMesh.instances.append(self)

        def init(self, vertices, faces):
            self.calls.append("init")

        def simplify(self, target_faces, *, verbose=False, options=None):
            self.calls.append(("simplify", self.num_faces, target_faces, verbose, options))
            if self.num_faces == 10:
                self.num_faces = 6
                return
            self.num_faces = 3

        def remove_duplicate_faces(self):
            self.calls.append("remove_duplicate_faces")

        def repair_non_manifold_edges(self):
            self.calls.append("repair_non_manifold_edges")

        def remove_small_connected_components(self, min_area):
            self.calls.append(("remove_small_connected_components", min_area))

        def fill_holes(self, max_hole_perimeter):
            self.calls.append(("fill_holes", max_hole_perimeter))

        def unify_face_orientations(self):
            self.calls.append("unify_face_orientations")

        def read(self):
            return (
                np.zeros((4, 3), dtype=np.float32),
                np.zeros((self.num_faces, 3), dtype=np.int32),
            )

    fake_torch = FakeTorch("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(source_mtlmesh, "_load_source_mesh_class", lambda **kwargs: FakeMesh)

    out_vertices, out_faces, trace = source_mtlmesh.postprocess_source_native(
        np.zeros((4, 3), dtype=np.float32),
        np.zeros((10, 3), dtype=np.int32),
        3,
        verbose=False,
    )

    assert len(FakeMesh.instances) == 1
    assert out_vertices.shape == (4, 3)
    assert out_faces.shape == (3, 3)
    assert [entry["operation"] for entry in trace] == [
        "prefill_holes_source_native",
        "simplify_coarse_source_native_qem",
        "cleanup_initial_source_native",
        "simplify_final_source_native_qem",
        "cleanup_final_source_native",
        "orient_faces_source_native",
    ]
    assert FakeMesh.instances[0].calls == [
        "init",
        ("fill_holes", 3e-2),
        ("simplify", 10, 9, False, {"thresh": 1e-08, "lambda_edge_length": 0.01, "lambda_skinny": 0.001}),
        "remove_duplicate_faces",
        "repair_non_manifold_edges",
        ("remove_small_connected_components", 1e-5),
        ("fill_holes", 3e-2),
        ("simplify", 6, 3, False, {"thresh": 1e-08, "lambda_edge_length": 0.01, "lambda_skinny": 0.001}),
        "remove_duplicate_faces",
        "repair_non_manifold_edges",
        ("remove_small_connected_components", 1e-5),
        ("fill_holes", 3e-2),
        "unify_face_orientations",
    ]
    assert trace[1]["simplifier_route"] == "high_level_simplify"
    assert trace[3]["simplifier_route"] == "high_level_simplify"

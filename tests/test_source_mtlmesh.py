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

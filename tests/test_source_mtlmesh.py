import subprocess
from pathlib import Path

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

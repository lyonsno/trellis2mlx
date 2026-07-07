"""Optional source-native mtlmesh simplification route."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from trellmlx.source_route_identity import (
    SourceRouteIdentityError,
    probe_cumesh_route_identity,
    validate_source_route_identity,
)


def _external_source_native_code() -> str:
    return r'''
import sys

import numpy as np

from trellmlx.source_mtlmesh import simplify_source_native


verbose = sys.argv[1] == "1"
input_npz, output_npz, target_faces_text, expected_root = sys.argv[2:6]
data = np.load(input_npz)
vertices, faces = simplify_source_native(
    np.asarray(data["vertices"], dtype=np.float32),
    np.asarray(data["faces"], dtype=np.int32),
    int(target_faces_text),
    verbose=verbose,
    expected_source_root=expected_root or None,
)
np.savez_compressed(output_npz, vertices=vertices, faces=faces)
'''


def _external_source_native_orient_code() -> str:
    return r'''
import sys

import numpy as np

from trellmlx.source_mtlmesh import orient_source_native


verbose = sys.argv[1] == "1"
input_npz, output_npz, expected_root = sys.argv[2:5]
data = np.load(input_npz)
vertices, faces = orient_source_native(
    np.asarray(data["vertices"], dtype=np.float32),
    np.asarray(data["faces"], dtype=np.int32),
    verbose=verbose,
    expected_source_root=expected_root or None,
)
np.savez_compressed(output_npz, vertices=vertices, faces=faces)
'''


def _external_source_native_cleanup_code() -> str:
    return r'''
import sys

import numpy as np

from trellmlx.source_mtlmesh import cleanup_source_native


verbose = sys.argv[1] == "1"
input_npz, output_npz, expected_root = sys.argv[2:5]
data = np.load(input_npz)
vertices, faces = cleanup_source_native(
    np.asarray(data["vertices"], dtype=np.float32),
    np.asarray(data["faces"], dtype=np.int32),
    verbose=verbose,
    expected_source_root=expected_root or None,
)
np.savez_compressed(output_npz, vertices=vertices, faces=faces)
'''


def _external_source_native_postprocess_code() -> str:
    return r'''
import json
import sys
from pathlib import Path

import numpy as np

from trellmlx.source_mtlmesh import postprocess_source_native


verbose = sys.argv[1] == "1"
input_npz, output_npz, trace_json, target_faces_text, expected_root = sys.argv[2:7]
data = np.load(input_npz)
vertices, faces, trace = postprocess_source_native(
    np.asarray(data["vertices"], dtype=np.float32),
    np.asarray(data["faces"], dtype=np.int32),
    int(target_faces_text),
    verbose=verbose,
    expected_source_root=expected_root or None,
)
np.savez_compressed(output_npz, vertices=vertices, faces=faces)
Path(trace_json).write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
'''


def _as_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    if callable(value):
        return int(value())
    return int(value)


def _mesh_face_count(mesh) -> int:
    return _as_int(getattr(mesh, "num_faces"))


def _simplify_with_step_loop(
    mesh,
    target_faces: int,
    *,
    lambda_edge_length: float,
    lambda_skinny: float,
    thresh: float,
) -> list[dict[str, int | float]]:
    num_face = _mesh_face_count(mesh)
    step_trace: list[dict[str, int | float]] = []
    if num_face <= target_faces:
        return step_trace

    iteration = 0
    while True:
        before = _mesh_face_count(mesh)
        new_num_vert, new_num_face = mesh.simplify_step(
            float(lambda_edge_length),
            float(lambda_skinny),
            float(thresh),
            False,
        )
        iteration += 1
        new_num_vert = _as_int(new_num_vert)
        new_num_face = _as_int(new_num_face)
        removed = before - new_num_face
        step_trace.append({
            "iteration": iteration,
            "threshold": float(thresh),
            "input_faces": int(before),
            "output_faces": int(new_num_face),
            "output_vertices": int(new_num_vert),
            "removed_faces": int(removed),
        })
        if new_num_face <= target_faces:
            break

        if removed / max(before, 1) < 1e-2:
            thresh *= 10
        num_face = new_num_face
    return step_trace


def _simplify_source_mesh(
    mesh,
    target_faces: int,
    *,
    verbose: bool,
    lambda_edge_length: float,
    lambda_skinny: float,
    thresh: float,
) -> dict:
    if hasattr(mesh, "simplify"):
        mesh.simplify(
            int(target_faces),
            verbose=bool(verbose),
            options={
                "thresh": float(thresh),
                "lambda_edge_length": float(lambda_edge_length),
                "lambda_skinny": float(lambda_skinny),
            },
        )
        return {"simplifier_route": "high_level_simplify"}

    step_trace = _simplify_with_step_loop(
        mesh,
        int(target_faces),
        lambda_edge_length=float(lambda_edge_length),
        lambda_skinny=float(lambda_skinny),
        thresh=float(thresh),
    )
    return {
        "simplifier_route": "simplify_step_loop",
        "simplifier_step_trace": step_trace,
    }


def _run_source_native_subprocess(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    *,
    reference_python: str | Path,
    verbose: bool,
    expected_source_root: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else str(repo_root) + os.pathsep + existing_pythonpath
    )

    with tempfile.TemporaryDirectory(prefix="trellis2mlx-source-native-") as tmp:
        tmp_path = Path(tmp)
        input_npz = tmp_path / "input_mesh.npz"
        output_npz = tmp_path / "output_mesh.npz"
        np.savez_compressed(
            input_npz,
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
        )
        cmd = [
            str(reference_python),
            "-c",
            _external_source_native_code(),
            "1" if verbose else "0",
            str(input_npz),
            str(output_npz),
            str(int(target_faces)),
            str(expected_source_root or ""),
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            detail = completed.stderr or completed.stdout or "no subprocess output"
            raise RuntimeError(
                f"qem_backend='source-native' reference_python {reference_python} "
                f"failed with exit code {completed.returncode}: {detail}"
            )
        if not output_npz.exists():
            raise RuntimeError(
                f"qem_backend='source-native' reference_python {reference_python} "
                "completed without writing output mesh"
            )
        output = np.load(output_npz)
        return (
            np.asarray(output["vertices"], dtype=np.float32),
            np.asarray(output["faces"], dtype=np.int32),
        )


def _run_source_native_orient_subprocess(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    reference_python: str | Path,
    verbose: bool,
    expected_source_root: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else str(repo_root) + os.pathsep + existing_pythonpath
    )

    with tempfile.TemporaryDirectory(prefix="trellis2mlx-source-native-orient-") as tmp:
        tmp_path = Path(tmp)
        input_npz = tmp_path / "input_mesh.npz"
        output_npz = tmp_path / "output_mesh.npz"
        np.savez_compressed(
            input_npz,
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
        )
        cmd = [
            str(reference_python),
            "-c",
            _external_source_native_orient_code(),
            "1" if verbose else "0",
            str(input_npz),
            str(output_npz),
            str(expected_source_root or ""),
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            detail = completed.stderr or completed.stdout or "no subprocess output"
            raise RuntimeError(
                f"source-native orientation reference_python {reference_python} "
                f"failed with exit code {completed.returncode}: {detail}"
            )
        if not output_npz.exists():
            raise RuntimeError(
                f"source-native orientation reference_python {reference_python} "
                "completed without writing output mesh"
            )
        output = np.load(output_npz)
        return (
            np.asarray(output["vertices"], dtype=np.float32),
            np.asarray(output["faces"], dtype=np.int32),
        )


def _run_source_native_cleanup_subprocess(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    reference_python: str | Path,
    verbose: bool,
    expected_source_root: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else str(repo_root) + os.pathsep + existing_pythonpath
    )

    with tempfile.TemporaryDirectory(prefix="trellis2mlx-source-native-cleanup-") as tmp:
        tmp_path = Path(tmp)
        input_npz = tmp_path / "input_mesh.npz"
        output_npz = tmp_path / "output_mesh.npz"
        np.savez_compressed(
            input_npz,
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
        )
        cmd = [
            str(reference_python),
            "-c",
            _external_source_native_cleanup_code(),
            "1" if verbose else "0",
            str(input_npz),
            str(output_npz),
            str(expected_source_root or ""),
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            detail = completed.stderr or completed.stdout or "no subprocess output"
            raise RuntimeError(
                f"source-native cleanup reference_python {reference_python} "
                f"failed with exit code {completed.returncode}: {detail}"
            )
        if not output_npz.exists():
            raise RuntimeError(
                f"source-native cleanup reference_python {reference_python} "
                "completed without writing output mesh"
            )
        output = np.load(output_npz)
        return (
            np.asarray(output["vertices"], dtype=np.float32),
            np.asarray(output["faces"], dtype=np.int32),
        )


def _run_source_native_postprocess_subprocess(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    *,
    reference_python: str | Path,
    verbose: bool,
    expected_source_root: str | Path | None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else str(repo_root) + os.pathsep + existing_pythonpath
    )

    with tempfile.TemporaryDirectory(prefix="trellis2mlx-source-native-postprocess-") as tmp:
        tmp_path = Path(tmp)
        input_npz = tmp_path / "input_mesh.npz"
        output_npz = tmp_path / "output_mesh.npz"
        trace_json = tmp_path / "operation_trace.json"
        np.savez_compressed(
            input_npz,
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
        )
        cmd = [
            str(reference_python),
            "-c",
            _external_source_native_postprocess_code(),
            "1" if verbose else "0",
            str(input_npz),
            str(output_npz),
            str(trace_json),
            str(int(target_faces)),
            str(expected_source_root or ""),
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            detail = completed.stderr or completed.stdout or "no subprocess output"
            raise RuntimeError(
                f"source-native postprocess reference_python {reference_python} "
                f"failed with exit code {completed.returncode}: {detail}"
            )
        if not output_npz.exists():
            raise RuntimeError(
                f"source-native postprocess reference_python {reference_python} "
                "completed without writing output mesh"
            )
        if not trace_json.exists():
            raise RuntimeError(
                f"source-native postprocess reference_python {reference_python} "
                "completed without writing operation trace"
            )
        output = np.load(output_npz)
        import json
        return (
            np.asarray(output["vertices"], dtype=np.float32),
            np.asarray(output["faces"], dtype=np.int32),
            json.loads(trace_json.read_text()),
        )


def _load_source_mesh_class(*, expected_source_root: str | Path | None = None):
    try:
        cumesh = importlib.import_module("cumesh")
    except ImportError as exc:
        raise RuntimeError(
            "qem_backend='source-native' requires the reference mtlmesh/cumesh "
            "package on PYTHONPATH"
        ) from exc

    mesh_cls = getattr(cumesh, "CuMesh", None)
    if mesh_cls is not None:
        try:
            validate_source_route_identity(
                probe_cumesh_route_identity(),
                expected_root=expected_source_root,
            )
        except SourceRouteIdentityError as exc:
            raise RuntimeError(f"qem_backend='source-native' rejected route: {exc}") from exc
        return mesh_cls

    try:
        metal_backend = importlib.import_module("cumesh.metal_backend")
    except ImportError as exc:
        raise RuntimeError(
            "qem_backend='source-native' found cumesh but not cumesh.metal_backend"
        ) from exc

    mesh_cls = getattr(metal_backend, "MtlMesh", None)
    if mesh_cls is None:
        raise RuntimeError(
            "qem_backend='source-native' requires cumesh.CuMesh or "
            "cumesh.metal_backend.MtlMesh"
        )
    try:
        validate_source_route_identity(
            probe_cumesh_route_identity(),
            expected_root=expected_source_root,
        )
    except SourceRouteIdentityError as exc:
        raise RuntimeError(f"qem_backend='source-native' rejected route: {exc}") from exc
    return mesh_cls


def simplify_source_native(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    *,
    verbose: bool = True,
    lambda_edge_length: float = 1e-2,
    lambda_skinny: float = 1e-3,
    thresh: float = 1e-8,
    expected_source_root: str | Path | None = None,
    reference_python: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplify with the reference mtlmesh/cumesh backend when installed.

    This is intentionally not a fallback path: callers select it when they want
    source-native QEM behavior rather than the local MLX parity probe.
    """
    if reference_python is not None:
        return _run_source_native_subprocess(
            vertices,
            faces,
            target_faces,
            reference_python=reference_python,
            verbose=verbose,
            expected_source_root=expected_source_root,
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "qem_backend='source-native' requires torch for the reference "
            "mtlmesh/cumesh backend"
        ) from exc

    mesh_cls = _load_source_mesh_class(expected_source_root=expected_source_root)
    mesh = mesh_cls()
    verts_t = torch.from_numpy(np.asarray(vertices, dtype=np.float32)).contiguous()
    faces_t = torch.from_numpy(np.asarray(faces, dtype=np.int32)).contiguous()
    mesh.init(verts_t, faces_t)

    options = {
        "lambda_edge_length": float(lambda_edge_length),
        "lambda_skinny": float(lambda_skinny),
        "thresh": float(thresh),
    }
    if hasattr(mesh, "simplify_step"):
        _simplify_with_step_loop(
            mesh,
            int(target_faces),
            lambda_edge_length=options["lambda_edge_length"],
            lambda_skinny=options["lambda_skinny"],
            thresh=options["thresh"],
        )
    else:
        try:
            mesh.simplify(int(target_faces), verbose=verbose, options=options)
        except TypeError:
            mesh.simplify(int(target_faces), verbose=verbose)

    out_vertices, out_faces = mesh.read()
    if hasattr(out_vertices, "detach"):
        out_vertices = out_vertices.detach().cpu().numpy()
    if hasattr(out_faces, "detach"):
        out_faces = out_faces.detach().cpu().numpy()
    return (
        np.asarray(out_vertices, dtype=np.float32),
        np.asarray(out_faces, dtype=np.int32),
    )


def orient_source_native(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    verbose: bool = True,
    expected_source_root: str | Path | None = None,
    reference_python: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unify face orientation with the reference mtlmesh/cumesh backend."""
    if reference_python is not None:
        return _run_source_native_orient_subprocess(
            vertices,
            faces,
            reference_python=reference_python,
            verbose=verbose,
            expected_source_root=expected_source_root,
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "source-native orientation requires torch for the reference "
            "mtlmesh/cumesh backend"
        ) from exc

    mesh_cls = _load_source_mesh_class(expected_source_root=expected_source_root)
    mesh = mesh_cls()
    verts_t = torch.from_numpy(np.asarray(vertices, dtype=np.float32)).contiguous()
    faces_t = torch.from_numpy(np.asarray(faces, dtype=np.int32)).contiguous()
    mesh.init(verts_t, faces_t)
    mesh.unify_face_orientations()

    out_vertices, out_faces = mesh.read()
    if hasattr(out_vertices, "detach"):
        out_vertices = out_vertices.detach().cpu().numpy()
    if hasattr(out_faces, "detach"):
        out_faces = out_faces.detach().cpu().numpy()
    return (
        np.asarray(out_vertices, dtype=np.float32),
        np.asarray(out_faces, dtype=np.int32),
    )


def cleanup_source_native(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    verbose: bool = True,
    expected_source_root: str | Path | None = None,
    reference_python: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run source cleanup primitives without final orientation fixing."""
    if reference_python is not None:
        return _run_source_native_cleanup_subprocess(
            vertices,
            faces,
            reference_python=reference_python,
            verbose=verbose,
            expected_source_root=expected_source_root,
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "source-native cleanup requires torch for the reference "
            "mtlmesh/cumesh backend"
        ) from exc

    mesh_cls = _load_source_mesh_class(expected_source_root=expected_source_root)
    mesh = mesh_cls()
    verts_t = torch.from_numpy(np.asarray(vertices, dtype=np.float32)).contiguous()
    faces_t = torch.from_numpy(np.asarray(faces, dtype=np.int32)).contiguous()
    mesh.init(verts_t, faces_t)
    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(1e-5)
    mesh.fill_holes(max_hole_perimeter=3e-2)

    out_vertices, out_faces = mesh.read()
    if hasattr(out_vertices, "detach"):
        out_vertices = out_vertices.detach().cpu().numpy()
    if hasattr(out_faces, "detach"):
        out_faces = out_faces.detach().cpu().numpy()
    return (
        np.asarray(out_vertices, dtype=np.float32),
        np.asarray(out_faces, dtype=np.int32),
    )


def _cleanup_source_mesh(mesh) -> None:
    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(1e-5)
    mesh.fill_holes(max_hole_perimeter=3e-2)


def _read_source_mesh(mesh) -> tuple[np.ndarray, np.ndarray]:
    out_vertices, out_faces = mesh.read()
    if hasattr(out_vertices, "detach"):
        out_vertices = out_vertices.detach().cpu().numpy()
    if hasattr(out_faces, "detach"):
        out_faces = out_faces.detach().cpu().numpy()
    return (
        np.asarray(out_vertices, dtype=np.float32),
        np.asarray(out_faces, dtype=np.int32),
    )


def postprocess_source_native(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    *,
    verbose: bool = True,
    lambda_edge_length: float = 1e-2,
    lambda_skinny: float = 1e-3,
    thresh: float = 1e-8,
    expected_source_root: str | Path | None = None,
    reference_python: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run source-native simplify/cleanup/orientation in one mesh object."""
    if reference_python is not None:
        return _run_source_native_postprocess_subprocess(
            vertices,
            faces,
            target_faces,
            reference_python=reference_python,
            verbose=verbose,
            expected_source_root=expected_source_root,
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "source-native postprocess requires torch for the reference "
            "mtlmesh/cumesh backend"
        ) from exc

    mesh_cls = _load_source_mesh_class(expected_source_root=expected_source_root)
    mesh = mesh_cls()
    verts_t = torch.from_numpy(np.asarray(vertices, dtype=np.float32)).contiguous()
    faces_t = torch.from_numpy(np.asarray(faces, dtype=np.int32)).contiguous()
    mesh.init(verts_t, faces_t)
    trace: list[dict] = []

    coarse_target = int(target_faces) * 3
    if _mesh_face_count(mesh) > coarse_target:
        input_faces = _mesh_face_count(mesh)
        simplify_trace = _simplify_source_mesh(
            mesh,
            coarse_target,
            verbose=verbose,
            lambda_edge_length=float(lambda_edge_length),
            lambda_skinny=float(lambda_skinny),
            thresh=float(thresh),
        )
        trace.append({
            "operation": "simplify_coarse_source_native_qem",
            "input_faces": int(input_faces),
            "requested_target_faces": int(coarse_target),
            "output_faces": int(_mesh_face_count(mesh)),
            **simplify_trace,
        })

    input_faces = _mesh_face_count(mesh)
    _cleanup_source_mesh(mesh)
    trace.append({
        "operation": "cleanup_initial_source_native",
        "input_faces": int(input_faces),
        "output_faces": int(_mesh_face_count(mesh)),
        "do_fix_normals": False,
    })

    if _mesh_face_count(mesh) > int(target_faces):
        input_faces = _mesh_face_count(mesh)
        simplify_trace = _simplify_source_mesh(
            mesh,
            int(target_faces),
            verbose=verbose,
            lambda_edge_length=float(lambda_edge_length),
            lambda_skinny=float(lambda_skinny),
            thresh=float(thresh),
        )
        trace.append({
            "operation": "simplify_final_source_native_qem",
            "input_faces": int(input_faces),
            "requested_target_faces": int(target_faces),
            "output_faces": int(_mesh_face_count(mesh)),
            **simplify_trace,
        })

    input_faces = _mesh_face_count(mesh)
    _cleanup_source_mesh(mesh)
    trace.append({
        "operation": "cleanup_final_source_native",
        "input_faces": int(input_faces),
        "output_faces": int(_mesh_face_count(mesh)),
        "do_fix_normals": False,
    })

    input_faces = _mesh_face_count(mesh)
    mesh.unify_face_orientations()
    trace.append({
        "operation": "orient_faces_source_native",
        "input_faces": int(input_faces),
        "output_faces": int(_mesh_face_count(mesh)),
    })

    out_vertices, out_faces = _read_source_mesh(mesh)
    return out_vertices, out_faces, trace

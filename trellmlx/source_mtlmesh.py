"""Optional source-native mtlmesh simplification route."""

from __future__ import annotations

from collections.abc import Callable
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from trellmlx.canonical_cumesh import (
    simplify_with_canonical_adjacency_step_loop,
)
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
(
    input_npz,
    output_npz,
    trace_json,
    target_faces_text,
    expected_root,
    expected_commit,
    rsqrt_lut_sha256,
) = sys.argv[2:9]
data = np.load(input_npz)
rsqrt_lut = (
    np.asarray(data["turing_rsqrt_delta_lut"], dtype=np.int8)
    if "turing_rsqrt_delta_lut" in data
    else None
)
vertices, faces, trace = postprocess_source_native(
    np.asarray(data["vertices"], dtype=np.float32),
    np.asarray(data["faces"], dtype=np.int32),
    int(target_faces_text),
    verbose=verbose,
    expected_source_root=expected_root or None,
    expected_source_commit=expected_commit or None,
    turing_rsqrt_delta_lut=rsqrt_lut,
    turing_rsqrt_lut_sha256=rsqrt_lut_sha256 or None,
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
    expected_source_commit: str | None,
    turing_rsqrt_delta_lut: np.ndarray | None,
    turing_rsqrt_lut_sha256: str | None,
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
        inputs = {
            "vertices": np.asarray(vertices, dtype=np.float32),
            "faces": np.asarray(faces, dtype=np.int32),
        }
        if turing_rsqrt_delta_lut is not None:
            inputs["turing_rsqrt_delta_lut"] = np.asarray(
                turing_rsqrt_delta_lut,
                dtype=np.int8,
            )
        np.savez_compressed(input_npz, **inputs)
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
            str(expected_source_commit or ""),
            str(turing_rsqrt_lut_sha256 or ""),
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


def _load_source_mesh_class(
    *,
    expected_source_root: str | Path | None = None,
    expected_source_commit: str | None = None,
    require_clean: bool = False,
):
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
                expected_commit=expected_source_commit,
                require_clean=require_clean,
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
            expected_commit=expected_source_commit,
            require_clean=require_clean,
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
    expected_source_commit: str | None = None,
    reference_python: str | Path | None = None,
    stage_callback: Callable[
        [str, int, int, dict, np.ndarray, np.ndarray],
        None,
    ]
    | None = None,
    simplification_runner: Callable[..., dict] | None = None,
    turing_rsqrt_delta_lut: np.ndarray | None = None,
    turing_rsqrt_lut_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run source-native simplify/cleanup/orientation in one mesh object."""
    if (turing_rsqrt_delta_lut is None) != (
        turing_rsqrt_lut_sha256 is None
    ):
        raise ValueError(
            "canonical source-native simplification requires both the "
            "Turing rsqrt LUT and its attested SHA256"
        )
    if simplification_runner is not None and turing_rsqrt_delta_lut is not None:
        raise ValueError(
            "explicit simplification_runner and Turing rsqrt LUT are "
            "mutually exclusive"
        )
    normalized_rsqrt_lut = None
    if turing_rsqrt_delta_lut is not None:
        normalized_rsqrt_lut = np.ascontiguousarray(
            np.asarray(turing_rsqrt_delta_lut)
        )
        if (
            normalized_rsqrt_lut.dtype != np.int8
            or normalized_rsqrt_lut.ndim != 1
        ):
            raise ValueError(
                "Turing rsqrt LUT must be a contiguous int8 vector"
            )
        if (
            not isinstance(turing_rsqrt_lut_sha256, str)
            or len(turing_rsqrt_lut_sha256) != 64
        ):
            raise ValueError(
                "Turing rsqrt LUT SHA256 must be 64 characters"
            )
    if reference_python is not None:
        if stage_callback is not None or simplification_runner is not None:
            raise ValueError(
                "stage_callback and simplification_runner require direct "
                "source-native execution; "
                "launch this function with the reference Python instead"
            )
        return _run_source_native_postprocess_subprocess(
            vertices,
            faces,
            target_faces,
            reference_python=reference_python,
            verbose=verbose,
            expected_source_root=expected_source_root,
            expected_source_commit=expected_source_commit,
            turing_rsqrt_delta_lut=normalized_rsqrt_lut,
            turing_rsqrt_lut_sha256=turing_rsqrt_lut_sha256,
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "source-native postprocess requires torch for the reference "
            "mtlmesh/cumesh backend"
        ) from exc

    mesh_cls = _load_source_mesh_class(
        expected_source_root=expected_source_root,
        expected_source_commit=expected_source_commit,
        require_clean=expected_source_commit is not None,
    )
    mesh = mesh_cls()
    verts_t = torch.from_numpy(np.asarray(vertices, dtype=np.float32)).contiguous()
    faces_t = torch.from_numpy(np.asarray(faces, dtype=np.int32)).contiguous()
    mesh.init(verts_t, faces_t)
    trace: list[dict] = []

    if normalized_rsqrt_lut is not None:
        rsqrt_lut = torch.from_numpy(normalized_rsqrt_lut)

        def canonical_turing_runner(
            source_mesh,
            requested_target,
            **options,
        ):
            step_trace = simplify_with_canonical_adjacency_step_loop(
                source_mesh,
                int(requested_target),
                lambda_edge_length=float(options["lambda_edge_length"]),
                lambda_skinny=float(options["lambda_skinny"]),
                thresh=float(options["thresh"]),
                rsqrt_lut=rsqrt_lut,
            )
            return {
                "simplifier_route": (
                    "canonical-adjacency-turing-rsqrt-step-loop"
                ),
                "adjacency_order": "ascending-face-id-per-vertex",
                "reuse_vertex_face_adjacency": True,
                "rsqrt_lut_sha256": turing_rsqrt_lut_sha256,
                "simplifier_step_trace": step_trace,
            }

        simplification_runner = canonical_turing_runner

    def emit_stage(
        operation: str,
        input_faces: int,
        details: dict | None = None,
    ) -> None:
        if stage_callback is None:
            return
        stage_vertices, stage_faces = _read_source_mesh(mesh)
        output_faces = _mesh_face_count(mesh)
        if len(stage_faces) != output_faces:
            raise RuntimeError(
                f"{operation} readback has {len(stage_faces)} faces, "
                f"but source mesh reports {output_faces}"
            )
        stage_callback(
            operation,
            int(input_faces),
            int(output_faces),
            details or {},
            stage_vertices,
            stage_faces,
        )

    input_faces = _mesh_face_count(mesh)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    emit_stage(
        "prefill_holes",
        input_faces,
        {"max_hole_perimeter": 3e-2},
    )
    trace.append({
        "operation": "prefill_holes_source_native",
        "input_faces": int(input_faces),
        "output_faces": int(_mesh_face_count(mesh)),
    })

    coarse_target = int(target_faces) * 3
    input_faces = _mesh_face_count(mesh)
    run_simplification = simplification_runner or _simplify_source_mesh
    simplify_trace = run_simplification(
        mesh,
        coarse_target,
        verbose=verbose,
        lambda_edge_length=float(lambda_edge_length),
        lambda_skinny=float(lambda_skinny),
        thresh=float(thresh),
    )
    simplify_details = {
        "requested_target_faces": int(coarse_target),
        **simplify_trace,
    }
    emit_stage("simplify_coarse", input_faces, simplify_details)
    trace.append({
        "operation": "simplify_coarse_source_native_qem",
        "input_faces": int(input_faces),
        "requested_target_faces": int(coarse_target),
        "output_faces": int(_mesh_face_count(mesh)),
        **simplify_trace,
    })

    cleanup_input_faces = _mesh_face_count(mesh)
    input_faces = _mesh_face_count(mesh)
    mesh.remove_duplicate_faces()
    emit_stage("remove_duplicate_faces_initial", input_faces)
    input_faces = _mesh_face_count(mesh)
    mesh.repair_non_manifold_edges()
    emit_stage("repair_non_manifold_edges_initial", input_faces)
    input_faces = _mesh_face_count(mesh)
    mesh.remove_small_connected_components(1e-5)
    emit_stage(
        "remove_small_connected_components_initial",
        input_faces,
        {"threshold": 1e-5},
    )
    input_faces = _mesh_face_count(mesh)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    emit_stage(
        "fill_holes_initial",
        input_faces,
        {"max_hole_perimeter": 3e-2},
    )
    trace.append({
        "operation": "cleanup_initial_source_native",
        "input_faces": int(cleanup_input_faces),
        "output_faces": int(_mesh_face_count(mesh)),
        "do_fix_normals": False,
    })

    input_faces = _mesh_face_count(mesh)
    simplify_trace = run_simplification(
        mesh,
        int(target_faces),
        verbose=verbose,
        lambda_edge_length=float(lambda_edge_length),
        lambda_skinny=float(lambda_skinny),
        thresh=float(thresh),
    )
    simplify_details = {
        "requested_target_faces": int(target_faces),
        **simplify_trace,
    }
    emit_stage("simplify_final", input_faces, simplify_details)
    trace.append({
        "operation": "simplify_final_source_native_qem",
        "input_faces": int(input_faces),
        "requested_target_faces": int(target_faces),
        "output_faces": int(_mesh_face_count(mesh)),
        **simplify_trace,
    })

    cleanup_input_faces = _mesh_face_count(mesh)
    input_faces = _mesh_face_count(mesh)
    mesh.remove_duplicate_faces()
    emit_stage("remove_duplicate_faces_final", input_faces)
    input_faces = _mesh_face_count(mesh)
    mesh.repair_non_manifold_edges()
    emit_stage("repair_non_manifold_edges_final", input_faces)
    input_faces = _mesh_face_count(mesh)
    mesh.remove_small_connected_components(1e-5)
    emit_stage(
        "remove_small_connected_components_final",
        input_faces,
        {"threshold": 1e-5},
    )
    input_faces = _mesh_face_count(mesh)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    emit_stage(
        "fill_holes_final",
        input_faces,
        {"max_hole_perimeter": 3e-2},
    )
    trace.append({
        "operation": "cleanup_final_source_native",
        "input_faces": int(cleanup_input_faces),
        "output_faces": int(_mesh_face_count(mesh)),
        "do_fix_normals": False,
    })

    input_faces = _mesh_face_count(mesh)
    mesh.unify_face_orientations()
    emit_stage("unify_face_orientations", input_faces)
    trace.append({
        "operation": "orient_faces_source_native",
        "input_faces": int(input_faces),
        "output_faces": int(_mesh_face_count(mesh)),
    })

    out_vertices, out_faces = _read_source_mesh(mesh)
    return out_vertices, out_faces, trace

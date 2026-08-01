"""Run the release-pinned TRELLIS.2 geometry postprocess on CUDA CuMesh.

This witness stops before UV unwrap and texture baking. It records every
geometry mutation in the standard non-remesh branch of
``o_voxel.postprocess.to_glb``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np


TRELLIS_REPOSITORY = "https://github.com/microsoft/TRELLIS.2.git"
TRELLIS_COMMIT = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
CUMESH_REPOSITORY = "https://github.com/JeffreyXiang/CuMesh.git"
CUMESH_COMMIT = "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87"
TRELLIS_POSTPROCESS_PATH = "o-voxel/o_voxel/postprocess.py"
TRELLIS_POSTPROCESS_SHA256 = (
    "ef51a1ba0f2748ffb4c265b47d382cee956f23c6a52d0f3587e6d8beccb7e54a"
)
EXPECTED_CUDA_DEVICE_NAME = "Tesla T4"
EXPECTED_CUDA_CAPABILITY = (7, 5)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")

STAGE_SPECS = (
    ("prefill_holes", "01_prefill_holes.ply"),
    ("simplify_coarse", "02_simplify_coarse.ply"),
    ("remove_duplicate_faces_initial", "03_remove_duplicate_faces_initial.ply"),
    ("repair_non_manifold_edges_initial", "04_repair_non_manifold_edges_initial.ply"),
    (
        "remove_small_connected_components_initial",
        "05_remove_small_connected_components_initial.ply",
    ),
    ("fill_holes_initial", "06_fill_holes_initial.ply"),
    ("simplify_final", "07_simplify_final.ply"),
    ("remove_duplicate_faces_final", "08_remove_duplicate_faces_final.ply"),
    ("repair_non_manifold_edges_final", "09_repair_non_manifold_edges_final.ply"),
    (
        "remove_small_connected_components_final",
        "10_remove_small_connected_components_final.ply",
    ),
    ("fill_holes_final", "11_fill_holes_final.ply"),
    ("unify_face_orientations", "12_unify_face_orientations.ply"),
)
STAGE_FILE_BY_OPERATION = dict(STAGE_SPECS)

FORBIDDEN_INFERENCES = [
    "not UV unwrap evidence",
    "not texture bake evidence",
    "not final material evidence",
    "not full o_voxel.postprocess.to_glb output",
    "not proof that the frozen raw extraction came from an unmodified official source tree",
]


class WitnessError(RuntimeError):
    pass


@dataclass
class CudaCuMeshRuntime:
    torch: Any
    cumesh: Any
    effective_route: dict[str, Any]

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Any:
        vertices_cuda = self.torch.from_numpy(
            np.ascontiguousarray(vertices, dtype=np.float32)
        ).cuda()
        faces_cuda = self.torch.from_numpy(
            np.ascontiguousarray(faces, dtype=np.int32)
        ).cuda()
        mesh = self.cumesh.CuMesh()
        mesh.init(vertices_cuda, faces_cuda)
        return mesh

    def read_mesh(self, mesh: Any) -> tuple[Any, Any]:
        return mesh.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-json", "--output-json", dest="report_json", required=True, type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--target-faces", required=True, type=int)
    parser.add_argument("--work-dir", required=True, type=Path)
    return parser


def execute_geometry_sequence(
    mesh: Any,
    target_faces: int,
    snapshot: Callable[[str, int, int, dict[str, Any]], None],
) -> None:
    def run(
        operation: str,
        mutation: Callable[[], None],
        details: dict[str, Any] | None = None,
    ) -> None:
        input_faces = int(mesh.num_faces)
        mutation()
        snapshot(operation, input_faces, int(mesh.num_faces), details or {})

    run(
        "prefill_holes",
        lambda: mesh.fill_holes(max_hole_perimeter=3e-2),
        {"max_hole_perimeter": 3e-2},
    )
    run(
        "simplify_coarse",
        lambda: mesh.simplify(target_faces * 3, verbose=False),
        {"requested_target_faces": int(target_faces * 3), "verbose": False},
    )
    run("remove_duplicate_faces_initial", mesh.remove_duplicate_faces)
    run("repair_non_manifold_edges_initial", mesh.repair_non_manifold_edges)
    run(
        "remove_small_connected_components_initial",
        lambda: mesh.remove_small_connected_components(1e-5),
        {"threshold": 1e-5},
    )
    run(
        "fill_holes_initial",
        lambda: mesh.fill_holes(max_hole_perimeter=3e-2),
        {"max_hole_perimeter": 3e-2},
    )
    run(
        "simplify_final",
        lambda: mesh.simplify(target_faces, verbose=False),
        {"requested_target_faces": int(target_faces), "verbose": False},
    )
    run("remove_duplicate_faces_final", mesh.remove_duplicate_faces)
    run("repair_non_manifold_edges_final", mesh.repair_non_manifold_edges)
    run(
        "remove_small_connected_components_final",
        lambda: mesh.remove_small_connected_components(1e-5),
        {"threshold": 1e-5},
    )
    run(
        "fill_holes_final",
        lambda: mesh.fill_holes(max_hole_perimeter=3e-2),
        {"max_hole_perimeter": 3e-2},
    )
    run("unify_face_orientations", mesh.unify_face_orientations)


def run_witness(
    *,
    input_ply: Path,
    output_dir: Path,
    report_json: Path,
    expected_input_sha256: str,
    target_faces: int,
    work_dir: Path,
    runtime_factory: Callable[..., Any] | None = None,
    sequence_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    output_dir = Path(output_dir)
    requested_report_json = Path(report_json)
    work_dir = Path(work_dir)
    stage_paths = {
        operation: output_dir / filename for operation, filename in STAGE_SPECS
    }
    effective_report_json, report_rerouted = _effective_report_path(
        requested_report_json,
        protected_paths=[input_ply, *stage_paths.values()],
    )
    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_cuda_cumesh_postprocess_witness.v1",
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "primary_output_status": "not_started",
        "requested_report_json": str(requested_report_json),
        "effective_report_json": str(effective_report_json),
        "report_rerouted": report_rerouted,
        "requested_route": {
            "input_ply": str(input_ply),
            "expected_input_sha256": expected_input_sha256,
            "output_dir": str(output_dir),
            "target_faces": int(target_faces),
            "trellis_repository": TRELLIS_REPOSITORY,
            "trellis_commit": TRELLIS_COMMIT,
            "trellis_postprocess_path": TRELLIS_POSTPROCESS_PATH,
            "trellis_postprocess_sha256": TRELLIS_POSTPROCESS_SHA256,
            "cumesh_repository": CUMESH_REPOSITORY,
            "cumesh_commit": CUMESH_COMMIT,
            "cuda_device_name": EXPECTED_CUDA_DEVICE_NAME,
            "cuda_capability": list(EXPECTED_CUDA_CAPABILITY),
        },
        "effective_route": None,
        "input_mesh": None,
        "stage_artifacts": [],
        "forbidden_inferences": FORBIDDEN_INFERENCES,
        "setup_commands": [],
    }
    phase = "request_validation"

    try:
        if _same_path(requested_report_json, input_ply):
            raise WitnessError("report path aliases protected input")
        for operation, stage_path in stage_paths.items():
            if _same_path(input_ply, stage_path):
                raise WitnessError(f"input aliases stage output for {operation}")
        if not HEX_SHA256.fullmatch(expected_input_sha256):
            raise WitnessError("--expected-input-sha256 must be 64 lowercase hex characters")
        if target_faces <= 0:
            raise WitnessError("--target-faces must be positive")

        phase = "input_validation"
        if not input_ply.is_file():
            raise WitnessError(f"input PLY does not exist: {input_ply}")
        actual_input_sha256 = sha256_file(input_ply)
        if actual_input_sha256 != expected_input_sha256:
            raise WitnessError(
                "input SHA256 mismatch: "
                f"expected {expected_input_sha256}, got {actual_input_sha256}"
            )
        input_vertices, input_faces = read_binary_ply(input_ply)
        report["input_mesh"] = {
            "sha256": actual_input_sha256,
            "size_bytes": input_ply.stat().st_size,
            **mesh_summary(input_vertices, input_faces),
        }
        report["last_trustworthy_phase"] = "input_validated"
        _write_report(effective_report_json, report)

        phase = "stale_output_cleanup"
        output_dir.mkdir(parents=True, exist_ok=True)
        for stage_path in stage_paths.values():
            if stage_path.exists():
                if not stage_path.is_file():
                    raise WitnessError(f"stale stage output is not a file: {stage_path}")
                stage_path.unlink()
        report["last_trustworthy_phase"] = "stale_outputs_removed"
        _write_report(effective_report_json, report)

        phase = "runtime_setup"
        runtime_builder = runtime_factory or prepare_release_runtime
        runtime = runtime_builder(work_dir=work_dir, report=report)
        report["effective_route"] = dict(runtime.effective_route)

        phase = "runtime_validation"
        _validate_effective_route(report["effective_route"])
        report["last_trustworthy_phase"] = "runtime_validated"
        report["status"] = "running"
        _write_report(effective_report_json, report)

        phase = "mesh_initialization"
        mesh = runtime.create_mesh(input_vertices, input_faces)
        if int(mesh.num_vertices) != len(input_vertices) or int(mesh.num_faces) != len(input_faces):
            raise WitnessError(
                "CuMesh initialization changed input dimensions before the first operation"
            )
        report["last_trustworthy_phase"] = "mesh_initialized"
        _write_report(effective_report_json, report)

        phase = "geometry_sequence"

        def snapshot(
            operation: str,
            operation_input_faces: int,
            operation_output_faces: int,
            details: dict[str, Any],
        ) -> None:
            if operation not in stage_paths:
                raise WitnessError(f"unexpected geometry operation: {operation}")
            if any(item["operation"] == operation for item in report["stage_artifacts"]):
                raise WitnessError(f"duplicate geometry operation: {operation}")
            raw_vertices, raw_faces = runtime.read_mesh(mesh)
            vertices = _tensor_to_numpy(raw_vertices, np.float32)
            faces = _tensor_to_numpy(raw_faces, np.int32)
            if vertices.shape != (int(mesh.num_vertices), 3):
                raise WitnessError(
                    f"{operation} vertex readback shape {vertices.shape} does not match CuMesh"
                )
            if faces.shape != (int(mesh.num_faces), 3):
                raise WitnessError(
                    f"{operation} face readback shape {faces.shape} does not match CuMesh"
                )
            if len(faces) != operation_output_faces:
                raise WitnessError(
                    f"{operation} callback output count {operation_output_faces} "
                    f"does not match readback {len(faces)}"
                )
            stage_path = stage_paths[operation]
            write_binary_ply(stage_path, vertices, faces)
            reopened_vertices, reopened_faces = read_binary_ply(stage_path)
            if not np.array_equal(reopened_vertices, vertices):
                raise WitnessError(f"{operation} reopened vertices differ from CUDA readback")
            if not np.array_equal(reopened_faces, faces):
                raise WitnessError(f"{operation} reopened faces differ from CUDA readback")
            report["stage_artifacts"].append(
                {
                    "index": len(report["stage_artifacts"]) + 1,
                    "operation": operation,
                    "input_faces": int(operation_input_faces),
                    "output_faces": int(operation_output_faces),
                    "output_vertices": int(len(vertices)),
                    "details": details,
                    "path": str(stage_path),
                    "sha256": sha256_file(stage_path),
                    "size_bytes": stage_path.stat().st_size,
                    "status": "validated",
                }
            )
            report["primary_output_status"] = "partial"
            report["last_trustworthy_phase"] = f"stage_validated:{operation}"
            _write_report(effective_report_json, report)

        runner = sequence_runner or execute_geometry_sequence
        runner(mesh, int(target_faces), snapshot)

        phase = "stage_validation"
        actual_operations = [item["operation"] for item in report["stage_artifacts"]]
        expected_operations = [operation for operation, _ in STAGE_SPECS]
        if actual_operations != expected_operations:
            raise WitnessError(
                "validated stage set does not match release sequence: "
                f"expected {expected_operations}, got {actual_operations}"
            )
        for item in report["stage_artifacts"]:
            path = Path(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise WitnessError(
                    f"validated stage artifact is missing or changed: {item['operation']}"
                )
            reopened_vertices, reopened_faces = read_binary_ply(path)
            if (
                len(reopened_vertices) != item["output_vertices"]
                or len(reopened_faces) != item["output_faces"]
            ):
                raise WitnessError(
                    f"validated stage artifact dimensions changed: {item['operation']}"
                )

        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "all_stages_validated",
                "primary_output_status": "validated",
                "elapsed_seconds": elapsed(started),
            }
        )
        _write_report(effective_report_json, report)
        return report
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "primary_output_status": (
                    "partial" if report["stage_artifacts"] else "not_started"
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": elapsed(started),
            }
        )
        _write_report(effective_report_json, report)
        raise


def prepare_release_runtime(
    *,
    work_dir: Path,
    report: dict[str, Any],
    cumesh_instrumentation: Callable[[Path, dict[str, Any]], dict[str, Any]]
    | None = None,
) -> CudaCuMeshRuntime:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    trellis_root = work_dir / "TRELLIS.2"
    cumesh_root = work_dir / "CuMesh"
    for path in (trellis_root, cumesh_root):
        if path.exists():
            shutil.rmtree(path)

    _run_setup_command(
        ["git", "clone", "--filter=blob:none", "--no-checkout", TRELLIS_REPOSITORY, str(trellis_root)],
        report,
    )
    _run_setup_command(
        ["git", "-C", str(trellis_root), "checkout", "--detach", TRELLIS_COMMIT],
        report,
    )
    _run_setup_command(
        ["git", "clone", "--filter=blob:none", "--no-checkout", CUMESH_REPOSITORY, str(cumesh_root)],
        report,
    )
    _run_setup_command(
        ["git", "-C", str(cumesh_root), "checkout", "--detach", CUMESH_COMMIT],
        report,
    )
    _run_setup_command(
        ["git", "-C", str(cumesh_root), "submodule", "update", "--init", "--recursive"],
        report,
    )

    trellis_head = _command_output(["git", "-C", str(trellis_root), "rev-parse", "HEAD"])
    cumesh_head = _command_output(["git", "-C", str(cumesh_root), "rev-parse", "HEAD"])
    if trellis_head != TRELLIS_COMMIT:
        raise WitnessError(f"TRELLIS checkout mismatch: {trellis_head}")
    if cumesh_head != CUMESH_COMMIT:
        raise WitnessError(f"CuMesh checkout mismatch: {cumesh_head}")
    trellis_status = _command_output(
        ["git", "-C", str(trellis_root), "status", "--porcelain", "--untracked-files=no"]
    )
    cumesh_status = _command_output(
        ["git", "-C", str(cumesh_root), "status", "--porcelain", "--untracked-files=no"]
    )
    if trellis_status:
        raise WitnessError(f"TRELLIS checkout is dirty before execution: {trellis_status}")
    if cumesh_status:
        raise WitnessError(f"CuMesh checkout is dirty before build: {cumesh_status}")
    postprocess_path = trellis_root / TRELLIS_POSTPROCESS_PATH
    actual_postprocess_sha256 = sha256_file(postprocess_path)
    if actual_postprocess_sha256 != TRELLIS_POSTPROCESS_SHA256:
        raise WitnessError(
            "TRELLIS postprocess source mismatch: "
            f"expected {TRELLIS_POSTPROCESS_SHA256}, got {actual_postprocess_sha256}"
        )

    instrumentation_identity = None
    if cumesh_instrumentation is not None:
        instrumentation_identity = cumesh_instrumentation(cumesh_root, report)
        if not isinstance(instrumentation_identity, dict):
            raise WitnessError("CuMesh instrumentation did not return route identity")

    _run_setup_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(cumesh_root),
            "--no-build-isolation",
        ],
        report,
    )
    importlib.invalidate_caches()
    import torch
    import cumesh

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        raise WitnessError("Torch CUDA is not available after CuMesh installation")
    device_name = str(torch.cuda.get_device_name(0))
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    effective_route = {
        "trellis_repository": TRELLIS_REPOSITORY,
        "trellis_commit": trellis_head,
        "trellis_source_clean": not bool(trellis_status),
        "trellis_postprocess_path": str(postprocess_path),
        "trellis_postprocess_sha256": actual_postprocess_sha256,
        "cumesh_repository": CUMESH_REPOSITORY,
        "cumesh_commit": cumesh_head,
        "cumesh_source_clean_before_build": not bool(cumesh_status),
        "cumesh_instrumentation": instrumentation_identity,
        "cumesh_module_path": str(Path(cumesh.__file__).resolve()),
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_runtime_version": str(torch.version.cuda),
        "cuda_device_name": device_name,
        "cuda_capability": list(capability),
        "device_type": "cuda",
        "geometry_route": "release-trellis2-cumesh-standard-non-remesh",
        "target_faces": int(report["requested_route"]["target_faces"]),
    }
    return CudaCuMeshRuntime(
        torch=torch,
        cumesh=cumesh,
        effective_route=effective_route,
    )


def _validate_effective_route(route: dict[str, Any]) -> None:
    required = {
        "trellis_commit": TRELLIS_COMMIT,
        "trellis_postprocess_sha256": TRELLIS_POSTPROCESS_SHA256,
        "cumesh_commit": CUMESH_COMMIT,
        "cuda_device_name": EXPECTED_CUDA_DEVICE_NAME,
        "cuda_capability": list(EXPECTED_CUDA_CAPABILITY),
        "device_type": "cuda",
    }
    for key, expected in required.items():
        actual = route.get(key)
        if actual != expected:
            if key in {"cuda_device_name", "cuda_capability"}:
                raise WitnessError(
                    "effective CUDA route is not the required Tesla T4 "
                    f"SM75 device: {key}={actual!r}"
                )
            raise WitnessError(
                f"effective route mismatch for {key}: expected {expected!r}, got {actual!r}"
            )
    if route.get("trellis_source_clean", True) is not True:
        raise WitnessError("effective TRELLIS source checkout is dirty")
    if route.get("cumesh_source_clean_before_build", True) is not True:
        raise WitnessError("effective CuMesh source checkout was dirty before build")


def read_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY ended before end_header")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError("only binary_little_endian PLY is supported")
        vertex_count = _header_count(header_lines, "vertex")
        face_count = _header_count(header_lines, "face")
        vertex_bytes = handle.read(vertex_count * 3 * 4)
        if len(vertex_bytes) != vertex_count * 3 * 4:
            raise ValueError("PLY ended before all vertices were read")
        vertices = np.frombuffer(vertex_bytes, dtype="<f4").reshape(vertex_count, 3).copy()
        face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
        face_bytes = handle.read(face_count * face_dtype.itemsize)
        if len(face_bytes) != face_count * face_dtype.itemsize:
            raise ValueError("PLY ended before all faces were read")
        face_records = np.frombuffer(face_bytes, dtype=face_dtype)
        if not np.all(face_records["count"] == 3):
            raise ValueError("only triangular PLY faces are supported")
        faces = np.asarray(face_records["indices"], dtype=np.int32).copy()
        if handle.read(1):
            raise ValueError("PLY contains trailing bytes")
    return vertices, faces


def write_binary_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype="<f4")
    faces = np.asarray(faces, dtype="<i4")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape [N, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape [F, 3], got {faces.shape}")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    records = np.empty(len(faces), dtype=face_dtype)
    records["count"] = 3
    records["indices"] = faces
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(np.ascontiguousarray(vertices).tobytes())
        handle.write(records.tobytes())


def mesh_summary(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "vertices_dtype": str(np.asarray(vertices).dtype),
        "faces_dtype": str(np.asarray(faces).dtype),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header_count(header_lines: list[str], element: str) -> int:
    prefix = f"element {element} "
    for line in header_lines:
        if line.startswith(prefix):
            return int(line.split()[-1])
    raise ValueError(f"missing PLY element count for {element}")


def _tensor_to_numpy(tensor: Any, dtype: np.dtype[Any]) -> np.ndarray:
    value = tensor.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def _effective_report_path(
    requested: Path,
    *,
    protected_paths: list[Path],
) -> tuple[Path, bool]:
    if not any(_same_path(requested, protected) for protected in protected_paths):
        return requested, False
    base = requested.with_name(requested.name + ".failure.json")
    candidate = base
    index = 1
    while candidate.exists() or any(
        _same_path(candidate, protected) for protected in protected_paths
    ):
        candidate = base.with_name(f"{base.stem}.{index}{base.suffix}")
        index += 1
    return candidate, True


def _same_path(left: Path, right: Path) -> bool:
    return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_setup_command(command: list[str], report: dict[str, Any]) -> None:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    command_report = {
        "command": command,
        "elapsed_seconds": elapsed(started),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    report["setup_commands"].append(command_report)
    if completed.returncode != 0:
        raise WitnessError(
            f"setup command failed with exit {completed.returncode}: {' '.join(command)}"
        )


def _command_output(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise WitnessError(
            f"identity command failed with exit {completed.returncode}: {' '.join(command)}"
        )
    return completed.stdout.strip()


def elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_witness(
            input_ply=args.input_ply,
            output_dir=args.output_dir,
            report_json=args.report_json,
            expected_input_sha256=args.expected_input_sha256,
            target_faces=args.target_faces,
            work_dir=args.work_dir,
        )
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

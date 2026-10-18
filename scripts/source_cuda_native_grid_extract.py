#!/usr/bin/env python3
"""Extract one saved TRELLIS.2 decoder field with pinned native CUDA o-voxel.

The parent keeps a failure report even when the native child segfaults. This
does not run inference, cleanup, CuMesh, texturing, or final GLB export.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


TRELLIS_REPOSITORY = "https://github.com/microsoft/TRELLIS.2.git"
TRELLIS_COMMIT = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
SCHEMA = "trellis2mlx.source_cuda_native_grid_extract.v1"
EXPECTED_DECODER_SHA256 = "ddd8cfd389c73e30e254df3066b09d9043bc06929e1df901f9ee61e69940cad5"
EXPECTED_KERNEL_SLUG = "t2mlx-kiln-1024-exact-slat-cuda-decode"


class NativeRouteError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_decoder_arrays(feats: np.ndarray, coords: np.ndarray, *, resolution: int) -> None:
    if feats.dtype != np.float32 or feats.ndim != 2 or feats.shape[1] != 7:
        raise ValueError("decoder feats must be float32[N,7]")
    if coords.dtype != np.int32 or coords.shape != (feats.shape[0], 4):
        raise ValueError("decoder coords must be int32[N,4]")
    if feats.shape[0] == 0 or not np.isfinite(feats).all():
        raise ValueError("decoder feats must be nonempty and finite")
    if not np.all(coords[:, 0] == 0) or np.any(coords[:, 1:] < 0) or np.any(coords[:, 1:] >= resolution):
        raise ValueError("decoder coords outside requested one-batch grid")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("decoder coords contain duplicate rows")


def require_native_convert(module_path: str, source_root: str) -> bool:
    path = Path(module_path).resolve()
    root = Path(source_root).resolve()
    if "override" in path.name or "stubs" in path.parts:
        raise NativeRouteError(f"override converter is not the native route: {path}")
    if not path.is_relative_to(root):
        raise NativeRouteError(f"converter path is outside pinned o-voxel source: {path}")
    return True


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"command {command!r} exited {result.returncode}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
    return result.stdout.strip()


def _native_child(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F

    child_report = {"schema": SCHEMA, "status": "failed", "phase": "native_import", "input_sha256": args.expected_sha256}
    write_report(args.child_report, child_report)
    if not torch.cuda.is_available():
        raise NativeRouteError("CUDA is unavailable")
    device = torch.cuda.get_device_name(0)
    if "T4" not in device:
        raise NativeRouteError(f"expected a Tesla T4, got {device}")
    import o_voxel
    from o_voxel.convert import flexible_dual_grid_to_mesh

    convert_path = inspect.getfile(flexible_dual_grid_to_mesh)
    require_native_convert(convert_path, str(args.source_root / "o-voxel"))
    spec = importlib.util.find_spec("o_voxel._C")
    if spec is None or not spec.origin or not Path(spec.origin).is_file():
        raise NativeRouteError("compiled native o_voxel._C extension is unavailable")
    child_report.update(phase="load_decoder", cuda_device=device, torch=torch.__version__, o_voxel_path=str(Path(o_voxel.__file__).resolve()), convert_path=str(Path(convert_path).resolve()), extension_path=str(Path(spec.origin).resolve()))
    write_report(args.child_report, child_report)
    with np.load(args.input_npz, allow_pickle=False) as saved:
        feats, coords = saved["feats"], saved["coords"]
    validate_decoder_arrays(feats, coords, resolution=args.resolution)
    features = torch.from_numpy(np.ascontiguousarray(feats)).to("cuda")
    grid = torch.from_numpy(np.ascontiguousarray(coords[:, 1:])).to("cuda")
    vertices = (2.0 * torch.sigmoid(features[:, :3])) - 0.5
    intersected = features[:, 3:6] > 0
    quad_lerp = F.softplus(features[:, 6:7])
    child_report.update(phase="native_call", decoder_rows=int(feats.shape[0]), active_edges=int(intersected.sum().item()))
    write_report(args.child_report, child_report)
    with torch.no_grad():
        mesh_vertices, mesh_faces = flexible_dual_grid_to_mesh(
            grid, vertices, intersected, quad_lerp,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=args.resolution, train=False,
        )
    torch.cuda.synchronize()
    child_report["phase"] = "save_mesh"
    write_report(args.child_report, child_report)
    v = mesh_vertices.cpu().numpy()
    f = mesh_faces.cpu().numpy()
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("native converter returned malformed mesh arrays")
    if not np.isfinite(v).all() or np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("native converter returned nonfinite or out-of-range geometry")
    np.savez(args.output_npz, vertices=v, faces=f)
    child_report.update(status="done", phase="mesh_saved", vertices=int(len(v)), faces=int(len(f)), output_sha256=sha256_file(args.output_npz))
    write_report(args.child_report, child_report)
    return 0


def _parent(args: argparse.Namespace) -> int:
    report = {"schema": SCHEMA, "status": "failed", "phase": "request", "input": str(args.input_npz), "expected_sha256": args.expected_sha256, "requested_route": "pinned-official-o-voxel-native-CUDA-on-saved-source-decoder-field", "resolution": args.resolution, "source_repository": TRELLIS_REPOSITORY, "source_commit": TRELLIS_COMMIT, "output": str(args.output_npz), "forbidden_inferences": ["not independent CUDA sampling", "not CuMesh cleanup", "not texture or final GLB"]}
    write_report(args.report, report)
    try:
        if args.output_npz.resolve() == args.input_npz.resolve() or any(
            path.resolve() in {args.input_npz.resolve(), args.output_npz.resolve(), args.report.resolve()}
            for path in (args.child_report, args.child_stdout, args.child_stderr)
        ):
            raise ValueError("output aliases a protected input or report")
        if sha256_file(args.input_npz) != args.expected_sha256:
            raise ValueError("saved source decoder SHA256 mismatch")
        report["phase"] = "source_setup"
        write_report(args.report, report)
        args.work_dir.mkdir(parents=True, exist_ok=True)
        if not args.source_root.exists():
            _run(["git", "clone", "--filter=blob:none", TRELLIS_REPOSITORY, str(args.source_root)])
        _run(["git", "checkout", "--detach", TRELLIS_COMMIT], cwd=args.source_root)
        _run(["git", "submodule", "update", "--init", "o-voxel"], cwd=args.source_root)
        report["source_head"] = _run(["git", "rev-parse", "HEAD"], cwd=args.source_root)
        report["o_voxel_head"] = _run(["git", "rev-parse", "HEAD"], cwd=args.source_root / "o-voxel")
        report["source_clean_before_build"] = not bool(_run(["git", "status", "--porcelain"], cwd=args.source_root))
        if report["source_head"] != TRELLIS_COMMIT or not report["source_clean_before_build"]:
            raise NativeRouteError("pinned source checkout is missing or dirty before build")
        _run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "ninja", "trimesh"])
        _run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-build-isolation", "--no-deps", "-e", str(args.source_root / "o-voxel")])
        report["phase"] = "native_child"
        write_report(args.report, report)
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--input-npz", str(args.input_npz), "--expected-sha256", args.expected_sha256, "--output-npz", str(args.output_npz), "--report", str(args.report), "--child-report", str(args.child_report), "--work-dir", str(args.work_dir), "--resolution", str(args.resolution)]
        with args.child_stdout.open("w") as stdout, args.child_stderr.open("w") as stderr:
            result = subprocess.run(command, cwd=args.work_dir, stdout=stdout, stderr=stderr)
        report["child_returncode"] = result.returncode
        report["child_report"] = str(args.child_report)
        if result.returncode != 0:
            raise NativeRouteError(f"native child exited {result.returncode}; see child report and stderr")
        child = json.loads(args.child_report.read_text())
        if child.get("status") != "done" or child.get("output_sha256") != sha256_file(args.output_npz):
            raise NativeRouteError("native child output did not validate")
        report.update(status="done", phase="complete", mesh={key: child[key] for key in ("vertices", "faces", "output_sha256")}, effective_route={key: child[key] for key in ("cuda_device", "torch", "o_voxel_path", "convert_path", "extension_path")})
        write_report(args.report, report)
        return 0
    except BaseException as exc:
        report.update(status="failed", failure_phase=report["phase"], error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_report(args.report, report)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--child-report", type=Path, default=Path("native_child.json"))
    parser.add_argument("--child-stdout", type=Path, default=Path("native_child.stdout.log"))
    parser.add_argument("--child-stderr", type=Path, default=Path("native_child.stderr.log"))
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args(argv)
    args.input_npz = args.input_npz.resolve()
    args.output_npz = args.output_npz.resolve()
    args.report = args.report.resolve()
    args.work_dir = args.work_dir.resolve()
    args.source_root = args.work_dir / "TRELLIS.2"
    args.child_report = args.child_report.resolve()
    args.child_stdout = args.child_stdout.resolve()
    args.child_stderr = args.child_stderr.resolve()
    if args.report in {args.input_npz, args.output_npz}:
        args.report = args.report.with_name(args.report.name + ".failure.json")
    return _native_child(args) if args.child else _parent(args)


def kaggle_main(*, mount: Path = Path("/kaggle/input"), out: Path = Path("/kaggle/working")) -> int:
    """Single-file Kaggle entrypoint; preserve a receipt even before native setup."""
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "native_grid_kernel_receipt.json"
    report_path = out / "native_grid_report.json"
    receipt = {
        "schema": "trellis2mlx.kaggle_native_grid_extract_receipt.v2",
        "status": "failed", "failure_phase": "mount_preflight",
        "requested_route": "pinned-official-o-voxel-native-CUDA-on-saved-source-decoder-field",
        "expected_input_sha256": EXPECTED_DECODER_SHA256,
        "expected_kernel_slug": EXPECTED_KERNEL_SLUG,
        "native_report": str(report_path),
    }
    write_report(receipt_path, receipt)
    try:
        candidates = sorted(mount.rglob("cuda_decoder_state.npz"))
        if len(candidates) != 1 or EXPECTED_KERNEL_SLUG not in candidates[0].parts:
            raise ValueError(f"expected one mounted prior-kernel decoder state under {EXPECTED_KERNEL_SLUG}, found {candidates}")
        decoder = candidates[0]
        receipt.update(mounted_input=str(decoder), failure_phase="native_assay")
        write_report(receipt_path, receipt)
        result = main([
            "--input-npz", str(decoder), "--expected-sha256", EXPECTED_DECODER_SHA256,
            "--output-npz", str(out / "native_grid_mesh.npz"),
            "--report", str(report_path),
            "--work-dir", str(out.parent / "temp" / "native-grid-work"),
            "--child-report", str(out / "native_grid_child.json"),
            "--child-stdout", str(out / "native_grid_child.stdout.log"),
            "--child-stderr", str(out / "native_grid_child.stderr.log"),
            "--resolution", "1024",
        ])
        receipt["assay_returncode"] = result
        native = json.loads(report_path.read_text()) if report_path.exists() else None
        receipt["native_status"] = native.get("status") if native else "missing-report"
        if result or native is None or native.get("status") != "done":
            raise RuntimeError(f"native assay failed: process={result}, phase={native.get('failure_phase') if native else 'missing-report'}")
        receipt.update(status="done", failure_phase=None)
        write_report(receipt_path, receipt)
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_report(receipt_path, receipt)
    # Kaggle publishes diagnostics for native crashes rather than dropping the run.
    return 0


if __name__ == "__main__":
    raise SystemExit(kaggle_main() if len(sys.argv) == 1 else main())

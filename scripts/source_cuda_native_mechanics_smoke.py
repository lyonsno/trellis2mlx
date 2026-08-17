#!/usr/bin/env python3
"""Qualify the exact native TRELLIS.2 T4 mechanics before model download."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import numpy as np


SCHEMA = "trellis2mlx.source_cuda_native_mechanics_smoke.v1"
EXPECTED_TORCH_VERSION = "2.10.0+cu128"
EXPECTED_DEVICE_NAME = "Tesla T4"
EXPECTED_CAPABILITY = (7, 5)
TRELLIS_REPOSITORY = "https://github.com/microsoft/TRELLIS.2.git"
TRELLIS_COMMIT = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
CUMESH_REPOSITORY = "https://github.com/JeffreyXiang/CuMesh.git"
CUMESH_COMMIT = "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87"
FLEX_GEMM_REPOSITORY = "https://github.com/JeffreyXiang/FlexGEMM.git"
FLEX_GEMM_COMMIT = "6dd94a859c26ee8246888502eada3dd8ad85532e"
NVDIFFRAST_REPOSITORY = "https://github.com/NVlabs/nvdiffrast.git"
NVDIFFRAST_COMMIT = "253ac4fcea7de5f396371124af597e6cc957bfae"
EXPECTED_CAPTURE_ORDER = (
    "postprocess_stage11_pre_orientation",
    "postprocess_stage12_post_orientation",
)
EXPECTED_ARTIFACT_FILENAMES = {
    stage: f"{index:02d}-{stage}.npz"
    for index, stage in enumerate(EXPECTED_CAPTURE_ORDER)
}
EXPECTED_ORIENTATION_STATE = {
    "call_count": 1,
    "native_method_return_preserved": True,
    "pre_readback_written": True,
    "post_readback_written": True,
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_run_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("run_id must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError("run_id must be a canonical lowercase UUID")
    return value


def _validate_native_source_identity(native: Any) -> None:
    expected = {
        "TRELLIS_COMMIT": TRELLIS_COMMIT,
        "CUMESH_COMMIT": CUMESH_COMMIT,
        "FLEX_GEMM_COMMIT": FLEX_GEMM_COMMIT,
        "NVDIFFRAST_COMMIT": NVDIFFRAST_COMMIT,
    }
    for field, value in expected.items():
        actual = getattr(native, field, None)
        if actual != value:
            raise ValueError(
                f"native witness source identity drift for {field}: expected {value}, got {actual}"
            )


def _prepare_runtime(native: Any, report: dict[str, Any]) -> dict[str, Path]:
    args = SimpleNamespace(work_dir=Path(report["requested_route"]["work_dir"]))
    return native.prepare_runtime(args, report)


def _canonical_source_roots(work_dir: Path) -> dict[str, Path]:
    work_dir = Path(work_dir).resolve()
    return {
        "source_root": work_dir / "TRELLIS.2",
        "cumesh_root": work_dir / "CuMesh",
        "flex_root": work_dir / "FlexGEMM",
        "nvdiffrast_root": work_dir / "nvdiffrast",
    }


def _module_path(module: Any, name: str) -> Path:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"effective import {name} has no module file")
    return Path(value).resolve()


def _direct_url_source_path(payload: Any) -> Path | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("url")
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return None
    return Path(unquote(parsed.path)).resolve()


def _distribution_provenance(
    module: Any,
    *,
    label: str,
    source_root: Path,
) -> dict[str, Any]:
    module_path = _module_path(module, label)
    top_level = str(getattr(module, "__name__", label)).split(".", 1)[0]
    preferred_names = importlib_metadata.packages_distributions().get(top_level, ())
    distributions: list[Any] = []
    seen: set[str] = set()
    for name in preferred_names:
        try:
            distribution = importlib_metadata.distribution(name)
        except importlib_metadata.PackageNotFoundError:
            continue
        key = str(distribution.metadata.get("Name", name)).lower()
        if key not in seen:
            seen.add(key)
            distributions.append(distribution)
    for distribution in importlib_metadata.distributions():
        key = str(distribution.metadata.get("Name", "")).lower()
        if key and key not in seen:
            seen.add(key)
            distributions.append(distribution)

    expected_source = Path(source_root).resolve()
    for distribution in distributions:
        try:
            direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if _direct_url_source_path(direct_url) != expected_source:
            continue
        distribution_root = Path(distribution.locate_file("")).resolve()
        matched_file = None
        for relative in distribution.files or ():
            if Path(distribution.locate_file(relative)).resolve() == module_path:
                matched_file = str(relative)
                break
        if matched_file is None:
            continue
        return {
            "mode": "pep610-direct-url",
            "source_root": str(expected_source),
            "import_name": str(getattr(module, "__name__", top_level)),
            "module_path": str(module_path),
            "distribution_name": str(distribution.metadata.get("Name", "")),
            "distribution_version": str(distribution.version),
            "distribution_root": str(distribution_root),
            "distribution_file": matched_file,
            "direct_url": direct_url,
        }
    raise RuntimeError(
        f"effective import {label} is not owned by a distribution installed from {expected_source}"
    )


def collect_build_import_provenance(
    imported: dict[str, Any],
    roots: dict[str, Path],
) -> dict[str, Any]:
    source_root = Path(roots["source_root"]).resolve()
    trellis_paths = {
        name: str(_module_path(imported[name], name))
        for name in ("attention_config", "sparse_config")
    }
    for name, value in trellis_paths.items():
        if source_root not in Path(value).parents:
            raise RuntimeError(
                f"effective TRELLIS import {name} is outside pinned source root {source_root}: {value}"
            )
    return {
        "trellis": {
            "mode": "source-tree",
            "source_root": str(source_root),
            "module_paths": trellis_paths,
        },
        "cumesh": _distribution_provenance(
            imported["cumesh"], label="cumesh", source_root=roots["cumesh_root"]
        ),
        "flex_gemm": _distribution_provenance(
            imported["flex_gemm"],
            label="flex_gemm",
            source_root=roots["flex_root"],
        ),
        "o_voxel": _distribution_provenance(
            imported["o_voxel"],
            label="o_voxel",
            source_root=source_root / "o-voxel",
        ),
        "nvdiffrast": _distribution_provenance(
            imported["nvdiffrast"],
            label="nvdiffrast",
            source_root=roots["nvdiffrast_root"],
        ),
    }


def import_effective_runtime(roots: dict[str, Path]) -> tuple[Any, dict[str, Any]]:
    os.environ["ATTN_BACKEND"] = "xformers"
    os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
    os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
    source_root = Path(roots["source_root"]).resolve()
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry or Path(entry).resolve() != source_root
    ]
    sys.path.insert(0, str(source_root))

    import torch
    import xformers
    import cumesh
    import flex_gemm
    import nvdiffrast.torch as nvdiffrast
    import o_voxel
    from trellis2.modules.attention import config as attention_config
    from trellis2.modules.sparse import config as sparse_config

    imported = {
        "xformers": xformers,
        "cumesh": cumesh,
        "flex_gemm": flex_gemm,
        "nvdiffrast": nvdiffrast,
        "o_voxel": o_voxel,
        "attention_config": attention_config,
        "sparse_config": sparse_config,
    }
    imported["build_import_provenance"] = collect_build_import_provenance(imported, roots)
    return torch, imported


def validate_runtime_identity(
    torch: Any,
    imported: dict[str, Any],
    roots: dict[str, Path],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"Torch runtime drift: expected {EXPECTED_TORCH_VERSION}, got {torch.__version__}"
        )
    device_name = str(torch.cuda.get_device_name(0))
    if device_name != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"expected {EXPECTED_DEVICE_NAME}, got {device_name}")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    if capability != EXPECTED_CAPABILITY:
        raise RuntimeError(f"expected T4 SM75 capability {EXPECTED_CAPABILITY}, got {capability}")

    attention = imported["attention_config"]
    sparse = imported["sparse_config"]
    if attention.BACKEND != "xformers":
        raise RuntimeError(f"dense attention backend fallback: {attention.BACKEND!r}")
    if sparse.ATTN != "xformers":
        raise RuntimeError(f"sparse attention backend fallback: {sparse.ATTN!r}")
    if sparse.CONV != "flex_gemm":
        raise RuntimeError(f"sparse convolution backend fallback: {sparse.CONV!r}")

    module_names = (
        "cumesh",
        "flex_gemm",
        "o_voxel",
        "nvdiffrast",
        "attention_config",
        "sparse_config",
    )
    provenance = imported.get("build_import_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("effective build/import provenance is missing")
    return {
        "device_type": "cuda",
        "cuda_device_name": device_name,
        "cuda_capability": list(capability),
        "cuda_runtime_version": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
        "xformers_version": getattr(imported["xformers"], "__version__", None),
        "attention_backend": attention.BACKEND,
        "sparse_attention_backend": sparse.ATTN,
        "sparse_conv_backend": sparse.CONV,
        "source_roots": {name: str(Path(path).resolve()) for name, path in roots.items()},
        "module_paths": {
            name: str(_module_path(imported[name], name)) for name in module_names
        },
        "build_import_provenance": provenance,
    }


class MechanicsRecorder:
    def __init__(self, output_dir: Path, native: Any, run_id: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.native = native
        self.run_id = run_id
        self.capture_order: list[str] = []
        self.artifacts: dict[str, dict[str, Any]] = {}

    def save_npz(self, stage: str, arrays: dict[str, Any]) -> Path:
        expected = EXPECTED_CAPTURE_ORDER[len(self.capture_order)]
        if stage != expected:
            raise RuntimeError(f"mechanics capture order mismatch: expected {expected}, got {stage}")
        path = self.output_dir / f"{len(self.capture_order):02d}-{stage}.npz"
        converted = {name: self.native._as_numpy(value) for name, value in arrays.items()}
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **converted)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"mechanics capture is missing or blank: {path}")
        self.capture_order.append(stage)
        self.artifacts[stage] = {
            "run_id": self.run_id,
            "path": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "arrays": {
                name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                for name, value in converted.items()
            },
        }
        return path


def run_orientation_probe(
    native: Any,
    cumesh: Any,
    torch: Any,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    recorder = MechanicsRecorder(output_dir, native, run_id)
    observer = native.install_orientation_observer(cumesh, recorder)
    try:
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        ).contiguous()
        faces = torch.tensor(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            dtype=torch.int32,
            device="cuda",
        ).contiguous()
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.unify_face_orientations()
        torch.cuda.synchronize()
    finally:
        native._restore_orientation_observer(observer)
    return {
        "state": dict(observer["state"]),
        "capture_order": list(recorder.capture_order),
        "artifacts": dict(recorder.artifacts),
    }


def run_smoke(
    *,
    run_id: str,
    output_json: Path,
    work_dir: Path,
    native_module: Any,
    no_download: bool = False,
    prepare: Callable[[Any, dict[str, Any]], dict[str, Path]] = _prepare_runtime,
    importer: Callable[[dict[str, Path]], tuple[Any, dict[str, Any]]] = import_effective_runtime,
    orientation_probe: Callable[[Any, Any, Any, Path, str], dict[str, Any]] = run_orientation_probe,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_json = Path(output_json).resolve()
    work_dir = Path(work_dir).resolve()
    output_dir = output_json.parent
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": run_id,
        "status": "running",
        "failure_phase": None,
        "last_trustworthy_phase": "arguments_parsed",
        "primary_output_status": "not_written",
        "requested_route": {
            "purpose": "native-runtime-mechanics-qualification-only",
            "work_dir": str(work_dir),
            "torch_version": EXPECTED_TORCH_VERSION,
            "cuda_device_name": EXPECTED_DEVICE_NAME,
            "cuda_capability": list(EXPECTED_CAPABILITY),
            "attention_backend": "xformers",
            "sparse_attention_backend": "xformers",
            "sparse_conv_backend": "flex_gemm",
            "trellis_repository": TRELLIS_REPOSITORY,
            "trellis_commit": getattr(native_module, "TRELLIS_COMMIT", None),
            "cumesh_repository": CUMESH_REPOSITORY,
            "cumesh_commit": getattr(native_module, "CUMESH_COMMIT", None),
            "flex_gemm_repository": FLEX_GEMM_REPOSITORY,
            "flex_gemm_commit": getattr(native_module, "FLEX_GEMM_COMMIT", None),
            "nvdiffrast_repository": NVDIFFRAST_REPOSITORY,
            "nvdiffrast_commit": getattr(native_module, "NVDIFFRAST_COMMIT", None),
        },
        "effective_route": {},
        "orientation_probe": {},
        "setup_commands": [],
        "claim_ceiling": (
            "qualifies the exact T4 native build/import/backend/orientation-observer mechanics; "
            "does not qualify model assets, inference, postprocess quality, or final GLB"
        ),
    }
    _atomic_write_json(output_json, report)
    phase = "request_validation"
    try:
        _canonical_run_id(run_id)
        _validate_native_source_identity(native_module)
        report["last_trustworthy_phase"] = "request_validated"
        _atomic_write_json(output_json, report)
        if no_download:
            report.update(
                {
                    "status": "preflight_stopped",
                    "failure_phase": None,
                    "last_trustworthy_phase": "request_validated",
                    "primary_output_status": "not_written_no_download",
                    "effective_route": {
                        "device_type": "not_loaded_no_download",
                        "native_dependencies": "not_built_no_download",
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            _atomic_write_json(output_json, report)
            return report

        phase = "prepare_runtime"
        roots = prepare(native_module, report)
        report["last_trustworthy_phase"] = "native_dependencies_built"
        _atomic_write_json(output_json, report)

        phase = "runtime_identity"
        torch, imported = importer(roots)
        report["effective_route"] = validate_runtime_identity(torch, imported, roots)
        report["last_trustworthy_phase"] = "runtime_identity_validated"
        _atomic_write_json(output_json, report)

        phase = "orientation_probe"
        probe = orientation_probe(
            native_module,
            imported["cumesh"],
            torch,
            output_dir,
            run_id,
        )
        if probe.get("state") != EXPECTED_ORIENTATION_STATE:
            raise RuntimeError(f"orientation observer incomplete: {probe.get('state')!r}")
        expected_order = list(EXPECTED_CAPTURE_ORDER)
        if probe.get("capture_order") != expected_order:
            raise RuntimeError(
                f"orientation capture order mismatch: {probe.get('capture_order')!r}"
            )
        report["orientation_probe"] = probe
        for stage in EXPECTED_CAPTURE_ORDER:
            record = probe["artifacts"].get(stage)
            if not isinstance(record, dict):
                raise RuntimeError(f"orientation artifact is missing for {stage}")
            expected_name = EXPECTED_ARTIFACT_FILENAMES[stage]
            if record.get("path") != expected_name:
                raise RuntimeError(
                    f"orientation artifact path for {stage} must be {expected_name!r}"
                )
            _validate_mesh_capture(output_dir / expected_name, record, run_id=run_id)
        report.update(
            {
                "status": "completed",
                "failure_phase": None,
                "last_trustworthy_phase": "mechanics_artifacts_reopened_and_validated",
                "primary_output_status": "validated",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _atomic_write_json(output_json, report)
        validate_completed_mechanics_report(
            output_json,
            expected_run_id=run_id,
        )
        return report
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "primary_output_status": "not_written",
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_write_json(output_json, report)
        raise


def _validate_requested_route(route: Any) -> None:
    if not isinstance(route, dict):
        raise ValueError("requested route is missing")
    expected = {
        "purpose": "native-runtime-mechanics-qualification-only",
        "torch_version": EXPECTED_TORCH_VERSION,
        "cuda_device_name": EXPECTED_DEVICE_NAME,
        "cuda_capability": list(EXPECTED_CAPABILITY),
        "attention_backend": "xformers",
        "sparse_attention_backend": "xformers",
        "sparse_conv_backend": "flex_gemm",
        "trellis_repository": TRELLIS_REPOSITORY,
        "trellis_commit": TRELLIS_COMMIT,
        "cumesh_repository": CUMESH_REPOSITORY,
        "cumesh_commit": CUMESH_COMMIT,
        "flex_gemm_repository": FLEX_GEMM_REPOSITORY,
        "flex_gemm_commit": FLEX_GEMM_COMMIT,
        "nvdiffrast_repository": NVDIFFRAST_REPOSITORY,
        "nvdiffrast_commit": NVDIFFRAST_COMMIT,
    }
    for field, value in expected.items():
        if route.get(field) != value:
            raise ValueError(
                f"requested route mismatch for {field}: expected {value!r}, got {route.get(field)!r}"
            )
    work_dir = route.get("work_dir")
    if not isinstance(work_dir, str) or not Path(work_dir).is_absolute():
        raise ValueError("requested route work_dir must be absolute")


def _validate_build_import_provenance(
    route: dict[str, Any],
    requested_route: dict[str, Any],
) -> None:
    work_dir = Path(requested_route["work_dir"]).resolve()
    roots = _canonical_source_roots(work_dir)
    effective_roots = route["source_roots"]
    expected_root_strings = {name: str(path) for name, path in roots.items()}
    if effective_roots != expected_root_strings:
        raise ValueError(
            "effective source roots do not match the exact requested work directory layout"
        )

    provenance = route.get("build_import_provenance")
    expected_keys = {"trellis", "cumesh", "flex_gemm", "o_voxel", "nvdiffrast"}
    if not isinstance(provenance, dict) or set(provenance) != expected_keys:
        raise ValueError("effective build/import provenance is incomplete")
    module_paths = route["module_paths"]

    trellis = provenance["trellis"]
    expected_trellis_paths = {
        "attention_config": module_paths["attention_config"],
        "sparse_config": module_paths["sparse_config"],
    }
    if not isinstance(trellis, dict) or trellis.get("mode") != "source-tree":
        raise ValueError("TRELLIS import provenance mode is invalid")
    if trellis.get("source_root") != str(roots["source_root"]):
        raise ValueError("TRELLIS import provenance source root is disconnected")
    if trellis.get("module_paths") != expected_trellis_paths:
        raise ValueError("TRELLIS import provenance module paths are disconnected")
    for name, value in expected_trellis_paths.items():
        path = Path(value)
        if roots["source_root"] not in path.parents:
            raise ValueError(
                f"TRELLIS module path {name} is outside the requested source root"
            )

    dependency_sources = {
        "cumesh": roots["cumesh_root"],
        "flex_gemm": roots["flex_root"],
        "o_voxel": roots["source_root"] / "o-voxel",
        "nvdiffrast": roots["nvdiffrast_root"],
    }
    for name, source_root in dependency_sources.items():
        record = provenance[name]
        if not isinstance(record, dict) or record.get("mode") != "pep610-direct-url":
            raise ValueError(f"{name} import provenance mode is invalid")
        if record.get("source_root") != str(source_root):
            raise ValueError(f"{name} import provenance source root is disconnected")
        if record.get("module_path") != module_paths[name]:
            raise ValueError(f"{name} import provenance module path is disconnected")
        for field in ("import_name", "distribution_name", "distribution_version"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} import provenance {field} is missing or blank")
        distribution_root_value = record.get("distribution_root")
        distribution_file_value = record.get("distribution_file")
        if not isinstance(distribution_root_value, str) or not Path(
            distribution_root_value
        ).is_absolute():
            raise ValueError(f"{name} import provenance distribution root is invalid")
        if not isinstance(distribution_file_value, str):
            raise ValueError(f"{name} import provenance distribution file is missing")
        distribution_file = Path(distribution_file_value)
        if distribution_file.is_absolute() or ".." in distribution_file.parts:
            raise ValueError(f"{name} import provenance distribution file is unsafe")
        owned_module = (Path(distribution_root_value) / distribution_file).resolve()
        if owned_module != Path(module_paths[name]).resolve():
            raise ValueError(f"{name} module path is not owned by the recorded distribution")
        if _direct_url_source_path(record.get("direct_url")) != source_root:
            raise ValueError(f"{name} import provenance direct URL is disconnected")


def _validate_effective_route(route: Any, requested_route: dict[str, Any]) -> None:
    if not isinstance(route, dict):
        raise ValueError("effective route is missing")
    expected = {
        "device_type": "cuda",
        "cuda_device_name": EXPECTED_DEVICE_NAME,
        "cuda_capability": list(EXPECTED_CAPABILITY),
        "torch_version": EXPECTED_TORCH_VERSION,
        "attention_backend": "xformers",
        "sparse_attention_backend": "xformers",
        "sparse_conv_backend": "flex_gemm",
    }
    for field, value in expected.items():
        if route.get(field) != value:
            raise ValueError(
                f"effective route mismatch for {field}: expected {value!r}, got {route.get(field)!r}"
            )
    for field in ("cuda_runtime_version", "xformers_version"):
        value = route.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"effective route {field} is missing or blank")
    roots = route.get("source_roots")
    if not isinstance(roots, dict) or set(roots) != {
        "source_root",
        "cumesh_root",
        "flex_root",
        "nvdiffrast_root",
    }:
        raise ValueError("effective route source roots are incomplete")
    module_paths = route.get("module_paths")
    if not isinstance(module_paths, dict) or set(module_paths) != {
        "cumesh",
        "flex_gemm",
        "o_voxel",
        "nvdiffrast",
        "attention_config",
        "sparse_config",
    }:
        raise ValueError("effective route module paths are incomplete")
    for label, paths in (("source root", roots), ("module path", module_paths)):
        for name, value in paths.items():
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"effective route {label} {name} must be absolute")
    _validate_build_import_provenance(route, requested_route)


def _validate_source_identities(report: dict[str, Any]) -> None:
    roots = report["effective_route"]["source_roots"]
    expected = {
        "trellis": {
            "path": roots["source_root"],
            "repository": TRELLIS_REPOSITORY,
            "commit": TRELLIS_COMMIT,
            "clean": True,
        },
        "cumesh": {
            "path": roots["cumesh_root"],
            "repository": CUMESH_REPOSITORY,
            "commit": CUMESH_COMMIT,
            "clean": True,
        },
        "flex_gemm": {
            "path": roots["flex_root"],
            "repository": FLEX_GEMM_REPOSITORY,
            "commit": FLEX_GEMM_COMMIT,
            "clean": True,
        },
        "nvdiffrast": {
            "path": roots["nvdiffrast_root"],
            "repository": NVDIFFRAST_REPOSITORY,
            "commit": NVDIFFRAST_COMMIT,
            "clean": True,
        },
    }
    for phase in ("before_build", "after_build"):
        field = f"source_identities_{phase}"
        if report.get(field) != expected:
            raise ValueError(
                f"source identity {phase.replace('_', ' ')} does not match the effective route"
            )

    cleanup = report.get("source_build_products_removed")
    if not isinstance(cleanup, dict) or set(cleanup) != {"nvdiffrast"}:
        raise ValueError("build-product cleanup record is missing or malformed")
    removed = cleanup["nvdiffrast"]
    allowed = {"build", "nvdiffrast.egg-info"}
    if (
        not isinstance(removed, list)
        or any(not isinstance(value, str) for value in removed)
        or removed != sorted(set(removed))
        or not set(removed).issubset(allowed)
    ):
        raise ValueError("build-product cleanup record is outside the admitted nvdiffrast set")


def _validate_mesh_capture(path: Path, record: Any, *, run_id: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"mechanics artifact record is missing for {path.name}")
    if record.get("run_id") != run_id:
        raise ValueError(f"mechanics artifact run identity mismatch for {path.name}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"mechanics artifact is missing or blank: {path}")
    if record.get("size_bytes") != path.stat().st_size:
        raise ValueError(f"mechanics artifact size mismatch: {path}")
    if record.get("sha256") != _sha256_file(path):
        raise ValueError(f"mechanics artifact digest mismatch: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["vertices", "faces"]:
                raise ValueError(
                    f"unexpected mechanics capture arrays for {path.name}: {archive.files}"
                )
            vertices = archive["vertices"]
            faces = archive["faces"]
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"invalid mechanics NPZ {path}: {exc}") from exc
    if vertices.dtype != np.dtype(np.float32):
        raise ValueError(f"vertices dtype must be float32, got {vertices.dtype}")
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise ValueError(f"vertices shape must be nonempty [N, 3], got {vertices.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite")
    if faces.dtype != np.dtype(np.int32):
        raise ValueError(f"faces dtype must be int32, got {faces.dtype}")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise ValueError(f"faces shape must be nonempty [M, 3], got {faces.shape}")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("face index is outside the captured vertex array")
    expected_arrays = {
        "vertices": {"dtype": "float32", "shape": list(vertices.shape)},
        "faces": {"dtype": "int32", "shape": list(faces.shape)},
    }
    if record.get("arrays") != expected_arrays:
        raise ValueError(f"mechanics artifact array metadata mismatch: {path}")


def validate_completed_mechanics_report(
    report_path: Path,
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    if not report_path.is_file() or report_path.stat().st_size <= 0:
        raise ValueError(f"mechanics report is missing or blank: {report_path}")
    try:
        report = json.loads(report_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid mechanics report JSON: {exc}") from exc
    if report.get("schema") != SCHEMA:
        raise ValueError(f"unexpected mechanics report schema: {report.get('schema')!r}")
    if report.get("status") != "completed":
        raise ValueError(f"mechanics report status is not completed: {report.get('status')!r}")
    if report.get("failure_phase") is not None:
        raise ValueError("completed mechanics report has a failure phase")
    if report.get("primary_output_status") != "validated":
        raise ValueError("mechanics primary output is not validated")
    if report.get("last_trustworthy_phase") != "mechanics_artifacts_reopened_and_validated":
        raise ValueError("mechanics report did not reach semantic artifact validation")
    run_id = _canonical_run_id(report.get("run_id"))
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(
            f"mechanics run identity mismatch: expected {expected_run_id}, got {run_id}"
        )
    requested_route = report.get("requested_route")
    _validate_requested_route(requested_route)
    _validate_effective_route(report.get("effective_route"), requested_route)
    _validate_source_identities(report)
    probe = report.get("orientation_probe")
    if not isinstance(probe, dict):
        raise ValueError("orientation probe is missing")
    if probe.get("state") != EXPECTED_ORIENTATION_STATE:
        raise ValueError(f"orientation observer state is invalid: {probe.get('state')!r}")
    if probe.get("capture_order") != list(EXPECTED_CAPTURE_ORDER):
        raise ValueError(f"orientation capture order is invalid: {probe.get('capture_order')!r}")
    artifacts = probe.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(EXPECTED_CAPTURE_ORDER):
        raise ValueError("orientation artifacts are incomplete")
    resolved_paths: set[Path] = set()
    for stage in EXPECTED_CAPTURE_ORDER:
        record = artifacts[stage]
        expected_name = EXPECTED_ARTIFACT_FILENAMES[stage]
        if record.get("path") != expected_name:
            raise ValueError(
                f"mechanics artifact path for {stage} must be {expected_name!r}"
            )
        path = (report_path.parent / expected_name).resolve()
        if report_path.parent not in path.parents:
            raise ValueError(f"mechanics artifact escapes report custody: {path}")
        if path in resolved_paths:
            raise ValueError(f"mechanics artifact path is duplicated: {path}")
        resolved_paths.add(path)
        _validate_mesh_capture(path, record, run_id=run_id)
    return report


def prepare_native_mechanics_packet(packet: Any) -> Any:
    """Prepare one mechanics-only attempt with explicit packet custody."""
    from trellmlx.kaggle_cuda_witness import WitnessPacketError, prepare_packet

    if packet.run_id is None:
        raise WitnessPacketError("native mechanics packet requires run identity")
    if packet.expected_image_sha256 is not None:
        raise WitnessPacketError("native mechanics packet must not claim image identity")
    if packet.enable_internet is not True:
        raise WitnessPacketError("native mechanics packet requires internet for pinned source builds")
    if packet.accelerator != "NvidiaTeslaT4":
        raise WitnessPacketError("native mechanics packet requires exact Nvidia Tesla T4 route")
    expected_outputs = tuple(EXPECTED_ARTIFACT_FILENAMES.values())
    if packet.output_json != "mechanics-report.json" or packet.output_npz is not None:
        raise WitnessPacketError("native mechanics packet output roles are not canonical")
    if packet.expected_outputs != expected_outputs:
        raise WitnessPacketError("native mechanics packet captures are not canonical")
    return prepare_packet(packet)


def validate_downloaded_native_mechanics_outputs(
    packet: Any,
    output_dir: Path,
) -> dict[str, Any]:
    from trellmlx.kaggle_cuda_witness import (
        WitnessPacketError,
        validate_downloaded_outputs,
    )

    if packet.run_id is None:
        raise WitnessPacketError("native mechanics packet is missing its run identity")
    if packet.enable_internet is not True or packet.accelerator != "NvidiaTeslaT4":
        raise WitnessPacketError("native mechanics packet effective route is not executable")
    expected_outputs = tuple(EXPECTED_ARTIFACT_FILENAMES.values())
    if packet.output_json != "mechanics-report.json" or packet.output_npz is not None:
        raise WitnessPacketError("native mechanics packet output roles are not canonical")
    if packet.expected_outputs != expected_outputs:
        raise WitnessPacketError("native mechanics packet captures are not canonical")
    records = validate_downloaded_outputs(packet, output_dir)
    try:
        report = validate_completed_mechanics_report(
            Path(output_dir) / packet.output_json,
            expected_run_id=packet.run_id,
        )
    except ValueError as exc:
        raise WitnessPacketError(str(exc)) from exc
    return {"downloaded_outputs": records, "report": report}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--no-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import source_cuda_native_image_to_glb_witness as native

    try:
        run_smoke(
            run_id=args.run_id,
            output_json=args.output_json,
            work_dir=args.work_dir,
            native_module=native,
            no_download=args.no_download,
        )
        return 0
    except BaseException as exc:
        print(f"native mechanics smoke failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Capture one native official TRELLIS.2 image-to-GLB CUDA trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCHEMA = "trellis2mlx.source_cuda_native_image_to_glb.v1"
TRELLIS_REPOSITORY = "https://github.com/microsoft/TRELLIS.2.git"
TRELLIS_COMMIT = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
CUMESH_REPOSITORY = "https://github.com/JeffreyXiang/CuMesh.git"
CUMESH_COMMIT = "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87"
FLEX_GEMM_REPOSITORY = "https://github.com/JeffreyXiang/FlexGEMM.git"
FLEX_GEMM_COMMIT = "6dd94a859c26ee8246888502eada3dd8ad85532e"
NVDIFFRAST_REPOSITORY = "https://github.com/NVlabs/nvdiffrast.git"
NVDIFFRAST_COMMIT = "253ac4fcea7de5f396371124af597e6cc957bfae"
MODEL_REPOSITORY = "microsoft/TRELLIS.2-4B"
MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
MODEL_PIPELINE_SHA256 = "222c359ab1ed9bc6735a640a34f95d47f8681b9bc4aaa101bfb80274676253c6"
SPARSE_DECODER_REPOSITORY = "microsoft/TRELLIS-image-large"
SPARSE_DECODER_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
DINOV3_REPOSITORY = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
DINOV3_FILES = {
    "model.safetensors": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    "config.json": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
    "preprocessor_config.json": "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
}
REMBG_REPOSITORY = "briaai/RMBG-2.0"
REMBG_REVISION = "5df4c9c76d8170882c34f6986e848ee07fd0ba43"
EXPECTED_TORCH_VERSION = "2.10.0+cu128"

EXPECTED_CAPTURE_ORDER = (
    "preprocessed_image",
    "conditioning_512",
    "sparse_flow",
    "sparse_support",
    "shape_flow",
    "shape_slat",
    "texture_flow",
    "decoder_raw_mesh",
    "texture_voxels",
    "pipeline_filled_mesh",
    "postprocess_stage11_pre_orientation",
    "postprocess_stage12_post_orientation",
    "consumer_glb",
)

EXPECTED_ARTIFACT_FILENAMES = {
    stage: f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
    for index, stage in enumerate(EXPECTED_CAPTURE_ORDER)
}

NPZ_STAGE_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "conditioning_512": {
        "cond": {"dtype": "float32", "ndim": 3},
        "neg_cond": {"dtype": "float32", "ndim": 3},
    },
    "sparse_flow": {
        "noise": {"dtype": "float32", "ndim": 5},
        "sample_next": {"dtype": "float32", "ndim": 6, "leading": 8},
        "pred_x0": {"dtype": "float32", "ndim": 6, "leading": 8},
    },
    "sparse_support": {
        "coords": {"dtype": "int32", "ndim": 2, "columns": 4, "unique_rows": True},
    },
    "shape_flow": {
        "noise_feats": {"dtype": "float32", "ndim": 2},
        "noise_coords": {"dtype": "int32", "ndim": 2, "columns": 4, "unique_rows": True},
        "coords": {"dtype": "int32", "ndim": 2, "columns": 4, "unique_rows": True},
        "sample_next": {"dtype": "float32", "ndim": 3, "leading": 8},
        "pred_x0": {"dtype": "float32", "ndim": 3, "leading": 8},
    },
    "shape_slat": {
        "shape_slat_feats": {"dtype": "float32", "ndim": 2},
        "shape_slat_coords": {"dtype": "int32", "ndim": 2, "columns": 4, "unique_rows": True},
    },
    "texture_flow": {
        "noise_feats": {"dtype": "float32", "ndim": 2},
        "noise_coords": {"dtype": "int32", "ndim": 2, "columns": 4, "unique_rows": True},
        "coords": {"dtype": "int32", "ndim": 2, "columns": 4, "unique_rows": True},
        "sample_next": {"dtype": "float32", "ndim": 3, "leading": 8},
        "pred_x0": {"dtype": "float32", "ndim": 3, "leading": 8},
    },
    "decoder_raw_mesh": {
        "vertices": {"dtype": "float32", "ndim": 2, "columns": 3},
        "faces": {"dtype": "int32", "ndim": 2, "columns": 3},
    },
    "texture_voxels": {
        "texture_voxels_feats": {"dtype": "float32", "ndim": 2},
        "texture_voxels_coords": {"dtype": "int32", "ndim": 2, "columns": 4, "unique_rows": True},
    },
    "pipeline_filled_mesh": {
        "vertices": {"dtype": "float32", "ndim": 2, "columns": 3},
        "faces": {"dtype": "int32", "ndim": 2, "columns": 3},
        "texture_coords": {"dtype": "int32", "ndim": 2, "columns": 3, "unique_rows": True},
        "texture_attrs": {"dtype": "float32", "ndim": 2},
    },
    "postprocess_stage11_pre_orientation": {
        "vertices": {"dtype": "float32", "ndim": 2, "columns": 3},
        "faces": {"dtype": "int32", "ndim": 2, "columns": 3},
    },
    "postprocess_stage12_post_orientation": {
        "vertices": {"dtype": "float32", "ndim": 2, "columns": 3},
        "faces": {"dtype": "int32", "ndim": 2, "columns": 3},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _run(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    receipt = {
        "command": command,
        "cwd": str(cwd) if cwd is not None else None,
        "elapsed_seconds": time.perf_counter() - started,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {command!r}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return receipt


def _requested_route(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "route": "official-source-native-image-to-glb-cuda",
        "trellis_repository": TRELLIS_REPOSITORY,
        "trellis_commit": TRELLIS_COMMIT,
        "cumesh_repository": CUMESH_REPOSITORY,
        "cumesh_commit": CUMESH_COMMIT,
        "flex_gemm_repository": FLEX_GEMM_REPOSITORY,
        "flex_gemm_commit": FLEX_GEMM_COMMIT,
        "nvdiffrast_repository": NVDIFFRAST_REPOSITORY,
        "nvdiffrast_commit": NVDIFFRAST_COMMIT,
        "model_repository": args.model_repository,
        "model_revision": MODEL_REVISION,
        "sparse_decoder_repository": SPARSE_DECODER_REPOSITORY,
        "sparse_decoder_revision": SPARSE_DECODER_REVISION,
        "dinov3_repository": DINOV3_REPOSITORY,
        "dinov3_revision": DINOV3_REVISION,
        "dinov3_model_path": (
            str(Path(args.dinov3_model_path).resolve())
            if args.dinov3_model_path is not None
            else None
        ),
        "rembg_repository": REMBG_REPOSITORY,
        "rembg_revision": REMBG_REVISION,
        "torch_version": EXPECTED_TORCH_VERSION,
        "image": str(Path(args.image).resolve()),
        "image_sha256": args.expected_image_sha256,
        "run_id": args.run_id,
        "pipeline_type": args.pipeline_type,
        "seed": args.seed,
        "sampler_steps": {
            "sparse": args.steps,
            "shape": args.steps,
            "texture": args.steps,
        },
        "attention_backend": args.attention_backend,
        "sparse_attention_backend": args.attention_backend,
        "sparse_conv_backend": args.sparse_conv_backend,
        "native_conditioning": True,
        "native_rng": True,
        "pipeline_run_called_once": True,
        "postprocess": {
            "decimation_target": args.target_faces,
            "texture_size": args.texture_size,
            "remesh": False,
        },
        "observation_only_instrumentation": True,
    }


def _validate_request(args: argparse.Namespace) -> None:
    image = Path(args.image).resolve()
    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    report_path = (
        Path(args.output_json).resolve()
        if args.output_json is not None
        else output_dir / "report.json"
    )
    if not image.is_file() or image.stat().st_size <= 0:
        raise ValueError(f"image is missing or blank: {image}")
    if len(args.expected_image_sha256) != 64:
        raise ValueError("--expected-image-sha256 must contain 64 hexadecimal characters")
    try:
        int(args.expected_image_sha256, 16)
    except ValueError as exc:
        raise ValueError("--expected-image-sha256 must be hexadecimal") from exc
    actual = sha256_file(image)
    if actual != args.expected_image_sha256:
        raise ValueError(
            f"image SHA256 mismatch: expected {args.expected_image_sha256}, got {actual}"
        )
    if args.pipeline_type != "512":
        raise ValueError("this witness admits only pipeline_type='512'")
    if args.seed != 42:
        raise ValueError("this witness admits only seed 42")
    if args.steps != 8:
        raise ValueError("this witness admits exactly 8 sparse, shape, and texture steps")
    if args.target_faces != 350000:
        raise ValueError("this witness admits target_faces=350000")
    if args.texture_size != 1024:
        raise ValueError("this witness admits texture_size=1024")
    if args.attention_backend != "xformers":
        raise ValueError("the T4 native source route requires attention_backend='xformers'")
    if args.sparse_conv_backend != "flex_gemm":
        raise ValueError("the native source route requires sparse_conv_backend='flex_gemm'")
    if args.model_repository != MODEL_REPOSITORY:
        raise ValueError(f"this witness admits only model_repository={MODEL_REPOSITORY!r}")
    try:
        parsed_run_id = uuid.UUID(args.run_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("--run-id must be a canonical UUID") from exc
    if str(parsed_run_id) != args.run_id:
        raise ValueError("--run-id must be a canonical lowercase UUID")
    if not args.no_download and args.dinov3_model_path is None:
        raise ValueError("live native conditioning requires --dinov3-model-path")
    if args.dinov3_model_path is not None:
        dinov3_root = Path(args.dinov3_model_path).resolve()
        if not dinov3_root.is_dir():
            raise ValueError(f"DINOv3 model directory is missing: {dinov3_root}")
        for name, expected in DINOV3_FILES.items():
            path = dinov3_root / name
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"DINOv3 model file is missing or blank: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"DINOv3 {name} SHA256 mismatch: expected {expected}, got {actual}"
                )
    if output_dir == output_dir.parent or work_dir == work_dir.parent:
        raise ValueError("output and work directories must not be filesystem roots")
    if report_path != output_dir / "report.json":
        raise ValueError(
            "--output-json must resolve exactly to <output-dir>/report.json: "
            f"{report_path} != {output_dir / 'report.json'}"
        )
    if image == report_path or (
        image.parent == output_dir and image.name in set(EXPECTED_ARTIFACT_FILENAMES.values())
    ):
        raise ValueError("image aliases a report or canonical output artifact")
    if output_dir == work_dir or output_dir in work_dir.parents or work_dir in output_dir.parents:
        raise ValueError("output and work directories must not overlap")


def _copy_admitted_file(source: Path, destination: Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"admitted-input destination already exists: {destination}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".admitting", dir=destination.parent
    )
    try:
        with source.open("rb") as reader, os.fdopen(fd, "wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        temporary = Path(temporary_name)
        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"authority changed while admitting {source.name}: "
                f"expected {expected_sha256}, copied {actual_sha256}"
            )
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "sha256": expected_sha256,
        "size_bytes": destination.stat().st_size,
    }


def admit_run_inputs(
    args: argparse.Namespace,
    report: dict[str, Any],
    *,
    expected_dinov3_files: dict[str, str] = DINOV3_FILES,
) -> argparse.Namespace:
    requested_image = Path(args.image).resolve()
    requested_dino = Path(args.dinov3_model_path).resolve()
    custody_root = Path(args.work_dir).resolve() / "admitted-inputs" / args.run_id
    if custody_root.exists():
        raise RuntimeError(f"run input custody already exists: {custody_root}")
    custody_root.mkdir(parents=True)

    requested = {
        "image": {
            "path": str(requested_image),
            "sha256": args.expected_image_sha256,
            "size_bytes": requested_image.stat().st_size,
        },
        "dinov3": {
            "path": str(requested_dino),
            "files": {
                name: {
                    "path": str(requested_dino / name),
                    "sha256": expected,
                    "size_bytes": (requested_dino / name).stat().st_size,
                }
                for name, expected in expected_dinov3_files.items()
            },
        },
    }
    admitted_image = custody_root / "image" / requested_image.name
    image_record = _copy_admitted_file(
        requested_image, admitted_image, args.expected_image_sha256
    )
    admitted_dino = custody_root / "dinov3"
    dino_records = {
        name: _copy_admitted_file(requested_dino / name, admitted_dino / name, expected)
        for name, expected in expected_dinov3_files.items()
    }
    report["requested_inputs"] = requested
    report["effective_inputs"] = {
        "run_id": args.run_id,
        "image": image_record,
        "dinov3": {
            "path": str(admitted_dino),
            "files": dino_records,
        },
    }
    admitted_args = argparse.Namespace(**vars(args))
    admitted_args.image = admitted_image
    admitted_args.dinov3_model_path = admitted_dino
    return admitted_args


def verify_admitted_inputs_before_use(
    args: argparse.Namespace,
    report: dict[str, Any],
    *,
    expected_dinov3_files: dict[str, str] = DINOV3_FILES,
) -> None:
    requested = report.get("requested_inputs", {})
    effective = report.get("effective_inputs", {})
    if effective.get("run_id") != args.run_id:
        raise RuntimeError("admitted input run identity mismatch")

    checks = [
        (
            "image",
            Path(requested.get("image", {}).get("path", "")),
            Path(args.image),
            args.expected_image_sha256,
        )
    ]
    for name, expected in expected_dinov3_files.items():
        checks.append(
            (
                name,
                Path(requested.get("dinov3", {}).get("files", {}).get(name, {}).get("path", "")),
                Path(args.dinov3_model_path) / name,
                expected,
            )
        )
    for role, requested_path, admitted_path, expected in checks:
        for label, path in (("requested", requested_path), ("admitted", admitted_path)):
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"{role} {label} authority is missing or blank: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"{role} {label} authority substitution: expected {expected}, got {actual}"
                )


def _remove_expected_build_products(
    path: Path,
    *,
    allowed_roots: tuple[str, ...],
) -> list[str]:
    path = Path(path).resolve()
    status = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"],
        cwd=path,
    )["stdout"]
    entries = [entry for entry in status.split("\0") if entry]
    if not entries:
        return []

    allowed = set(allowed_roots)
    if any(
        not root or Path(root).is_absolute() or len(Path(root).parts) != 1
        for root in allowed
    ):
        raise ValueError("allowed build-product roots must be relative top-level names")

    observed_roots: set[str] = set()
    rejected: list[str] = []
    for entry in entries:
        if not entry.startswith("?? "):
            rejected.append(entry)
            continue
        relative = Path(entry[3:])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            rejected.append(entry)
            continue
        root = relative.parts[0]
        if root not in allowed:
            rejected.append(entry)
            continue
        observed_roots.add(root)
    if rejected:
        raise RuntimeError(
            f"unattributable post-build source changes at {path}: "
            + "; ".join(rejected)
        )

    for root in sorted(observed_roots):
        target = path / root
        if target.is_symlink() or not target.is_dir():
            raise RuntimeError(f"invalid expected build-product tree at {target}")
    for root in sorted(observed_roots):
        shutil.rmtree(path / root)

    remaining = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=path,
    )["stdout"]
    if remaining:
        raise RuntimeError(f"source checkout remains dirty after build cleanup at {path}: {remaining}")
    return sorted(observed_roots)


def _git_identity(path: Path, repository: str, commit: str) -> dict[str, Any]:
    head = _run(["git", "rev-parse", "HEAD"], cwd=path)["stdout"].strip()
    if head != commit:
        raise RuntimeError(f"checkout commit mismatch at {path}: expected {commit}, got {head}")
    status = _run(["git", "status", "--porcelain"], cwd=path)["stdout"]
    if status:
        raise RuntimeError(f"source checkout is dirty at {path}: {status}")
    remotes = _run(["git", "remote", "get-url", "origin"], cwd=path)["stdout"].strip()
    if remotes.rstrip("/") != repository.rstrip("/"):
        raise RuntimeError(
            f"checkout origin mismatch at {path}: expected {repository}, got {remotes}"
        )
    return {
        "path": str(path),
        "repository": repository,
        "commit": commit,
        "clean": True,
    }


def _clone_checkout(
    root: Path,
    *,
    repository: str,
    commit: str,
    recursive: bool = False,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        command = ["git", "clone", "--filter=blob:none", "--no-checkout"]
        if recursive:
            command.append("--recursive")
        command.extend([repository, str(root)])
        receipts.append(_run(command))
    receipts.append(_run(["git", "fetch", "origin", commit], cwd=root))
    receipts.append(_run(["git", "checkout", "--detach", commit], cwd=root))
    if recursive:
        receipts.append(
            _run(["git", "submodule", "update", "--init", "--recursive"], cwd=root)
        )
    return receipts


def prepare_runtime(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Path]:
    work_dir = Path(args.work_dir).resolve()
    source_root = work_dir / "TRELLIS.2"
    cumesh_root = work_dir / "CuMesh"
    flex_root = work_dir / "FlexGEMM"
    nvdiffrast_root = work_dir / "nvdiffrast"
    report["setup_commands"] = []
    report["setup_commands"].extend(
        _clone_checkout(
            source_root,
            repository=TRELLIS_REPOSITORY,
            commit=TRELLIS_COMMIT,
            recursive=True,
        )
    )
    report["setup_commands"].extend(
        _clone_checkout(
            cumesh_root,
            repository=CUMESH_REPOSITORY,
            commit=CUMESH_COMMIT,
            recursive=True,
        )
    )
    report["setup_commands"].extend(
        _clone_checkout(
            flex_root,
            repository=FLEX_GEMM_REPOSITORY,
            commit=FLEX_GEMM_COMMIT,
            recursive=True,
        )
    )
    report["setup_commands"].extend(
        _clone_checkout(
            nvdiffrast_root,
            repository=NVDIFFRAST_REPOSITORY,
            commit=NVDIFFRAST_COMMIT,
        )
    )

    source_identity = _git_identity(source_root, TRELLIS_REPOSITORY, TRELLIS_COMMIT)
    cumesh_identity = _git_identity(cumesh_root, CUMESH_REPOSITORY, CUMESH_COMMIT)
    flex_identity = _git_identity(flex_root, FLEX_GEMM_REPOSITORY, FLEX_GEMM_COMMIT)
    nvdiffrast_identity = _git_identity(
        nvdiffrast_root, NVDIFFRAST_REPOSITORY, NVDIFFRAST_COMMIT
    )
    report["source_identities_before_build"] = {
        "trellis": source_identity,
        "cumesh": cumesh_identity,
        "flex_gemm": flex_identity,
        "nvdiffrast": nvdiffrast_identity,
    }

    python = sys.executable
    report["setup_commands"].append(
        _run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "imageio",
                "imageio-ffmpeg",
                "tqdm",
                "easydict",
                "plyfile",
                "zstandard",
                "opencv-python-headless",
                "ninja",
                "trimesh",
                "transformers",
                "huggingface-hub",
                "kornia",
                "timm",
            ]
        )
    )
    for package_root in (
        cumesh_root,
        flex_root,
        source_root / "o-voxel",
        nvdiffrast_root,
    ):
        report["setup_commands"].append(
            _run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-build-isolation",
                    "--no-deps",
                    str(package_root),
                ]
            )
        )

    report["source_build_products_removed"] = {
        "nvdiffrast": _remove_expected_build_products(
            nvdiffrast_root,
            allowed_roots=("build", "nvdiffrast.egg-info"),
        )
    }

    report["source_identities_after_build"] = {
        "trellis": _git_identity(source_root, TRELLIS_REPOSITORY, TRELLIS_COMMIT),
        "cumesh": _git_identity(cumesh_root, CUMESH_REPOSITORY, CUMESH_COMMIT),
        "flex_gemm": _git_identity(flex_root, FLEX_GEMM_REPOSITORY, FLEX_GEMM_COMMIT),
        "nvdiffrast": _git_identity(
            nvdiffrast_root, NVDIFFRAST_REPOSITORY, NVDIFFRAST_COMMIT
        ),
    }
    return {
        "source_root": source_root,
        "cumesh_root": cumesh_root,
        "flex_root": flex_root,
        "nvdiffrast_root": nvdiffrast_root,
    }


def prepare_model_view(args: argparse.Namespace, report: dict[str, Any]) -> Path:
    from huggingface_hub import snapshot_download

    verify_admitted_inputs_before_use(args, report)
    work_dir = Path(args.work_dir).resolve()
    hf_cache = work_dir / "huggingface"
    snapshots = {
        "trellis": Path(
            snapshot_download(
                repo_id=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                cache_dir=hf_cache,
            )
        ).resolve(),
        "sparse_decoder": Path(
            snapshot_download(
                repo_id=SPARSE_DECODER_REPOSITORY,
                revision=SPARSE_DECODER_REVISION,
                cache_dir=hf_cache,
            )
        ).resolve(),
        "rembg": Path(
            snapshot_download(
                repo_id=REMBG_REPOSITORY,
                revision=REMBG_REVISION,
                cache_dir=hf_cache,
            )
        ).resolve(),
        "dinov3": Path(args.dinov3_model_path).resolve(),
    }
    pipeline_path = snapshots["trellis"] / "pipeline.json"
    if sha256_file(pipeline_path) != MODEL_PIPELINE_SHA256:
        raise RuntimeError("pinned TRELLIS.2 pipeline.json digest mismatch")
    pipeline_config = json.loads(pipeline_path.read_text())
    model_specs = pipeline_config["args"]["models"]
    for name, spec in list(model_specs.items()):
        if spec.startswith("ckpts/"):
            model_specs[name] = str(snapshots["trellis"] / spec)
        elif spec.startswith(f"{SPARSE_DECODER_REPOSITORY}/"):
            relative = spec.removeprefix(f"{SPARSE_DECODER_REPOSITORY}/")
            model_specs[name] = str(snapshots["sparse_decoder"] / relative)
        else:
            raise RuntimeError(f"unadmitted model path in pinned pipeline config: {spec}")
    pipeline_config["args"]["image_cond_model"]["args"]["model_name"] = str(
        snapshots["dinov3"]
    )
    pipeline_config["args"]["rembg_model"]["args"]["model_name"] = str(
        snapshots["rembg"]
    )
    model_view = work_dir / "pinned-model-view"
    model_view.mkdir(parents=True, exist_ok=True)
    rewritten_path = model_view / "pipeline.json"
    _atomic_write_json(rewritten_path, pipeline_config)
    report["model_assets"] = {
        "trellis": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "snapshot_path": str(snapshots["trellis"]),
            "pipeline_json_sha256": MODEL_PIPELINE_SHA256,
        },
        "sparse_decoder": {
            "repository": SPARSE_DECODER_REPOSITORY,
            "revision": SPARSE_DECODER_REVISION,
            "snapshot_path": str(snapshots["sparse_decoder"]),
        },
        "dinov3": {
            "repository": DINOV3_REPOSITORY,
            "revision": DINOV3_REVISION,
            "snapshot_path": str(snapshots["dinov3"]),
            "files": dict(DINOV3_FILES),
        },
        "rembg": {
            "repository": REMBG_REPOSITORY,
            "revision": REMBG_REVISION,
            "snapshot_path": str(snapshots["rembg"]),
        },
        "path_rewrite_only": True,
        "rewritten_pipeline_json": str(rewritten_path),
        "rewritten_pipeline_json_sha256": sha256_file(rewritten_path),
    }
    return model_view


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        dtype_text = str(getattr(tensor, "dtype", ""))
        if "bfloat16" in dtype_text:
            tensor = tensor.float()
        return np.ascontiguousarray(tensor.numpy())
    return np.ascontiguousarray(np.asarray(value))


def _sparse_arrays(prefix: str, value: Any) -> dict[str, np.ndarray]:
    if hasattr(value, "feats") and hasattr(value, "coords"):
        return {
            f"{prefix}_feats": _as_numpy(value.feats),
            f"{prefix}_coords": _as_numpy(value.coords),
        }
    return {prefix: _as_numpy(value)}


class ArtifactRecorder:
    def __init__(
        self,
        output_dir: Path,
        *,
        run_id: str | None = None,
        on_capture: Callable[["ArtifactRecorder"], None] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.capture_order: list[str] = []
        self.run_id = run_id
        self.on_capture = on_capture

    def _next_path(self, stage: str, suffix: str) -> Path:
        if stage in self.artifacts:
            raise RuntimeError(f"stage captured more than once: {stage}")
        expected = EXPECTED_CAPTURE_ORDER[len(self.capture_order)]
        if stage != expected:
            raise RuntimeError(f"capture order mismatch: expected {expected}, got {stage}")
        return self.output_dir / f"{len(self.capture_order):02d}-{stage}{suffix}"

    def _register(self, stage: str, path: Path, metadata: dict[str, Any] | None = None) -> None:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"captured artifact is missing or blank: {path}")
        self.capture_order.append(stage)
        self.artifacts[stage] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            **({"run_id": self.run_id} if self.run_id is not None else {}),
            **(metadata or {}),
        }
        if self.on_capture is not None:
            self.on_capture(self)

    def save_npz(self, stage: str, arrays: dict[str, Any]) -> Path:
        path = self._next_path(stage, ".npz")
        converted = {key: _as_numpy(value) for key, value in arrays.items()}
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".npz.tmp", dir=self.output_dir
        )
        os.close(fd)
        try:
            with Path(tmp_name).open("wb") as handle:
                np.savez(handle, **converted)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        self._register(
            stage,
            path,
            {
                "arrays": {
                    key: {
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                    }
                    for key, value in converted.items()
                }
            },
        )
        return path

    def save_image(self, stage: str, image: Any) -> Path:
        path = self._next_path(stage, ".png")
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".png.tmp", dir=self.output_dir
        )
        os.close(fd)
        try:
            image.save(tmp_name, format="PNG")
            with Path(tmp_name).open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        self._register(stage, path, {"mode": image.mode, "size": list(image.size)})
        return path

    def register_glb(self, stage: str, path: Path) -> None:
        expected_path = self._next_path(stage, ".glb")
        if Path(path).resolve() != expected_path.resolve():
            raise RuntimeError(f"consumer GLB path mismatch: {path} != {expected_path}")
        self._register(stage, expected_path)


def _flow_arrays(noise: Any, result: Any) -> dict[str, np.ndarray]:
    arrays = _sparse_arrays("noise", noise)
    sample_next = list(result.pred_x_t)
    pred_x0 = list(result.pred_x_0)
    if not sample_next or len(sample_next) != len(pred_x0):
        raise RuntimeError("official sampler returned an incomplete recurrence")
    first = sample_next[0]
    if hasattr(first, "feats"):
        arrays["coords"] = _as_numpy(first.coords)
        arrays["sample_next"] = np.stack([_as_numpy(value.feats) for value in sample_next])
        arrays["pred_x0"] = np.stack([_as_numpy(value.feats) for value in pred_x0])
    else:
        arrays["sample_next"] = np.stack([_as_numpy(value) for value in sample_next])
        arrays["pred_x0"] = np.stack([_as_numpy(value) for value in pred_x0])
    return arrays


class SamplerObserver:
    def __init__(self, delegate: Any, stage: str, recorder: ArtifactRecorder, steps: int):
        self.delegate = delegate
        self.stage = stage
        self.recorder = recorder
        self.steps = steps
        self.call_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def sample(self, model: Any, noise: Any, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        if self.call_count != 1:
            raise RuntimeError(f"{self.stage} sampler called more than once")
        result = self.delegate.sample(model, noise, *args, **kwargs)
        arrays = _flow_arrays(noise, result)
        if arrays["sample_next"].shape[0] != self.steps:
            raise RuntimeError(
                f"{self.stage} recurrence has {arrays['sample_next'].shape[0]} steps; "
                f"expected {self.steps}"
            )
        self.recorder.save_npz(self.stage, arrays)
        return result


def _bind_method(instance: Any, name: str, wrapper: Callable[..., Any]) -> Callable[..., Any]:
    original = getattr(instance, name)
    setattr(instance, name, types.MethodType(wrapper, instance))
    return original


def install_pipeline_observers(
    pipeline: Any,
    recorder: ArtifactRecorder,
    *,
    steps: int,
) -> dict[str, Any]:
    sparse_observer = SamplerObserver(
        pipeline.sparse_structure_sampler, "sparse_flow", recorder, steps
    )
    shape_observer = SamplerObserver(pipeline.shape_slat_sampler, "shape_flow", recorder, steps)
    texture_observer = SamplerObserver(
        pipeline.tex_slat_sampler, "texture_flow", recorder, steps
    )
    pipeline.sparse_structure_sampler = sparse_observer
    pipeline.shape_slat_sampler = shape_observer
    pipeline.tex_slat_sampler = texture_observer

    original_preprocess = pipeline.preprocess_image

    def preprocess(self: Any, image: Any) -> Any:
        output = original_preprocess(image)
        recorder.save_image("preprocessed_image", output)
        return output

    _bind_method(pipeline, "preprocess_image", preprocess)

    original_get_cond = pipeline.get_cond

    def get_cond(self: Any, image: Any, resolution: int, include_neg_cond: bool = True) -> Any:
        output = original_get_cond(image, resolution, include_neg_cond)
        if resolution != 512:
            raise RuntimeError(f"unexpected conditioning resolution on 512 route: {resolution}")
        recorder.save_npz(
            "conditioning_512",
            {key: _as_numpy(value) for key, value in output.items()},
        )
        return output

    _bind_method(pipeline, "get_cond", get_cond)

    original_sparse = pipeline.sample_sparse_structure

    def sample_sparse(self: Any, *args: Any, **kwargs: Any) -> Any:
        coords = original_sparse(*args, **kwargs)
        recorder.save_npz("sparse_support", {"coords": coords})
        return coords

    _bind_method(pipeline, "sample_sparse_structure", sample_sparse)

    original_shape = pipeline.sample_shape_slat

    def sample_shape(self: Any, *args: Any, **kwargs: Any) -> Any:
        slat = original_shape(*args, **kwargs)
        recorder.save_npz("shape_slat", _sparse_arrays("shape_slat", slat))
        return slat

    _bind_method(pipeline, "sample_shape_slat", sample_shape)

    original_decode_shape = pipeline.decode_shape_slat

    def decode_shape(self: Any, *args: Any, **kwargs: Any) -> Any:
        meshes, subs = original_decode_shape(*args, **kwargs)
        if len(meshes) != 1:
            raise RuntimeError(f"expected one decoded mesh, got {len(meshes)}")
        mesh = meshes[0]
        recorder.save_npz(
            "decoder_raw_mesh",
            {"vertices": mesh.vertices, "faces": mesh.faces},
        )
        return meshes, subs

    _bind_method(pipeline, "decode_shape_slat", decode_shape)

    original_decode_tex = pipeline.decode_tex_slat

    def decode_tex(self: Any, *args: Any, **kwargs: Any) -> Any:
        voxels = original_decode_tex(*args, **kwargs)
        recorder.save_npz("texture_voxels", _sparse_arrays("texture_voxels", voxels))
        return voxels

    _bind_method(pipeline, "decode_tex_slat", decode_tex)

    original_decode_latent = pipeline.decode_latent

    def decode_latent(self: Any, *args: Any, **kwargs: Any) -> Any:
        meshes = original_decode_latent(*args, **kwargs)
        if len(meshes) != 1:
            raise RuntimeError(f"expected one pipeline mesh, got {len(meshes)}")
        mesh = meshes[0]
        recorder.save_npz(
            "pipeline_filled_mesh",
            {
                "vertices": mesh.vertices,
                "faces": mesh.faces,
                "texture_coords": mesh.coords,
                "texture_attrs": mesh.attrs,
            },
        )
        return meshes

    _bind_method(pipeline, "decode_latent", decode_latent)
    return {
        "sparse": sparse_observer,
        "shape": shape_observer,
        "texture": texture_observer,
    }


def install_orientation_observer(cumesh_module: Any, recorder: ArtifactRecorder) -> dict[str, Any]:
    native_class = cumesh_module.CuMesh
    state = {
        "call_count": 0,
        "native_method_return_preserved": False,
        "pre_readback_written": False,
        "post_readback_written": False,
    }

    class ObservedCuMesh:
        def __init__(self, *args: Any, **kwargs: Any):
            self._native = native_class(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._native, name)

        def unify_face_orientations(self, *args: Any, **kwargs: Any) -> Any:
            state["call_count"] += 1
            if state["call_count"] != 1:
                raise RuntimeError("unify_face_orientations called more than once")
            pre_vertices, pre_faces = self._native.read()
            recorder.save_npz(
                "postprocess_stage11_pre_orientation",
                {"vertices": pre_vertices, "faces": pre_faces},
            )
            state["pre_readback_written"] = True
            result = self._native.unify_face_orientations(*args, **kwargs)
            state["native_method_return_preserved"] = True
            post_vertices, post_faces = self._native.read()
            recorder.save_npz(
                "postprocess_stage12_post_orientation",
                {"vertices": post_vertices, "faces": post_faces},
            )
            state["post_readback_written"] = True
            return result

    cumesh_module.CuMesh = ObservedCuMesh
    return {
        "module": cumesh_module,
        "native_class": native_class,
        "observed_class": ObservedCuMesh,
        "state": state,
    }


def _restore_orientation_observer(observer: dict[str, Any]) -> None:
    observer["module"].CuMesh = observer["native_class"]


def _cleanup_stale_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stage in EXPECTED_CAPTURE_ORDER:
        for path in output_dir.glob(f"??-{stage}.*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


def run_live(args: argparse.Namespace, report: dict[str, Any]) -> None:
    output_dir = Path(args.output_dir).resolve()
    verify_admitted_inputs_before_use(args, report)
    phase_started = time.perf_counter()
    report["phase"] = "prepare_runtime"
    roots = prepare_runtime(args, report)
    report.setdefault("phase_timings", {})["prepare_runtime"] = time.perf_counter() - phase_started
    _atomic_write_json(output_dir / "report.json", report)

    os.environ["ATTN_BACKEND"] = args.attention_backend
    os.environ["SPARSE_ATTN_BACKEND"] = args.attention_backend
    os.environ["SPARSE_CONV_BACKEND"] = args.sparse_conv_backend
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    source_root = roots["source_root"]
    sys.path.insert(0, str(source_root))

    report["phase"] = "runtime_identity"
    import torch
    import xformers
    import cumesh
    import flex_gemm
    import o_voxel
    from PIL import Image
    from trellis2.modules.attention import config as attention_config
    from trellis2.modules.sparse import config as sparse_config
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"Torch runtime drift: expected {EXPECTED_TORCH_VERSION}, got {torch.__version__}"
        )
    device_name = torch.cuda.get_device_name(0)
    if "T4" not in device_name:
        raise RuntimeError(f"expected Tesla T4, got {device_name}")
    if attention_config.BACKEND != args.attention_backend:
        raise RuntimeError(
            f"dense attention backend fallback: {attention_config.BACKEND!r}"
        )
    if sparse_config.ATTN != args.attention_backend:
        raise RuntimeError(f"sparse attention backend fallback: {sparse_config.ATTN!r}")
    if sparse_config.CONV != args.sparse_conv_backend:
        raise RuntimeError(f"sparse convolution backend fallback: {sparse_config.CONV!r}")

    report["effective_route"] = {
        "device_type": "cuda",
        "cuda_device_name": device_name,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_runtime_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "xformers_version": getattr(xformers, "__version__", None),
        "attention_backend": attention_config.BACKEND,
        "sparse_attention_backend": sparse_config.ATTN,
        "sparse_conv_backend": sparse_config.CONV,
        "trellis_commit": TRELLIS_COMMIT,
        "trellis_source_clean": report["source_identities_after_build"]["trellis"]["clean"],
        "cumesh_commit": CUMESH_COMMIT,
        "cumesh_source_clean_before_build": report["source_identities_before_build"]["cumesh"]["clean"],
        "flex_gemm_commit": FLEX_GEMM_COMMIT,
        "nvdiffrast_commit": NVDIFFRAST_COMMIT,
        "model_revision": MODEL_REVISION,
        "sparse_decoder_revision": SPARSE_DECODER_REVISION,
        "dinov3_revision": DINOV3_REVISION,
        "rembg_revision": REMBG_REVISION,
        "pipeline_type": args.pipeline_type,
        "seed": args.seed,
        "sampler_steps": {"sparse": args.steps, "shape": args.steps, "texture": args.steps},
        "pipeline_run_call_count": 0,
        "native_conditioning": True,
        "native_rng": True,
        "observation_only_instrumentation": True,
        "run_id": args.run_id,
        "cumesh_module_path": str(Path(cumesh.__file__).resolve()),
        "flex_gemm_module_path": str(Path(flex_gemm.__file__).resolve()),
        "o_voxel_module_path": str(Path(o_voxel.__file__).resolve()),
    }
    _atomic_write_json(output_dir / "report.json", report)

    report["phase"] = "load_pipeline"
    phase_started = time.perf_counter()
    model_view = prepare_model_view(args, report)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(model_view))
    pipeline.cuda()
    report["phase_timings"]["load_pipeline"] = time.perf_counter() - phase_started

    def publish_capture(current: ArtifactRecorder) -> None:
        report["capture_order"] = list(current.capture_order)
        report["artifacts"] = dict(current.artifacts)
        report["last_trustworthy_phase"] = f"{current.capture_order[-1]}_captured"
        _atomic_write_json(output_dir / "report.json", report)

    recorder = ArtifactRecorder(output_dir, run_id=args.run_id, on_capture=publish_capture)
    sampler_observers = install_pipeline_observers(pipeline, recorder, steps=args.steps)
    verify_admitted_inputs_before_use(args, report)
    image = Image.open(args.image)

    report["phase"] = "pipeline_run"
    phase_started = time.perf_counter()
    report["effective_route"]["pipeline_run_call_count"] += 1
    meshes, latent = pipeline.run(
        image,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
        sparse_structure_sampler_params={"steps": args.steps},
        shape_slat_sampler_params={"steps": args.steps},
        tex_slat_sampler_params={"steps": args.steps},
        return_latent=True,
    )
    torch.cuda.synchronize()
    report["phase_timings"]["pipeline_run"] = time.perf_counter() - phase_started
    if report["effective_route"]["pipeline_run_call_count"] != 1:
        raise RuntimeError("official pipeline.run was not called exactly once")
    if any(observer.call_count != 1 for observer in sampler_observers.values()):
        raise RuntimeError("one or more native samplers did not execute exactly once")
    if len(meshes) != 1 or len(latent) != 3:
        raise RuntimeError("official pipeline returned an unexpected result shape")

    mesh = meshes[0]
    report["phase"] = "official_to_glb"
    phase_started = time.perf_counter()
    orientation = install_orientation_observer(cumesh, recorder)
    try:
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.target_faces,
            texture_size=args.texture_size,
            remesh=False,
            verbose=True,
        )
    finally:
        _restore_orientation_observer(orientation)
    report["orientation_observer"] = orientation["state"]
    expected_orientation = {
        "call_count": 1,
        "native_method_return_preserved": True,
        "pre_readback_written": True,
        "post_readback_written": True,
    }
    if report["orientation_observer"] != expected_orientation:
        raise RuntimeError(f"orientation observer incomplete: {report['orientation_observer']}")

    glb_path = output_dir / f"{len(recorder.capture_order):02d}-consumer_glb.glb"
    tmp_glb = glb_path.with_name(f".{glb_path.name}.tmp.glb")
    tmp_glb.unlink(missing_ok=True)
    glb.export(tmp_glb, extension_webp=True)
    if not tmp_glb.is_file() or tmp_glb.stat().st_size <= 0:
        raise RuntimeError("official GLB export produced no artifact")
    os.replace(tmp_glb, glb_path)
    recorder.register_glb("consumer_glb", glb_path)
    torch.cuda.synchronize()
    report["phase_timings"]["official_to_glb"] = time.perf_counter() - phase_started

    if tuple(recorder.capture_order) != EXPECTED_CAPTURE_ORDER:
        raise RuntimeError(
            f"capture order mismatch: expected {EXPECTED_CAPTURE_ORDER}, got {recorder.capture_order}"
        )
    report["capture_order"] = recorder.capture_order
    report["artifacts"] = recorder.artifacts
    report["status"] = "running"
    report["failure_phase"] = None
    report["last_trustworthy_phase"] = "consumer_glb_written_unvalidated"
    report["primary_output_status"] = "written_unvalidated"
    report["phase"] = "artifact_admission"
    candidate_report_path = output_dir / "report.json"
    _atomic_write_json(candidate_report_path, report)
    _validate_completed_bundle(
        candidate_report_path,
        report,
        expected_run_id=args.run_id,
        expected_image_sha256=args.expected_image_sha256,
    )
    report["status"] = "completed"
    report["failure_phase"] = None
    report["last_trustworthy_phase"] = "consumer_glb_validated"
    report["primary_output_status"] = "validated"
    report["phase"] = "completed"


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _validate_npz_artifact(stage: str, path: Path, record: dict[str, Any]) -> None:
    schema = NPZ_STAGE_SCHEMAS[stage]
    try:
        with np.load(path, allow_pickle=False) as reopened:
            reopened_files = set(reopened.files)
            arrays = {name: np.ascontiguousarray(reopened[name]) for name in reopened.files}
    except BaseException as exc:
        raise ValueError(f"NPZ artifact structure is invalid for {stage}: {exc}") from exc
    if reopened_files != set(schema):
        raise ValueError(
            f"NPZ keys for {stage} must be {sorted(schema)!r}, got {sorted(reopened_files)!r}"
        )

    recorded_arrays = record.get("arrays")
    if not isinstance(recorded_arrays, dict) or set(recorded_arrays) != set(schema):
        raise ValueError(f"NPZ array receipts are incomplete for {stage}")
    for name, spec in schema.items():
        value = arrays[name]
        if str(value.dtype) != spec["dtype"]:
            raise ValueError(
                f"NPZ dtype for {stage}.{name} must be {spec['dtype']}, got {value.dtype}"
            )
        if value.ndim != spec["ndim"]:
            raise ValueError(
                f"NPZ rank for {stage}.{name} must be {spec['ndim']}, got {value.ndim}"
            )
        if value.size == 0 or any(dimension <= 0 for dimension in value.shape):
            raise ValueError(f"NPZ array is empty for {stage}.{name}")
        if spec.get("columns") is not None and value.shape[1] != spec["columns"]:
            raise ValueError(
                f"NPZ columns for {stage}.{name} must be {spec['columns']}, got {value.shape[1]}"
            )
        if spec.get("leading") is not None and value.shape[0] != spec["leading"]:
            raise ValueError(
                f"NPZ leading dimension for {stage}.{name} must be {spec['leading']}, got {value.shape[0]}"
            )
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"NPZ array contains non-finite values for {stage}.{name}")
        if spec.get("unique_rows") and len(np.unique(value, axis=0)) != len(value):
            raise ValueError(f"NPZ coordinate rows are not unique for {stage}.{name}")
        metadata = recorded_arrays[name]
        expected_metadata = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
        if metadata != expected_metadata:
            raise ValueError(
                f"NPZ array receipt mismatch for {stage}.{name}: "
                f"expected {expected_metadata!r}, got {metadata!r}"
            )

    if stage == "conditioning_512" and arrays["cond"].shape != arrays["neg_cond"].shape:
        raise ValueError("conditioning positive/negative shapes differ")
    if stage == "sparse_flow":
        if arrays["sample_next"].shape != arrays["pred_x0"].shape:
            raise ValueError("sparse flow recurrence shapes differ")
        if arrays["sample_next"].shape[1:] != arrays["noise"].shape:
            raise ValueError("sparse flow recurrence is not bound to noise shape")
    if stage in {"shape_flow", "texture_flow"}:
        if not np.array_equal(arrays["coords"], arrays["noise_coords"]):
            raise ValueError(f"{stage} recurrence coordinates differ from noise coordinates")
        feature_shape = arrays["noise_feats"].shape
        if arrays["sample_next"].shape != (8, *feature_shape):
            raise ValueError(f"{stage} sample recurrence is not row-bound to noise features")
        if arrays["pred_x0"].shape != (8, *feature_shape):
            raise ValueError(f"{stage} x0 recurrence is not row-bound to noise features")
        if len(arrays["coords"]) != feature_shape[0]:
            raise ValueError(f"{stage} coordinates are not row-bound to features")
    for feature_name, coordinate_name in (
        ("shape_slat_feats", "shape_slat_coords"),
        ("texture_voxels_feats", "texture_voxels_coords"),
        ("texture_attrs", "texture_coords"),
    ):
        if feature_name in arrays and len(arrays[feature_name]) != len(arrays[coordinate_name]):
            raise ValueError(
                f"{stage} {feature_name} rows are not bound to {coordinate_name} rows"
            )
    if "vertices" in arrays and "faces" in arrays:
        vertices = arrays["vertices"]
        faces = arrays["faces"]
        if np.any(faces < 0) or int(faces.max()) >= len(vertices):
            raise ValueError(f"{stage} face indices escape the vertex array")


def _validate_png_artifact(path: Path, record: dict[str, Any]) -> None:
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode not in {"RGB", "RGBA"}:
                raise ValueError(
                    f"PNG artifact must be RGB/RGBA PNG, got {image.format}/{image.mode}"
                )
            if image.width <= 0 or image.height <= 0:
                raise ValueError("PNG artifact has empty dimensions")
            if record.get("mode") != image.mode or record.get("size") != [image.width, image.height]:
                raise ValueError("PNG artifact metadata does not match reopened image")
    except ValueError:
        raise
    except BaseException as exc:
        raise ValueError(f"PNG artifact structure is invalid: {exc}") from exc


def _validate_glb_artifact(path: Path) -> None:
    import trimesh

    try:
        scene = trimesh.load(path, force="scene", process=False)
    except BaseException as exc:
        raise ValueError(f"GLB artifact structure is invalid: {exc}") from exc
    geometries = list(getattr(scene, "geometry", {}).values())
    if not geometries:
        raise ValueError("GLB contains no geometry")
    textured = False
    for geometry in geometries:
        vertices = np.asarray(getattr(geometry, "vertices", ()))
        faces = np.asarray(getattr(geometry, "faces", ()))
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
            raise ValueError("GLB geometry has no nonempty vertex array")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
            raise ValueError("GLB geometry has no nonempty triangle array")
        if not np.isfinite(vertices).all():
            raise ValueError("GLB geometry contains non-finite vertices")
        visual = getattr(geometry, "visual", None)
        uv = getattr(visual, "uv", None)
        material = getattr(visual, "material", None)
        texture = None
        if material is not None:
            texture = getattr(material, "baseColorTexture", None)
            if texture is None:
                texture = getattr(material, "image", None)
        if uv is not None and len(np.asarray(uv)) == len(vertices) and texture is not None:
            width, height = getattr(texture, "size", (0, 0))
            textured = width > 0 and height > 0
    if not textured:
        raise ValueError("GLB has no geometry with bound UVs and a nonempty texture material")


def _validate_authority(report: dict[str, Any], *, expected_run_id: str | None, expected_image_sha256: str | None) -> str:
    run_id = report.get("run_id")
    try:
        parsed = uuid.UUID(run_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("run identity is missing or invalid") from exc
    if str(parsed) != run_id:
        raise ValueError("run identity is not canonical")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(f"run identity mismatch: expected {expected_run_id}, got {run_id}")
    requested_route = report.get("requested_route", {})
    if requested_route.get("run_id") != run_id:
        raise ValueError("requested route run identity mismatch")
    requested_image = requested_route.get("image_sha256")
    if expected_image_sha256 is not None and requested_image != expected_image_sha256:
        raise ValueError("requested image identity does not match external admission")
    effective_inputs = report.get("effective_inputs", {})
    image = effective_inputs.get("image", {})
    if image.get("sha256") != requested_image or not isinstance(image.get("size_bytes"), int) or image["size_bytes"] <= 0:
        raise ValueError("effective image identity does not match requested image")
    if effective_inputs.get("run_id") != run_id:
        raise ValueError("effective input run identity mismatch")
    dino = effective_inputs.get("dinov3", {}).get("files", {})
    if set(dino) != set(DINOV3_FILES):
        raise ValueError("effective DINOv3 identity set is incomplete")
    for name, expected in DINOV3_FILES.items():
        record = dino[name]
        if record.get("sha256") != expected or not isinstance(record.get("size_bytes"), int) or record["size_bytes"] <= 0:
            raise ValueError(f"effective DINOv3 identity mismatch for {name}")

    model_assets = report.get("model_assets")
    if not isinstance(model_assets, dict):
        raise ValueError("model assets authority is missing")
    required_assets = {
        "trellis": (MODEL_REPOSITORY, MODEL_REVISION),
        "sparse_decoder": (SPARSE_DECODER_REPOSITORY, SPARSE_DECODER_REVISION),
        "dinov3": (DINOV3_REPOSITORY, DINOV3_REVISION),
        "rembg": (REMBG_REPOSITORY, REMBG_REVISION),
    }
    for role, (repository, revision) in required_assets.items():
        record = model_assets.get(role, {})
        if record.get("repository") != repository or record.get("revision") != revision:
            raise ValueError(f"model assets authority mismatch for {role}")
    if model_assets["trellis"].get("pipeline_json_sha256") != MODEL_PIPELINE_SHA256:
        raise ValueError("model assets pipeline identity mismatch")
    if model_assets["dinov3"].get("files") != DINOV3_FILES:
        raise ValueError("model assets DINOv3 file identity mismatch")
    if model_assets.get("path_rewrite_only") is not True:
        raise ValueError("model assets path rewrite identity is missing")

    source_identities = report.get("source_identities_after_build")
    if not isinstance(source_identities, dict):
        raise ValueError("source checkout identities are missing")
    required_sources = {
        "trellis": (TRELLIS_REPOSITORY, TRELLIS_COMMIT),
        "cumesh": (CUMESH_REPOSITORY, CUMESH_COMMIT),
        "flex_gemm": (FLEX_GEMM_REPOSITORY, FLEX_GEMM_COMMIT),
        "nvdiffrast": (NVDIFFRAST_REPOSITORY, NVDIFFRAST_COMMIT),
    }
    for role, (repository, commit) in required_sources.items():
        record = source_identities.get(role, {})
        if record.get("repository") != repository or record.get("commit") != commit or record.get("clean") is not True:
            raise ValueError(f"source checkout identity mismatch for {role}")
    return run_id


def _validate_completed_bundle(
    report_path: Path,
    report: dict[str, Any],
    *,
    expected_run_id: str | None = None,
    expected_image_sha256: str | None = None,
) -> None:
    run_id = _validate_authority(
        report,
        expected_run_id=expected_run_id,
        expected_image_sha256=expected_image_sha256,
    )
    route = report.get("effective_route", {})
    required_route = {
        "device_type": "cuda",
        "torch_version": EXPECTED_TORCH_VERSION,
        "attention_backend": "xformers",
        "sparse_attention_backend": "xformers",
        "sparse_conv_backend": "flex_gemm",
        "trellis_commit": TRELLIS_COMMIT,
        "trellis_source_clean": True,
        "cumesh_commit": CUMESH_COMMIT,
        "cumesh_source_clean_before_build": True,
        "nvdiffrast_commit": NVDIFFRAST_COMMIT,
        "model_revision": MODEL_REVISION,
        "sparse_decoder_revision": SPARSE_DECODER_REVISION,
        "dinov3_revision": DINOV3_REVISION,
        "rembg_revision": REMBG_REVISION,
        "pipeline_type": "512",
        "seed": 42,
        "sampler_steps": {"sparse": 8, "shape": 8, "texture": 8},
        "pipeline_run_call_count": 1,
        "native_conditioning": True,
        "native_rng": True,
        "observation_only_instrumentation": True,
        "run_id": run_id,
    }
    for key, expected in required_route.items():
        if route.get(key) != expected:
            raise ValueError(
                f"effective route {key} must be {expected!r}, got {route.get(key)!r}"
            )
    device_name = route.get("cuda_device_name")
    if not isinstance(device_name, str) or "T4" not in device_name:
        raise ValueError(f"effective route cuda_device_name is not Tesla T4: {device_name!r}")
    if report.get("capture_order") != list(EXPECTED_CAPTURE_ORDER):
        raise ValueError(
            f"capture order must be {list(EXPECTED_CAPTURE_ORDER)!r}, "
            f"got {report.get('capture_order')!r}"
        )
    orientation = report.get("orientation_observer", {})
    required_orientation = {
        "call_count": 1,
        "native_method_return_preserved": True,
        "pre_readback_written": True,
        "post_readback_written": True,
    }
    if orientation != required_orientation:
        raise ValueError(f"orientation observer is incomplete: {orientation!r}")
    artifacts = report.get("artifacts", {})
    if set(artifacts) != set(EXPECTED_CAPTURE_ORDER):
        raise ValueError("artifact set does not match capture order")
    recorded_paths = [Path(artifacts[stage].get("path", "")).as_posix() for stage in EXPECTED_CAPTURE_ORDER]
    if len(set(recorded_paths)) != len(recorded_paths):
        raise ValueError("artifact paths must be distinct; duplicate stage path recorded")
    resolved_paths: set[Path] = set()
    for stage in EXPECTED_CAPTURE_ORDER:
        record = artifacts[stage]
        recorded = Path(record.get("path", ""))
        expected_name = EXPECTED_ARTIFACT_FILENAMES[stage]
        if recorded.as_posix() != expected_name:
            raise ValueError(
                f"artifact path for {stage} must be canonical {expected_name!r}, got {recorded.as_posix()!r}"
            )
        path = recorded if recorded.is_absolute() else report_path.parent / recorded
        path = path.resolve()
        if report_path.parent not in path.parents:
            raise ValueError(f"artifact path escapes report custody for {stage}: {path}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"artifact is missing or blank for {stage}: {path}")
        actual_digest = sha256_file(path)
        if actual_digest != record.get("sha256"):
            raise ValueError(
                f"artifact digest mismatch for {stage}: {actual_digest} != {record.get('sha256')}"
            )
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"artifact size mismatch for {stage}")
        if path in resolved_paths:
            raise ValueError(f"artifact paths must be distinct; duplicate at {stage}: {path}")
        resolved_paths.add(path)
        if record.get("run_id") != run_id:
            raise ValueError(f"artifact run identity mismatch for {stage}")
        if stage == "preprocessed_image":
            _validate_png_artifact(path, record)
        elif stage == "consumer_glb":
            _validate_glb_artifact(path)
        else:
            _validate_npz_artifact(stage, path, record)


def validate_completed_report(
    report_path: Path,
    *,
    expected_run_id: str | None = None,
    expected_image_sha256: str | None = None,
) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    if not report_path.is_file() or report_path.stat().st_size <= 0:
        raise ValueError(f"report is missing or blank: {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("schema") != SCHEMA:
        raise ValueError(f"unexpected schema: {report.get('schema')!r}")
    if report.get("status") != "completed":
        raise ValueError(f"report status is not completed: {report.get('status')!r}")
    if report.get("primary_output_status") != "validated":
        raise ValueError("primary output is not validated")
    if report.get("last_trustworthy_phase") != "consumer_glb_validated":
        raise ValueError("report did not reach consumer_glb_validated")
    _validate_completed_bundle(
        report_path,
        report,
        expected_run_id=expected_run_id,
        expected_image_sha256=expected_image_sha256,
    )
    return report


def validate_downloaded_native_image_to_glb_outputs(
    packet: Any,
    output_dir: Path,
) -> dict[str, Any]:
    from trellmlx.kaggle_cuda_witness import (
        WitnessPacketError,
        validate_downloaded_outputs,
    )

    if packet.run_id is None:
        raise WitnessPacketError("native image-to-GLB packet is missing its run identity")
    if packet.expected_image_sha256 is None:
        raise WitnessPacketError("native image-to-GLB packet is missing its image identity")
    expected_stage_outputs = tuple(
        EXPECTED_ARTIFACT_FILENAMES[stage] for stage in EXPECTED_CAPTURE_ORDER
    )
    if packet.output_json != "report.json" or packet.output_npz is not None:
        raise WitnessPacketError("native image-to-GLB packet output roles are not canonical")
    if packet.expected_outputs != expected_stage_outputs:
        raise WitnessPacketError("native image-to-GLB packet stage outputs are not canonical")
    records = validate_downloaded_outputs(packet, output_dir)
    report = validate_completed_report(
        Path(output_dir) / packet.output_json,
        expected_run_id=packet.run_id,
        expected_image_sha256=packet.expected_image_sha256,
    )
    return {"downloaded_outputs": records, "report": report}


def prepare_native_image_to_glb_packet(packet: Any) -> Any:
    """Prepare one native image-to-GLB attempt with precommitted authority."""
    from trellmlx.kaggle_cuda_witness import WitnessPacketError, prepare_packet

    if packet.run_id is None or packet.expected_image_sha256 is None:
        raise WitnessPacketError(
            "native image-to-GLB packet requires run identity and image identity"
        )
    return prepare_packet(packet)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--expected-image-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--model-repository", default=MODEL_REPOSITORY)
    parser.add_argument("--dinov3-model-path", type=Path)
    parser.add_argument("--pipeline-type", default="512")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--steps", default=8, type=int)
    parser.add_argument("--target-faces", default=350000, type=int)
    parser.add_argument("--texture-size", default=1024, type=int)
    parser.add_argument("--attention-backend", default="xformers")
    parser.add_argument("--sparse-conv-backend", default="flex_gemm")
    parser.add_argument("--no-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_id is None:
        args.run_id = str(uuid.uuid4())
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    report_path = (
        Path(args.output_json).resolve()
        if args.output_json is not None
        else output_dir / "report.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "status": "running",
        "failure_phase": None,
        "last_trustworthy_phase": "arguments_parsed",
        "primary_output_status": "not_attempted",
        "requested_route": _requested_route(args),
        "effective_route": {},
        "expected_capture_order": list(EXPECTED_CAPTURE_ORDER),
        "capture_order": [],
        "artifacts": {},
    }
    phase = "request_validation"
    try:
        _validate_request(args)
        report["last_trustworthy_phase"] = "request_validated"
        if args.no_download:
            report.update(
                {
                    "status": "preflight_stopped",
                    "failure_phase": None,
                    "last_trustworthy_phase": "request_validated",
                    "primary_output_status": "not_written_no_download",
                    "effective_route": {
                        "device_type": "not_loaded_no_download",
                        "trellis_source_clean": "not_checked_no_download",
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            _atomic_write_json(report_path, report)
            return 0

        phase = "input_admission"
        args = admit_run_inputs(args, report)
        report["last_trustworthy_phase"] = "inputs_admitted_to_run_custody"
        _atomic_write_json(report_path, report)
        _cleanup_stale_outputs(output_dir)
        phase = "runtime"
        run_live(args, report)
        report["elapsed_seconds"] = time.perf_counter() - started
        _atomic_write_json(report_path, report)
        validate_completed_report(
            report_path,
            expected_run_id=args.run_id,
            expected_image_sha256=args.expected_image_sha256,
        )
        return 0
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase if report.get("phase") is None else report.get("phase"),
                "primary_output_status": (
                    "partial" if report.get("artifacts") else "not_attempted"
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        try:
            _atomic_write_json(report_path, report)
        except BaseException as report_exc:
            print(
                f"failed to write witness report {report_path}: {report_exc}",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

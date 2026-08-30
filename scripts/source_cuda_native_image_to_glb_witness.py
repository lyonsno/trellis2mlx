#!/usr/bin/env python3
"""Capture one native official TRELLIS.2 image-to-GLB CUDA trajectory."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from dataclasses import replace
import hashlib
from importlib import metadata as importlib_metadata
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import unquote, urlparse
from urllib.request import urlretrieve

import numpy as np

try:
    import trellmlx.witness_authority as witness_authority_module
except ModuleNotFoundError as exc:
    if exc.name not in {"trellmlx", "trellmlx.witness_authority"}:
        raise
    import witness_authority as witness_authority_module

AuthorityCoordinate = witness_authority_module.AuthorityCoordinate
AuthorityCoordinateError = witness_authority_module.AuthorityCoordinateError


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
MODEL_SOURCE_MARKER = (
    "runtime/huggingface/models--microsoft--TRELLIS.2-4B/"
    "blobs/f5ec14c7f71b3d7f2cb0221c5f568a6871dc5e90"
)
SPARSE_DECODER_REPOSITORY = "microsoft/TRELLIS-image-large"
SPARSE_DECODER_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
MODEL_BLOB_MANIFEST = {
    "trellis": {
        "cache_dir": "models--microsoft--TRELLIS.2-4B",
        "files": {
            "pipeline.json": {
                "blob": "f5ec14c7f71b3d7f2cb0221c5f568a6871dc5e90",
                "sha256": MODEL_PIPELINE_SHA256,
                "size_bytes": 4186,
            },
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.json": {
                "blob": "f71c87cda44a9a940fa3a2d250a0944d606d2245",
                "sha256": "6cdfb636854f60ffcadd8c49244169cd449f34c5ccdb55b5ff7c04678f9c399e",
                "size_bytes": 467,
            },
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors": {
                "blob": "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6",
                "sha256": "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6",
                "size_bytes": 2584426920,
            },
            "ckpts/shape_dec_next_dc_f16c32_fp16.json": {
                "blob": "e46164e905de0e68328f5f4b1f128b8dac608d97",
                "sha256": "5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209",
                "size_bytes": 678,
            },
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors": {
                "blob": "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581",
                "sha256": "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581",
                "size_bytes": 948490494,
            },
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json": {
                "blob": "6e39aacbf7bd43d16d3dad35c95d43c0b17d6a4f",
                "sha256": "ddbd7a1d34ce9a8e7af2f6ef9ae686d9ab49f69e62dacb98d8409b511cdbd5e2",
                "size_bytes": 458,
            },
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors": {
                "blob": "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f",
                "sha256": "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f",
                "size_bytes": 2584574424,
            },
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json": {
                "blob": "cfd8d761e89aad595e97ec0cfe4b8469c9053583",
                "sha256": "7ebd6d367393f52fee9e92c2b3d727a9026a338549c6a6fa84c7ea122140ffdd",
                "size_bytes": 458,
            },
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors": {
                "blob": "07cd0596f634c5adc1890023d16023afc5eed02fb84b22bb23aff5bf0030fbbd",
                "sha256": "07cd0596f634c5adc1890023d16023afc5eed02fb84b22bb23aff5bf0030fbbd",
                "size_bytes": 2584574424,
            },
            "ckpts/tex_dec_next_dc_f16c32_fp16.json": {
                "blob": "e17107bef522e2aded7b3bc22ef421b57e16e779",
                "sha256": "22074e4ab5b28e2b72d33a3fe61bf416134b89e12d0fc5c729f37da75da0c4d1",
                "size_bytes": 705,
            },
            "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors": {
                "blob": "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c",
                "sha256": "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c",
                "size_bytes": 948458812,
            },
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json": {
                "blob": "b260bbc6cce80ccd7b61e32a40474225522b2043",
                "sha256": "3093796257a62f838d2e5578ac4580ca3b7d88d10b87d348065d3fdfcc4c2077",
                "size_bytes": 458,
            },
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors": {
                "blob": "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d",
                "sha256": "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d",
                "size_bytes": 2584672728,
            },
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json": {
                "blob": "d949ce887c62645f6a927ee66bba2a0df56e383e",
                "sha256": "31f4fdad9974930e5e9324ac9dfb6ef7de6b30cea51854ba515ef827a8d353b2",
                "size_bytes": 458,
            },
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors": {
                "blob": "580401269059a339b8318ab9ced459a13ba63391721c83a6c383198c29e77686",
                "sha256": "580401269059a339b8318ab9ced459a13ba63391721c83a6c383198c29e77686",
                "size_bytes": 2584672728,
            },
        },
    },
    "sparse_decoder": {
        "cache_dir": "models--microsoft--TRELLIS-image-large",
        "files": {
            "ckpts/ss_dec_conv3d_16l8_fp16.json": {
                "blob": "9f3affaf13ab29fe48105229da9fab72ea8de716",
                "sha256": "646781293f1cda74720de85d1cef50a957fb4aebd9a4bd014e454e32f2330ac5",
                "size_bytes": 245,
            },
            "ckpts/ss_dec_conv3d_16l8_fp16.safetensors": {
                "blob": "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
                "sha256": "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
                "size_bytes": 147591972,
            },
        },
    },
}
MODEL_REQUIRED_BYTES = sum(
    record["size_bytes"]
    for family in MODEL_BLOB_MANIFEST.values()
    for record in family["files"].values()
)
MODEL_OUTPUT_RESERVE_BYTES = 2 * 1024**3
DINOV3_REPOSITORY = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
DINOV3_FILES = {
    "model.safetensors": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    "config.json": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
    "preprocessor_config.json": "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
}
REMBG_REPOSITORY = "briaai/RMBG-2.0"
REMBG_REVISION = "5df4c9c76d8170882c34f6986e848ee07fd0ba43"
REMBG_FILES = {
    "model.safetensors": "566ed80c3d95f87ada6864d4cbe2290a1c5eb1c7bb0b123e984f60f76b02c3a7",
    "config.json": "c97ea21569daf66b205491a4635147dd3bc42c7c168b89d7d75b53f67ef548ae",
    "birefnet.py": "e499d75224b8819e985e68fb78b7a8e8c99316840474e74e16b5529f03ca2860",
    "BiRefNet_config.py": "e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52",
}
REMBG_FILE_ARGUMENTS = {
    "model.safetensors": "rembg_model_file",
    "config.json": "rembg_config_file",
    "birefnet.py": "rembg_birefnet_file",
    "BiRefNet_config.py": "rembg_birefnet_config_file",
}
EXPECTED_TORCH_VERSION = "2.10.0+cu128"
EXPECTED_XFORMERS_VERSION = "0.0.35"
XFORMERS_WHEEL_FILENAME = "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
XFORMERS_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/a4/85/"
    "6d71f9b16f2ac647877e66ed4af723b3fbd477806ab8b8a89d39a362b85f/"
    "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
)
XFORMERS_WHEEL_SHA256 = "ccc73c7db9890224ab05f5fb60e2034f9e6c8672a10be0cf00e95cbbae3eda7c"
XFORMERS_WHEEL_SIZE_BYTES = 3264751
KAGGLE_KERNEL_WORKING_DIRECTORY = PurePosixPath("/kaggle/working")

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

CAPTURE_PROFILES = {
    "full": EXPECTED_CAPTURE_ORDER,
    "final-consumer": (
        "decoder_raw_mesh",
        "postprocess_stage11_pre_orientation",
        "postprocess_stage12_post_orientation",
        "consumer_glb",
    ),
}


def capture_order_for_profile(profile: str) -> tuple[str, ...]:
    try:
        return CAPTURE_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported capture profile: {profile}") from exc

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
        "model_blob_root": (
            str(Path(args.model_blob_root).resolve())
            if args.model_blob_root is not None
            else None
        ),
        "model_source_kernel": args.model_source_kernel,
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
        "rembg_files": {
            name: (
                str(Path(getattr(args, attribute)).resolve())
                if getattr(args, attribute) is not None
                else None
            )
            for name, attribute in REMBG_FILE_ARGUMENTS.items()
        },
        "rembg_declared_files": {
            name: (
                str(getattr(args, attribute))
                if getattr(args, attribute) is not None
                else None
            )
            for name, attribute in REMBG_FILE_ARGUMENTS.items()
        },
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
    if args.seed < 0:
        raise ValueError("this witness requires a nonnegative seed")
    if args.steps != 8:
        raise ValueError("this witness admits exactly 8 sparse, shape, and texture steps")
    if args.target_faces <= 0:
        raise ValueError("this witness requires positive target_faces")
    if args.texture_size <= 0 or args.texture_size & (args.texture_size - 1):
        raise ValueError("this witness requires a positive power-of-two texture_size")
    if args.attention_backend != "xformers":
        raise ValueError("the T4 native source route requires attention_backend='xformers'")
    if args.sparse_conv_backend != "flex_gemm":
        raise ValueError("the native source route requires sparse_conv_backend='flex_gemm'")
    if args.model_repository != MODEL_REPOSITORY:
        raise ValueError(f"this witness admits only model_repository={MODEL_REPOSITORY!r}")
    if (args.model_blob_root is None) != (args.model_source_kernel is None):
        raise ValueError(
            "--model-blob-root and --model-source-kernel must be provided together"
        )
    if args.model_source_kernel is not None:
        parts = args.model_source_kernel.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError("--model-source-kernel must be a Kaggle ref like owner/slug")
        if not Path(args.model_blob_root).resolve().is_dir():
            raise ValueError(
                f"mounted model blob root is missing: {Path(args.model_blob_root).resolve()}"
            )
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
    rembg_paths = {
        name: getattr(args, attribute)
        for name, attribute in REMBG_FILE_ARGUMENTS.items()
    }
    supplied_rembg = {name for name, path in rembg_paths.items() if path is not None}
    if supplied_rembg and supplied_rembg != set(REMBG_FILES):
        missing = sorted(set(REMBG_FILES) - supplied_rembg)
        raise ValueError(
            "local RMBG inputs must be a complete local RMBG file group; "
            f"missing {missing}"
        )
    if not args.no_download and supplied_rembg != set(REMBG_FILES):
        raise ValueError("live native background removal requires a complete local RMBG file group")
    for name, expected in REMBG_FILES.items():
        if name not in supplied_rembg:
            continue
        try:
            coordinate = AuthorityCoordinate.bind_path(
                rembg_paths[name],
                label=f"requested RMBG coordinate for {name}",
            )
        except AuthorityCoordinateError as exc:
            raise ValueError(str(exc)) from exc
        path = coordinate.resolved
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"RMBG model file is missing or blank: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"RMBG {name} SHA256 mismatch: expected {expected}, got {actual}")
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
    expected_rembg_files: dict[str, str] = REMBG_FILES,
) -> argparse.Namespace:
    requested_image = Path(args.image).resolve()
    requested_dino = Path(args.dinov3_model_path).resolve()
    requested_rembg = {
        name: AuthorityCoordinate.bind_path(
            (
                str(getattr(args, REMBG_FILE_ARGUMENTS[name]))
                if isinstance(getattr(args, REMBG_FILE_ARGUMENTS[name]), Path)
                else getattr(args, REMBG_FILE_ARGUMENTS[name])
            ),
            label=f"requested RMBG coordinate for {name}",
        )
        for name in expected_rembg_files
    }
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
        "rembg": {
            "repository": REMBG_REPOSITORY,
            "revision": REMBG_REVISION,
            "files": {
                name: {
                    "path": str(requested_rembg[name].resolved),
                    "declared_path": requested_rembg[name].raw,
                    "sha256": expected,
                    "size_bytes": requested_rembg[name].resolved.stat().st_size,
                }
                for name, expected in expected_rembg_files.items()
            },
        },
    }
    try:
        admitted_image = custody_root / "image" / requested_image.name
        image_record = _copy_admitted_file(
            requested_image, admitted_image, args.expected_image_sha256
        )
        admitted_dino = custody_root / "dinov3"
        dino_records = {
            name: _copy_admitted_file(
                requested_dino / name, admitted_dino / name, expected
            )
            for name, expected in expected_dinov3_files.items()
        }
        admitted_rembg = custody_root / "rembg"
        rembg_records = {
            name: _copy_admitted_file(
                requested_rembg[name].resolved, admitted_rembg / name, expected
            )
            for name, expected in expected_rembg_files.items()
        }
        report["requested_inputs"] = requested
        report["effective_inputs"] = {
            "run_id": args.run_id,
            "image": image_record,
            "dinov3": {
                "path": str(admitted_dino),
                "files": dino_records,
            },
            "rembg": {
                "path": str(admitted_rembg),
                "repository": REMBG_REPOSITORY,
                "revision": REMBG_REVISION,
                "files": rembg_records,
            },
        }
        admitted_args = argparse.Namespace(**vars(args))
        admitted_args.image = admitted_image
        admitted_args.dinov3_model_path = admitted_dino
        admitted_args.rembg_model_path = admitted_rembg
        for name, attribute in REMBG_FILE_ARGUMENTS.items():
            setattr(admitted_args, attribute, admitted_rembg / name)
        return admitted_args
    except BaseException:
        shutil.rmtree(custody_root, ignore_errors=True)
        raise


def verify_admitted_inputs_before_use(
    args: argparse.Namespace,
    report: dict[str, Any],
    *,
    expected_dinov3_files: dict[str, str] = DINOV3_FILES,
    expected_rembg_files: dict[str, str] = REMBG_FILES,
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
    for name, expected in expected_rembg_files.items():
        checks.append(
            (
                name,
                Path(requested.get("rembg", {}).get("files", {}).get(name, {}).get("path", "")),
                Path(args.rembg_model_path) / name,
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


def _install_pinned_xformers(
    *,
    python: str,
    work_dir: Path,
    report: dict[str, Any],
    wheel_url: str = XFORMERS_WHEEL_URL,
    expected_sha256: str = XFORMERS_WHEEL_SHA256,
    downloader: Callable[[str, str], Any] = urlretrieve,
    runner: Callable[..., dict[str, Any]] = _run,
) -> dict[str, Any]:
    wheel_dir = Path(work_dir).resolve() / "pinned-wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / XFORMERS_WHEEL_FILENAME
    temporary = wheel_path.with_name(f".{wheel_path.name}.downloading")
    temporary.unlink(missing_ok=True)
    try:
        downloader(wheel_url, str(temporary))
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("xformers wheel download is missing or blank")
        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "xformers wheel digest mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        os.replace(temporary, wheel_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    pip_version_receipt = runner(
        [python, "-m", "pip", "--disable-pip-version-check", "--version"]
    )
    pip_version = str(pip_version_receipt.get("stdout", "")).strip()
    if not pip_version.startswith("pip "):
        raise RuntimeError(f"effective pip identity is invalid: {pip_version!r}")
    report["setup_commands"].append(pip_version_receipt)

    command = [
        python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--force-reinstall",
        "--no-deps",
        str(wheel_path),
    ]
    report["setup_commands"].append(runner(command))
    record = {
        "version": EXPECTED_XFORMERS_VERSION,
        "url": wheel_url,
        "path": str(wheel_path),
        "sha256": expected_sha256,
        "size_bytes": wheel_path.stat().st_size,
        "install_mode": "forced-local-wheel-no-deps",
        "pip_version": pip_version,
    }
    report["xformers_wheel"] = record
    return record


def read_xformers_build_identity(xformers: Any) -> dict[str, Any]:
    version = str(getattr(xformers, "__version__", ""))
    if version != EXPECTED_XFORMERS_VERSION:
        raise RuntimeError(
            f"xformers version drift: expected {EXPECTED_XFORMERS_VERSION}, got {version}"
        )
    module_path = Path(getattr(xformers, "__file__", "")).resolve()
    identity_path = module_path.parent / "cpp_lib.json"
    if not identity_path.is_file():
        raise RuntimeError(f"xformers build identity is missing: {identity_path}")
    try:
        payload = json.loads(identity_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"xformers build identity is invalid: {exc}") from exc
    build_version = payload.get("version", {})
    environment = payload.get("env", {})
    torch_version = build_version.get("torch")
    cuda_version = build_version.get("cuda")
    arch_list = environment.get("TORCH_CUDA_ARCH_LIST")
    package_from = environment.get("XFORMERS_PACKAGE_FROM")
    if torch_version != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"xformers build Torch drift: expected {EXPECTED_TORCH_VERSION}, got {torch_version}"
        )
    if cuda_version != 1208:
        raise RuntimeError(f"xformers build CUDA drift: expected 1208, got {cuda_version}")
    if not isinstance(arch_list, str) or "7.5" not in arch_list.split():
        raise RuntimeError(f"xformers build does not admit Tesla T4 SM75: {arch_list!r}")
    if package_from != f"wheel-v{EXPECTED_XFORMERS_VERSION}":
        raise RuntimeError(f"xformers package origin drift: {package_from!r}")
    return {
        "version": version,
        "torch": torch_version,
        "cuda": cuda_version,
        "torch_cuda_arch_list": arch_list,
        "package_from": package_from,
    }


def _module_file(module: Any, label: str) -> Path:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"effective import {label} has no module file")
    return Path(value).resolve()


def _direct_url_file_path(payload: Any) -> Path | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("url")
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return None
    return Path(unquote(parsed.path)).resolve()


def _direct_url_archive_sha256(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    archive_info = payload.get("archive_info")
    if not isinstance(archive_info, dict):
        return None
    hashes = archive_info.get("hashes")
    if isinstance(hashes, dict) and isinstance(hashes.get("sha256"), str):
        return hashes["sha256"]
    legacy = archive_info.get("hash")
    if isinstance(legacy, str) and legacy.startswith("sha256="):
        return legacy.removeprefix("sha256=")
    return None


def read_xformers_install_provenance(
    xformers: Any,
    xformers_ops: Any,
    *,
    wheel_path: Path,
    expected_sha256: str = XFORMERS_WHEEL_SHA256,
    distribution_loader: Callable[[str], Any] = importlib_metadata.distribution,
) -> dict[str, Any]:
    wheel_path = Path(wheel_path).resolve()
    if not wheel_path.is_file() or sha256_file(wheel_path) != expected_sha256:
        raise RuntimeError("installed xformers wheel bytes no longer match the admitted digest")
    try:
        distribution = distribution_loader("xformers")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError("installed xformers distribution is missing") from exc
    distribution_name = str(distribution.metadata.get("Name", ""))
    if distribution_name.lower().replace("_", "-") != "xformers":
        raise RuntimeError(f"xformers distribution ownership drift: {distribution_name!r}")
    if str(distribution.version) != EXPECTED_XFORMERS_VERSION:
        raise RuntimeError(f"xformers distribution version drift: {distribution.version!r}")
    try:
        direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"xformers direct-install provenance is invalid: {exc}") from exc
    if _direct_url_file_path(direct_url) != wheel_path:
        raise RuntimeError("xformers direct-install wheel path is disconnected")
    if _direct_url_archive_sha256(direct_url) != expected_sha256:
        raise RuntimeError("xformers direct-install wheel digest is disconnected")

    distribution_root = Path(distribution.locate_file("")).resolve()
    modules = {"xformers": xformers, "xformers.ops": xformers_ops}
    module_paths = {name: _module_file(module, name) for name, module in modules.items()}
    distribution_files: dict[str, str] = {}
    for name, module_path in module_paths.items():
        for relative_value in distribution.files or ():
            relative = Path(str(relative_value))
            if relative.is_absolute() or ".." in relative.parts:
                continue
            if Path(distribution.locate_file(relative_value)).resolve() == module_path:
                distribution_files[name] = str(relative)
                break
        if name not in distribution_files:
            raise RuntimeError(f"effective import {name} is not owned by xformers distribution")
    return {
        "mode": "pep610-local-wheel",
        "wheel_path": str(wheel_path),
        "wheel_sha256": expected_sha256,
        "distribution_name": distribution_name,
        "distribution_version": str(distribution.version),
        "distribution_root": str(distribution_root),
        "module_paths": {name: str(path) for name, path in module_paths.items()},
        "distribution_files": distribution_files,
        "direct_url": direct_url,
    }


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
    _install_pinned_xformers(
        python=python,
        work_dir=work_dir,
        report=report,
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


def _selective_model_patterns(family: str) -> tuple[str, ...]:
    return tuple(MODEL_BLOB_MANIFEST[family]["files"])


def _verify_and_link_mounted_models(
    *,
    blob_root: Path,
    work_dir: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    blob_root = Path(blob_root).resolve(strict=True)
    assets_root = work_dir / "pinned-model-assets"
    if assets_root.exists():
        shutil.rmtree(assets_root)
    records: dict[str, Any] = {}
    mounted_blob_bytes = 0
    snapshots: dict[str, Path] = {}
    for family, family_manifest in MODEL_BLOB_MANIFEST.items():
        snapshot = assets_root / family
        snapshots[family] = snapshot
        family_records = {}
        blob_dir = blob_root / family_manifest["cache_dir"] / "blobs"
        for coordinate, expected in family_manifest["files"].items():
            source = blob_dir / expected["blob"]
            try:
                source_resolved = source.resolve(strict=True)
                size_bytes = source_resolved.stat().st_size
                actual_sha256 = sha256_file(source_resolved)
            except OSError as exc:
                raise RuntimeError(
                    f"mounted model blob is missing for {family}:{coordinate}: {source}"
                ) from exc
            if (
                size_bytes != expected["size_bytes"]
                or actual_sha256 != expected["sha256"]
            ):
                raise RuntimeError(
                    "mounted model blob identity mismatch for "
                    f"{family}:{coordinate}: expected sha256={expected['sha256']} "
                    f"size={expected['size_bytes']}, got sha256={actual_sha256} "
                    f"size={size_bytes}"
                )
            destination = snapshot / coordinate
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source_resolved)
            family_records[coordinate] = {
                "source_path": str(source),
                "resolved_source_path": str(source_resolved),
                "linked_path": str(destination),
                "blob": expected["blob"],
                "sha256": actual_sha256,
                "size_bytes": size_bytes,
            }
            mounted_blob_bytes += size_bytes
        records[family] = family_records
    return snapshots, {
        "source_mode": "mounted-kernel-output",
        "source_root": str(blob_root),
        "mounted_blob_bytes": mounted_blob_bytes,
        "writable_model_bytes": 0,
        "files": records,
    }


def _download_model_snapshots(
    *,
    work_dir: Path,
    report: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    from huggingface_hub import snapshot_download

    usage = shutil.disk_usage(work_dir)
    required_bytes = MODEL_REQUIRED_BYTES + MODEL_OUTPUT_RESERVE_BYTES
    admission = {
        "source_mode": "selective-snapshot-download",
        "free_bytes": usage.free,
        "model_required_bytes": MODEL_REQUIRED_BYTES,
        "output_reserve_bytes": MODEL_OUTPUT_RESERVE_BYTES,
        "required_bytes": required_bytes,
    }
    report["model_storage_admission"] = admission
    if usage.free < required_bytes:
        raise RuntimeError(
            "insufficient writable storage for pinned model acquisition and output "
            f"reserve: free={usage.free}, required={required_bytes}, "
            f"models={MODEL_REQUIRED_BYTES}, outputs={MODEL_OUTPUT_RESERVE_BYTES}"
        )
    hf_cache = work_dir / "huggingface"
    snapshots = {
        "trellis": Path(
            snapshot_download(
                repo_id=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                cache_dir=hf_cache,
                allow_patterns=list(_selective_model_patterns("trellis")),
            )
        ).resolve(),
        "sparse_decoder": Path(
            snapshot_download(
                repo_id=SPARSE_DECODER_REPOSITORY,
                revision=SPARSE_DECODER_REVISION,
                cache_dir=hf_cache,
                allow_patterns=list(_selective_model_patterns("sparse_decoder")),
            )
        ).resolve(),
    }
    return snapshots, admission


def prepare_model_view(args: argparse.Namespace, report: dict[str, Any]) -> Path:

    verify_admitted_inputs_before_use(args, report)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    model_blob_root = getattr(args, "model_blob_root", None)
    model_source_kernel = getattr(args, "model_source_kernel", None)
    if (model_blob_root is None) != (model_source_kernel is None):
        raise RuntimeError(
            "mounted model source requires both model blob root and source kernel"
        )
    if model_blob_root is not None:
        snapshots, source_record = _verify_and_link_mounted_models(
            blob_root=Path(model_blob_root),
            work_dir=work_dir,
        )
        source_record["source_kernel"] = model_source_kernel
        report["model_storage_admission"] = {
            **source_record,
            "free_bytes": shutil.disk_usage(work_dir).free,
            "output_reserve_bytes": MODEL_OUTPUT_RESERVE_BYTES,
        }
    else:
        snapshots, source_record = _download_model_snapshots(
            work_dir=work_dir,
            report=report,
        )
    snapshots.update(
        rembg=Path(args.rembg_model_path).resolve(),
        dinov3=Path(args.dinov3_model_path).resolve(),
    )
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
        **source_record,
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
            "files": dict(REMBG_FILES),
            "source": "admitted-local-files",
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


def _normalize_child_model_source_authority(
    requested_route: dict[str, Any],
    model_assets: dict[str, Any] | None,
    model_storage_admission: dict[str, Any] | None,
) -> tuple[str, str] | None:
    if not isinstance(requested_route, dict):
        raise ValueError("mounted model requested route is invalid")
    for label, record in (
        ("assets", model_assets),
        ("storage admission", model_storage_admission),
    ):
        if record is not None and not isinstance(record, dict):
            raise ValueError(f"mounted model {label} authority is invalid")

    assets = model_assets or {}
    storage = model_storage_admission or {}
    requested_root = requested_route.get("model_blob_root")
    requested_kernel = requested_route.get("model_source_kernel")
    assets_source = (
        assets.get("source_mode"),
        assets.get("source_root"),
        assets.get("source_kernel"),
    )
    storage_source = (
        storage.get("source_mode"),
        storage.get("source_root"),
        storage.get("source_kernel"),
    )

    if requested_root is None and requested_kernel is None:
        for label, (mode, root, kernel) in (
            ("assets", assets_source),
            ("storage admission", storage_source),
        ):
            if (
                mode not in (None, "selective-snapshot-download")
                or root is not None
                or kernel is not None
            ):
                raise ValueError(
                    f"mounted model {label} authority contradicts generic route"
                )
        return None

    if not isinstance(requested_root, str) or not Path(requested_root).is_absolute():
        raise ValueError("requested mounted model blob root is missing or non-absolute")
    if (
        not isinstance(requested_kernel, str)
        or len(requested_kernel.split("/")) != 2
        or not all(requested_kernel.split("/"))
    ):
        raise ValueError("requested mounted model source kernel identity is invalid")
    expected_source = ("mounted-kernel-output", requested_root, requested_kernel)
    if assets_source != expected_source:
        raise ValueError("mounted model assets source authority mismatch")
    if storage_source != expected_source:
        raise ValueError("mounted model storage admission source authority mismatch")
    return requested_root, requested_kernel


def _validate_mounted_model_authority(
    requested_route: dict[str, Any],
    report: dict[str, Any],
    model_assets: dict[str, Any],
) -> None:
    authority = _normalize_child_model_source_authority(
        requested_route,
        model_assets,
        report.get("model_storage_admission"),
    )
    if authority is None:
        return
    requested_root, requested_kernel = authority
    if model_assets.get("mounted_blob_bytes") != MODEL_REQUIRED_BYTES:
        raise ValueError("mounted model assets byte total mismatch")
    if model_assets.get("writable_model_bytes") != 0:
        raise ValueError("mounted model assets wrote model payload bytes")
    families = model_assets.get("files")
    if not isinstance(families, dict) or set(families) != set(MODEL_BLOB_MANIFEST):
        raise ValueError("mounted model assets family set is incomplete")
    for family, expected_family in MODEL_BLOB_MANIFEST.items():
        actual_files = families.get(family)
        expected_files = expected_family["files"]
        if not isinstance(actual_files, dict) or set(actual_files) != set(expected_files):
            raise ValueError(f"mounted model assets file set is incomplete for {family}")
        for coordinate, expected in expected_files.items():
            actual = actual_files[coordinate]
            if not isinstance(actual, dict) or any(
                actual.get(field) != expected[field]
                for field in ("blob", "sha256", "size_bytes")
            ):
                raise ValueError(
                    f"mounted model blob identity mismatch for {family}:{coordinate}"
                )
    storage = report.get("model_storage_admission")
    if not isinstance(storage, dict):
        raise ValueError("mounted model storage admission is missing")
    for field, expected in (
        ("mounted_blob_bytes", MODEL_REQUIRED_BYTES),
        ("writable_model_bytes", 0),
        ("output_reserve_bytes", MODEL_OUTPUT_RESERVE_BYTES),
    ):
        if storage.get(field) != expected:
            raise ValueError(f"mounted model storage admission {field} mismatch")
    free_bytes = storage.get("free_bytes")
    if type(free_bytes) is not int or free_bytes < MODEL_OUTPUT_RESERVE_BYTES:
        raise ValueError("mounted model storage admission does not preserve output reserve")


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
        expected_capture_order: tuple[str, ...] = EXPECTED_CAPTURE_ORDER,
        run_id: str | None = None,
        on_capture: Callable[["ArtifactRecorder"], None] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.capture_order: list[str] = []
        self.expected_capture_order = tuple(expected_capture_order)
        self.run_id = run_id
        self.on_capture = on_capture

    def _next_path(self, stage: str, suffix: str) -> Path:
        if stage in self.artifacts:
            raise RuntimeError(f"stage captured more than once: {stage}")
        expected = self.expected_capture_order[len(self.capture_order)]
        if stage != expected:
            raise RuntimeError(f"capture order mismatch: expected {expected}, got {stage}")
        expected_name = EXPECTED_ARTIFACT_FILENAMES[stage]
        if Path(expected_name).suffix != suffix:
            raise RuntimeError(f"artifact suffix mismatch for {stage}: {suffix}")
        return self.output_dir / expected_name

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
    def __init__(
        self,
        delegate: Any,
        stage: str,
        recorder: ArtifactRecorder,
        steps: int,
        *,
        capture: bool,
    ):
        self.delegate = delegate
        self.stage = stage
        self.recorder = recorder
        self.steps = steps
        self.capture = capture
        self.call_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def sample(self, model: Any, noise: Any, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        if self.call_count != 1:
            raise RuntimeError(f"{self.stage} sampler called more than once")
        result = self.delegate.sample(model, noise, *args, **kwargs)
        if not self.capture:
            return result
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
    capture_stages = set(recorder.expected_capture_order)
    sparse_observer = SamplerObserver(
        pipeline.sparse_structure_sampler,
        "sparse_flow",
        recorder,
        steps,
        capture="sparse_flow" in capture_stages,
    )
    shape_observer = SamplerObserver(
        pipeline.shape_slat_sampler,
        "shape_flow",
        recorder,
        steps,
        capture="shape_flow" in capture_stages,
    )
    texture_observer = SamplerObserver(
        pipeline.tex_slat_sampler,
        "texture_flow",
        recorder,
        steps,
        capture="texture_flow" in capture_stages,
    )
    pipeline.sparse_structure_sampler = sparse_observer
    pipeline.shape_slat_sampler = shape_observer
    pipeline.tex_slat_sampler = texture_observer

    original_preprocess = pipeline.preprocess_image

    def preprocess(self: Any, image: Any) -> Any:
        output = original_preprocess(image)
        if "preprocessed_image" in capture_stages:
            recorder.save_image("preprocessed_image", output)
        return output

    _bind_method(pipeline, "preprocess_image", preprocess)

    original_get_cond = pipeline.get_cond

    def get_cond(self: Any, image: Any, resolution: int, include_neg_cond: bool = True) -> Any:
        output = original_get_cond(image, resolution, include_neg_cond)
        if resolution != 512:
            raise RuntimeError(f"unexpected conditioning resolution on 512 route: {resolution}")
        if "conditioning_512" in capture_stages:
            recorder.save_npz(
                "conditioning_512",
                {key: _as_numpy(value) for key, value in output.items()},
            )
        return output

    _bind_method(pipeline, "get_cond", get_cond)

    original_sparse = pipeline.sample_sparse_structure

    def sample_sparse(self: Any, *args: Any, **kwargs: Any) -> Any:
        coords = original_sparse(*args, **kwargs)
        if "sparse_support" in capture_stages:
            recorder.save_npz("sparse_support", {"coords": coords})
        return coords

    _bind_method(pipeline, "sample_sparse_structure", sample_sparse)

    original_shape = pipeline.sample_shape_slat

    def sample_shape(self: Any, *args: Any, **kwargs: Any) -> Any:
        slat = original_shape(*args, **kwargs)
        if "shape_slat" in capture_stages:
            recorder.save_npz("shape_slat", _sparse_arrays("shape_slat", slat))
        return slat

    _bind_method(pipeline, "sample_shape_slat", sample_shape)

    original_decode_shape = pipeline.decode_shape_slat

    def decode_shape(self: Any, *args: Any, **kwargs: Any) -> Any:
        meshes, subs = original_decode_shape(*args, **kwargs)
        if len(meshes) != 1:
            raise RuntimeError(f"expected one decoded mesh, got {len(meshes)}")
        mesh = meshes[0]
        if "decoder_raw_mesh" in capture_stages:
            recorder.save_npz(
                "decoder_raw_mesh",
                {"vertices": mesh.vertices, "faces": mesh.faces},
            )
        return meshes, subs

    _bind_method(pipeline, "decode_shape_slat", decode_shape)

    original_decode_tex = pipeline.decode_tex_slat

    def decode_tex(self: Any, *args: Any, **kwargs: Any) -> Any:
        voxels = original_decode_tex(*args, **kwargs)
        if "texture_voxels" in capture_stages:
            recorder.save_npz("texture_voxels", _sparse_arrays("texture_voxels", voxels))
        return voxels

    _bind_method(pipeline, "decode_tex_slat", decode_tex)

    original_decode_latent = pipeline.decode_latent

    def decode_latent(self: Any, *args: Any, **kwargs: Any) -> Any:
        meshes = original_decode_latent(*args, **kwargs)
        if len(meshes) != 1:
            raise RuntimeError(f"expected one pipeline mesh, got {len(meshes)}")
        mesh = meshes[0]
        if "pipeline_filled_mesh" in capture_stages:
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


def _record_storage_snapshot(
    args: argparse.Namespace,
    report: dict[str, Any],
    *,
    label: str,
) -> None:
    work_dir = Path(args.work_dir).resolve()
    usage = shutil.disk_usage(work_dir)
    report.setdefault("storage_snapshots", []).append(
        {
            "label": label,
            "path": str(work_dir),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "observed_at": time.time(),
        }
    )


def run_live(args: argparse.Namespace, report: dict[str, Any]) -> None:
    output_dir = Path(args.output_dir).resolve()
    expected_capture_order = capture_order_for_profile(args.capture_profile)
    verify_admitted_inputs_before_use(args, report)
    phase_started = time.perf_counter()
    report["phase"] = "prepare_runtime"
    _record_storage_snapshot(args, report, label="before_prepare_runtime")
    _atomic_write_json(output_dir / "report.json", report)
    roots = prepare_runtime(args, report)
    report.setdefault("phase_timings", {})["prepare_runtime"] = time.perf_counter() - phase_started
    _record_storage_snapshot(args, report, label="after_prepare_runtime")
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
    import xformers.ops as xformers_ops
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
        "xformers_build_identity": read_xformers_build_identity(xformers),
        "xformers_import_provenance": read_xformers_install_provenance(
            xformers,
            xformers_ops,
            wheel_path=Path(report["xformers_wheel"]["path"]),
            expected_sha256=report["xformers_wheel"]["sha256"],
        ),
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
    _record_storage_snapshot(args, report, label="after_runtime_identity")
    _atomic_write_json(output_dir / "report.json", report)

    report["phase"] = "load_pipeline"
    phase_started = time.perf_counter()
    _record_storage_snapshot(args, report, label="before_model_view")
    _atomic_write_json(output_dir / "report.json", report)
    model_view = prepare_model_view(args, report)
    _record_storage_snapshot(args, report, label="after_model_view")
    _atomic_write_json(output_dir / "report.json", report)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(model_view))
    pipeline.cuda()
    report["phase_timings"]["load_pipeline"] = time.perf_counter() - phase_started

    def publish_capture(current: ArtifactRecorder) -> None:
        report["capture_order"] = list(current.capture_order)
        report["artifacts"] = dict(current.artifacts)
        report["last_trustworthy_phase"] = f"{current.capture_order[-1]}_captured"
        _atomic_write_json(output_dir / "report.json", report)

    recorder = ArtifactRecorder(
        output_dir,
        expected_capture_order=expected_capture_order,
        run_id=args.run_id,
        on_capture=publish_capture,
    )
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

    glb_path = output_dir / EXPECTED_ARTIFACT_FILENAMES["consumer_glb"]
    tmp_glb = glb_path.with_name(f".{glb_path.name}.tmp.glb")
    tmp_glb.unlink(missing_ok=True)
    glb.export(tmp_glb, extension_webp=True)
    if not tmp_glb.is_file() or tmp_glb.stat().st_size <= 0:
        raise RuntimeError("official GLB export produced no artifact")
    os.replace(tmp_glb, glb_path)
    recorder.register_glb("consumer_glb", glb_path)
    torch.cuda.synchronize()
    report["phase_timings"]["official_to_glb"] = time.perf_counter() - phase_started

    if tuple(recorder.capture_order) != expected_capture_order:
        raise RuntimeError(
            f"capture order mismatch: expected {expected_capture_order}, got {recorder.capture_order}"
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
    if (
        requested_route.get("rembg_repository") != REMBG_REPOSITORY
        or requested_route.get("rembg_revision") != REMBG_REVISION
    ):
        raise ValueError("requested RMBG route repository or revision identity mismatch")
    requested_route_files = requested_route.get("rembg_files")
    if not isinstance(requested_route_files, dict) or set(requested_route_files) != set(REMBG_FILES):
        raise ValueError("requested RMBG route coordinate set is incomplete")

    def canonical_requested_rembg_path(value: Any, name: str) -> PurePosixPath:
        try:
            return AuthorityCoordinate.bind_absolute(
                value,
                label=f"requested RMBG route coordinate for {name}",
            ).lexical
        except AuthorityCoordinateError as exc:
            raise ValueError(str(exc)) from exc

    canonical_requested_route_files = {
        name: canonical_requested_rembg_path(path, name)
        for name, path in requested_route_files.items()
    }
    if len(set(canonical_requested_route_files.values())) != len(REMBG_FILES):
        raise ValueError("requested RMBG route coordinates must be distinct")

    requested_inputs = report.get("requested_inputs")
    if not isinstance(requested_inputs, dict):
        raise ValueError("requested inputs authority is missing")
    requested_rembg = requested_inputs.get("rembg")
    if not isinstance(requested_rembg, dict):
        raise ValueError("requested RMBG input authority is missing")
    if (
        requested_rembg.get("repository") != REMBG_REPOSITORY
        or requested_rembg.get("revision") != REMBG_REVISION
    ):
        raise ValueError("requested RMBG repository or revision identity mismatch")
    requested_rembg_files = requested_rembg.get("files")
    if not isinstance(requested_rembg_files, dict) or set(requested_rembg_files) != set(REMBG_FILES):
        raise ValueError("requested RMBG identity set is incomplete")

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
    rembg = effective_inputs.get("rembg", {})
    if (
        rembg.get("repository") != REMBG_REPOSITORY
        or rembg.get("revision") != REMBG_REVISION
    ):
        raise ValueError("effective RMBG repository or revision identity mismatch")
    rembg_files = rembg.get("files", {})
    if set(rembg_files) != set(REMBG_FILES):
        raise ValueError("effective RMBG identity set is incomplete")
    rembg_file_coordinates: dict[str, AuthorityCoordinate] = {}
    for name in REMBG_FILES:
        record = rembg_files[name]
        try:
            rembg_file_coordinates[name] = AuthorityCoordinate.bind_absolute(
                record.get("path") if isinstance(record, dict) else None,
                label=f"effective RMBG file coordinate for {name}",
            )
        except AuthorityCoordinateError as exc:
            raise ValueError(str(exc)) from exc
    declared_roots = {
        coordinate.lexical.parent.as_posix()
        for coordinate in rembg_file_coordinates.values()
    }
    if len(declared_roots) != 1:
        raise ValueError("effective RMBG files do not declare one canonical root")
    declared_root = declared_roots.pop()
    try:
        rembg_coordinate = AuthorityCoordinate.bind_absolute(
            rembg.get("path"),
            label="effective RMBG run-custody path",
            expected_raw=declared_root,
        )
    except AuthorityCoordinateError as exc:
        raise ValueError(str(exc)) from exc
    rembg_root = rembg_coordinate.lexical
    if (
        not rembg_root.is_absolute()
        or rembg_root.name != "rembg"
        or rembg_root.parent.name != run_id
        or rembg_root.parent.parent.name != "admitted-inputs"
    ):
        raise ValueError("effective RMBG path is not under canonical run custody")
    for name, expected in REMBG_FILES.items():
        requested_record = requested_rembg_files[name]
        if not isinstance(requested_record, dict):
            raise ValueError(f"requested RMBG identity record is invalid for {name}")
        requested_path = requested_record.get("path")
        if (
            not isinstance(requested_path, str)
            or not requested_path
            or requested_path != requested_route_files[name]
        ):
            raise ValueError(f"requested RMBG path identity mismatch for {name}")
        canonical_requested_rembg_path(requested_path, name)
        if (
            requested_record.get("sha256") != expected
            or type(requested_record.get("size_bytes")) is not int
            or requested_record["size_bytes"] <= 0
        ):
            raise ValueError(f"requested RMBG identity mismatch for {name}")
        record = rembg_files[name]
        if (
            record.get("sha256") != expected
            or record.get("sha256") != requested_record.get("sha256")
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
            or record["size_bytes"] != requested_record["size_bytes"]
            or rembg_file_coordinates[name].raw != (rembg_root / name).as_posix()
        ):
            raise ValueError(f"effective RMBG identity mismatch for {name}")

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
    if model_assets["rembg"].get("files") != REMBG_FILES:
        raise ValueError("model assets RMBG file identity mismatch")
    if model_assets["rembg"].get("source") != "admitted-local-files":
        raise ValueError("model assets RMBG source identity mismatch")
    if model_assets["rembg"].get("snapshot_path") != rembg_coordinate.raw:
        raise ValueError("model assets RMBG snapshot path identity mismatch")
    if model_assets.get("path_rewrite_only") is not True:
        raise ValueError("model assets path rewrite identity is missing")
    _validate_mounted_model_authority(requested_route, report, model_assets)

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


def _validate_xformers_wheel_record(report: dict[str, Any]) -> None:
    record = report.get("xformers_wheel")
    expected = {
        "version": EXPECTED_XFORMERS_VERSION,
        "url": XFORMERS_WHEEL_URL,
        "sha256": XFORMERS_WHEEL_SHA256,
        "size_bytes": XFORMERS_WHEEL_SIZE_BYTES,
        "install_mode": "forced-local-wheel-no-deps",
    }
    if not isinstance(record, dict):
        raise ValueError("xformers wheel identity is missing")
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise ValueError(
                f"xformers wheel identity {field} must be {expected_value!r}"
            )
    path = record.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("xformers wheel identity path must be absolute")
    if Path(path).name != XFORMERS_WHEEL_FILENAME:
        raise ValueError(
            f"xformers wheel identity filename must be {XFORMERS_WHEEL_FILENAME!r}"
        )
    pip_version = record.get("pip_version")
    if not isinstance(pip_version, str) or not pip_version.startswith("pip "):
        raise ValueError("xformers wheel effective pip identity is missing")


def _validate_xformers_import_provenance(
    route: dict[str, Any],
    wheel_record: dict[str, Any],
    *,
    effective_module_paths: dict[str, str] | None = None,
) -> None:
    provenance = route.get("xformers_import_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("effective route xformers import provenance is missing")
    expected = {
        "mode": "pep610-local-wheel",
        "wheel_path": wheel_record["path"],
        "wheel_sha256": wheel_record["sha256"],
        "distribution_version": EXPECTED_XFORMERS_VERSION,
    }
    for field, expected_value in expected.items():
        if provenance.get(field) != expected_value:
            raise ValueError(
                f"xformers import provenance {field} must be {expected_value!r}"
            )
    distribution_name = provenance.get("distribution_name")
    if not isinstance(distribution_name, str) or distribution_name.lower().replace(
        "_", "-"
    ) != "xformers":
        raise ValueError("xformers import provenance distribution name is invalid")
    distribution_root_value = provenance.get("distribution_root")
    if not isinstance(distribution_root_value, str) or not Path(
        distribution_root_value
    ).is_absolute():
        raise ValueError("xformers import provenance distribution root is invalid")
    distribution_root = Path(distribution_root_value).resolve()
    module_paths = provenance.get("module_paths")
    distribution_files = provenance.get("distribution_files")
    expected_modules = {"xformers", "xformers.ops"}
    if not isinstance(module_paths, dict) or set(module_paths) != expected_modules:
        raise ValueError("xformers import provenance module paths are incomplete")
    if not isinstance(distribution_files, dict) or set(distribution_files) != expected_modules:
        raise ValueError("xformers import provenance distribution files are incomplete")
    for name in sorted(expected_modules):
        module_value = module_paths[name]
        relative_value = distribution_files[name]
        if not isinstance(module_value, str) or not Path(module_value).is_absolute():
            raise ValueError(f"xformers import provenance module path {name} is invalid")
        if not isinstance(relative_value, str):
            raise ValueError(f"xformers import provenance distribution file {name} is invalid")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"xformers import provenance distribution file {name} is unsafe")
        if (distribution_root / relative).resolve() != Path(module_value).resolve():
            raise ValueError(f"xformers import provenance module {name} is not distribution-owned")
    if effective_module_paths is not None:
        expected_effective = {
            "xformers": effective_module_paths.get("xformers"),
            "xformers.ops": effective_module_paths.get("xformers_ops"),
        }
        if module_paths != expected_effective:
            raise ValueError("xformers import provenance module paths are disconnected")
    direct_url = provenance.get("direct_url")
    if _direct_url_file_path(direct_url) != Path(wheel_record["path"]).resolve():
        raise ValueError("xformers import provenance wheel path is disconnected")
    if _direct_url_archive_sha256(direct_url) != wheel_record["sha256"]:
        raise ValueError("xformers import provenance wheel digest is disconnected")


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
    _validate_xformers_wheel_record(report)
    route = report.get("effective_route", {})
    _validate_xformers_import_provenance(route, report["xformers_wheel"])
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
    expected_xformers = {
        "version": EXPECTED_XFORMERS_VERSION,
        "torch": EXPECTED_TORCH_VERSION,
        "cuda": 1208,
        "package_from": f"wheel-v{EXPECTED_XFORMERS_VERSION}",
    }
    xformers_identity = route.get("xformers_build_identity")
    if not isinstance(xformers_identity, dict):
        raise ValueError("effective route xformers build identity is missing")
    for field, expected in expected_xformers.items():
        if xformers_identity.get(field) != expected:
            raise ValueError(
                f"effective route xformers build identity {field} must be {expected!r}"
            )
    arch_list = xformers_identity.get("torch_cuda_arch_list")
    if not isinstance(arch_list, str) or "7.5" not in arch_list.split():
        raise ValueError("effective route xformers build identity does not include SM75")
    capture_profile = report.get("capture_profile", "full")
    expected_capture_order = capture_order_for_profile(capture_profile)
    if report.get("expected_capture_order") != list(expected_capture_order):
        raise ValueError("report expected capture order does not match capture profile")
    if report.get("capture_order") != list(expected_capture_order):
        raise ValueError(
            f"capture order must be {list(expected_capture_order)!r}, "
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
    if set(artifacts) != set(expected_capture_order):
        raise ValueError("artifact set does not match capture order")
    recorded_paths = [
        Path(artifacts[stage].get("path", "")).as_posix()
        for stage in expected_capture_order
    ]
    if len(set(recorded_paths)) != len(recorded_paths):
        raise ValueError("artifact paths must be distinct; duplicate stage path recorded")
    resolved_paths: set[Path] = set()
    for stage in expected_capture_order:
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


def _packet_capture_contract(packet: Any) -> Any:
    from trellmlx.kaggle_cuda_witness import WitnessPacketError
    from trellmlx.native_image_to_glb_attempt import (
        AttemptSpecError,
        capture_contract_from_entrypoint_args,
    )

    try:
        return capture_contract_from_entrypoint_args(
            packet.entrypoint_args,
            packet.expected_outputs,
            context="native image-to-GLB packet",
        )
    except AttemptSpecError as exc:
        raise WitnessPacketError(str(exc)) from exc


def _packet_capture_profile(packet: Any) -> str:
    return _packet_capture_contract(packet).capture_profile


def _normalize_model_source_authority(
    packet: Any,
    report: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], str, str] | None:
    requested_route = report.get("requested_route", {})
    packet_sources = tuple(packet.kernel_sources)
    receipt_mount = receipt.get("model_source_mount")
    child_authority = _normalize_child_model_source_authority(
        requested_route,
        report.get("model_assets"),
        report.get("model_storage_admission"),
    )

    if not packet_sources and child_authority is None and receipt_mount is None:
        return None

    if len(packet_sources) != 1:
        raise ValueError("mounted model authority requires exactly one packet source")
    expected_kernel = packet_sources[0]
    if child_authority is None:
        raise ValueError("mounted model child report authority is missing")
    report_blob_root, report_kernel = child_authority
    if report_kernel != expected_kernel:
        raise ValueError("mounted model report kernel source mismatch")
    if not isinstance(receipt_mount, dict):
        raise ValueError("mounted model receipt authority is missing")
    if receipt_mount.get("requested_kernel_source") != expected_kernel:
        raise ValueError("mounted model receipt kernel source mismatch")
    return receipt_mount, expected_kernel, report_blob_root


def _validate_mounted_model_receipt(
    packet: Any,
    output_dir: Path,
    report: dict[str, Any],
) -> None:
    receipt_path = Path(output_dir) / "kaggle_cuda_witness_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    authority = _normalize_model_source_authority(packet, report, receipt)
    if authority is None:
        return
    mount, _expected_kernel, expected_blob_root = authority
    effective_mount_root = mount.get("effective_mount_root")
    if not isinstance(effective_mount_root, str):
        raise ValueError("mounted model receipt effective mount root is missing")
    mount_path = PurePosixPath(effective_mount_root)
    kaggle_input_root = PurePosixPath("/kaggle/input")
    if not mount_path.is_absolute() or mount_path.parent != kaggle_input_root:
        raise ValueError("mounted model receipt effective mount root is invalid")
    derived_blob_root = str(mount_path / "runtime" / "huggingface")
    if mount.get("effective_blob_root") != derived_blob_root:
        raise ValueError("mounted model receipt blob root is not derived from mount root")
    if derived_blob_root != expected_blob_root:
        raise ValueError("mounted model receipt blob root mismatch")
    if mount.get("marker") != MODEL_SOURCE_MARKER:
        raise ValueError("mounted model receipt marker coordinate mismatch")
    if mount.get("candidate_count") != 1:
        raise ValueError("mounted model receipt source selection is ambiguous")
    marker_sha256 = mount.get("marker_sha256")
    if marker_sha256 != MODEL_PIPELINE_SHA256:
        raise ValueError("mounted model receipt pipeline marker mismatch")


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
    capture_contract = _packet_capture_contract(packet)
    capture_profile = capture_contract.capture_profile
    expected_stage_outputs = tuple(
        EXPECTED_ARTIFACT_FILENAMES[stage]
        for stage in capture_order_for_profile(capture_profile)
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
    _validate_mounted_model_receipt(packet, output_dir, report)
    if report.get("capture_profile", "full") != capture_profile:
        raise WitnessPacketError("downloaded report capture profile mismatch")
    return {"downloaded_outputs": records, "report": report}


def prepare_native_image_to_glb_packet(packet: Any) -> Any:
    """Prepare one native image-to-GLB attempt with precommitted authority."""
    from trellmlx.kaggle_cuda_witness import WitnessPacketError, prepare_packet

    if packet.run_id is None or packet.expected_image_sha256 is None:
        raise WitnessPacketError(
            "native image-to-GLB packet requires run identity and image identity"
        )

    capture_contract = _packet_capture_contract(packet)
    capture_profile = capture_contract.capture_profile
    expected_stage_outputs = tuple(
        EXPECTED_ARTIFACT_FILENAMES[stage]
        for stage in capture_order_for_profile(capture_profile)
    )

    capsule_root = Path(packet.capsule_dir).resolve()
    requested_output = Path(packet.output_dir).resolve()
    if (
        capsule_root == requested_output
        or capsule_root in requested_output.parents
        or requested_output in capsule_root.parents
    ):
        raise WitnessPacketError(
            "native image-to-GLB packet capsule and output topology overlaps"
        )

    attempt_payload = None
    attempt_bytes: bytes | None = None
    attempt_sha256: str | None = None
    manifest_declares_outputs = False
    if packet.attempt_manifest is not None:
        from trellmlx.native_image_to_glb_attempt import (
            AttemptSpecError,
            capture_contract_from_manifest,
            load_attempt_manifest_bytes,
            validate_attempt_manifest,
        )

        attempt_path = capsule_root / packet.attempt_manifest
        try:
            attempt_bytes = attempt_path.read_bytes()
            attempt_sha256 = hashlib.sha256(attempt_bytes).hexdigest()
            attempt_payload = load_attempt_manifest_bytes(attempt_bytes)
            validate_attempt_manifest(packet, attempt_payload)
            manifest_declares_outputs = not capture_contract_from_manifest(
                attempt_payload
            ).profile_binds_outputs
        except (OSError, AttemptSpecError) as exc:
            raise WitnessPacketError(f"native attempt manifest rejected: {exc}") from exc

    if not manifest_declares_outputs and packet.expected_outputs != expected_stage_outputs:
        raise WitnessPacketError(
            "native image-to-GLB packet outputs do not match capture profile"
        )

    authority_helper_name = "witness_authority.py"
    if packet.inputs.count(authority_helper_name) != 1:
        raise WitnessPacketError(
            "native image-to-GLB packet requires exactly one authority helper input"
        )
    authority_helper = (capsule_root / authority_helper_name).resolve()
    expected_authority_helper = Path(witness_authority_module.__file__).resolve()
    if (
        authority_helper.parent != capsule_root
        or not authority_helper.is_file()
        or authority_helper.stat().st_size <= 0
        or sha256_file(authority_helper) != sha256_file(expected_authority_helper)
    ):
        raise WitnessPacketError(
            "native image-to-GLB packet authority helper is missing or does not match the reviewed implementation"
        )

    try:
        with redirect_stderr(io.StringIO()):
            build_parser().parse_args(packet.entrypoint_args)
    except SystemExit as exc:
        raise WitnessPacketError(
            "native image-to-GLB packet entrypoint arguments are invalid"
        ) from exc

    def required_argument(flag: str) -> str:
        values: list[str] = []
        for index, argument in enumerate(packet.entrypoint_args):
            if argument.startswith(f"{flag}="):
                values.append(argument.split("=", 1)[1])
            elif argument == flag:
                if (
                    index + 1 >= len(packet.entrypoint_args)
                    or packet.entrypoint_args[index + 1].startswith("--")
                ):
                    raise WitnessPacketError(f"{flag} is missing its value")
                values.append(packet.entrypoint_args[index + 1])
        if len(values) != 1 or not values[0]:
            raise WitnessPacketError(
                f"native image-to-GLB packet requires exactly one {flag} value"
            )
        return values[0]

    def effective_kaggle_path(value: str) -> PurePosixPath:
        declared = PurePosixPath(value)
        rooted = (
            declared
            if declared.is_absolute()
            else KAGGLE_KERNEL_WORKING_DIRECTORY / declared
        )
        normalized = os.path.normpath(str(rooted))
        if normalized.startswith("//"):
            normalized = f"/{normalized.lstrip('/')}"
        return PurePosixPath(normalized)

    def required_flat_packet_input(flag: str, role: str) -> str:
        if any(argument.startswith(f"{flag}=") for argument in packet.entrypoint_args):
            raise WitnessPacketError(
                f"native image-to-GLB packet RMBG coordinate {flag} must use separate-token form"
            )
        positions = [
            index
            for index, argument in enumerate(packet.entrypoint_args)
            if argument == flag
        ]
        if len(positions) != 1:
            raise WitnessPacketError(
                f"native image-to-GLB packet requires exactly one RMBG coordinate {flag}"
            )
        index = positions[0]
        if (
            index + 1 >= len(packet.entrypoint_args)
            or not packet.entrypoint_args[index + 1]
            or packet.entrypoint_args[index + 1].startswith("-")
        ):
            raise WitnessPacketError(
                f"native image-to-GLB packet RMBG coordinate {flag} is missing its value"
            )
        value = packet.entrypoint_args[index + 1]
        coordinate = PurePosixPath(value)
        if (
            coordinate.is_absolute()
            or len(coordinate.parts) != 1
            or coordinate.name in {"", ".", ".."}
            or str(coordinate) != value
        ):
            raise WitnessPacketError(
                f"native image-to-GLB packet RMBG coordinate for {role} must be flat and relative"
            )
        if value not in packet.inputs:
            raise WitnessPacketError(
                f"native image-to-GLB packet RMBG coordinate for {role} is not a staged input"
            )
        capsule_root = Path(packet.capsule_dir).resolve()
        source = (capsule_root / value).resolve()
        if source.parent != capsule_root:
            raise WitnessPacketError(
                f"native image-to-GLB packet RMBG coordinate for {role} escapes packet custody"
            )
        if not source.is_file() or source.stat().st_size <= 0:
            raise WitnessPacketError(
                f"native image-to-GLB packet RMBG source is missing or blank for {role}: {source}"
            )
        actual = sha256_file(source)
        expected = REMBG_FILES[role]
        if actual != expected:
            raise WitnessPacketError(
                f"native image-to-GLB packet RMBG {role} SHA256 mismatch: "
                f"expected {expected}, got {actual}"
            )
        return value

    rembg_coordinates = {
        role: required_flat_packet_input(
            f"--{attribute.replace('_', '-')}", role
        )
        for role, attribute in REMBG_FILE_ARGUMENTS.items()
    }
    if len(set(rembg_coordinates.values())) != len(REMBG_FILES):
        raise WitnessPacketError(
            "native image-to-GLB packet RMBG coordinates must be distinct"
        )

    if packet.attempt_manifest is not None:
        if any(
            argument.startswith("--dinov3-model-path=")
            for argument in packet.entrypoint_args
        ):
            raise WitnessPacketError(
                "native image-to-GLB packet DINOv3 model coordinate must use separate-token form"
            )
        if required_argument("--dinov3-model-path") != ".":
            raise WitnessPacketError(
                "native image-to-GLB packet DINOv3 model coordinate must be '.'"
            )
        dinov3_sources: dict[str, Path] = {}
        for role, expected in DINOV3_FILES.items():
            if packet.inputs.count(role) != 1:
                raise WitnessPacketError(
                    f"native image-to-GLB packet requires exactly one DINOv3 {role} input"
                )
            source = (capsule_root / role).resolve()
            if source.parent != capsule_root:
                raise WitnessPacketError(
                    f"native image-to-GLB packet DINOv3 source escapes packet custody for {role}"
                )
            if not source.is_file() or source.stat().st_size <= 0:
                raise WitnessPacketError(
                    f"native image-to-GLB packet DINOv3 source is missing or blank for {role}: {source}"
                )
            actual = sha256_file(source)
            if actual != expected:
                raise WitnessPacketError(
                    f"native image-to-GLB packet DINOv3 {role} SHA256 mismatch: "
                    f"expected {expected}, got {actual}"
                )
            dinov3_sources[role] = source
        if len(set(dinov3_sources.values())) != len(DINOV3_FILES):
            raise WitnessPacketError(
                "native image-to-GLB packet DINOv3 source identities must be distinct"
            )

    output_dir = effective_kaggle_path(required_argument("--output-dir"))
    work_dir = effective_kaggle_path(required_argument("--work-dir"))
    if (
        output_dir == work_dir
        or output_dir in work_dir.parents
        or work_dir in output_dir.parents
    ):
        raise WitnessPacketError(
            "native image-to-GLB Kaggle output and work directories overlap: "
            f"output={output_dir}, work={work_dir}"
        )

    requested_output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_capsule = Path(
        tempfile.mkdtemp(
            prefix=f".{capsule_root.name}.admitted-",
            dir=capsule_root.parent,
        )
    )
    try:
        for relative_name in packet.inputs:
            relative = Path(relative_name)
            if (
                relative.is_absolute()
                or len(relative.parts) != 1
                or ".." in relative.parts
            ):
                raise WitnessPacketError(
                    f"native packet input is not flat capsule custody: {relative_name}"
                )
            source = capsule_root / relative
            destination = snapshot_capsule / relative
            if packet.attempt_manifest == relative_name and attempt_bytes is not None:
                destination.write_bytes(attempt_bytes)
            else:
                shutil.copy2(source, destination)
    except BaseException:
        shutil.rmtree(snapshot_capsule, ignore_errors=True)
        raise
    candidate_output = Path(
        tempfile.mkdtemp(
            prefix=f".{requested_output.name}.candidate-",
            dir=requested_output.parent,
        )
    )
    candidate_packet = replace(
        packet,
        capsule_dir=snapshot_capsule,
        output_dir=candidate_output,
    )
    backup_output: Path | None = None
    try:
        prepare_packet(candidate_packet)
        manifest_path = candidate_packet.dataset_dir / "witness-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise WitnessPacketError(
                "native image-to-GLB packet candidate manifest is missing or invalid"
            ) from exc
        manifest_files = manifest.get("files")
        if not isinstance(manifest_files, dict):
            raise WitnessPacketError(
                "native image-to-GLB packet candidate manifest file authority is missing"
            )
        if attempt_payload is not None:
            try:
                copied_attempt = candidate_packet.dataset_dir / packet.attempt_manifest
                copied_bytes = copied_attempt.read_bytes()
                copied_payload = load_attempt_manifest_bytes(copied_bytes)
                if (
                    copied_bytes != attempt_bytes
                    or hashlib.sha256(copied_bytes).hexdigest() != attempt_sha256
                    or copied_payload != attempt_payload
                ):
                    raise AttemptSpecError(
                        "copied attempt manifest differs from admitted byte snapshot"
                    )
                validate_attempt_manifest(
                    candidate_packet,
                    copied_payload,
                    file_records=manifest_files,
                )
            except AttemptSpecError as exc:
                raise WitnessPacketError(
                    f"native attempt manifest does not reconcile to packet manifest: {exc}"
                ) from exc
        for role, coordinate in rembg_coordinates.items():
            admitted = candidate_packet.dataset_dir / coordinate
            record = manifest_files.get(coordinate)
            if not admitted.is_file() or admitted.stat().st_size <= 0:
                raise WitnessPacketError(
                    f"native image-to-GLB packet admitted RMBG file is missing or blank for {role}"
                )
            actual_sha256 = sha256_file(admitted)
            actual_size = admitted.stat().st_size
            if actual_sha256 != REMBG_FILES[role]:
                raise WitnessPacketError(
                    f"native image-to-GLB packet admitted RMBG {role} SHA256 mismatch: "
                    f"expected {REMBG_FILES[role]}, got {actual_sha256}"
                )
            if (
                not isinstance(record, dict)
                or record.get("sha256") != actual_sha256
                or type(record.get("size_bytes")) is not int
                or record["size_bytes"] != actual_size
            ):
                raise WitnessPacketError(
                    f"native image-to-GLB packet manifest RMBG digest or size mismatch for {role}"
                )
        if packet.attempt_manifest is not None:
            for role, expected in DINOV3_FILES.items():
                admitted = candidate_packet.dataset_dir / role
                record = manifest_files.get(role)
                if not admitted.is_file() or admitted.stat().st_size <= 0:
                    raise WitnessPacketError(
                        f"native image-to-GLB packet admitted DINOv3 file is missing or blank for {role}"
                    )
                actual_sha256 = sha256_file(admitted)
                actual_size = admitted.stat().st_size
                if actual_sha256 != expected:
                    raise WitnessPacketError(
                        f"native image-to-GLB packet admitted DINOv3 {role} SHA256 mismatch: "
                        f"expected {expected}, got {actual_sha256}"
                    )
                if (
                    not isinstance(record, dict)
                    or record.get("sha256") != actual_sha256
                    or type(record.get("size_bytes")) is not int
                    or record["size_bytes"] != actual_size
                ):
                    raise WitnessPacketError(
                        f"native image-to-GLB packet manifest DINOv3 digest or size mismatch for {role}"
                    )

        if packet.attempt_manifest is not None and (
            (capsule_root / packet.attempt_manifest).read_bytes() != attempt_bytes
        ):
            raise WitnessPacketError(
                "native attempt manifest authority changed during packet preparation"
            )
        if requested_output.exists():
            backup_output = requested_output.with_name(
                f".{requested_output.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(requested_output, backup_output)
        try:
            os.replace(candidate_output, requested_output)
        except BaseException:
            if backup_output is not None:
                os.replace(backup_output, requested_output)
                backup_output = None
            raise
        if backup_output is not None:
            shutil.rmtree(backup_output)
            backup_output = None
    finally:
        shutil.rmtree(candidate_output, ignore_errors=True)
        shutil.rmtree(snapshot_capsule, ignore_errors=True)
        if backup_output is not None and not requested_output.exists():
            os.replace(backup_output, requested_output)
    return packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--expected-image-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--model-repository", default=MODEL_REPOSITORY)
    parser.add_argument("--model-blob-root", type=Path)
    parser.add_argument("--model-source-kernel")
    parser.add_argument("--dinov3-model-path", type=Path)
    parser.add_argument("--rembg-model-file")
    parser.add_argument("--rembg-config-file")
    parser.add_argument("--rembg-birefnet-file")
    parser.add_argument("--rembg-birefnet-config-file")
    parser.add_argument("--pipeline-type", default="512")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--steps", default=8, type=int)
    parser.add_argument("--target-faces", default=350000, type=int)
    parser.add_argument("--texture-size", default=1024, type=int)
    parser.add_argument("--attention-backend", default="xformers")
    parser.add_argument("--sparse-conv-backend", default="flex_gemm")
    parser.add_argument(
        "--capture-profile",
        choices=tuple(CAPTURE_PROFILES),
        default="full",
    )
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
        "capture_profile": args.capture_profile,
        "expected_capture_order": list(capture_order_for_profile(args.capture_profile)),
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

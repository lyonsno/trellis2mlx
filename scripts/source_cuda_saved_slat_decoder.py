"""Official CUDA raw shape-decoder assay for an exact saved MLX shape SLat.

This stops before mesh extraction, cleanup, texture, and final-asset export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

try:
    from scripts.source_cuda_postcond_full_decode_timing import (
        apply_sparse_backend_env, decode_shape_slat_raw, extract_source,
        install_mesh_override, parameter_count, sha256_file, sync_cuda,
        write_decoder_state_npz,
    )
except ModuleNotFoundError:
    from source_cuda_postcond_full_decode_timing import (
        apply_sparse_backend_env, decode_shape_slat_raw, extract_source,
        install_mesh_override, parameter_count, sha256_file, sync_cuda,
        write_decoder_state_npz,
    )


def validate_saved_slat(path: Path, expected_sha256: str, *, resolution: int) -> dict:
    path = Path(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected SLat SHA256 must be lowercase hexadecimal")
    if resolution not in (512, 768, 1024):
        raise ValueError("unsupported decoder resolution")
    if not path.is_file():
        raise ValueError(f"saved SLat is missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"saved SLat SHA256 mismatch: {actual_sha256}")
    with np.load(path, allow_pickle=False) as saved:
        if "feats" not in saved or "coords" not in saved:
            raise ValueError("saved SLat requires feats and coords")
        feats, coords = saved["feats"], saved["coords"]
        if feats.dtype != np.float32 or feats.ndim != 2 or feats.shape[1] != 32:
            raise ValueError("saved SLat feats must be float32[N,32]")
        if coords.dtype != np.int32 or coords.shape != (feats.shape[0], 4):
            raise ValueError("saved SLat coords must be int32[N,4]")
        if feats.shape[0] == 0 or not np.isfinite(feats).all():
            raise ValueError("saved SLat feats must be nonempty and finite")
        if not np.all(coords[:, 0] == 0):
            raise ValueError("saved SLat requires one batch at index zero")
        if np.any(coords[:, 1:] < 0) or np.any(coords[:, 1:] >= resolution):
            raise ValueError("saved SLat coordinates exceed the requested grid")
        if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
            raise ValueError("saved SLat coordinates must be unique")
        return {
            "path": str(path), "sha256": actual_sha256,
            "shape": list(feats.shape), "coords_shape": list(coords.shape),
            "feats_sha256": hashlib.sha256(feats.tobytes()).hexdigest(),
            "coords_sha256": hashlib.sha256(coords.tobytes()).hexdigest(),
            "resolution": resolution,
        }


def validate_request(path: Path, expected_sha256: str, metadata_path: Path, *, resolution: int) -> dict:
    identity = validate_saved_slat(path, expected_sha256, resolution=resolution)
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("mesh_grid_size") != resolution:
        raise ValueError("saved SLat mesh_grid_size disagrees with decoder resolution")
    if metadata.get("cascade") is not True:
        raise ValueError("saved SLat does not declare the expected cascade route")
    identity.update(metadata_path=str(metadata_path), metadata_sha256=sha256_file(metadata_path), metadata=metadata)
    return identity


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--shape-slat", required=True, type=Path)
    parser.add_argument("--shape-slat-metadata", required=True, type=Path)
    parser.add_argument("--shape-slat-sha256", required=True)
    parser.add_argument("--source-tar", required=True, type=Path)
    parser.add_argument("--source-tar-sha256", required=True)
    parser.add_argument("--mesh-override", required=True, type=Path)
    parser.add_argument("--mesh-override-sha256", required=True)
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--model-repo", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--pipeline-config", default="pipeline.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    phase = "request_validation"
    protected = {args.shape_slat.resolve(), args.shape_slat_metadata.resolve(), args.source_tar.resolve(), args.mesh_override.resolve()}
    report_path = args.output_json
    if report_path.resolve() in protected or report_path.resolve() == args.output_npz.resolve():
        report_path = report_path.with_name(f"{report_path.name}.failure.json")
    report = {
        "schema": "trellis2mlx.source_cuda_saved_slat_decoder.v1",
        "status": "failed", "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "requested_route": "official-TRELLIS.2-CUDA-saved-SLat-raw-decoder",
        "requested_resolution": args.resolution, "model_repo": args.model_repo,
        "model_revision": args.model_revision,
        "input": str(args.shape_slat), "input_expected_sha256": args.shape_slat_sha256,
        "source_tar": str(args.source_tar), "source_tar_expected_sha256": args.source_tar_sha256,
        "mesh_override": str(args.mesh_override), "mesh_override_expected_sha256": args.mesh_override_sha256,
        "output": str(args.output_npz),
        "effective_report_path": str(report_path),
        "forbidden_inferences": [
            "not source CUDA inference or sampling",
            "not mesh extraction, cleanup, texturing, or final GLB",
            "not parity until same-input MLX arrays are independently compared",
        ],
    }
    try:
        if args.output_json.resolve() == args.output_npz.resolve() or args.output_json.resolve() in protected or args.output_npz.resolve() in protected:
            raise ValueError("output aliases another output or a protected input")
        report["input_identity"] = validate_request(args.shape_slat, args.shape_slat_sha256, args.shape_slat_metadata, resolution=args.resolution)
        if not re.fullmatch(r"[0-9a-f]{64}", args.source_tar_sha256):
            raise ValueError("expected source tar SHA256 must be lowercase hexadecimal")
        actual_tar_sha256 = sha256_file(args.source_tar)
        if actual_tar_sha256 != args.source_tar_sha256:
            raise ValueError(f"source tar SHA256 mismatch: {actual_tar_sha256}")
        report["source_tar_sha256"] = actual_tar_sha256
        if not re.fullmatch(r"[0-9a-f]{64}", args.mesh_override_sha256):
            raise ValueError("expected mesh override SHA256 must be lowercase hexadecimal")
        actual_override_sha256 = sha256_file(args.mesh_override)
        if actual_override_sha256 != args.mesh_override_sha256:
            raise ValueError(f"mesh override SHA256 mismatch: {actual_override_sha256}")
        report["mesh_override_sha256"] = actual_override_sha256
        report["last_trustworthy_phase"] = phase
        _write_report(report_path, report)

        phase = "official_source_import"
        source_root = extract_source(args.source_tar, Path.cwd())
        sys.path.insert(0, str(source_root))
        report["source_root"] = str(source_root)
        report["mesh_override_install"] = install_mesh_override(source_root, args.mesh_override)
        report["sparse_backends_requested"] = apply_sparse_backend_env("none", "sdpa")
        import torch
        from huggingface_hub import snapshot_download
        from trellis2 import models as source_models
        from trellis2.modules.sparse import SparseTensor
        from trellis2.modules.sparse import config as sparse_config
        if not torch.cuda.is_available():
            raise RuntimeError("official source CUDA decoder requires a CUDA device")
        report["effective_route"] = {
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "sparse_conv_backend": getattr(sparse_config, "CONV", None),
            "sparse_attention_backend": getattr(sparse_config, "ATTN", None),
            "resolution": args.resolution, "source_tar_sha256": actual_tar_sha256,
        }
        report["last_trustworthy_phase"] = phase
        _write_report(report_path, report)

        phase = "load_official_decoder"
        if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
            raise ValueError("model revision must be a pinned 40-character commit")
        model_root = Path(snapshot_download(
            args.model_repo, revision=args.model_revision,
            allow_patterns=[args.pipeline_config, "ckpts/shape_dec_next_dc_f16c32_fp16.*"],
        ))
        config_path = model_root / args.pipeline_config
        pipeline_args = json.loads(config_path.read_text())["args"]
        model_name = pipeline_args["models"]["shape_slat_decoder"]
        if model_name != "ckpts/shape_dec_next_dc_f16c32_fp16":
            raise ValueError(f"pinned pipeline names an unexpected shape decoder: {model_name}")
        model_ref = model_root / model_name
        model_json = Path(f"{model_ref}.json")
        model_weights = Path(f"{model_ref}.safetensors")
        if not model_json.is_file() or not model_weights.is_file():
            raise ValueError("pinned shape decoder files are missing")
        decoder = source_models.from_pretrained(str(model_ref))
        decoder.eval()
        decoder.set_resolution(args.resolution)
        decoder.to(torch.device("cuda"))
        decoder.low_vram = True
        if decoder.training:
            raise RuntimeError("official decoder remained in training mode")
        report["model"] = {
            "pipeline_config_sha256": sha256_file(config_path),
            "model_ref": str(model_ref), "parameter_count": parameter_count(decoder),
            "model_config_sha256": sha256_file(model_json),
            "model_weights_sha256": sha256_file(model_weights),
            "resolution": args.resolution,
        }
        report["last_trustworthy_phase"] = phase
        _write_report(report_path, report)

        phase = "decode_saved_slat"
        with np.load(args.shape_slat, allow_pickle=False) as saved:
            feats = torch.from_numpy(np.ascontiguousarray(saved["feats"])).to("cuda")
            coords = torch.from_numpy(np.ascontiguousarray(saved["coords"])).to("cuda")
        shape_slat = SparseTensor(feats=feats, coords=coords)
        with torch.no_grad():
            decoded, subs = decode_shape_slat_raw(decoder, shape_slat)
        sync_cuda(torch)
        report["last_trustworthy_phase"] = phase
        _write_report(report_path, report)

        phase = "publish_decoder_state"
        validation = write_decoder_state_npz(args.output_npz, decoded, subs)
        report["output_identity"] = {
            "path": str(args.output_npz), "sha256": sha256_file(args.output_npz),
            "size_bytes": args.output_npz.stat().st_size, "validation": validation,
        }
        report["last_trustworthy_phase"] = phase
        report["status"] = "done"
        report["elapsed_seconds"] = time.monotonic() - started
        _write_report(report_path, report)
        return 0
    except BaseException as exc:
        report.update(status="failed", failure_phase=phase, error_type=type(exc).__name__, error_message=str(exc), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        _write_report(report_path, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

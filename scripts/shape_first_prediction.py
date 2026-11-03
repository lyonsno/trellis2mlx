#!/usr/bin/env python3
"""Compare the first 1024 HR shape-flow prediction and step-1 Euler update on identical retained inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np

MODEL_REPO = "microsoft/TRELLIS.2-4B"
MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
TRELLIS_REVISION = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
STEM = "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16"
WEIGHT_HASHES = {
    ".json": "7ebd6d367393f52fee9e92c2b3d727a9026a338549c6a6fa84c7ea122140ffdd",
    ".safetensors": "07cd0596f634c5adc1890023d16023afc5eed02fb84b22bb23aff5bf0030fbbd",
}
NATIVE_PROFILES = (
    "fp16-control",
    "production-bf16",
    "bf16-f32-modulation",
    "bf16-mlx-sum",
)
ACCEPTED_TORCH_CUDA_VERSIONS = ("2.11.0+cu128", "2.11.0+cu130")
SAMPLER_PARAMS = {
    "steps": 12,
    "guidance_strength": 7.5,
    "guidance_rescale": 0.5,
    "guidance_interval": (0.6, 1.0),
    "rescale_t": 3.0,
    "sigma_min": 1e-5,
}
STEP1_T = np.float32(1.0)
STEP1_T_PREV = np.float32(33.0 / 34.0)
STEP1_DT = np.float32(1.0 / 34.0)

FP16_CONTROL_CLAIM = (
    "First-step 1024 HR shape-flow prediction diagnostic only: same retained "
    "24578-row support/noise, same 4101x1024 conditioning and zero negative "
    "conditioning, same t=1000, and explicit float16 block computation because "
    "Tesla T4 refuses BF16 xFormers BlockDiagonalMask. Not native BF16 production "
    "parity, not full 12-step trajectory evidence, and not hosted SourceCUDA "
    "random-draw reproduction."
)
PRODUCTION_BF16_CLAIM = (
    "Native production BF16 first-step 1024 HR shape-flow prediction control on "
    "the identical retained 24578-row support/noise and 4101x1024 conditioning at "
    "t=1000. Useful for quantifying the MLX BF16 vs FP16 delta and verifying exact "
    "equivalence to retained hr_shape_euler_01.npz; not a T4 CUDA BF16 equivalence "
    "claim."
)
BF16_CONTROL_CLAIM = (
    "Native BF16 first-step 1024 HR shape-flow prediction isolation control on the "
    "identical retained 24578-row support/noise and 4101x1024 conditioning at "
    "t=1000: block Linear weights remain bfloat16 while only the named "
    "sub-operation is varied. Diagnostic attribution only; not a production "
    "default change or T4 CUDA BF16 equivalence claim."
)

STEP1_MATRIX_KEYS = (
    "sample_in",
    "pred_pos",
    "pred_neg",
    "pred_cfg",
    "x0_pos",
    "x0_cfg",
    "x0_rescaled",
    "x0_after_rescale",
    "pred_final",
    "sample_next",
)
STEP1_SCALAR_KEYS = (
    "std_pos",
    "std_cfg",
    "ratio_raw",
    "std_ratio",
    "ratio_effective",
    "t",
    "t_prev",
)


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(p)


def claim_ceiling_for(backend: str, native_profile: str) -> str:
    if backend == "mlx" and native_profile == "production-bf16":
        return PRODUCTION_BF16_CLAIM
    if backend == "mlx" and native_profile in {
        "bf16-f32-modulation",
        "bf16-mlx-sum",
    }:
        return BF16_CONTROL_CLAIM
    return FP16_CONTROL_CLAIM


def validate_output(
    pred: np.ndarray, coords: np.ndarray, expected_coords: np.ndarray
) -> None:
    if pred.dtype != np.float32 or coords.dtype != np.int32:
        raise ValueError(f"unexpected dtypes: pred={pred.dtype} coords={coords.dtype}")
    if (
        expected_coords.ndim != 2
        or expected_coords.shape[0] <= 0
        or expected_coords.shape[1] != 4
    ):
        raise ValueError(f"invalid expected_coords shape: {expected_coords.shape}")
    if pred.shape != (expected_coords.shape[0], 32):
        raise ValueError(f"prediction shape mismatch: {pred.shape}")
    if coords.shape != expected_coords.shape or not np.array_equal(
        coords, expected_coords
    ):
        raise ValueError("output coordinates do not match input support order")
    if not np.isfinite(pred).all():
        raise ValueError("nonfinite values in prediction")


def sparse_cfg_rescale_std_fp32(x0: np.ndarray) -> np.ndarray:
    """Reproduce SparseTensor.std's row-tree and serial segment reduction order in FP32."""
    x0 = np.ascontiguousarray(x0, dtype=np.float32)
    if x0.ndim != 2 or x0.shape[0] <= 0 or x0.shape[1] != 32:
        raise ValueError(f"sparse_cfg_rescale_std_fp32 expects [N,32], got {x0.shape}")
    values = x0.copy()
    squares = (values * values).astype(np.float32)
    for offset in (16, 8, 4, 2, 1):
        values[:, :offset] = (
            values[:, :offset] + values[:, offset : 2 * offset]
        ).astype(np.float32)
        squares[:, :offset] = (
            squares[:, :offset] + squares[:, offset : 2 * offset]
        ).astype(np.float32)
    inv_channels = np.float32(1.0 / 32.0)
    row_means = (values[:, 0] * inv_channels).astype(np.float32)
    row_mean2s = (squares[:, 0] * inv_channels).astype(np.float32)
    mean_sum = np.float32(0.0)
    mean2_sum = np.float32(0.0)
    for idx in range(x0.shape[0]):
        mean_sum = np.float32(mean_sum + row_means[idx])
        mean2_sum = np.float32(mean2_sum + row_mean2s[idx])
    inv_rows = np.float32(x0.shape[0])
    mean = np.float32(mean_sum / inv_rows)
    mean2 = np.float32(mean2_sum / inv_rows)
    variance = np.float32(mean2 - np.float32(mean * mean))
    return np.sqrt(variance).astype(np.float32).reshape(())


def compute_step1_euler(
    sample_in: np.ndarray,
    pred_pos: np.ndarray,
    pred_neg: np.ndarray,
    coords: np.ndarray,
) -> dict[str, np.ndarray]:
    sample = np.ascontiguousarray(sample_in, dtype=np.float32)
    pos = np.ascontiguousarray(pred_pos, dtype=np.float32)
    neg = np.ascontiguousarray(pred_neg, dtype=np.float32)
    coords_i32 = np.ascontiguousarray(coords, dtype=np.int32)
    g = np.float32(SAMPLER_PARAMS["guidance_strength"])
    r = np.float32(SAMPLER_PARAMS["guidance_rescale"])
    sigma_min = float(SAMPLER_PARAMS["sigma_min"])
    one_minus_sigma = np.float32(1.0 - sigma_min)
    coeff_val = sigma_min + (1.0 - sigma_min) * 1.0
    coeff = np.float32(coeff_val)
    inv_coeff = np.float32(1.0 / coeff_val)

    pred_cfg = (g * pos + (np.float32(1.0) - g) * neg).astype(np.float32)
    x0_pos = (one_minus_sigma * sample - coeff * pos).astype(np.float32)
    x0_cfg = (one_minus_sigma * sample - coeff * pred_cfg).astype(np.float32)
    std_pos = sparse_cfg_rescale_std_fp32(x0_pos)
    std_cfg = sparse_cfg_rescale_std_fp32(x0_cfg)
    ratio = (
        np.float32(std_pos / std_cfg).reshape(())
        if float(std_cfg) > 0.0
        else np.float32(1.0).reshape(())
    )
    x0_rescaled = (x0_cfg * ratio).astype(np.float32)
    x0_after_rescale = (
        r * x0_rescaled + (np.float32(1.0) - r) * x0_cfg
    ).astype(np.float32)
    pred_final = ((one_minus_sigma * sample - x0_after_rescale) * inv_coeff).astype(
        np.float32
    )
    sample_next = (sample - STEP1_DT * pred_final).astype(np.float32)
    return {
        "coords": coords_i32,
        "sample_in": sample,
        "pred_pos": pos,
        "pred_neg": neg,
        "pred_cfg": pred_cfg,
        "x0_pos": x0_pos,
        "x0_cfg": x0_cfg,
        "std_pos": std_pos,
        "std_cfg": std_cfg,
        "ratio_raw": ratio,
        "std_ratio": ratio,
        "ratio_effective": ratio,
        "x0_rescaled": x0_rescaled,
        "x0_after_rescale": x0_after_rescale,
        "pred_final": pred_final,
        "sample_next": sample_next,
        "t": STEP1_T.reshape(()),
        "t_prev": STEP1_T_PREV.reshape(()),
    }


def validate_step1_bundle(
    bundle: dict[str, np.ndarray],
    expected_coords: np.ndarray,
    expected_sample_in: np.ndarray,
) -> None:
    required = {"coords", *STEP1_MATRIX_KEYS, *STEP1_SCALAR_KEYS}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"step-1 bundle missing keys: {missing}")
    coords = bundle["coords"]
    for key in STEP1_MATRIX_KEYS:
        validate_output(bundle[key], coords, expected_coords)
    if not np.array_equal(bundle["sample_in"], expected_sample_in):
        raise ValueError("step-1 bundle sample_in does not match input noise")
    for key in STEP1_SCALAR_KEYS:
        arr = np.asarray(bundle[key])
        if arr.dtype != np.float32 or arr.shape != () or not np.isfinite(arr).all():
            raise ValueError(f"invalid scalar {key}: shape={arr.shape} dtype={arr.dtype}")
    for key in ("std_pos", "std_cfg", "ratio_raw", "std_ratio", "ratio_effective"):
        if float(bundle[key]) <= 0.0:
            raise ValueError(f"non-positive guidance scalar {key}: {float(bundle[key])}")
    if float(bundle["t"]) != float(STEP1_T) or float(bundle["t_prev"]) != float(
        STEP1_T_PREV
    ):
        raise ValueError(
            f"unexpected step-1 schedule scalars: t={bundle['t']} t_prev={bundle['t_prev']}"
        )


def validate_route(
    route: dict[str, Any], backend: str, native_profile: str = "fp16-control"
) -> None:
    if backend == "cuda":
        if native_profile != "fp16-control":
            raise ValueError("native profile requires MLX backend")
        expected = dict(
            cuda_available=True,
            attention="xformers",
            sparse_container="none",
            trellis=TRELLIS_REVISION,
            xformers="0.0.35",
            torso_dtype="float16",
            hr_resolution=64,
        )
        if not all(route.get(k) == v for k, v in expected.items()):
            raise ValueError(f"invalid CUDA route: {route}")
        if route.get("torch") not in ACCEPTED_TORCH_CUDA_VERSIONS:
            raise ValueError(f"unsupported CUDA torch runtime: {route.get('torch')!r}")
    elif backend == "mlx":
        if native_profile not in NATIVE_PROFILES:
            raise ValueError(f"unknown native profile: {native_profile}")
        expected_dtype = "float16" if native_profile == "fp16-control" else "bfloat16"
        expected_qk = (
            "mlx-sum"
            if native_profile in {"fp16-control", "bf16-mlx-sum"}
            else "source-cuda-warp32"
        )
        expected_modulation = (
            "float32"
            if native_profile in {"fp16-control", "bf16-f32-modulation"}
            else "bfloat16"
        )
        if route.get("torso_dtype") != expected_dtype:
            raise ValueError(
                f"MLX precision does not match {native_profile}: {route}"
            )
        if (
            "learned_modulation_dtype" in route
            and route.get("learned_modulation_dtype") != expected_modulation
        ):
            raise ValueError(
                f"MLX modulation precision does not match {native_profile}: {route}"
            )
        expected_mlx = dict(
            mlx="0.32.4",
            device="gpu",
            attention="fast",
            query_chunk=1024,
            hr_attention_active=True,
            shape_flow_layernorm="mlx-two-pass",
            qk_norm=expected_qk,
            torso_dtype=expected_dtype,
        )
        if not all(route.get(k) == v for k, v in expected_mlx.items()):
            raise ValueError(f"invalid MLX route: {route}")
    else:
        raise ValueError(f"unknown backend: {backend}")


def load_inputs(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    expected_keys = {"noise", "cond", "neg_cond", "coords", "t", "t_prev"}
    if set(arrays) != expected_keys:
        raise ValueError(f"unexpected input keys: {sorted(arrays)}")
    coords = arrays["coords"]
    if (
        coords.dtype != np.int32
        or coords.ndim != 2
        or coords.shape[0] <= 0
        or coords.shape[1] != 4
    ):
        raise ValueError(f"invalid coords: shape={coords.shape} dtype={coords.dtype}")
    if not np.all(coords[:, 0] == 0):
        raise ValueError("coords batch column must be 0")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("duplicate coordinates in input support")
    rows = coords.shape[0]
    for k, shape in [
        ("noise", (rows, 32)),
        ("cond", (1, 4101, 1024)),
        ("neg_cond", (1, 4101, 1024)),
        ("t", (1,)),
        ("t_prev", (1,)),
    ]:
        arr = arrays[k]
        if arr.dtype != np.float32 or arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"invalid {k}: shape={arr.shape} dtype={arr.dtype}")
    if not np.all(arrays["neg_cond"] == 0.0):
        raise ValueError("neg_cond must be all zeros for shape-flow step 1")
    if float(arrays["t"][0]) != 1000.0:
        raise ValueError(f"unexpected timestep t: {arrays['t']}")
    if float(arrays["t_prev"][0]) != float(STEP1_T_PREV):
        raise ValueError(f"unexpected t_prev: {arrays['t_prev']}")
    return arrays


def run_cuda(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    save: Any = None,
) -> dict[str, np.ndarray]:
    def _save() -> None:
        if callable(save):
            save()

    def command(argv: list[str], cwd: Path | None = None) -> str:
        done = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
        report.setdefault("commands", []).append(
            dict(
                argv=argv,
                cwd=str(cwd),
                exit_code=done.returncode,
                stdout=done.stdout,
                stderr=done.stderr,
            )
        )
        _save()
        if done.returncode:
            raise RuntimeError(f"command failed: {argv}")
        return done.stdout.strip()

    report["phase"] = "cuda_setup"
    _save()
    import torch

    if not torch.cuda.is_available() or torch.__version__ not in ACCEPTED_TORCH_CUDA_VERSIONS:
        raise ValueError(f"expected measured CUDA Torch route, got {torch.__version__!r}")
    os.environ.update(
        SPARSE_CONV_BACKEND="none",
        SPARSE_ATTN_BACKEND="xformers",
        ATTN_BACKEND="xformers",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    command([sys.executable, "-m", "pip", "install", "--no-deps", "xformers==0.0.35"])
    command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "easydict",
            "einops",
            "tqdm",
            "huggingface-hub",
            "safetensors",
        ]
    )
    source = args.work_dir / "TRELLIS.2"
    if source.exists():
        raise ValueError("fresh CUDA source directory required")
    source.parent.mkdir(parents=True, exist_ok=True)
    command(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/microsoft/TRELLIS.2.git",
            str(source),
        ]
    )
    command(["git", "checkout", "--detach", TRELLIS_REVISION], source)
    sys.path.insert(0, str(source))
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from trellis2.models.structured_latent_flow import SLatFlowModel
    from trellis2.modules import sparse as sp
    from trellis2.modules.sparse import config

    files = {}
    for ext, expected in WEIGHT_HASHES.items():
        name = f"{STEM}{ext}"
        path = Path(
            hf_hub_download(
                MODEL_REPO,
                name,
                revision=MODEL_REVISION,
                cache_dir=str(args.work_dir / "hf"),
            )
        )
        got = digest(path)
        if got != expected:
            raise ValueError(f"model hash differs for {name}: {got} != {expected}")
        files[ext] = path

    cfg = json.loads(files[".json"].read_text())
    report["source_model_config"] = cfg
    kwargs = dict(cfg["args"], dtype="float16")
    route = dict(
        cuda_available=torch.cuda.is_available(),
        device=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),
        torch=torch.__version__,
        xformers=importlib.metadata.version("xformers"),
        attention=config.ATTN,
        sparse_container=config.CONV,
        torso_dtype="float16",
        hr_resolution=int(kwargs.get("resolution", -1)),
        trellis=command(["git", "rev-parse", "HEAD"], source),
        model_revision=MODEL_REVISION,
        source_path=inspect.getfile(SLatFlowModel),
        precision_change="original dtype argument bfloat16 replaced by float16; original convert_to retained",
        weights={k: digest(v) for k, v in files.items()},
    )
    validate_route(route, "cuda")
    if not Path(route["source_path"]).is_relative_to(source):
        raise ValueError("wrong source import")
    command(["git", "diff", "--exit-code", "HEAD"], source)
    report["effective_route"] = route
    report["phase"] = "model_load"
    _save()

    t0 = time.perf_counter()
    model = SLatFlowModel(**kwargs)
    model.load_state_dict(load_file(str(files[".safetensors"])), strict=True)
    model.eval().cuda()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    report["parameter_dtypes"] = {
        k: str(v.dtype) for k, v in model.named_parameters()
    }
    if (
        model.dtype != torch.float16
        or model.input_layer.weight.dtype != torch.float32
        or model.out_layer.weight.dtype != torch.float32
        or any(
            b.modulation.dtype != torch.float32
            or b.self_attn.to_qkv.weight.dtype != torch.float16
            for b in model.blocks
        )
    ):
        raise ValueError("source mixed precision contract differs")
    report["last_trustworthy_phase"] = "model_loaded"
    report["phase"] = "prediction"
    _save()

    coords = torch.from_numpy(arrays["coords"]).cuda()
    x = sp.SparseTensor(
        feats=torch.from_numpy(arrays["noise"]).cuda(),
        coords=coords,
    )
    t_tensor = torch.from_numpy(arrays["t"]).cuda()
    cond = torch.from_numpy(arrays["cond"]).cuda()
    neg_cond = torch.from_numpy(arrays["neg_cond"]).cuda()
    with torch.inference_mode():
        t1 = time.perf_counter()
        out_pos = model(x, t_tensor, cond)
        out_neg = model(x, t_tensor, neg_cond)
        torch.cuda.synchronize()
        pred_s = time.perf_counter() - t1

    pos_coords = out_pos.coords.cpu().numpy().astype(np.int32)
    neg_coords = out_neg.coords.cpu().numpy().astype(np.int32)
    if not np.array_equal(pos_coords, neg_coords):
        raise ValueError("CUDA pos and neg branch output coordinates differ")
    pred_pos = out_pos.feats.float().cpu().numpy()
    pred_neg = out_neg.feats.float().cpu().numpy()
    report["timing_seconds"] = {
        "model_load": load_s,
        "prediction_both_branches": pred_s,
    }
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    return compute_step1_euler(arrays["noise"], pred_pos, pred_neg, pos_coords)


def run_mlx(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    save: Any = None,
) -> dict[str, np.ndarray]:
    def _save() -> None:
        if callable(save):
            save()

    dtype = "float16" if args.native_profile == "fp16-control" else "bfloat16"
    qk = (
        "mlx-sum"
        if args.native_profile in {"fp16-control", "bf16-mlx-sum"}
        else "source-cuda-warp32"
    )
    modulation = (
        "float32"
        if args.native_profile in {"fp16-control", "bf16-f32-modulation"}
        else "bfloat16"
    )
    native_bf16 = dtype == "bfloat16"
    root = (args.source_root or Path(__file__).resolve().parents[1]).resolve()
    if args.native_revision:
        actual_rev = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual_rev != args.native_revision:
            raise ValueError(f"native revision differs: {actual_rev} != {args.native_revision}")
        subprocess.run(
            ["git", "-C", str(root), "diff", "--exit-code", "HEAD"],
            check=True,
            capture_output=True,
        )
    sys.path.insert(0, str(root))
    os.environ.update(
        TRELLIS2MLX_ATTENTION_BACKEND="fast",
        TRELLIS2MLX_FAST_ATTENTION_QUERY_CHUNK_SIZE="2048",
        TRELLIS2MLX_HR_SHAPE_FAST_ATTENTION_QUERY_CHUNK_SIZE="1024",
        TRELLIS2MLX_HR_SHAPE_ATTENTION_ACTIVE="1",
        TRELLIS2MLX_QK_NORM_BACKEND=qk,
    )
    import mlx.core as mx
    import mlx.utils
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import generate
    from trellmlx import sparse_flow_attention as sfa
    from trellmlx.greenroom_pause import pause_session
    from trellmlx.models.slat_flow import SLatFlowModel
    from trellmlx.models.sparse_structure_flow import _cast_block_linears
    import trellmlx.modules.attention as attention_mod
    from trellmlx.shape_flow_layernorm import get_shape_flow_layernorm_backend
    from trellmlx.weight_loader import _remap_key, load_weights

    sfa.configure_sparse_flow_attention_from_env()
    sfa.set_hr_shape_attention_active(True)
    attention_mod._qk_norm_backend = None
    try:
        ckpt_path = args.checkpoint
        json_path = None
        if ckpt_path is None:
            ckpt_path = Path(
                hf_hub_download(MODEL_REPO, f"{STEM}.safetensors", revision=MODEL_REVISION)
            )
            json_path = Path(
                hf_hub_download(MODEL_REPO, f"{STEM}.json", revision=MODEL_REVISION)
            )
        else:
            ckpt_path = Path(ckpt_path)
            candidate_json = ckpt_path.with_suffix(".json")
            json_path = (
                candidate_json
                if candidate_json.is_file()
                else Path(
                    hf_hub_download(MODEL_REPO, f"{STEM}.json", revision=MODEL_REVISION)
                )
            )
        if digest(ckpt_path) != WEIGHT_HASHES[".safetensors"]:
            raise ValueError("native checkpoint digest differs")
        if digest(json_path) != WEIGHT_HASHES[".json"]:
            raise ValueError("native config digest differs")

        actual = generate._shape_flow_attention_route_from_env()
        route = dict(
            mlx=importlib.metadata.version("mlx"),
            device="gpu" if mx.default_device() == mx.gpu else str(mx.default_device()),
            attention=actual["shape_flow_attention_backend_effective"],
            query_chunk=actual["shape_flow_fast_attention_query_chunk_size_effective"],
            hr_attention_active=sfa.get_hr_shape_attention_active(),
            shape_flow_layernorm=get_shape_flow_layernorm_backend(),
            qk_norm=generate.get_qk_norm_backend(),
            torso_dtype=dtype,
            learned_modulation_dtype=modulation,
            native_profile=args.native_profile,
            native_revision=args.native_revision,
            source_path=inspect.getfile(SLatFlowModel),
            model_revision=MODEL_REVISION,
            weights={
                ".json": digest(json_path),
                ".safetensors": digest(ckpt_path),
            },
            precision_change={
                "production-bf16": "none: unmodified for_shape constructor and production weight loader",
                "fp16-control": "block linears float16, source-equivalent float32 learned block modulation; outer layers stay float32",
                "bf16-f32-modulation": "only learned block modulation held float32 before loading; production BF16 linears and QK retained",
                "bf16-mlx-sum": "none: production parameter dtypes; only QK normalization selects mlx-sum",
            }[args.native_profile],
        )
        validate_route(route, "mlx", native_profile=args.native_profile)
        if Path(route["source_path"]).resolve() != root / "trellmlx/models/slat_flow.py":
            raise ValueError("wrong native source import")
        report["effective_route"] = route
        report["phase"] = "model_load"
        _save()

        with pause_session(args.queue_dir, args.output_json.parent / "pause.json"):
            t0 = time.perf_counter()
            model = SLatFlowModel.for_shape()
            if not native_bf16:
                for block in model.blocks:
                    _cast_block_linears(block, mx.float16)
                    block.modulation = block.modulation.astype(mx.float32)
            elif modulation == "float32":
                for block in model.blocks:
                    block.modulation = block.modulation.astype(mx.float32)
            params = dict(mlx.utils.tree_flatten(model.parameters()))
            with safe_open(str(ckpt_path), framework="numpy") as sf:
                keys = {_remap_key(k) for k in sf.keys()}
            if set(params) != keys:
                raise ValueError("native/checkpoint parameter keys differ")
            if load_weights(model, str(ckpt_path), verbose=False):
                raise ValueError("unloaded native weights")
            mx.eval(model.parameters())
            load_s = time.perf_counter() - t0

            linear_dtype = mx.bfloat16 if native_bf16 else mx.float16
            modulation_dtype = mx.bfloat16 if modulation == "bfloat16" else mx.float32
            if (
                model.input_layer.weight.dtype != mx.float32
                or model.out_layer.weight.dtype != mx.float32
                or any(
                    b.modulation.dtype != modulation_dtype
                    or b.self_attn.to_qkv.weight.dtype != linear_dtype
                    for b in model.blocks
                )
            ):
                raise ValueError("native mixed precision contract differs")
            report["parameter_dtypes"] = {
                k: str(v.dtype) for k, v in mlx.utils.tree_flatten(model.parameters())
            }
            report["last_trustworthy_phase"] = "model_loaded"
            report["phase"] = "prediction"
            _save()

            t1 = time.perf_counter()
            x_mx = mx.array(arrays["noise"])
            t_mx = mx.array(arrays["t"])
            coords3_mx = mx.array(arrays["coords"][:, 1:])
            cond_mx = mx.array(arrays["cond"])
            neg_cond_mx = mx.array(arrays["neg_cond"])
            out_pos = model(x_mx, t_mx, cond_mx, coords=coords3_mx)
            out_neg = model(x_mx, t_mx, neg_cond_mx, coords=coords3_mx)
            mx.eval(out_pos, out_neg)
            pred_s = time.perf_counter() - t1
            report["peak_allocated_bytes"] = int(mx.get_peak_memory())

        pred_pos = np.array(out_pos.astype(mx.float32), dtype=np.float32, copy=True)
        pred_neg = np.array(out_neg.astype(mx.float32), dtype=np.float32, copy=True)
        report["timing_seconds"] = {
            "model_load": load_s,
            "prediction_both_branches": pred_s,
        }
        return compute_step1_euler(
            arrays["noise"], pred_pos, pred_neg, arrays["coords"].copy()
        )
    finally:
        sfa.set_hr_shape_attention_active(False)


def compare_bundles(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, Any]:
    if not np.array_equal(left["coords"], right["coords"]):
        raise ValueError("cannot compare bundles with mismatched coords")
    metrics: dict[str, Any] = {"matrices": {}, "scalars": {}}
    for key in STEP1_MATRIX_KEYS:
        a = np.asarray(left[key], dtype=np.float32)
        b = np.asarray(right[key], dtype=np.float32)
        diff = np.abs(a - b)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cosine = float(np.sum(a * b) / denom) if denom > 0.0 else 1.0
        metrics["matrices"][key] = {
            "shape": list(a.shape),
            "exact": bool(np.array_equal(a, b)),
            "nonzero_count": int(np.count_nonzero(diff)),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "p99_abs": float(np.quantile(diff, 0.99)),
            "p999_abs": float(np.quantile(diff, 0.999)),
            "cosine": cosine,
        }
    for key in STEP1_SCALAR_KEYS:
        av = float(left[key])
        bv = float(right[key])
        metrics["scalars"][key] = {
            "left": av,
            "right": bv,
            "abs_diff": abs(av - bv),
            "exact": bool(np.array_equal(left[key], right[key])),
        }
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["cuda", "mlx"], required=True)
    ap.add_argument(
        "--native-profile", choices=NATIVE_PROFILES, default="fp16-control"
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--inputs-sha256", required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-npz", type=Path, required=True)
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/kaggle/temp/shape-first-prediction"),
    )
    ap.add_argument("--source-root", type=Path)
    ap.add_argument("--native-revision")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument(
        "--queue-dir",
        type=Path,
        default=Path.home() / ".local/state/gpu-greenroom",
    )
    args = ap.parse_args(argv)

    phase = "argument_parsing"
    last_trustworthy_phase: str | None = None
    report: dict[str, Any] = {
        "schema": "trellis2mlx.shape_first_prediction_witness.v1",
        "run_id": args.run_id,
        "backend_requested": args.backend,
        "native_profile_requested": args.native_profile,
        "status": "running",
        "claim_ceiling": claim_ceiling_for(args.backend, args.native_profile),
        "inputs": {
            "path": str(args.inputs),
            "expected_sha256": args.inputs_sha256,
        },
    }

    def save() -> None:
        write_json(args.output_json, report)

    try:
        phase = "profile_admission"
        if args.backend != "mlx" and args.native_profile != "fp16-control":
            raise ValueError("native profile requires MLX backend")
        last_trustworthy_phase = phase

        phase = "output_path_admission"
        if args.output_npz.exists():
            raise ValueError(
                f"refusing to overwrite preexisting output NPZ: {args.output_npz}"
            )
        if args.output_json.resolve() == args.output_npz.resolve():
            raise ValueError("output JSON and output NPZ paths must not alias")
        if args.inputs.exists() and (
            args.output_json.resolve() == args.inputs.resolve()
            or args.output_npz.resolve() == args.inputs.resolve()
        ):
            raise ValueError("output paths must not alias input archive")
        last_trustworthy_phase = phase

        phase = "input_admission"
        if not args.inputs.is_file() or args.inputs.stat().st_size <= 0:
            raise ValueError(f"missing or empty inputs file: {args.inputs}")
        got = digest(args.inputs)
        report["inputs"]["observed_sha256"] = got
        if got != args.inputs_sha256:
            raise ValueError(f"input digest mismatch: {got} != {args.inputs_sha256}")
        arrays = load_inputs(args.inputs)
        report["inputs"]["rows"] = int(arrays["coords"].shape[0])
        last_trustworthy_phase = phase

        phase = "backend_execution"
        runner = run_cuda if args.backend == "cuda" else run_mlx
        bundle = runner(args, arrays, report, save)
        last_trustworthy_phase = phase

        phase = "output_capture"
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_npz, **bundle)
        report["output"] = {
            "path": str(args.output_npz),
            "sha256": digest(args.output_npz),
            "bytes": args.output_npz.stat().st_size,
        }
        validate_step1_bundle(bundle, arrays["coords"], arrays["noise"])
        report["output"].update(
            rows=int(bundle["coords"].shape[0]),
            pred_pos_sha256=hashlib.sha256(
                np.ascontiguousarray(bundle["pred_pos"]).tobytes()
            ).hexdigest(),
            pred_neg_sha256=hashlib.sha256(
                np.ascontiguousarray(bundle["pred_neg"]).tobytes()
            ).hexdigest(),
            pred_final_sha256=hashlib.sha256(
                np.ascontiguousarray(bundle["pred_final"]).tobytes()
            ).hexdigest(),
            sample_next_sha256=hashlib.sha256(
                np.ascontiguousarray(bundle["sample_next"]).tobytes()
            ).hexdigest(),
            std_pos=float(bundle["std_pos"]),
            std_cfg=float(bundle["std_cfg"]),
            std_ratio=float(bundle["std_ratio"]),
        )
        last_trustworthy_phase = phase
        report["last_trustworthy_phase"] = last_trustworthy_phase
        report["status"] = "completed"
        save()
        return 0
    except BaseException as exc:
        report.update(
            status="failed",
            failure_phase=phase,
            last_trustworthy_phase=last_trustworthy_phase,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        save()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

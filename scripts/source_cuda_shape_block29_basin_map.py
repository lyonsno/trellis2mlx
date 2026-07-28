#!/usr/bin/env python3
"""Continue a fixed block29 MLX/source endpoint plane through source CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.source_cuda_shape_block29_basin_map.v1"
ENDPOINT_SCHEMA = "trellis2mlx.shape_block29_cuda_basin_endpoints.v1"
COMPARISON_CLASS = "fixed_block29_endpoint_affine_plane"
ENDPOINT_SEMANTICS = "current + scale * (source - current)"
BRANCH_STAGE_KEYS = (
    "pos_block29_after_self",
    "pos_block29_cross_attention_raw",
    "neg_block29_after_self",
    "neg_block29_cross_attention_raw",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parse_axis_values(value: str, *, name: str) -> list[float]:
    values: list[float] = []
    for part in value.split(","):
        if not part.strip():
            continue
        parsed = float(part.strip())
        if not math.isfinite(parsed):
            raise ValueError(f"--{name}s must contain only finite {name} values")
        if parsed in values:
            raise ValueError(f"duplicate {name} coordinate: {parsed}")
        values.append(parsed)
    if not values:
        raise ValueError(f"--{name}s must contain at least one value")
    return values


def cartesian_coordinates(alphas: list[float], betas: list[float]) -> list[tuple[float, float]]:
    return [(float(alpha), float(beta)) for alpha in alphas for beta in betas]


def prioritized_execution_coordinates(
    coordinates: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    requested = [(float(alpha), float(beta)) for alpha, beta in coordinates]
    priority = ((1.0, 1.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    return [point for point in priority if point in requested] + [
        point for point in requested if point not in priority
    ]


def _float_slug(value: float) -> str:
    rendered = format(float(value), ".17g")
    return (
        rendered.replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
        .replace("e", "e")
    )


def coordinate_key(alpha: float, beta: float) -> str:
    return f"alpha-{_float_slug(alpha)}_beta-{_float_slug(beta)}"


def interpolate_endpoint(current: np.ndarray, source: np.ndarray, scale: float) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    if current.shape != source.shape:
        raise ValueError(f"endpoint shapes differ: {current.shape} != {source.shape}")
    if not math.isfinite(float(scale)):
        raise ValueError(f"endpoint scale must be finite, got {scale}")
    if float(scale) == 0.0:
        return np.ascontiguousarray(current)
    if float(scale) == 1.0:
        return np.ascontiguousarray(source)
    return np.ascontiguousarray(
        current + np.float32(scale) * (source - current),
        dtype=np.float32,
    )


def endpoint_target_shape(
    target_shape: tuple[int, ...], live_shape: tuple[int, ...]
) -> tuple[int, ...]:
    if target_shape == live_shape:
        return live_shape
    if math.prod(target_shape) != math.prod(live_shape):
        raise ValueError(
            f"endpoint/live element count differs: {target_shape} vs {live_shape}"
        )
    return live_shape


def _decode_bf16_words(words: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(np.asarray(words, dtype=np.uint16))
    return np.ascontiguousarray((packed.astype(np.uint32) << np.uint32(16)).view(np.float32))


def _compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "left_shape": [int(v) for v in left.shape],
            "right_shape": [int(v) for v in right.shape],
        }
    diff = np.abs(left - right)
    return {
        "shape_match": True,
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "nonzero": int(np.count_nonzero(diff)),
        "exact": bool(np.array_equal(left, right)),
    }


def validate_result_manifest(
    payload: dict[str, Any], *, coordinates: list[tuple[float, float]]
) -> None:
    if payload.get("status") != "done":
        raise ValueError("basin result is not done")
    route = payload.get("effective_route", {})
    required_route = {
        "device_type": "cuda",
        "attention_backend": "sdpa",
        "steps": 8,
        "block_index": 29,
        "step_index": 0,
    }
    for key, expected in required_route.items():
        if route.get(key) != expected:
            raise ValueError(f"effective route {key} must be {expected!r}")
    requested = {(float(alpha), float(beta)) for alpha, beta in coordinates}
    observed = {
        (float(point["coordinate"]["alpha"]), float(point["coordinate"]["beta"]))
        for point in payload.get("points", [])
    }
    if observed != requested:
        raise ValueError(f"result coordinates differ: observed={observed}, requested={requested}")
    source_control = payload.get("source_control", {})
    if source_control.get("coordinate") != {"alpha": 1.0, "beta": 1.0}:
        raise ValueError("source control must be coordinate (1,1)")
    if (
        source_control.get("exact") is not True
        or source_control.get("max_abs") != 0.0
        or source_control.get("nonzero") != 0
    ):
        raise ValueError("source control is not exact")


def _load_endpoints(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files or "coords" not in archive.files:
            raise ValueError("endpoint packet must contain metadata_json and coords")
        raw_metadata = np.asarray(archive["metadata_json"])
        if raw_metadata.size != 1:
            raise ValueError("endpoint metadata_json must contain one value")
        metadata = json.loads(str(raw_metadata.reshape(-1)[0]))
        required_metadata = {
            "schema": ENDPOINT_SCHEMA,
            "status": "done",
            "comparison_class": COMPARISON_CLASS,
            "endpoint_semantics": ENDPOINT_SEMANTICS,
            "steps": 8,
            "block_index": 29,
            "step_index": 0,
        }
        for key, expected in required_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(f"endpoint metadata {key} must be {expected!r}")
        coords = np.asarray(archive["coords"], dtype=np.int32)
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"endpoint coords must have shape [N,4], got {coords.shape}")
        endpoints: dict[str, np.ndarray] = {}
        endpoint_digests = metadata.get("endpoint_digests")
        if not isinstance(endpoint_digests, dict):
            raise ValueError("endpoint metadata has no endpoint_digests object")
        for key in BRANCH_STAGE_KEYS:
            key_digests = endpoint_digests.get(key)
            if not isinstance(key_digests, dict):
                raise ValueError(f"endpoint metadata has no digest object for {key}")
            for endpoint in ("current", "source"):
                packed_key = f"{key}_{endpoint}_bf16_words"
                if packed_key not in archive.files:
                    raise ValueError(f"endpoint packet has no {packed_key}")
                values = _decode_bf16_words(archive[packed_key])
                if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] != coords.shape[0]:
                    raise ValueError(
                        f"endpoint {packed_key} must have shape [1,N,C], got {values.shape}"
                    )
                digest_key = f"{endpoint}_float32_sha256"
                expected_digest = key_digests.get(digest_key)
                if not isinstance(expected_digest, str) or len(expected_digest) != 64:
                    raise ValueError(
                        f"endpoint metadata has no valid {key}.{digest_key}"
                    )
                actual_digest = hashlib.sha256(values.tobytes()).hexdigest()
                if actual_digest != expected_digest:
                    raise ValueError(
                        f"endpoint digest mismatch for {packed_key}: "
                        f"{actual_digest} != {expected_digest}"
                    )
                endpoints[f"{key}_{endpoint}"] = values
    return endpoints, np.ascontiguousarray(coords), metadata


def _validate_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} is blank: {path}")


def _invalidate_primary_output(path: Path, *, protected: dict[str, Path]) -> None:
    output = path.resolve()
    for label, candidate in protected.items():
        if output == candidate.resolve():
            raise ValueError(f"primary output path collides with {label}: {output}")
    if not path.exists() and not path.is_symlink():
        return
    if not path.is_file() and not path.is_symlink():
        raise ValueError(f"primary output path is not a file: {path}")
    path.unlink()


def _route_digest(metadata: dict[str, Any], route_name: str, key: str) -> str:
    route = metadata.get(route_name, {})
    value = route.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"endpoint {route_name} has no valid {key}")
    return value


def _load_conditioning(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if "cond" not in archive.files or "neg_cond" not in archive.files:
            raise ValueError("conditioning must contain cond and neg_cond")
        cond = np.asarray(archive["cond"], dtype=np.float32)
        neg_cond = np.asarray(archive["neg_cond"], dtype=np.float32)
    if not np.isfinite(cond).all() or not np.isfinite(neg_cond).all():
        raise ValueError("conditioning contains non-finite values")
    return cond, neg_cond


def _load_noise(path: Path, *, expected_coords: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if "noise" not in archive.files or "coords" not in archive.files:
            raise ValueError("shape-flow noise sample must contain noise and coords")
        noise = np.asarray(archive["noise"], dtype=np.float32)
        coords = np.asarray(archive["coords"], dtype=np.int32)
        scalar = {
            "steps": int(np.asarray(archive["steps"]).reshape(-1)[0]),
            "guidance_strength": float(np.asarray(archive["guidance_strength"]).reshape(-1)[0]),
            "guidance_rescale": float(np.asarray(archive["guidance_rescale"]).reshape(-1)[0]),
            "guidance_interval": [float(v) for v in np.asarray(archive["guidance_interval"]).reshape(-1)],
            "rescale_t": float(np.asarray(archive["rescale_t"]).reshape(-1)[0]),
        }
    if coords.shape != expected_coords.shape or not np.array_equal(coords, expected_coords):
        raise ValueError("shape-flow noise coordinates differ from endpoint coordinates")
    if noise.ndim != 2 or noise.shape[0] != coords.shape[0]:
        raise ValueError(f"shape-flow noise has invalid shape {noise.shape}")
    if not np.isfinite(noise).all():
        raise ValueError("shape-flow noise contains non-finite values")
    if scalar["steps"] != 8 or len(scalar["guidance_interval"]) != 2:
        raise ValueError(f"shape-flow noise route metadata is unsupported: {scalar}")
    return np.ascontiguousarray(noise), scalar


def _schedule_pairs(steps: int, rescale_t: float) -> list[tuple[float, float]]:
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    return [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]


def route_values_match(actual: Any, expected: Any) -> bool:
    try:
        actual_values = np.asarray(actual, dtype=np.float32)
        expected_values = np.asarray(expected, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return actual_values.shape == expected_values.shape and np.array_equal(
        actual_values, expected_values
    )


def _target_tensors(
    endpoints: dict[str, np.ndarray], *, alpha: float, beta: float, torch: Any, device: Any
) -> dict[tuple[str, str], Any]:
    targets: dict[tuple[str, str], Any] = {}
    for branch in ("pos", "neg"):
        for stage, scale in (("after_self", alpha), ("cross_attention_raw", beta)):
            prefix = f"{branch}_block29_{stage}"
            values = interpolate_endpoint(
                endpoints[f"{prefix}_current"],
                endpoints[f"{prefix}_source"],
                scale,
            )[0]
            targets[(branch, stage)] = torch.from_numpy(values).to(
                device=device, dtype=torch.bfloat16
            )
    return targets


def _run_intervened_block(torch: Any, block: Any, x: Any, mod: Any, context: Any, targets: dict[str, Any]):
    from source_cuda_shape_block_trace import (
        _source_mlp_gelu,
        _source_mlp_linears,
        split_block_modulation,
    )
    from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = split_block_modulation(
        block, mod
    )
    h = x.replace(block.norm1(x.feats))
    h = h * (1 + scale_msa) + shift_msa
    attn = block.self_attn
    qkv = attn._linear(attn.to_qkv, h)
    qkv = attn._fused_pre(qkv, num_fused=3)
    q, k, v = qkv.unbind(dim=-3)
    if getattr(attn, "qk_rms_norm", False):
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    if getattr(attn, "use_rope", False):
        q, k = attn.rope(q, k)
    qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
    h = sparse_scaled_dot_product_attention(qkv)
    h = attn._reshape_chs(h, (-1,))
    h = attn._linear(attn.to_out, h)
    x = x + h * gate_msa
    after_self = targets["after_self"]
    if tuple(after_self.shape) != tuple(x.feats.shape):
        raise ValueError(
            f"after_self target shape {tuple(after_self.shape)} != live {tuple(x.feats.shape)}"
        )
    x = x.replace(after_self)

    h = x.replace(block.norm2(x.feats))
    attn = block.cross_attn
    q = attn._linear(attn.to_q, h)
    q = attn._reshape_chs(q, (attn.num_heads, -1))
    kv = attn._linear(attn.to_kv, context)
    kv = attn._fused_pre(kv, num_fused=2)
    k, v = kv.unbind(dim=-3)
    if getattr(attn, "qk_rms_norm", False):
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    h = sparse_scaled_dot_product_attention(q, k, v)
    cross_raw = targets["cross_attention_raw"]
    target_shape = endpoint_target_shape(tuple(cross_raw.shape), tuple(h.feats.shape))
    if tuple(cross_raw.shape) != target_shape:
        cross_raw = cross_raw.reshape(target_shape)
    h = h.replace(cross_raw)
    h = attn._reshape_chs(h, (-1,))
    h = attn._linear(attn.to_out, h)
    x = x + h

    h = x.replace(block.norm3(x.feats))
    h = h * (1 + scale_mlp) + shift_mlp
    fc1, fc2 = _source_mlp_linears(block.mlp)
    h = fc2(_source_mlp_gelu(torch, block.mlp)(fc1(h)))
    return x + h * gate_mlp


def _flow_forward(
    torch: Any,
    flow_model: Any,
    x: Any,
    t_model: Any,
    context: Any,
    *,
    branch: str,
    targets: dict[tuple[str, str], Any] | None,
) -> Any:
    from trellis2.modules.utils import manual_cast

    h = flow_model.input_layer(x)
    h = manual_cast(h, flow_model.dtype)
    t_emb = flow_model.t_embedder(t_model)
    if flow_model.share_mod:
        t_emb = flow_model.adaLN_modulation(t_emb)
    t_emb = manual_cast(t_emb, flow_model.dtype)
    context = manual_cast(context, flow_model.dtype)
    if flow_model.pe_mode == "ape":
        pe = flow_model.pos_embedder(h.coords[:, 1:])
        h = h + manual_cast(pe, flow_model.dtype)
    for block_index, block in enumerate(flow_model.blocks):
        if targets is not None and block_index == 29:
            h = _run_intervened_block(
                torch,
                block,
                h,
                t_emb,
                context,
                {
                    "after_self": targets[(branch, "after_self")],
                    "cross_attention_raw": targets[(branch, "cross_attention_raw")],
                },
            )
        else:
            h = block(h, t_emb, context)
    h = manual_cast(h, x.dtype)
    h = h.replace(torch.nn.functional.layer_norm(h.feats, h.feats.shape[-1:]))
    return flow_model.out_layer(h)


def _guided_prediction(
    *,
    sampler: Any,
    sample: Any,
    pred_pos: Any,
    pred_neg: Any,
    t: float,
    guidance_strength: float,
    guidance_rescale: float,
    guidance_interval: tuple[float, float],
    capture: dict[str, Any] | None = None,
) -> Any:
    guidance_active = guidance_interval[0] <= t <= guidance_interval[1]
    pred_cfg = (
        guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
        if guidance_active
        else pred_pos
    )
    if capture is not None:
        capture["guidance_active"] = guidance_active and guidance_rescale > 0
        capture["pred_cfg"] = pred_cfg
    if not guidance_active or guidance_rescale <= 0:
        return pred_cfg
    x0_pos = sampler._pred_to_xstart(sample, t, pred_pos)
    x0_cfg = sampler._pred_to_xstart(sample, t, pred_cfg)
    std_pos = x0_pos.std(dim=list(range(1, x0_pos.ndim)), keepdim=True)
    std_cfg = x0_cfg.std(dim=list(range(1, x0_cfg.ndim)), keepdim=True)
    std_ratio = std_pos / std_cfg
    x0_rescaled = x0_cfg * std_ratio
    x0 = guidance_rescale * x0_rescaled + (1 - guidance_rescale) * x0_cfg
    if capture is not None:
        capture.update(
            {
                "x0_pos": x0_pos,
                "x0_cfg": x0_cfg,
                "std_pos": std_pos,
                "std_cfg": std_cfg,
                "std_ratio": std_ratio,
                "x0_rescaled": x0_rescaled,
                "x0_after_rescale": x0,
            }
        )
    return sampler._xstart_to_pred(sample, t, x0)


def _guidance_capture_to_numpy(capture: dict[str, Any]) -> dict[str, np.ndarray]:
    converted: dict[str, np.ndarray] = {
        "guidance_active": np.asarray(
            bool(capture.get("guidance_active", False)), dtype=np.bool_
        )
    }
    for name, value in capture.items():
        if name == "guidance_active":
            continue
        tensor = value.feats if hasattr(value, "feats") else value
        converted[name] = (
            tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
        )
    return converted


def _run_shape_flow(
    *,
    torch: Any,
    flow_model: Any,
    sampler: Any,
    coords: Any,
    noise: Any,
    cond: Any,
    neg_cond: Any,
    params: dict[str, Any],
    targets: dict[tuple[str, str], Any] | None,
) -> tuple[Any, list[float]]:
    from trellis2.modules.sparse import SparseTensor

    sample = SparseTensor(feats=noise.clone(), coords=coords)
    step_timings: list[float] = []
    guidance_interval = tuple(float(v) for v in params["guidance_interval"])
    for step_index, (t, t_prev) in enumerate(
        _schedule_pairs(int(params["steps"]), float(params["rescale_t"]))
    ):
        step_started = time.perf_counter()
        t_model = torch.tensor(
            [1000.0 * t] * sample.shape[0], device=sample.device, dtype=torch.float32
        )
        step_targets = targets if step_index == 0 else None
        pred_pos = _flow_forward(
            torch,
            flow_model,
            sample,
            t_model,
            cond,
            branch="pos",
            targets=step_targets,
        )
        pred_neg = _flow_forward(
            torch,
            flow_model,
            sample,
            t_model,
            neg_cond,
            branch="neg",
            targets=step_targets,
        )
        pred = _guided_prediction(
            sampler=sampler,
            sample=sample,
            pred_pos=pred_pos,
            pred_neg=pred_neg,
            t=t,
            guidance_strength=float(params["guidance_strength"]),
            guidance_rescale=float(params["guidance_rescale"]),
            guidance_interval=guidance_interval,
        )
        sample = sample - (t - t_prev) * pred
        torch.cuda.synchronize()
        step_timings.append(time.perf_counter() - step_started)
    return sample, step_timings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--endpoints", required=True, type=Path)
    parser.add_argument("--conditioning", required=True, type=Path)
    parser.add_argument("--shape-flow-noise-sample", required=True, type=Path)
    parser.add_argument("--source-tar", required=True, type=Path)
    parser.add_argument("--model-repo", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--pipeline-config", default="pipeline.json")
    parser.add_argument("--alphas", default="0,0.5,1")
    parser.add_argument("--betas", default="0,0.5,1")
    parser.add_argument("--sparse-conv-backend", default="none")
    parser.add_argument("--sparse-attn-backend", default="sdpa")
    parser.add_argument("--no-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    phase = "arguments_parsed"
    last_trustworthy_phase: str | None = phase
    requested_route = {
        "route": "official-source-cuda-full-eight-step-shape-flow-with-fixed-block29-endpoints",
        "alphas_requested": args.alphas,
        "betas_requested": args.betas,
        "steps": 8,
        "block_index": 29,
        "step_index": 0,
        "attention_backend": args.sparse_attn_backend,
        "conv_backend": args.sparse_conv_backend,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "requested_route": requested_route,
        "effective_route": "not-established",
        "primary_output_status": "missing",
        "failure_phase": None,
        "last_trustworthy_phase": last_trustworthy_phase,
        "phase_timings": {},
    }
    primary_written_this_run = False
    try:
        phase = "request_validation"
        phase_started = time.perf_counter()
        try:
            _invalidate_primary_output(
                args.output_npz,
                protected={
                    "output report": args.output_json,
                    "endpoint packet": args.endpoints,
                    "conditioning": args.conditioning,
                    "shape-flow noise sample": args.shape_flow_noise_sample,
                    "source tar": args.source_tar,
                },
            )
        except ValueError as exc:
            if "collides with" in str(exc) or "not a file" in str(exc):
                report["primary_output_status"] = "not_owned_due_to_path_collision"
            raise
        report["primary_output_status"] = "missing"
        alphas = parse_axis_values(args.alphas, name="alpha")
        betas = parse_axis_values(args.betas, name="beta")
        coordinates = cartesian_coordinates(alphas, betas)
        requested_route.update(
            {
                "alphas": alphas,
                "betas": betas,
                "coordinates": [
                    {"alpha": alpha, "beta": beta} for alpha, beta in coordinates
                ],
            }
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "input_validation"
        phase_started = time.perf_counter()
        for path, label in (
            (args.endpoints, "endpoint packet"),
            (args.conditioning, "conditioning"),
            (args.shape_flow_noise_sample, "shape-flow noise sample"),
            (args.source_tar, "source tar"),
        ):
            _validate_file(Path(path), label=label)
        endpoints, coords_np, endpoint_metadata = _load_endpoints(args.endpoints)
        cond_np, neg_cond_np = _load_conditioning(args.conditioning)
        noise_np, noise_route = _load_noise(
            args.shape_flow_noise_sample, expected_coords=coords_np
        )
        digest_expectations = (
            (
                args.conditioning,
                _route_digest(endpoint_metadata, "current_route", "conditioning_sample_sha256"),
                "conditioning",
            ),
            (
                args.shape_flow_noise_sample,
                _route_digest(endpoint_metadata, "current_route", "shape_flow_noise_sample_sha256"),
                "shape-flow noise",
            ),
            (
                args.source_tar,
                _route_digest(endpoint_metadata, "source_route", "source_tar_sha256"),
                "source tar",
            ),
        )
        input_digests: dict[str, str] = {"endpoints": _sha256(args.endpoints)}
        for path, expected, label in digest_expectations:
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(f"{label} digest mismatch: {actual} != {expected}")
            input_digests[label] = actual
        report["inputs"] = {
            "digests": input_digests,
            "endpoint_metadata": endpoint_metadata,
            "noise_route": noise_route,
            "coords_shape": [int(v) for v in coords_np.shape],
            "noise_shape": [int(v) for v in noise_np.shape],
        }
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        if args.no_download:
            raise RuntimeError("--no-download stops after validated local inputs by request")

        phase = "extract_source"
        phase_started = time.perf_counter()
        from source_cuda_shape_block_trace import extract_source

        source_root = extract_source(args.source_tar, Path.cwd())
        sys.path.insert(0, str(source_root))
        report["source_root"] = str(source_root)
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "import_runtime"
        phase_started = time.perf_counter()
        os.environ["SPARSE_CONV_BACKEND"] = args.sparse_conv_backend
        os.environ["SPARSE_ATTN_BACKEND"] = args.sparse_attn_backend
        os.environ["ATTN_BACKEND"] = args.sparse_attn_backend
        import torch
        from huggingface_hub import hf_hub_download
        from trellis2 import models as source_models
        from trellis2.modules.sparse import config as sparse_config
        from trellis2.pipelines import samplers

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.set_grad_enabled(False)
        device = torch.device("cuda")
        report.update(
            {
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_device_count": torch.cuda.device_count(),
                "sparse_attention_backend": getattr(sparse_config, "ATTN", None),
                "sparse_conv_backend": getattr(sparse_config, "CONV", None),
            }
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "load_pipeline_config"
        phase_started = time.perf_counter()
        config_path = Path(hf_hub_download(args.model_repo, args.pipeline_config))
        pipeline_args = json.loads(config_path.read_text())["args"]
        sampler_params = {
            **pipeline_args["shape_slat_sampler"]["params"],
            "steps": 8,
        }
        for key in (
            "steps",
            "guidance_strength",
            "guidance_rescale",
            "guidance_interval",
            "rescale_t",
        ):
            expected = noise_route[key]
            actual = sampler_params[key]
            if not route_values_match(actual, expected):
                raise ValueError(f"pipeline/noise route mismatch for {key}: {actual} != {expected}")
        sampler = getattr(samplers, pipeline_args["shape_slat_sampler"]["name"])(
            **pipeline_args["shape_slat_sampler"]["args"]
        )
        report["pipeline_config"] = {
            "path": str(config_path),
            "model_repo": args.model_repo,
            "pipeline_config": args.pipeline_config,
            "sampler_name": pipeline_args["shape_slat_sampler"]["name"],
            "sampler_args": pipeline_args["shape_slat_sampler"]["args"],
            "sampler_params": sampler_params,
            "shape_slat_normalization": pipeline_args["shape_slat_normalization"],
        }
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "load_model"
        phase_started = time.perf_counter()
        from source_cuda_shape_block_trace import resolve_model_ref

        model_ref = resolve_model_ref(
            args.model_repo, pipeline_args["models"]["shape_slat_flow_model_512"]
        )
        flow_model = source_models.from_pretrained(model_ref).to(device).eval()
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - phase_started
        report["model"] = {
            "model_ref": model_ref,
            "parameter_count": int(sum(parameter.numel() for parameter in flow_model.parameters())),
            "load_seconds": model_load_seconds,
        }
        report["phase_timings"][phase] = model_load_seconds
        last_trustworthy_phase = phase

        phase = "continuation"
        phase_started = time.perf_counter()
        coords = torch.from_numpy(coords_np).to(device=device, dtype=torch.int32)
        noise = torch.from_numpy(noise_np).to(device=device, dtype=torch.float32)
        cond = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)
        neg_cond = torch.from_numpy(neg_cond_np).to(device=device, dtype=torch.float32)
        normalization_std = torch.tensor(
            pipeline_args["shape_slat_normalization"]["std"], device=device
        )[None]
        normalization_mean = torch.tensor(
            pipeline_args["shape_slat_normalization"]["mean"], device=device
        )[None]

        baseline_started = time.perf_counter()
        baseline_raw, baseline_step_timings = _run_shape_flow(
            torch=torch,
            flow_model=flow_model,
            sampler=sampler,
            coords=coords,
            noise=noise,
            cond=cond,
            neg_cond=neg_cond,
            params=sampler_params,
            targets=None,
        )
        baseline = baseline_raw * normalization_std + normalization_mean
        baseline_np = baseline.feats.detach().float().cpu().numpy().astype(np.float32)
        baseline_seconds = time.perf_counter() - baseline_started

        arrays: dict[str, np.ndarray] = {
            "coords": coords_np,
            "source_control_shape_slat": baseline_np,
            "alpha_values": np.asarray(alphas, dtype=np.float64),
            "beta_values": np.asarray(betas, dtype=np.float64),
        }
        points: list[dict[str, Any]] = []
        coordinate_seconds = 0.0
        report["points"] = points
        report["timing"] = {
            "model_load_seconds": model_load_seconds,
            "source_baseline_seconds": baseline_seconds,
            "source_baseline_step_seconds": baseline_step_timings,
            "coordinate_continuation_seconds": coordinate_seconds,
            "full_decode_count_completed": 1,
            "full_decode_count_requested": 1 + len(coordinates),
            "t4_compute_seconds_through_continuation": time.perf_counter() - started,
        }
        for alpha, beta in prioritized_execution_coordinates(coordinates):
            point_started = time.perf_counter()
            targets = _target_tensors(
                endpoints,
                alpha=alpha,
                beta=beta,
                torch=torch,
                device=device,
            )
            result_raw, step_timings = _run_shape_flow(
                torch=torch,
                flow_model=flow_model,
                sampler=sampler,
                coords=coords,
                noise=noise,
                cond=cond,
                neg_cond=neg_cond,
                params=sampler_params,
                targets=targets,
            )
            result = result_raw * normalization_std + normalization_mean
            result_np = result.feats.detach().float().cpu().numpy().astype(np.float32)
            elapsed_seconds = time.perf_counter() - point_started
            coordinate_seconds += elapsed_seconds
            key = coordinate_key(alpha, beta)
            output_key = f"point_{key}_shape_slat"
            arrays[output_key] = result_np
            point = {
                "coordinate": {"alpha": alpha, "beta": beta},
                "coordinate_key": key,
                "output_key": output_key,
                "shape": [int(v) for v in result_np.shape],
                "sha256": hashlib.sha256(result_np.tobytes()).hexdigest(),
                "elapsed_seconds": elapsed_seconds,
                "step_elapsed_seconds": step_timings,
                "vs_source_control": _compare_arrays(result_np, baseline_np),
            }
            points.append(point)
            report["timing"].update(
                {
                    "coordinate_continuation_seconds": coordinate_seconds,
                    "full_decode_count_completed": 1 + len(points),
                    "t4_compute_seconds_through_continuation": time.perf_counter() - started,
                }
            )
            if (alpha, beta) == (1.0, 1.0):
                source_control = {
                    "coordinate": {"alpha": 1.0, "beta": 1.0},
                    **point["vs_source_control"],
                    "baseline_output_key": "source_control_shape_slat",
                    "intervened_output_key": point["output_key"],
                }
                report["source_control"] = source_control
                if not source_control["exact"]:
                    raise RuntimeError(
                        "intervened (1,1) source control is not bit-exact with the normal source route: "
                        f"mean_abs={source_control['mean_abs']} "
                        f"max_abs={source_control['max_abs']} "
                        f"nonzero={source_control['nonzero']}"
                    )
            del targets, result_raw, result, result_np
            torch.cuda.empty_cache()

        source_control = report["source_control"]
        effective_route = {
            "route": requested_route["route"],
            "device_type": next(flow_model.parameters()).device.type,
            "cuda_device": torch.cuda.get_device_name(0),
            "attention_backend": getattr(sparse_config, "ATTN", None),
            "conv_backend": getattr(sparse_config, "CONV", None),
            "steps": 8,
            "block_index": 29,
            "step_index": 0,
            "model_ref": model_ref,
            "one_model_load": True,
            "endpoint_semantics": ENDPOINT_SEMANTICS,
        }
        report.update(
            {
                "status": "done",
                "effective_route": effective_route,
                "points": points,
                "source_control": source_control,
                "timing": {
                    "model_load_seconds": model_load_seconds,
                    "source_baseline_seconds": baseline_seconds,
                    "source_baseline_step_seconds": baseline_step_timings,
                    "coordinate_continuation_seconds": coordinate_seconds,
                    "full_decode_count_completed": 1 + len(coordinates),
                    "full_decode_count_requested": 1 + len(coordinates),
                    "t4_compute_seconds_through_continuation": time.perf_counter() - started,
                },
                "forbidden_inferences": [
                    "not final mesh, texture, winding, or GLB evidence",
                    "not proof of a distinct visual basin until selective source decode",
                    "not a production implementation patch",
                ],
            }
        )
        validate_result_manifest(report, coordinates=coordinates)
        arrays["metadata_json"] = np.asarray(
            json.dumps(report, sort_keys=True, allow_nan=False)
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "write_outputs"
        phase_started = time.perf_counter()
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_npz, **arrays)
        primary_written_this_run = True
        report.update(
            {
                "primary_output_status": "written",
                "primary_output": {
                    "path": str(args.output_npz),
                    "sha256": _sha256(args.output_npz),
                    "size_bytes": args.output_npz.stat().st_size,
                    "keys": sorted(arrays),
                },
                "failure_phase": None,
                "last_trustworthy_phase": "source_control_exact_and_all_points_saved",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        _write_json(args.output_json, report)
        return 0
    except Exception as exc:
        if report.get("primary_output_status") != "not_owned_due_to_path_collision":
            report["primary_output_status"] = (
                "written" if primary_written_this_run and args.output_npz.exists() else "missing"
            )
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "last_trustworthy_phase": last_trustworthy_phase,
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(args.output_json, report)
        print(f"source CUDA block29 basin map failed in {phase}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

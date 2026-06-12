"""Production MLX model loaders for interleaved TRELLIS route contracts.

This module adapts the real TRELLIS model constructors and weight loaders into
the declarative `StageHandleLoaderRequest` contract. Building requests is cheap:
models are constructed and weights are loaded only when a runner opens the
stage-handle context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from .interleaved_generation import StageArtifactValue
from .interleaved_production import (
    build_trellis_production_loader_context,
    build_trellis_production_loader_requests,
    production_model_role_ids,
)
from .model_handle_roles import ModelHandleRole, TRELLIS_MODEL_HANDLE_ROLES
from .stage_handle_loader import (
    LoadedStageHandle,
    StageHandleLoaderCloser,
    StageHandleLoaderFactory,
    StageHandleLoaderRequest,
    StageHandleLoaderRuntime,
)


FLOW_MODEL_ROLE_IDS: tuple[str, ...] = (
    "sparse_structure_flow",
    "shape_flow_lr",
    "shape_flow_hr",
    "texture_flow",
)

ModelConstructor = Callable[[], object]
ModelCleanup = Callable[[object], None]


class ModelWeightLoader(Protocol):
    def __call__(self, model: object, checkpoint_path: str, *, verbose: bool) -> object:
        """Load model weights from a concrete local path."""


class DINOv3WeightLoader(Protocol):
    def __call__(self, model: object, checkpoint_path: str) -> object:
        """Load DINOv3 weights from a concrete local path or snapshot dir."""


class ModelQuantizer(Protocol):
    def __call__(self, model: object, *, bits: int) -> object:
        """Quantize a model in-place."""


HF_CACHE_ROOT = Path("~/.cache/huggingface/hub").expanduser()
TRELLIS_4B_SNAPSHOT = "af44b45f2e35a493886929c6d786e563ec68364d"
TRELLIS_IMAGE_LARGE_SNAPSHOT = "25e0d31ffbebe4b5a97464dd851910efc3002d96"

_TRELLIS_4B_CHECKPOINT_FILES: dict[str, str] = {
    "sparse_structure_flow": "ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "shape_flow_lr": "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "shape_flow_hr": "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    "shape_decoder": "shape_dec_next_dc_f16c32_fp16.safetensors",
    "texture_flow": "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
    "texture_decoder": "tex_dec_next_dc_f16c32_fp16.safetensors",
}
_TRELLIS_IMAGE_LARGE_CHECKPOINT_FILES: dict[str, str] = {
    "sparse_structure_decoder": "ss_dec_conv3d_16l8_fp16.safetensors",
}

_ROLE_BY_ID: dict[str, ModelHandleRole] = {
    role.handle_id: role for role in TRELLIS_MODEL_HANDLE_ROLES
}


def build_trellis_production_model_loader_requests(
    *,
    role_ids: Sequence[str] | None = None,
    constructors: Mapping[str, ModelConstructor] | None = None,
    checkpoint_paths: Mapping[str, str | Path] | None = None,
    hf_cache_root: str | Path | None = None,
    load_model_weights: ModelWeightLoader | None = None,
    load_dinov3_weights: DINOv3WeightLoader | None = None,
    quantize_model: ModelQuantizer | None = None,
    compile_models: bool = False,
    quantize_bits: int = 0,
    requested_loader_route: str | Mapping[str, str] = "mlx",
    closes: Mapping[str, StageHandleLoaderCloser] | None = None,
    metadata: Mapping[str, Mapping[str, StageArtifactValue]] | None = None,
    verbose: bool = False,
) -> tuple[StageHandleLoaderRequest, ...]:
    """Build lazy production model-loader requests for interleaved routes."""

    selected_role_ids = tuple(role_ids) if role_ids is not None else production_model_role_ids()
    factories = build_trellis_production_model_loader_factories(
        role_ids=selected_role_ids,
        constructors=constructors,
        checkpoint_paths=checkpoint_paths,
        hf_cache_root=hf_cache_root,
        load_model_weights=load_model_weights,
        load_dinov3_weights=load_dinov3_weights,
        quantize_model=quantize_model,
        compile_models=compile_models,
        quantize_bits=quantize_bits,
        verbose=verbose,
    )
    return build_trellis_production_loader_requests(
        factories=factories,
        closes=closes
        if closes is not None
        else build_trellis_production_model_loader_closers(role_ids=selected_role_ids),
        role_ids=selected_role_ids,
        requested_loader_route=requested_loader_route,
        metadata=metadata,
    )


def build_trellis_production_model_loader_context(
    *,
    role_ids: Sequence[str] | None = None,
    constructors: Mapping[str, ModelConstructor] | None = None,
    checkpoint_paths: Mapping[str, str | Path] | None = None,
    hf_cache_root: str | Path | None = None,
    load_model_weights: ModelWeightLoader | None = None,
    load_dinov3_weights: DINOv3WeightLoader | None = None,
    quantize_model: ModelQuantizer | None = None,
    compile_models: bool = False,
    quantize_bits: int = 0,
    requested_loader_route: str | Mapping[str, str] = "mlx",
    closes: Mapping[str, StageHandleLoaderCloser] | None = None,
    metadata: Mapping[str, Mapping[str, StageArtifactValue]] | None = None,
    report_path=None,
    run_id: str = "trellis-production-model-loader",
    verbose: bool = False,
):
    """Build stage-runner context callbacks from lazy production loaders."""

    selected_role_ids = tuple(role_ids) if role_ids is not None else production_model_role_ids()
    factories = build_trellis_production_model_loader_factories(
        role_ids=selected_role_ids,
        constructors=constructors,
        checkpoint_paths=checkpoint_paths,
        hf_cache_root=hf_cache_root,
        load_model_weights=load_model_weights,
        load_dinov3_weights=load_dinov3_weights,
        quantize_model=quantize_model,
        compile_models=compile_models,
        quantize_bits=quantize_bits,
        verbose=verbose,
    )
    return build_trellis_production_loader_context(
        factories=factories,
        closes=closes
        if closes is not None
        else build_trellis_production_model_loader_closers(role_ids=selected_role_ids),
        role_ids=selected_role_ids,
        requested_loader_route=requested_loader_route,
        metadata=metadata,
        report_path=report_path,
        run_id=run_id,
    )


def build_trellis_production_model_loader_factories(
    *,
    role_ids: Sequence[str] | None = None,
    constructors: Mapping[str, ModelConstructor] | None = None,
    checkpoint_paths: Mapping[str, str | Path] | None = None,
    hf_cache_root: str | Path | None = None,
    load_model_weights: ModelWeightLoader | None = None,
    load_dinov3_weights: DINOv3WeightLoader | None = None,
    quantize_model: ModelQuantizer | None = None,
    compile_models: bool = False,
    quantize_bits: int = 0,
    verbose: bool = False,
) -> dict[str, StageHandleLoaderFactory]:
    """Build lazy factories keyed by canonical production model role id."""

    if quantize_bits < 0:
        raise ValueError("quantize_bits cannot be negative")
    selected_role_ids = tuple(role_ids) if role_ids is not None else production_model_role_ids()
    constructors = constructors or {}
    checkpoint_paths = checkpoint_paths or {}
    default_checkpoint_paths = build_trellis_production_default_checkpoint_paths(
        hf_cache_root=hf_cache_root,
    )

    factories: dict[str, StageHandleLoaderFactory] = {}
    for role_id in selected_role_ids:
        role = _role_for_id(role_id)
        constructor = constructors.get(role_id)
        checkpoint_path = str(checkpoint_paths.get(role_id, default_checkpoint_paths[role_id]))
        factories[role_id] = _build_role_factory(
            role=role,
            constructor=constructor,
            checkpoint_path=checkpoint_path,
            load_model_weights=load_model_weights,
            load_dinov3_weights=load_dinov3_weights,
            quantize_model=quantize_model,
            compile_models=compile_models,
            quantize_bits=quantize_bits,
            verbose=verbose,
        )
    return factories


def build_trellis_production_model_loader_closers(
    *,
    cleanup_model: Callable[[object], None] | None = None,
    role_ids: Sequence[str] | None = None,
) -> dict[str, StageHandleLoaderCloser]:
    """Build model closers that release handles through the cleanup hook."""

    selected_role_ids = tuple(role_ids) if role_ids is not None else production_model_role_ids()
    closers: dict[str, StageHandleLoaderCloser] = {}
    for role_id in selected_role_ids:
        _role_for_id(role_id)
        closers[role_id] = _build_role_closer(cleanup_model=cleanup_model)
    return closers


def build_trellis_production_default_checkpoint_paths(
    *,
    hf_cache_root: str | Path | None = None,
) -> dict[str, str]:
    """Return generate.py-style local checkpoint paths for production roles."""

    cache_root = Path(hf_cache_root).expanduser() if hf_cache_root is not None else HF_CACHE_ROOT
    trellis_4b_ckpts = (
        cache_root
        / "models--microsoft--TRELLIS.2-4B"
        / "snapshots"
        / TRELLIS_4B_SNAPSHOT
        / "ckpts"
    )
    trellis_large_ckpts = (
        cache_root
        / "models--microsoft--TRELLIS-image-large"
        / "snapshots"
        / TRELLIS_IMAGE_LARGE_SNAPSHOT
        / "ckpts"
    )
    paths: dict[str, str] = {
        "dinov3_image_encoder": _default_dinov3_checkpoint_path(cache_root),
        **{
            role_id: str(trellis_4b_ckpts / filename)
            for role_id, filename in _TRELLIS_4B_CHECKPOINT_FILES.items()
        },
        **{
            role_id: str(trellis_large_ckpts / filename)
            for role_id, filename in _TRELLIS_IMAGE_LARGE_CHECKPOINT_FILES.items()
        },
    }
    missing_roles = sorted(set(_ROLE_BY_ID) - set(paths))
    if missing_roles:
        raise RuntimeError("missing default checkpoint path for role: " + ", ".join(missing_roles))
    return paths


def _default_dinov3_checkpoint_path(cache_root: Path) -> str:
    snapshots_root = (
        cache_root
        / "models--facebook--dinov3-vitl16-pretrain-lvd1689m"
        / "snapshots"
    )
    if snapshots_root.is_dir():
        snapshots = sorted(
            path for path in snapshots_root.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
        if snapshots:
            return str(snapshots[0])
    return str(snapshots_root)


def _build_role_factory(
    *,
    role: ModelHandleRole,
    constructor: ModelConstructor | None,
    checkpoint_path: str,
    load_model_weights: ModelWeightLoader | None,
    load_dinov3_weights: DINOv3WeightLoader | None,
    quantize_model: ModelQuantizer | None,
    compile_models: bool,
    quantize_bits: int,
    verbose: bool,
) -> StageHandleLoaderFactory:
    def factory(runtime: StageHandleLoaderRuntime) -> LoadedStageHandle:
        model = (constructor or _default_constructor_for_role(role.handle_id))()
        loaded_weight_arrays = _load_role_weights(
            role_id=role.handle_id,
            model=model,
            checkpoint_path=checkpoint_path,
            load_model_weights=load_model_weights,
            load_dinov3_weights=load_dinov3_weights,
            verbose=verbose,
        )
        optimized = _maybe_optimize_flow_model(
            role_id=role.handle_id,
            model=model,
            quantize_model=quantize_model,
            quantize_bits=quantize_bits,
            compile_models=compile_models,
        )
        metadata: dict[str, StageArtifactValue] = {
            "weights_path": checkpoint_path,
            "loader_kind": "mlx_model",
            "compiled": optimized["compiled"],
            "quantize_bits": optimized["quantize_bits"],
        }
        if isinstance(loaded_weight_arrays, int):
            metadata["loaded_weight_arrays"] = loaded_weight_arrays
        return LoadedStageHandle(
            handle=model,
            effective_loader_route=runtime.requested_loader_route,
            metadata=metadata,
        )

    return factory


def _build_role_closer(
    *,
    cleanup_model: Callable[[object], None] | None,
) -> StageHandleLoaderCloser:
    def close(runtime: StageHandleLoaderRuntime, handle: object) -> None:
        del runtime
        cleanup = cleanup_model or _default_cleanup_model
        cleanup(handle)

    return close


def _load_role_weights(
    *,
    role_id: str,
    model: object,
    checkpoint_path: str,
    load_model_weights: ModelWeightLoader | None,
    load_dinov3_weights: DINOv3WeightLoader | None,
    verbose: bool,
) -> object:
    if role_id == "dinov3_image_encoder":
        loader = load_dinov3_weights or _default_load_dinov3_weights
        return loader(model, checkpoint_path)
    loader = load_model_weights or _default_load_model_weights
    return loader(model, checkpoint_path, verbose=verbose)


def _maybe_optimize_flow_model(
    *,
    role_id: str,
    model: object,
    quantize_model: ModelQuantizer | None,
    quantize_bits: int,
    compile_models: bool,
) -> dict[str, bool | int]:
    if role_id not in FLOW_MODEL_ROLE_IDS:
        return {"compiled": False, "quantize_bits": 0}

    if quantize_bits:
        quantizer = quantize_model or _default_quantize_model
        quantizer(model, bits=quantize_bits)
    compiled = False
    if compile_models:
        compile_method = getattr(model, "compile", None)
        if compile_method is None:
            raise TypeError(f"model role {role_id} does not expose compile()")
        compile_method()
        compiled = True
    return {"compiled": compiled, "quantize_bits": quantize_bits if quantize_bits else 0}


def _default_constructor_for_role(role_id: str) -> ModelConstructor:
    if role_id == "dinov3_image_encoder":
        from .models.dinov3 import DINOv3ViT

        return DINOv3ViT
    if role_id == "sparse_structure_flow":
        from .models.sparse_structure_flow import SparseStructureFlowModel

        return SparseStructureFlowModel
    if role_id == "sparse_structure_decoder":
        from .models.sparse_structure_decoder import SparseStructureDecoder

        return SparseStructureDecoder
    if role_id in {"shape_flow_lr", "shape_flow_hr"}:
        from .models.slat_flow import SLatFlowModel

        return SLatFlowModel
    if role_id == "shape_decoder":
        from .models.shape_slat_decoder import SLatDecoder

        return lambda: SLatDecoder(out_channels=7, pred_subdiv=True)
    if role_id == "texture_flow":
        from .models.slat_flow import SLatFlowModel

        return lambda: SLatFlowModel(in_channels=64, out_channels=32)
    if role_id == "texture_decoder":
        from .models.shape_slat_decoder import SLatDecoder

        return lambda: SLatDecoder(out_channels=6, pred_subdiv=False)
    raise ValueError(f"unknown production model role: {role_id}")


def _default_load_model_weights(model: object, checkpoint_path: str, *, verbose: bool) -> object:
    from .weight_loader import load_weights

    return load_weights(model, checkpoint_path, verbose=verbose)


def _default_load_dinov3_weights(model: object, checkpoint_path: str) -> object:
    from .models.dinov3 import load_dinov3_weights

    return load_dinov3_weights(model, checkpoint_path)


def _default_quantize_model(model: object, *, bits: int) -> object:
    from .quantize import quantize_model

    return quantize_model(model, bits=bits)


def _default_cleanup_model(model: object) -> None:
    from .cleanup import cleanup_model

    cleanup_model(model)


def _role_for_id(role_id: str) -> ModelHandleRole:
    try:
        return _ROLE_BY_ID[role_id]
    except KeyError as exc:
        raise ValueError(f"unknown production model role: {role_id}") from exc

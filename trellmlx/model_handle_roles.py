"""Declarative TRELLIS model-handle roles for stage-runner contexts.

This module names model roles and checkpoint identities only. It does not load
weights, construct models, run samplers, or call `generate.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .interleaved_generation import StageArtifactValue, _validate_artifacts
from .stage_handle_loader import (
    StageHandleLoaderCloser,
    StageHandleLoaderFactory,
    StageHandleLoaderRequest,
)


@dataclass(frozen=True)
class ModelHandleRole:
    """Portable declaration for one reusable model-handle role."""

    handle_id: str
    stage: str
    model_family: str
    checkpoint: str
    consumer_stages: Sequence[str] = field(default_factory=tuple)
    kind: str = "model"
    default_loader_route: str = "mlx"
    extra_metadata: Mapping[str, StageArtifactValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.handle_id:
            raise ValueError("ModelHandleRole requires a handle_id")
        if not self.stage:
            raise ValueError("ModelHandleRole requires a stage")
        if not self.model_family:
            raise ValueError("ModelHandleRole requires a model_family")
        if not self.checkpoint:
            raise ValueError("ModelHandleRole requires a checkpoint")
        consumer_stages = tuple(self.consumer_stages) or (self.stage,)
        if self.stage not in consumer_stages:
            raise ValueError("ModelHandleRole consumer_stages must include stage")
        for consumer_stage in consumer_stages:
            if not consumer_stage:
                raise ValueError("ModelHandleRole consumer_stages cannot contain empty stages")
        object.__setattr__(self, "consumer_stages", consumer_stages)
        if not self.kind:
            raise ValueError("ModelHandleRole requires a kind")
        if not self.default_loader_route:
            raise ValueError("ModelHandleRole requires a default_loader_route")
        object.__setattr__(self, "extra_metadata", _validate_artifacts(self.extra_metadata))

    def metadata(self) -> dict[str, StageArtifactValue]:
        metadata = {
            "role": self.handle_id,
            "stage": self.stage,
            "model_family": self.model_family,
            "checkpoint": self.checkpoint,
            **self.extra_metadata,
        }
        if len(self.consumer_stages) > 1:
            metadata["consumer_stages"] = ",".join(self.consumer_stages)
        return metadata


TRELLIS_MODEL_HANDLE_ROLES: tuple[ModelHandleRole, ...] = (
    ModelHandleRole(
        handle_id="dinov3_image_encoder",
        stage="image_conditioning",
        model_family="dinov3",
        checkpoint="facebook/dinov3-vitl16-pretrain-lvd1689m",
    ),
    ModelHandleRole(
        handle_id="sparse_structure_flow",
        stage="sparse_structure",
        model_family="sparse_structure_flow",
        checkpoint="microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    ),
    ModelHandleRole(
        handle_id="sparse_structure_decoder",
        stage="sparse_structure",
        model_family="sparse_structure_decoder",
        checkpoint="microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    ),
    ModelHandleRole(
        handle_id="shape_flow_lr",
        stage="lr_shape_latent",
        model_family="slat_flow_shape",
        checkpoint="microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        extra_metadata={"resolution": 512},
    ),
    ModelHandleRole(
        handle_id="shape_flow_hr",
        stage="hr_shape_latent",
        model_family="slat_flow_shape",
        checkpoint="microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        extra_metadata={"resolution": 1024},
    ),
    ModelHandleRole(
        handle_id="shape_decoder",
        stage="shape_decode",
        model_family="slat_decoder_shape",
        checkpoint="microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        consumer_stages=("hr_coordinates", "shape_decode"),
    ),
    ModelHandleRole(
        handle_id="texture_flow",
        stage="texture_latent",
        model_family="slat_flow_texture",
        checkpoint="microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
        extra_metadata={"resolution": 512},
    ),
    ModelHandleRole(
        handle_id="texture_decoder",
        stage="texture_decode",
        model_family="slat_decoder_texture",
        checkpoint="microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    ),
)


def build_trellis_model_role_requests(
    *,
    factories: Mapping[str, StageHandleLoaderFactory],
    closes: Mapping[str, StageHandleLoaderCloser] | None = None,
    role_ids: Sequence[str] | None = None,
    requested_loader_route: str | Mapping[str, str] = "mlx",
    metadata: Mapping[str, Mapping[str, StageArtifactValue]] | None = None,
) -> tuple[StageHandleLoaderRequest, ...]:
    """Build loader requests for canonical TRELLIS model roles."""

    selected_roles = _select_roles(TRELLIS_MODEL_HANDLE_ROLES, role_ids)
    return build_model_role_requests(
        selected_roles,
        factories=factories,
        closes=closes,
        requested_loader_route=requested_loader_route,
        metadata=metadata,
    )


def build_model_role_requests(
    roles: Sequence[ModelHandleRole],
    *,
    factories: Mapping[str, StageHandleLoaderFactory],
    closes: Mapping[str, StageHandleLoaderCloser] | None = None,
    requested_loader_route: str | Mapping[str, str] = "mlx",
    metadata: Mapping[str, Mapping[str, StageArtifactValue]] | None = None,
) -> tuple[StageHandleLoaderRequest, ...]:
    """Convert declarative model roles into stage handle loader requests."""

    closes = closes or {}
    metadata = metadata or {}
    seen: set[str] = set()
    requests: list[StageHandleLoaderRequest] = []
    for role in roles:
        if role.handle_id in seen:
            raise ValueError(f"duplicate model handle role: {role.handle_id}")
        seen.add(role.handle_id)
        if role.handle_id not in factories:
            raise ValueError(f"missing factory for model handle role: {role.handle_id}")
        role_metadata = role.metadata()
        caller_metadata = _validate_artifacts(metadata.get(role.handle_id, {}))
        duplicate_metadata = sorted(set(role_metadata) & set(caller_metadata))
        if duplicate_metadata:
            raise ValueError(
                "model role metadata cannot override canonical keys: "
                + ", ".join(duplicate_metadata)
            )
        request_metadata = {**role_metadata, **caller_metadata}
        requests.append(
            StageHandleLoaderRequest(
                handle_id=role.handle_id,
                kind=role.kind,
                requested_loader_route=_route_for_role(role, requested_loader_route),
                factory=factories[role.handle_id],
                close=closes.get(role.handle_id),
                metadata=request_metadata,
            )
        )
    return tuple(requests)


def _select_roles(
    roles: Sequence[ModelHandleRole],
    role_ids: Sequence[str] | None,
) -> tuple[ModelHandleRole, ...]:
    if role_ids is None:
        return tuple(roles)
    by_id = {role.handle_id: role for role in roles}
    selected: list[ModelHandleRole] = []
    seen: set[str] = set()
    for role_id in role_ids:
        if role_id in seen:
            raise ValueError(f"duplicate model handle role: {role_id}")
        seen.add(role_id)
        try:
            selected.append(by_id[role_id])
        except KeyError as exc:
            raise ValueError(f"unknown model handle role: {role_id}") from exc
    return tuple(selected)


def _route_for_role(
    role: ModelHandleRole,
    requested_loader_route: str | Mapping[str, str],
) -> str:
    if isinstance(requested_loader_route, str):
        return requested_loader_route or role.default_loader_route
    try:
        route = requested_loader_route[role.handle_id]
    except KeyError as exc:
        raise ValueError(f"missing requested loader route for model handle role: {role.handle_id}") from exc
    if not route:
        raise ValueError(f"empty requested loader route for model handle role: {role.handle_id}")
    return route

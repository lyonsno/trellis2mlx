"""No-generation TRELLIS production route wiring for interleaved runs.

This module declares how the existing TRELLIS stage order maps to reusable
model roles and handler construction. It does not load weights, run samplers,
invoke `generate.py`, or claim runtime speedup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .interleaved_generation import (
    DEFAULT_STAGE_SEQUENCE,
    GenerationStageInvocation,
    GenerationStageResult,
    JobState,
    StageArtifactValue,
    StageContextCloser,
    StageContextFactory,
    StageExecutionContext,
    StageHandler,
    StageRunnerOutput,
)
from .model_handle_roles import TRELLIS_MODEL_HANDLE_ROLES, build_trellis_model_role_requests
from .stage_handle_loader import (
    StageHandleLoaderCloser,
    StageHandleLoaderFactory,
    StageHandleLoaderRequest,
    build_stage_loader_context,
)
from .stage_handlers import (
    StageHandlerFixture,
    StageHandlerRuntime,
    build_model_role_stage_handler,
)


@dataclass(frozen=True)
class TrellisProductionStageRoute:
    """Static stage route entry for production-shaped interleaved execution."""

    stage: str
    role_ids: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("TrellisProductionStageRoute requires a stage")
        object.__setattr__(self, "role_ids", _validate_unique_strings("role_id", self.role_ids))
        object.__setattr__(
            self,
            "required_artifacts",
            _validate_unique_strings("required artifact", self.required_artifacts),
        )


def _validate_unique_strings(label: str, values: Sequence[str]) -> tuple[str, ...]:
    values_tuple = tuple(values)
    seen: set[str] = set()
    for value in values_tuple:
        if not value:
            raise ValueError(f"{label} entries cannot be empty")
        if value in seen:
            raise ValueError(f"duplicate {label}: {value}")
        seen.add(value)
    return values_tuple


TRELLIS_PRODUCTION_STAGE_ROUTES: tuple[TrellisProductionStageRoute, ...] = (
    TrellisProductionStageRoute("image_conditioning", ("dinov3_image_encoder",)),
    TrellisProductionStageRoute(
        "sparse_structure",
        ("sparse_structure_flow", "sparse_structure_decoder"),
        ("conditioning_key",),
    ),
    TrellisProductionStageRoute("lr_shape_latent", ("shape_flow_lr",), ("sparse_structure_key",)),
    TrellisProductionStageRoute("hr_coordinates", ("shape_decoder",), ("lr_shape_latent_key",)),
    TrellisProductionStageRoute("hr_shape_latent", ("shape_flow_hr",), ("hr_coordinate_key",)),
    TrellisProductionStageRoute("shape_decode", ("shape_decoder",), ("hr_shape_latent_key",)),
    TrellisProductionStageRoute("mesh_extract", required_artifacts=("shape_key",)),
    TrellisProductionStageRoute("mesh_postprocess", required_artifacts=("raw_mesh_key",)),
    TrellisProductionStageRoute(
        "texture_latent",
        ("texture_flow",),
        ("mesh_key", "conditioning_key"),
    ),
    TrellisProductionStageRoute("texture_decode", ("texture_decoder",), ("texture_latent_key",)),
    TrellisProductionStageRoute("texture_bake", required_artifacts=("mesh_key", "texture_key")),
    TrellisProductionStageRoute("export", required_artifacts=("mesh_key", "texture_bake_key")),
)


def production_stage_sequence() -> tuple[str, ...]:
    """Return the production stage order used by interleaved TRELLIS runs."""

    return tuple(route.stage for route in TRELLIS_PRODUCTION_STAGE_ROUTES)


def production_model_role_ids(
    routes: Sequence[TrellisProductionStageRoute] = TRELLIS_PRODUCTION_STAGE_ROUTES,
) -> tuple[str, ...]:
    """Return canonical model role ids used by the route plan.

    Role ids are emitted in canonical `TRELLIS_MODEL_HANDLE_ROLES` order rather
    than stage occurrence order so shared roles such as `shape_decoder` are
    loaded exactly once with stable request ordering.
    """

    used_role_ids = {role_id for route in routes for role_id in route.role_ids}
    canonical_role_ids = tuple(
        role.handle_id
        for role in TRELLIS_MODEL_HANDLE_ROLES
        if role.handle_id in used_role_ids
    )
    unknown_role_ids = sorted(used_role_ids - set(canonical_role_ids))
    if unknown_role_ids:
        raise ValueError(f"unknown production model role: {', '.join(unknown_role_ids)}")
    return canonical_role_ids


def build_trellis_production_loader_requests(
    *,
    factories: Mapping[str, StageHandleLoaderFactory],
    closes: Mapping[str, StageHandleLoaderCloser] | None = None,
    role_ids: Sequence[str] | None = None,
    requested_loader_route: str | Mapping[str, str] = "mlx",
    metadata: Mapping[str, Mapping[str, StageArtifactValue]] | None = None,
) -> tuple[StageHandleLoaderRequest, ...]:
    """Build no-generation loader requests for production route model roles."""

    selected_role_ids = tuple(role_ids) if role_ids is not None else production_model_role_ids()
    return build_trellis_model_role_requests(
        factories=factories,
        closes=closes,
        role_ids=selected_role_ids,
        requested_loader_route=requested_loader_route,
        metadata=metadata,
    )


def build_trellis_production_loader_context(
    *,
    factories: Mapping[str, StageHandleLoaderFactory],
    closes: Mapping[str, StageHandleLoaderCloser] | None = None,
    role_ids: Sequence[str] | None = None,
    requested_loader_route: str | Mapping[str, str] = "mlx",
    metadata: Mapping[str, Mapping[str, StageArtifactValue]] | None = None,
    report_path=None,
    run_id: str = "trellis-production-route",
) -> tuple[StageContextFactory, StageContextCloser]:
    """Build `StageRunner` context callbacks for production route roles."""

    requests = build_trellis_production_loader_requests(
        factories=factories,
        closes=closes,
        role_ids=role_ids,
        requested_loader_route=requested_loader_route,
        metadata=metadata,
    )
    return build_stage_loader_context(requests, report_path=report_path, run_id=run_id)


def build_trellis_production_stage_handlers(
    *,
    fixtures: Mapping[str, StageHandlerFixture],
    stages: Sequence[str] = DEFAULT_STAGE_SEQUENCE,
    routes: Sequence[TrellisProductionStageRoute] = TRELLIS_PRODUCTION_STAGE_ROUTES,
) -> dict[str, StageHandler]:
    """Build fixture-backed handlers for the requested production stages."""

    route_by_stage = _route_by_stage(routes)
    handlers: dict[str, StageHandler] = {}
    for stage in stages:
        try:
            route = route_by_stage[stage]
        except KeyError as exc:
            raise ValueError(f"unknown production stage: {stage}") from exc
        try:
            fixture = fixtures[stage]
        except KeyError as exc:
            raise ValueError(f"missing fixture for production stage: {stage}") from exc
        if route.role_ids:
            handlers[stage] = build_model_role_stage_handler(
                stage=stage,
                role_ids=route.role_ids,
                required_artifacts=route.required_artifacts,
                fixture=fixture,
            )
        else:
            handlers[stage] = _build_no_model_stage_handler(route=route, fixture=fixture)
    return handlers


def _build_no_model_stage_handler(
    *,
    route: TrellisProductionStageRoute,
    fixture: StageHandlerFixture,
) -> StageHandler:
    def handler(
        invocation: GenerationStageInvocation,
        state: JobState,
        context: StageExecutionContext,
    ) -> StageRunnerOutput | GenerationStageResult:
        if invocation.stage != route.stage:
            raise ValueError(
                f"handler for stage {route.stage} cannot run invocation stage {invocation.stage}"
            )
        missing_artifacts = [
            artifact for artifact in route.required_artifacts if artifact not in state.artifacts
        ]
        if missing_artifacts:
            raise KeyError(
                f"missing required state artifact for {route.stage}: "
                + ", ".join(missing_artifacts)
            )
        return fixture(
            StageHandlerRuntime(
                invocation=invocation,
                state=state,
                context=context,
                handles={},
                handle_metadata={},
            )
        )

    return handler


def _route_by_stage(
    routes: Sequence[TrellisProductionStageRoute],
) -> dict[str, TrellisProductionStageRoute]:
    route_by_stage: dict[str, TrellisProductionStageRoute] = {}
    for route in routes:
        if route.stage in route_by_stage:
            raise ValueError(f"duplicate production stage: {route.stage}")
        route_by_stage[route.stage] = route
    return route_by_stage


if production_stage_sequence() != DEFAULT_STAGE_SEQUENCE:
    raise RuntimeError("TRELLIS production stage routes must match DEFAULT_STAGE_SEQUENCE")

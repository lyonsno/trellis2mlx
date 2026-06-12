"""No-heavy production assembly for interleaved TRELLIS runners.

This module composes the production route, lazy model-loader, and stage-handler
contracts into one object that callers can hand to `StageRunner` or
`run_interleaved_batch` without manually threading factories and closers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .interleaved_generation import (
    DEFAULT_STAGE_SEQUENCE,
    InterleavedBatchPlan,
    StageContextCloser,
    StageContextFactory,
    StageHandler,
    StageRunner,
)
from .interleaved_production import (
    TRELLIS_PRODUCTION_STAGE_ROUTES,
    build_trellis_production_stage_handlers,
    production_model_role_ids,
    production_stage_sequence,
)
from .production_model_loader import (
    DINOv3WeightLoader,
    ModelConstructor,
    ModelQuantizer,
    ModelWeightLoader,
    build_trellis_production_model_loader_closers,
    build_trellis_production_model_loader_requests,
)
from .stage_handle_loader import StageHandleLoaderCloser, StageHandleLoaderRequest, build_stage_loader_context
from .stage_handlers import StageHandlerFixture


@dataclass(frozen=True)
class TrellisProductionAssembly:
    """Runner-ready production components for interleaved TRELLIS execution."""

    stages: tuple[str, ...]
    role_ids: tuple[str, ...]
    loader_requests: tuple[StageHandleLoaderRequest, ...]
    handlers: Mapping[str, StageHandler]
    context_factory: StageContextFactory
    context_closer: StageContextCloser

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "role_ids", tuple(self.role_ids))
        object.__setattr__(self, "loader_requests", tuple(self.loader_requests))
        object.__setattr__(self, "handlers", dict(self.handlers))

    def build_runner(self, plan: InterleavedBatchPlan) -> StageRunner:
        """Build a `StageRunner` using this assembly's handlers and context."""

        return StageRunner(
            plan,
            handlers=self.handlers,
            context_factory=self.context_factory,
            context_closer=self.context_closer,
        )


def build_trellis_production_assembly(
    *,
    fixtures: Mapping[str, StageHandlerFixture],
    stages: Sequence[str] = DEFAULT_STAGE_SEQUENCE,
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
    cleanup_model=None,
    closes: Mapping[str, StageHandleLoaderCloser] | None = None,
    report_path=None,
    run_id: str = "trellis-production-assembly",
    verbose: bool = False,
) -> TrellisProductionAssembly:
    """Build runner-ready production handlers and lazy model-loader context.

    The returned assembly does not construct models or load weights. Those
    effects happen only when the context factory is invoked by a runner.
    """

    selected_stages = tuple(stages)
    selected_role_ids = (
        tuple(role_ids)
        if role_ids is not None
        else _production_model_role_ids_for_stages(selected_stages)
    )
    if cleanup_model is not None and closes is not None:
        raise ValueError("cleanup_model cannot be combined with explicit closes")
    selected_closes = (
        closes
        if closes is not None
        else build_trellis_production_model_loader_closers(
            cleanup_model=cleanup_model,
            role_ids=selected_role_ids,
        )
    )
    loader_requests = build_trellis_production_model_loader_requests(
        role_ids=selected_role_ids,
        constructors=constructors,
        checkpoint_paths=checkpoint_paths,
        hf_cache_root=hf_cache_root,
        load_model_weights=load_model_weights,
        load_dinov3_weights=load_dinov3_weights,
        quantize_model=quantize_model,
        compile_models=compile_models,
        quantize_bits=quantize_bits,
        requested_loader_route=requested_loader_route,
        closes=selected_closes,
        verbose=verbose,
    )
    context_factory, context_closer = build_stage_loader_context(
        loader_requests,
        report_path=report_path,
        run_id=run_id,
    )
    handlers = build_trellis_production_stage_handlers(
        fixtures=fixtures,
        stages=selected_stages,
    )
    return TrellisProductionAssembly(
        stages=selected_stages,
        role_ids=selected_role_ids,
        loader_requests=loader_requests,
        handlers=handlers,
        context_factory=context_factory,
        context_closer=context_closer,
    )


def _production_model_role_ids_for_stages(stages: Sequence[str]) -> tuple[str, ...]:
    route_by_stage = {route.stage: route for route in TRELLIS_PRODUCTION_STAGE_ROUTES}
    selected_routes = []
    for stage in stages:
        try:
            selected_routes.append(route_by_stage[stage])
        except KeyError as exc:
            raise ValueError(f"unknown production stage: {stage}") from exc
    return production_model_role_ids(selected_routes)


if production_stage_sequence() != DEFAULT_STAGE_SEQUENCE:
    raise RuntimeError("TRELLIS production assembly stages must match DEFAULT_STAGE_SEQUENCE")

"""No-generation stage handle loader probe.

This module routes loader-shaped handle specs through the shared
`StageExecutionContext` contract and writes a durable report. It deliberately
does not call `generate.py`, run samplers, or execute stage handlers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .interleaved_generation import (
    InterleavedBatchPlan,
    StageArtifactValue,
    StageHandleCloseError,
    StageHandleCloseReport,
    StageHandleFactoryError,
    StageHandleFactoryResult,
    StageHandleLoadError,
    StageHandleLoadReport,
    StageHandleRuntime,
    StageHandleSpec,
    _validate_artifacts,
    build_stage_context_factory,
)


SCHEMA = "trellis2mlx.stage_handle_loader_report.v1"
_RESERVED_LOADER_METADATA_KEYS = frozenset({"effective_loader_route", "requested_loader_route"})


@dataclass(frozen=True)
class StageHandleLoaderRuntime:
    """Inputs passed to a no-generation handle loader callback."""

    handle_id: str
    requested_loader_route: str
    stage_runtime: StageHandleRuntime


@dataclass(frozen=True)
class LoadedStageHandle:
    """A loaded handle plus route identity observed by the loader."""

    handle: object
    effective_loader_route: str
    metadata: Mapping[str, StageArtifactValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.effective_loader_route:
            raise ValueError("LoadedStageHandle requires an effective_loader_route")
        metadata = _validate_loader_metadata(self.metadata)
        object.__setattr__(self, "metadata", metadata)


StageHandleLoaderFactory = Callable[[StageHandleLoaderRuntime], LoadedStageHandle]
StageHandleLoaderCloser = Callable[[StageHandleLoaderRuntime, object], None]


@dataclass(frozen=True)
class StageHandleLoaderRequest:
    """Declarative no-generation loader request for one stage handle."""

    handle_id: str
    kind: str
    requested_loader_route: str
    factory: StageHandleLoaderFactory
    close: StageHandleLoaderCloser | None = None
    metadata: Mapping[str, StageArtifactValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.handle_id:
            raise ValueError("StageHandleLoaderRequest requires a handle_id")
        if not self.kind:
            raise ValueError("StageHandleLoaderRequest requires a kind")
        if not self.requested_loader_route:
            raise ValueError("StageHandleLoaderRequest requires a requested_loader_route")
        object.__setattr__(self, "metadata", _validate_loader_metadata(self.metadata))

    def to_stage_handle_spec(self) -> StageHandleSpec:
        static_metadata = {"requested_loader_route": self.requested_loader_route, **self.metadata}

        def factory(stage_runtime: StageHandleRuntime) -> StageHandleFactoryResult:
            loader_runtime = StageHandleLoaderRuntime(
                handle_id=self.handle_id,
                requested_loader_route=self.requested_loader_route,
                stage_runtime=stage_runtime,
            )
            loaded = self.factory(loader_runtime)
            if loaded.effective_loader_route != self.requested_loader_route:
                raise StageHandleFactoryError(
                    f"effective loader route mismatch for {self.handle_id}: "
                    f"requested {self.requested_loader_route}, got {loaded.effective_loader_route}",
                    metadata={"effective_loader_route": loaded.effective_loader_route},
                )
            return StageHandleFactoryResult(
                handle=loaded.handle,
                metadata={"effective_loader_route": loaded.effective_loader_route, **loaded.metadata},
            )

        def close(stage_runtime: StageHandleRuntime, handle: object) -> None:
            if self.close is None:
                return
            loader_runtime = StageHandleLoaderRuntime(
                handle_id=self.handle_id,
                requested_loader_route=self.requested_loader_route,
                stage_runtime=stage_runtime,
            )
            self.close(loader_runtime, handle)

        return StageHandleSpec(
            handle_id=self.handle_id,
            kind=self.kind,
            factory=factory,
            close=close,
            metadata=static_metadata,
        )


@dataclass(frozen=True)
class StageHandleLoaderReport:
    """Durable report for a no-generation handle-loader probe."""

    schema: str
    run_id: str
    ok: bool
    requested_handle_ids: tuple[str, ...]
    loaded_handle_ids: tuple[str, ...]
    load_reports: tuple[StageHandleLoadReport, ...] = ()
    close_reports: tuple[StageHandleCloseReport, ...] = ()
    failure_phase: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_stage_handles(
    plan: InterleavedBatchPlan,
    requests: Sequence[StageHandleLoaderRequest],
    *,
    report_path: Path | str | None = None,
    run_id: str = "stage-handle-loader",
) -> StageHandleLoaderReport:
    """Load requested handles, close them, and optionally write a report."""

    requested_handle_ids = tuple(request.handle_id for request in requests)
    try:
        specs = [request.to_stage_handle_spec() for request in requests]
        context_factory, context_closer = build_stage_context_factory(specs, run_id=run_id)
        context = context_factory(plan)
        close_error: StageHandleCloseError | None = None
        try:
            context_closer(context)
        except StageHandleCloseError as exc:
            close_error = exc
        report = StageHandleLoaderReport(
            schema=SCHEMA,
            run_id=run_id,
            ok=close_error is None,
            requested_handle_ids=requested_handle_ids,
            loaded_handle_ids=context.handle_ids,
            load_reports=context.load_reports,
            close_reports=context.close_reports,
            failure_phase="close" if close_error is not None else None,
            error=str(close_error) if close_error is not None else None,
        )
    except StageHandleLoadError as exc:
        report = StageHandleLoaderReport(
            schema=SCHEMA,
            run_id=run_id,
            ok=False,
            requested_handle_ids=requested_handle_ids,
            loaded_handle_ids=(),
            load_reports=context_factory.last_load_reports,
            close_reports=context_factory.last_close_reports,
            failure_phase="load",
            error=str(exc),
        )
    except Exception as exc:
        report = StageHandleLoaderReport(
            schema=SCHEMA,
            run_id=run_id,
            ok=False,
            requested_handle_ids=requested_handle_ids,
            loaded_handle_ids=(),
            failure_phase="setup",
            error=str(exc),
        )

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return report


def _validate_loader_metadata(metadata: Mapping[str, StageArtifactValue]) -> dict[str, StageArtifactValue]:
    metadata = _validate_artifacts(metadata)
    reserved_keys = sorted(set(metadata) & _RESERVED_LOADER_METADATA_KEYS)
    if reserved_keys:
        raise ValueError(f"loader metadata cannot use reserved route keys: {', '.join(reserved_keys)}")
    return metadata

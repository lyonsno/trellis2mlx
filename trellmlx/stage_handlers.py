"""Fixture-backed stage handler contracts for reusable model-role handles.

This module wires `StageRunner` handlers to already-loaded model-role handles.
It deliberately does not construct models, load weights, run samplers, or call
`generate.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .interleaved_generation import (
    GenerationStageInvocation,
    GenerationStageResult,
    JobState,
    StageExecutionContext,
    StageHandler,
    StageRunnerOutput,
    StageArtifactValue,
)


StageHandlerFixture = Callable[["StageHandlerRuntime"], StageRunnerOutput | GenerationStageResult]


@dataclass(frozen=True)
class StageHandlerRuntime:
    """Inputs passed to a fixture-backed stage handler."""

    invocation: GenerationStageInvocation
    state: JobState
    context: StageExecutionContext
    handles: Mapping[str, object]
    handle_metadata: Mapping[str, Mapping[str, StageArtifactValue]]


def build_model_role_stage_handler(
    *,
    stage: str,
    role_ids: Sequence[str],
    fixture: StageHandlerFixture,
    required_artifacts: Sequence[str] = (),
) -> StageHandler:
    """Build a `StageRunner` handler backed by declared model-role handles."""

    if not stage:
        raise ValueError("build_model_role_stage_handler requires a stage")
    role_ids_tuple = _validate_unique_strings("model role", role_ids)
    if not role_ids_tuple:
        raise ValueError("build_model_role_stage_handler requires at least one role_id")
    required_artifacts_tuple = _validate_unique_strings("required artifact", required_artifacts)

    def handler(
        invocation: GenerationStageInvocation,
        state: JobState,
        context: StageExecutionContext,
    ) -> StageRunnerOutput | GenerationStageResult:
        if invocation.stage != stage:
            raise ValueError(f"handler for stage {stage} cannot run invocation stage {invocation.stage}")
        missing_artifacts = [artifact for artifact in required_artifacts_tuple if artifact not in state.artifacts]
        if missing_artifacts:
            raise KeyError(
                f"missing required state artifact for {stage}: "
                + ", ".join(missing_artifacts)
            )

        handles: dict[str, object] = {}
        handle_metadata: dict[str, Mapping[str, StageArtifactValue]] = {}
        for role_id in role_ids_tuple:
            handles[role_id] = context.require_handle(role_id)
            try:
                metadata = context.handle_metadata[role_id]
            except KeyError as exc:
                raise KeyError(f"missing stage handle metadata: {role_id}") from exc
            _validate_model_role_metadata(stage=stage, role_id=role_id, metadata=metadata)
            handle_metadata[role_id] = metadata

        runtime = StageHandlerRuntime(
            invocation=invocation,
            state=state,
            context=context,
            handles=handles,
            handle_metadata=handle_metadata,
        )
        return fixture(runtime)

    return handler


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


def _validate_model_role_metadata(
    *,
    stage: str,
    role_id: str,
    metadata: Mapping[str, StageArtifactValue],
) -> None:
    if metadata.get("role") != role_id:
        raise ValueError(f"model role metadata mismatch for {role_id}")
    required_keys = ("stage", "model_family", "checkpoint", "requested_loader_route", "effective_loader_route")
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:
        raise ValueError(f"model role {role_id} metadata missing keys: {', '.join(missing_keys)}")
    for key in ("model_family", "checkpoint", "requested_loader_route", "effective_loader_route"):
        _require_nonempty_string_metadata(role_id=role_id, key=key, value=metadata[key])
    _validate_loader_route_identity(
        role_id=role_id,
        requested=metadata["requested_loader_route"],
        effective=metadata["effective_loader_route"],
    )

    declared_stages = _declared_consumer_stages(metadata)
    if stage not in declared_stages:
        raise ValueError(f"model role {role_id} is not declared for stage {stage}")


def _require_nonempty_string_metadata(*, role_id: str, key: str, value: StageArtifactValue) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"model role {role_id} metadata {key} must be a nonempty string")


def _validate_loader_route_identity(
    *,
    role_id: str,
    requested: StageArtifactValue,
    effective: StageArtifactValue,
) -> None:
    if requested != effective:
        raise ValueError(
            f"model role {role_id} loader route mismatch: "
            f"requested {requested}, got {effective}"
        )


def _declared_consumer_stages(metadata: Mapping[str, StageArtifactValue]) -> tuple[str, ...]:
    if "consumer_stages" in metadata:
        consumer_stages = metadata["consumer_stages"]
        if not isinstance(consumer_stages, str) or not consumer_stages:
            raise ValueError(
                f"model role {metadata['role']} metadata consumer_stages must be a nonempty string"
            )
        stages = tuple(stage.strip() for stage in consumer_stages.split(",") if stage.strip())
        if not stages:
            raise ValueError(
                f"model role {metadata['role']} metadata consumer_stages must name at least one stage"
            )
        return stages
    stage = metadata["stage"]
    if not isinstance(stage, str) or not stage:
        raise ValueError("model role metadata stage must be a nonempty string")
    return (stage,)

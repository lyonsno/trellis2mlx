"""No-generation mesh postprocess stage adapter.

This module maps post-decode mesh cleanup/simplification into the
fixture-backed `StageRunner` contract. It deliberately does not import
`generate.py`, mesh cleanup code, fast simplification, xatlas, or model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .interleaved_generation import (
    GenerationStageInvocation,
    GenerationStageResult,
    JobState,
    StageExecutionContext,
    StageRunnerOutput,
)


MESH_POSTPROCESS_STAGE = "mesh_postprocess"
RAW_MESH_KEY_ARTIFACT = "raw_mesh_key"


@dataclass(frozen=True)
class MeshPostprocessRuntime:
    """Inputs passed to the injected no-generation mesh postprocess fixture."""

    invocation: GenerationStageInvocation
    state: JobState
    context: StageExecutionContext
    input_mesh_key: str
    mesh_object: object


@dataclass(frozen=True)
class MeshPostprocessFixtureResult:
    """Scalar witness for a mesh postprocess fixture result."""

    mesh_key: str
    vertices: int
    faces: int
    elapsed_seconds: float = 0.0
    mesh_object: object | None = None

    def __post_init__(self) -> None:
        if not self.mesh_key:
            raise ValueError("mesh_key must be nonempty")
        _require_integer_count("vertices", self.vertices)
        _require_integer_count("faces", self.faces)
        if self.vertices <= 0:
            raise ValueError("vertices must be positive")
        if self.faces <= 0:
            raise ValueError("faces must be positive")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")


MeshPostprocessFixture = Callable[[MeshPostprocessRuntime], MeshPostprocessFixtureResult]


def build_mesh_postprocess_stage_handler(
    *,
    fixture: MeshPostprocessFixture,
    input_artifact: str = RAW_MESH_KEY_ARTIFACT,
    route: str = "fixture",
):
    """Build the mesh-postprocess `StageRunner` handler.

    The fixture stands in for cleanup/simplification and returns scalar mesh
    facts plus an optional runtime object. Runtime mesh objects stay in
    `StageExecutionContext`, not portable job artifacts.
    """

    if not input_artifact:
        raise ValueError("input_artifact must be nonempty")
    if not route:
        raise ValueError("route must be nonempty")

    def handler(
        invocation: GenerationStageInvocation,
        state: JobState,
        context: StageExecutionContext,
    ) -> StageRunnerOutput:
        if invocation.stage != MESH_POSTPROCESS_STAGE:
            raise ValueError(
                f"handler for stage {MESH_POSTPROCESS_STAGE} cannot run invocation stage {invocation.stage}"
            )
        if input_artifact not in state.artifacts:
            raise KeyError(
                f"missing required state artifact for {MESH_POSTPROCESS_STAGE}: {input_artifact}"
            )

        input_mesh_key = state.artifacts[input_artifact]
        if not isinstance(input_mesh_key, str) or not input_mesh_key:
            raise ValueError(f"{input_artifact} must be a nonempty string")
        mesh_object = context.require_runtime_object(input_mesh_key)

        fixture_result = fixture(
            MeshPostprocessRuntime(
                invocation=invocation,
                state=state,
                context=context,
                input_mesh_key=input_mesh_key,
                mesh_object=mesh_object,
            )
        )
        if not isinstance(fixture_result, MeshPostprocessFixtureResult):
            raise TypeError("mesh postprocess fixture must return MeshPostprocessFixtureResult")

        if fixture_result.mesh_object is None:
            context.require_runtime_object(fixture_result.mesh_key)
        else:
            if fixture_result.mesh_key in context.runtime_object_keys:
                raise ValueError(
                    "mesh postprocess key collision for "
                    f"{fixture_result.mesh_key} on job {invocation.job_id}"
                )
            context.register_runtime_object(fixture_result.mesh_key, fixture_result.mesh_object)

        return StageRunnerOutput(
            result=GenerationStageResult(
                invocation.stage,
                elapsed_seconds=fixture_result.elapsed_seconds,
                output_counts={
                    "vertices": fixture_result.vertices,
                    "faces": fixture_result.faces,
                },
            ),
            artifacts={
                input_artifact: input_mesh_key,
                "mesh_key": fixture_result.mesh_key,
                "mesh_postprocess_route": route,
                "mesh_target_faces": state.config["target_faces"],
                "mesh_no_cleanup": state.config["no_cleanup"],
                "mesh_keep_largest": state.config["keep_largest"],
                "mesh_vertices": fixture_result.vertices,
                "mesh_faces": fixture_result.faces,
            },
        )

    return handler


def _require_integer_count(field: str, value: object) -> None:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")

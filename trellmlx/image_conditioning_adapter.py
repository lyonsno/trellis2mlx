"""No-generation image-conditioning stage adapter.

This module maps the current DINOv3 image-conditioning boundary into the
fixture-backed `StageRunner` contract. It deliberately does not import model
code, load checkpoints, extract features, or store tensors in job artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .interleaved_generation import (
    GenerationStageResult,
    StageArtifactValue,
    StageRunnerOutput,
)
from .stage_handlers import (
    StageHandlerRuntime,
    build_model_role_stage_handler,
)


IMAGE_CONDITIONING_STAGE = "image_conditioning"
IMAGE_ENCODER_ROLE = "dinov3_image_encoder"


@dataclass(frozen=True)
class ImageConditioningRuntime:
    """Inputs passed to the injected no-generation image-conditioning fixture."""

    stage_runtime: StageHandlerRuntime
    images: tuple[str, ...]
    image_encoder: object
    image_encoder_metadata: Mapping[str, StageArtifactValue]

    @property
    def invocation(self):
        return self.stage_runtime.invocation

    @property
    def state(self):
        return self.stage_runtime.state

    @property
    def context(self):
        return self.stage_runtime.context


@dataclass(frozen=True)
class ImageConditioningFixtureResult:
    """Scalar witness for an image-conditioning fixture result."""

    conditioning_key: str
    context_tokens: int
    channels: int
    views: int
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.conditioning_key:
            raise ValueError("conditioning_key must be nonempty")
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.views <= 0:
            raise ValueError("views must be positive")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")


ImageConditioningFixture = Callable[[ImageConditioningRuntime], ImageConditioningFixtureResult]


def build_image_conditioning_stage_handler(
    *,
    fixture: ImageConditioningFixture,
):
    """Build the image-conditioning `StageRunner` handler.

    The fixture stands in for real feature extraction and returns only scalar
    shape/provenance facts. The actual feature tensor/object handoff is left for
    a later runtime registry slice.
    """

    def image_fixture(runtime: StageHandlerRuntime) -> StageRunnerOutput:
        invocation = runtime.invocation
        if invocation.conditioning_route != "image":
            raise ValueError("image conditioning adapter requires image conditioning route")
        if not invocation.images:
            raise ValueError("image conditioning adapter requires at least one image")

        metadata = runtime.handle_metadata[IMAGE_ENCODER_ROLE]
        fixture_runtime = ImageConditioningRuntime(
            stage_runtime=runtime,
            images=invocation.images,
            image_encoder=runtime.handles[IMAGE_ENCODER_ROLE],
            image_encoder_metadata=metadata,
        )
        fixture_result = fixture(fixture_runtime)
        if fixture_result.views != len(invocation.images):
            raise ValueError("fixture views must match image count")

        return StageRunnerOutput(
            result=GenerationStageResult(
                invocation.stage,
                elapsed_seconds=fixture_result.elapsed_seconds,
                output_counts={
                    "images": len(invocation.images),
                    "context_tokens": fixture_result.context_tokens,
                },
            ),
            artifacts={
                "conditioning_key": fixture_result.conditioning_key,
                "conditioning_route": invocation.conditioning_route,
                "conditioning_role": metadata["role"],
                "conditioning_model_family": metadata["model_family"],
                "conditioning_checkpoint": metadata["checkpoint"],
                "conditioning_loader_route": metadata["effective_loader_route"],
                "conditioning_image_count": len(invocation.images),
                "conditioning_view_count": fixture_result.views,
                "conditioning_context_tokens": fixture_result.context_tokens,
                "conditioning_channels": fixture_result.channels,
            },
        )

    return build_model_role_stage_handler(
        stage=IMAGE_CONDITIONING_STAGE,
        role_ids=(IMAGE_ENCODER_ROLE,),
        fixture=image_fixture,
    )

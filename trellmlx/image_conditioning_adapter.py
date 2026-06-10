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
    conditioning_object: object | None = None

    def __post_init__(self) -> None:
        if not self.conditioning_key:
            raise ValueError("conditioning_key must be nonempty")
        _require_integer_count("context_tokens", self.context_tokens)
        _require_integer_count("channels", self.channels)
        _require_integer_count("views", self.views)
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.views <= 0:
            raise ValueError("views must be positive")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")


def _require_integer_count(field: str, value: object) -> None:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")


@dataclass(frozen=True)
class ImageEncoderFeatureResult:
    """Runtime feature object plus scalar shape facts from an image encoder."""

    features: object
    context_tokens: int
    channels: int
    views: int
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.features is None:
            raise ValueError("features cannot be None")
        _require_integer_count("context_tokens", self.context_tokens)
        _require_integer_count("channels", self.channels)
        _require_integer_count("views", self.views)
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.views <= 0:
            raise ValueError("views must be positive")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")


ImageConditioningFixture = Callable[[ImageConditioningRuntime], ImageConditioningFixtureResult]
ImageEncoderExtractor = Callable[[ImageConditioningRuntime], ImageEncoderFeatureResult]


def build_image_encoder_fixture(
    *,
    extract_features: ImageEncoderExtractor | None = None,
    key_template: str = "cond://{job_id}/{stage}",
) -> ImageConditioningFixture:
    """Build a fixture that adapts an image encoder handle into conditioning state.

    The default extractor calls `runtime.image_encoder.extract_features(images)`.
    Tests can pass a fixture encoder handle without importing or constructing
    production DINO code.
    """

    if not key_template:
        raise ValueError("key_template must be nonempty")

    def fixture(runtime: ImageConditioningRuntime) -> ImageConditioningFixtureResult:
        feature_result = (
            extract_features(runtime)
            if extract_features is not None
            else _extract_features_from_handle(runtime)
        )
        if not isinstance(feature_result, ImageEncoderFeatureResult):
            raise TypeError("image encoder fixture must return ImageEncoderFeatureResult")
        conditioning_key = key_template.format(
            job_id=runtime.invocation.job_id,
            stage=runtime.invocation.stage,
        )
        return ImageConditioningFixtureResult(
            conditioning_key=conditioning_key,
            context_tokens=feature_result.context_tokens,
            channels=feature_result.channels,
            views=feature_result.views,
            elapsed_seconds=feature_result.elapsed_seconds,
            conditioning_object=feature_result.features,
        )

    return fixture


def _extract_features_from_handle(runtime: ImageConditioningRuntime) -> ImageEncoderFeatureResult:
    extractor = getattr(runtime.image_encoder, "extract_features", None)
    if extractor is None:
        raise TypeError("image encoder handle must provide extract_features(...)")
    result = extractor(runtime.images)
    if not isinstance(result, ImageEncoderFeatureResult):
        raise TypeError("image encoder fixture must return ImageEncoderFeatureResult")
    return result


def build_image_conditioning_stage_handler(
    *,
    fixture: ImageConditioningFixture,
):
    """Build the image-conditioning `StageRunner` handler.

    The fixture stands in for feature extraction and returns scalar
    shape/provenance facts plus an optional runtime object. Runtime objects stay
    in `StageExecutionContext`, not portable job artifacts.
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
        if fixture_result.conditioning_object is not None:
            if fixture_result.conditioning_key in runtime.context.runtime_object_keys:
                raise ValueError(
                    "image conditioning key collision for "
                    f"{fixture_result.conditioning_key} on job {invocation.job_id}"
                )
            runtime.context.register_runtime_object(
                fixture_result.conditioning_key,
                fixture_result.conditioning_object,
            )

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

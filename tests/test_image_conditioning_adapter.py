"""No-generation image-conditioning stage adapter contract."""

import pytest


def _plan(tmp_path, *, images=("front.png",), random_conditioning=False):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan

    job = GenerationJob(
        "seed-101",
        images,
        101,
        tmp_path / "seed-101.glb",
        random_conditioning=random_conditioning,
    )
    return InterleavedBatchPlan(jobs=(job,), stages=("image_conditioning",))


def _context(*, handle=None, route="fixture-dino"):
    from trellmlx.interleaved_generation import StageExecutionContext

    return StageExecutionContext(
        run_id="image-conditioning-probe",
        handles={"dinov3_image_encoder": handle if handle is not None else object()},
        handle_metadata={
            "dinov3_image_encoder": {
                "role": "dinov3_image_encoder",
                "stage": "image_conditioning",
                "model_family": "dinov3",
                "checkpoint": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "requested_loader_route": route,
                "effective_loader_route": route,
            }
        },
    )


def test_image_conditioning_adapter_records_feature_shape_and_route_identity(tmp_path):
    from trellmlx.interleaved_generation import JobState
    from trellmlx.image_conditioning_adapter import (
        ImageConditioningFixtureResult,
        build_image_conditioning_stage_handler,
    )

    plan = _plan(tmp_path, images=("front.png", "side.png"))
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])
    encoder = object()
    calls = []

    def extract(runtime):
        calls.append(
            (
                runtime.invocation.job_id,
                runtime.images,
                runtime.image_encoder is encoder,
                runtime.image_encoder_metadata["effective_loader_route"],
            )
        )
        return ImageConditioningFixtureResult(
            conditioning_key="cond://seed-101",
            context_tokens=257 * len(runtime.images),
            channels=1024,
            views=len(runtime.images),
            elapsed_seconds=0.25,
        )

    handler = build_image_conditioning_stage_handler(fixture=extract)
    output = handler(invocation, state, _context(handle=encoder))

    assert output.result.stage == "image_conditioning"
    assert output.result.elapsed_seconds == 0.25
    assert output.result.output_counts == {"images": 2, "context_tokens": 514}
    assert output.artifacts == {
        "conditioning_key": "cond://seed-101",
        "conditioning_route": "image",
        "conditioning_role": "dinov3_image_encoder",
        "conditioning_model_family": "dinov3",
        "conditioning_checkpoint": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "conditioning_loader_route": "fixture-dino",
        "conditioning_image_count": 2,
        "conditioning_view_count": 2,
        "conditioning_context_tokens": 514,
        "conditioning_channels": 1024,
    }
    assert calls == [("seed-101", ("front.png", "side.png"), True, "fixture-dino")]


def test_image_conditioning_adapter_rejects_random_route_without_calling_fixture(tmp_path):
    from trellmlx.interleaved_generation import JobState
    from trellmlx.image_conditioning_adapter import build_image_conditioning_stage_handler

    plan = _plan(tmp_path, images=(), random_conditioning=True)
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])
    called = False

    def extract(runtime):
        nonlocal called
        called = True
        raise AssertionError("fixture should not run for random conditioning")

    handler = build_image_conditioning_stage_handler(fixture=extract)

    with pytest.raises(ValueError, match="requires image conditioning route"):
        handler(invocation, state, _context())
    assert called is False


def test_image_conditioning_adapter_rejects_malformed_fixture_shape(tmp_path):
    from trellmlx.interleaved_generation import JobState
    from trellmlx.image_conditioning_adapter import (
        ImageConditioningFixtureResult,
        build_image_conditioning_stage_handler,
    )

    plan = _plan(tmp_path)
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])

    def extract(runtime):
        return ImageConditioningFixtureResult(
            conditioning_key="cond://bad",
            context_tokens=0,
            channels=1024,
            views=1,
        )

    handler = build_image_conditioning_stage_handler(fixture=extract)

    with pytest.raises(ValueError, match="context_tokens must be positive"):
        handler(invocation, state, _context())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("context_tokens", True),
        ("channels", 1024.5),
        ("views", True),
        ("views", 1.0),
    ),
)
def test_image_conditioning_fixture_result_rejects_non_integer_counts(field, value):
    from trellmlx.image_conditioning_adapter import ImageConditioningFixtureResult

    kwargs = {
        "conditioning_key": "cond://bad-count",
        "context_tokens": 257,
        "channels": 1024,
        "views": 1,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        ImageConditioningFixtureResult(**kwargs)


def test_image_conditioning_adapter_rejects_fixture_view_count_mismatch(tmp_path):
    from trellmlx.interleaved_generation import JobState
    from trellmlx.image_conditioning_adapter import (
        ImageConditioningFixtureResult,
        build_image_conditioning_stage_handler,
    )

    plan = _plan(tmp_path, images=("front.png", "side.png"))
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])

    def extract(runtime):
        return ImageConditioningFixtureResult(
            conditioning_key="cond://bad-views",
            context_tokens=514,
            channels=1024,
            views=1,
        )

    handler = build_image_conditioning_stage_handler(fixture=extract)

    with pytest.raises(ValueError, match="fixture views must match image count"):
        handler(invocation, state, _context())

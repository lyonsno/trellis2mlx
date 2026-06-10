"""No-generation image-conditioning stage adapter contract."""

import pytest


def _plan(tmp_path, *, images=("front.png",), random_conditioning=False, stages=("image_conditioning",)):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan

    job = GenerationJob(
        "seed-101",
        images,
        101,
        tmp_path / "seed-101.glb",
        random_conditioning=random_conditioning,
    )
    return InterleavedBatchPlan(jobs=(job,), stages=stages)


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


def test_image_conditioning_adapter_registers_runtime_object_for_downstream_stage(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationStageResult,
        StageRunner,
        StageRunnerOutput,
    )
    from trellmlx.image_conditioning_adapter import (
        ImageConditioningFixtureResult,
        build_image_conditioning_stage_handler,
    )

    plan = _plan(tmp_path, stages=("image_conditioning", "sparse_structure"))
    feature_object = object()
    downstream_calls = []

    def extract(runtime):
        return ImageConditioningFixtureResult(
            conditioning_key="cond://seed-101",
            context_tokens=257,
            channels=1024,
            views=1,
            conditioning_object=feature_object,
        )

    def sparse_structure(invocation, state, context):
        key = state.artifacts["conditioning_key"]
        downstream_calls.append(
            (
                key,
                context.require_runtime_object(key) is feature_object,
                "conditioning_object" in state.artifacts,
            )
        )
        return StageRunnerOutput(
            result=GenerationStageResult(invocation.stage, elapsed_seconds=0.0),
            artifacts={"consumed_conditioning_key": key},
        )

    context = _context()
    result = StageRunner(
        plan,
        handlers={
            "image_conditioning": build_image_conditioning_stage_handler(fixture=extract),
            "sparse_structure": sparse_structure,
        },
        context_factory=lambda plan: context,
    ).run()

    assert result.ok is True
    assert downstream_calls == [("cond://seed-101", True, False)]
    assert result.job_states["seed-101"].artifacts["conditioning_key"] == "cond://seed-101"
    assert "conditioning_object" not in result.job_states["seed-101"].artifacts


def test_image_conditioning_adapter_rejects_duplicate_conditioning_key_across_jobs(tmp_path):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan, StageRunner
    from trellmlx.image_conditioning_adapter import (
        ImageConditioningFixtureResult,
        build_image_conditioning_stage_handler,
    )

    jobs = (
        GenerationJob("seed-101", ("front.png",), 101, tmp_path / "seed-101.glb"),
        GenerationJob("seed-202", ("side.png",), 202, tmp_path / "seed-202.glb"),
    )
    plan = InterleavedBatchPlan(jobs=jobs, stages=("image_conditioning",))
    first_object = object()
    second_object = object()

    def extract(runtime):
        conditioning_object = (
            first_object if runtime.invocation.job_id == "seed-101" else second_object
        )
        return ImageConditioningFixtureResult(
            conditioning_key="cond://shared",
            context_tokens=257,
            channels=1024,
            views=1,
            conditioning_object=conditioning_object,
        )

    context = _context()
    runner = StageRunner(
        plan,
        handlers={"image_conditioning": build_image_conditioning_stage_handler(fixture=extract)},
        context_factory=lambda plan: context,
    )

    with pytest.raises(
        ValueError,
        match="image conditioning key collision for cond://shared on job seed-202",
    ):
        runner.run()

    assert context.require_runtime_object("cond://shared") is first_object


def test_image_encoder_fixture_registers_typed_feature_object(tmp_path):
    from trellmlx.interleaved_generation import JobState
    from trellmlx.image_conditioning_adapter import (
        ImageEncoderFeatureResult,
        build_image_conditioning_stage_handler,
        build_image_encoder_fixture,
    )

    plan = _plan(tmp_path, images=("front.png", "side.png"))
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])
    feature_object = object()
    calls = []

    class FakeImageEncoder:
        def extract_features(self, images):
            calls.append(tuple(images))
            return ImageEncoderFeatureResult(
                features=feature_object,
                context_tokens=514,
                channels=1024,
                views=2,
                elapsed_seconds=0.31,
            )

    context = _context(handle=FakeImageEncoder())
    handler = build_image_conditioning_stage_handler(fixture=build_image_encoder_fixture())
    output = handler(invocation, state, context)

    assert calls == [("front.png", "side.png")]
    assert output.result.elapsed_seconds == 0.31
    assert output.result.output_counts == {"images": 2, "context_tokens": 514}
    assert output.artifacts["conditioning_key"] == "cond://seed-101/image_conditioning"
    assert output.artifacts["conditioning_context_tokens"] == 514
    assert output.artifacts["conditioning_channels"] == 1024
    assert output.artifacts["conditioning_view_count"] == 2
    assert "conditioning_object" not in output.artifacts
    assert context.require_runtime_object("cond://seed-101/image_conditioning") is feature_object


def test_image_encoder_fixture_rejects_non_feature_result(tmp_path):
    from trellmlx.interleaved_generation import JobState
    from trellmlx.image_conditioning_adapter import (
        build_image_conditioning_stage_handler,
        build_image_encoder_fixture,
    )

    plan = _plan(tmp_path)
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])

    class FakeImageEncoder:
        def extract_features(self, images):
            return object()

    handler = build_image_conditioning_stage_handler(fixture=build_image_encoder_fixture())

    with pytest.raises(TypeError, match="image encoder fixture must return ImageEncoderFeatureResult"):
        handler(invocation, state, _context(handle=FakeImageEncoder()))


def test_image_encoder_fixture_rejects_callable_handle_without_extract_features(tmp_path):
    from trellmlx.interleaved_generation import JobState
    from trellmlx.image_conditioning_adapter import (
        ImageEncoderFeatureResult,
        build_image_conditioning_stage_handler,
        build_image_encoder_fixture,
    )

    plan = _plan(tmp_path)
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])

    class CallableImageEncoder:
        def __call__(self, images):
            return ImageEncoderFeatureResult(
                features=object(),
                context_tokens=257,
                channels=1024,
                views=1,
            )

    handler = build_image_conditioning_stage_handler(fixture=build_image_encoder_fixture())

    with pytest.raises(
        TypeError,
        match=r"image encoder handle must provide extract_features\(\.\.\.\)",
    ):
        handler(invocation, state, _context(handle=CallableImageEncoder()))


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

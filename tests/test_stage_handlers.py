"""Contracts for fixture-backed model-role stage handlers."""

import json

import pytest


def _single_job_plan(tmp_path, stages=("image_conditioning",)):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    return InterleavedBatchPlan(jobs=(job,), stages=stages)


def test_model_role_stage_handler_consumes_loader_context_and_records_artifacts(tmp_path):
    from trellmlx.interleaved_generation import GenerationStageResult, StageRunner, StageRunnerOutput
    from trellmlx.model_handle_roles import build_trellis_model_role_requests
    from trellmlx.stage_handle_loader import LoadedStageHandle, build_stage_loader_context
    from trellmlx.stage_handlers import StageHandlerRuntime, build_model_role_stage_handler

    plan = _single_job_plan(tmp_path)
    image_encoder = object()
    calls = []

    def load_dinov3(runtime):
        return LoadedStageHandle(
            handle=image_encoder,
            effective_loader_route="fixture-dino",
            metadata={"weights_path": "fixture://dinov3"},
        )

    def fixture(runtime):
        assert isinstance(runtime, StageHandlerRuntime)
        metadata = runtime.handle_metadata["dinov3_image_encoder"]
        calls.append(
            (
                runtime.invocation.job_id,
                runtime.handles["dinov3_image_encoder"] is image_encoder,
                metadata["checkpoint"],
                metadata["requested_loader_route"],
                metadata["effective_loader_route"],
            )
        )
        return StageRunnerOutput(
            result=GenerationStageResult(
                runtime.invocation.stage,
                elapsed_seconds=0.0,
                output_counts={"images": len(runtime.invocation.images)},
            ),
            artifacts={
                "conditioning_role": metadata["role"],
                "conditioning_route": runtime.invocation.conditioning_route,
                "conditioning_loader_route": metadata["effective_loader_route"],
            },
        )

    requests = build_trellis_model_role_requests(
        factories={"dinov3_image_encoder": load_dinov3},
        role_ids=("dinov3_image_encoder",),
        requested_loader_route={"dinov3_image_encoder": "fixture-dino"},
    )
    report_path = tmp_path / "stage-handler-loader-report.json"
    context_factory, context_closer = build_stage_loader_context(
        requests,
        report_path=report_path,
        run_id="stage-handler-probe",
    )

    result = StageRunner(
        plan,
        handlers={
            "image_conditioning": build_model_role_stage_handler(
                stage="image_conditioning",
                role_ids=("dinov3_image_encoder",),
                fixture=fixture,
            )
        },
        context_factory=context_factory,
        context_closer=context_closer,
    ).run()

    assert result.ok is True
    assert result.job_states["seed-101"].artifacts == {
        "conditioning_role": "dinov3_image_encoder",
        "conditioning_route": "image",
        "conditioning_loader_route": "fixture-dino",
    }
    assert result.job_states["seed-101"].stage_results[0].output_counts == {"images": 1}
    assert calls == [
        (
            "seed-101",
            True,
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "fixture-dino",
            "fixture-dino",
        )
    ]

    persisted = json.loads(report_path.read_text())
    assert persisted["ok"] is True
    assert persisted["load_reports"][0]["metadata"]["checkpoint"] == (
        "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )


def test_model_role_stage_handler_rejects_missing_required_artifacts(tmp_path):
    from trellmlx.interleaved_generation import JobState, StageExecutionContext
    from trellmlx.stage_handlers import build_model_role_stage_handler

    plan = _single_job_plan(tmp_path, stages=("sparse_structure",))
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])
    context = StageExecutionContext(
        run_id="stage-handler-probe",
        handles={"sparse_structure_flow": object()},
        handle_metadata={
            "sparse_structure_flow": {
                "role": "sparse_structure_flow",
                "stage": "sparse_structure",
                "model_family": "sparse_structure_flow",
                "checkpoint": "fixture://ss-flow",
                "requested_loader_route": "fixture",
                "effective_loader_route": "fixture",
            }
        },
    )
    called = False

    def fixture(runtime):
        nonlocal called
        called = True
        raise AssertionError("fixture should not run")

    handler = build_model_role_stage_handler(
        stage="sparse_structure",
        role_ids=("sparse_structure_flow",),
        required_artifacts=("conditioning_key",),
        fixture=fixture,
    )

    with pytest.raises(KeyError, match="missing required state artifact for sparse_structure: conditioning_key"):
        handler(invocation, state, context)
    assert called is False


def test_model_role_stage_handler_accepts_secondary_consumer_stage(tmp_path):
    from trellmlx.interleaved_generation import GenerationStageResult, StageRunner, StageRunnerOutput
    from trellmlx.model_handle_roles import build_trellis_model_role_requests
    from trellmlx.stage_handle_loader import LoadedStageHandle, build_stage_loader_context
    from trellmlx.stage_handlers import build_model_role_stage_handler

    plan = _single_job_plan(tmp_path, stages=("hr_coordinates",))
    decoder = object()

    def load_decoder(runtime):
        return LoadedStageHandle(handle=decoder, effective_loader_route="fixture")

    def fixture(runtime):
        metadata = runtime.handle_metadata["shape_decoder"]
        return StageRunnerOutput(
            result=GenerationStageResult(runtime.invocation.stage, elapsed_seconds=0.0),
            artifacts={
                "decoder_role": metadata["role"],
                "decoder_consumers": metadata["consumer_stages"],
            },
        )

    requests = build_trellis_model_role_requests(
        factories={"shape_decoder": load_decoder},
        role_ids=("shape_decoder",),
        requested_loader_route="fixture",
    )
    context_factory, context_closer = build_stage_loader_context(requests, run_id="stage-handler-probe")

    result = StageRunner(
        plan,
        handlers={
            "hr_coordinates": build_model_role_stage_handler(
                stage="hr_coordinates",
                role_ids=("shape_decoder",),
                fixture=fixture,
            )
        },
        context_factory=context_factory,
        context_closer=context_closer,
    ).run()

    assert result.ok is True
    assert result.job_states["seed-101"].artifacts == {
        "decoder_role": "shape_decoder",
        "decoder_consumers": "hr_coordinates,shape_decode",
    }


def test_model_role_stage_handler_rejects_role_for_wrong_stage(tmp_path):
    from trellmlx.interleaved_generation import JobState, StageExecutionContext
    from trellmlx.stage_handlers import build_model_role_stage_handler

    plan = _single_job_plan(tmp_path, stages=("image_conditioning",))
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])
    context = StageExecutionContext(
        run_id="stage-handler-probe",
        handles={"sparse_structure_flow": object()},
        handle_metadata={
            "sparse_structure_flow": {
                "role": "sparse_structure_flow",
                "stage": "sparse_structure",
                "model_family": "sparse_structure_flow",
                "checkpoint": "fixture://ss-flow",
                "requested_loader_route": "fixture",
                "effective_loader_route": "fixture",
            }
        },
    )

    def fixture(runtime):
        raise AssertionError("fixture should not run")

    handler = build_model_role_stage_handler(
        stage="image_conditioning",
        role_ids=("sparse_structure_flow",),
        fixture=fixture,
    )

    with pytest.raises(ValueError, match="model role sparse_structure_flow is not declared for stage image_conditioning"):
        handler(invocation, state, context)


def test_model_role_stage_handler_rejects_malformed_consumer_stages_metadata(tmp_path):
    from trellmlx.interleaved_generation import JobState, StageExecutionContext
    from trellmlx.stage_handlers import build_model_role_stage_handler

    plan = _single_job_plan(tmp_path, stages=("shape_decode",))
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])
    context = StageExecutionContext(
        run_id="stage-handler-probe",
        handles={"shape_decoder": object()},
        handle_metadata={
            "shape_decoder": {
                "role": "shape_decoder",
                "stage": "shape_decode",
                "consumer_stages": "",
                "model_family": "slat_decoder_shape",
                "checkpoint": "fixture://shape-decoder",
                "requested_loader_route": "fixture",
                "effective_loader_route": "fixture",
            }
        },
    )

    def fixture(runtime):
        raise AssertionError("fixture should not run")

    handler = build_model_role_stage_handler(
        stage="shape_decode",
        role_ids=("shape_decoder",),
        fixture=fixture,
    )

    with pytest.raises(ValueError, match="model role shape_decoder metadata consumer_stages must be a nonempty string"):
        handler(invocation, state, context)

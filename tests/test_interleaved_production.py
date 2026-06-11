"""Contracts for no-generation TRELLIS production route wiring."""

import json


def _single_job_plan(tmp_path, *, stages=("image_conditioning",)):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    return InterleavedBatchPlan(jobs=(job,), stages=stages)


def _minimal_model_metadata(role_id, stage):
    return {
        "role": role_id,
        "stage": stage,
        "model_family": "fixture",
        "checkpoint": f"fixture://{role_id}",
        "requested_loader_route": "fixture",
        "effective_loader_route": "fixture",
    }


def test_production_route_plan_declares_stage_order_and_model_roles():
    from trellmlx.interleaved_generation import DEFAULT_STAGE_SEQUENCE
    from trellmlx.interleaved_production import (
        TRELLIS_PRODUCTION_STAGE_ROUTES,
        production_model_role_ids,
        production_stage_sequence,
    )

    assert production_stage_sequence() == DEFAULT_STAGE_SEQUENCE
    routes_by_stage = {route.stage: route for route in TRELLIS_PRODUCTION_STAGE_ROUTES}
    assert tuple(routes_by_stage) == DEFAULT_STAGE_SEQUENCE
    assert routes_by_stage["image_conditioning"].role_ids == ("dinov3_image_encoder",)
    assert routes_by_stage["sparse_structure"].role_ids == (
        "sparse_structure_flow",
        "sparse_structure_decoder",
    )
    assert routes_by_stage["hr_coordinates"].role_ids == ("shape_decoder",)
    assert routes_by_stage["shape_decode"].role_ids == ("shape_decoder",)
    assert routes_by_stage["mesh_extract"].role_ids == ()
    assert routes_by_stage["mesh_postprocess"].role_ids == ()
    assert routes_by_stage["texture_bake"].role_ids == ()
    assert routes_by_stage["export"].role_ids == ()
    assert routes_by_stage["image_conditioning"].required_artifacts == ()
    assert routes_by_stage["sparse_structure"].required_artifacts == ("conditioning_key",)
    assert routes_by_stage["lr_shape_latent"].required_artifacts == (
        "sparse_structure_key",
        "conditioning_key",
    )
    assert routes_by_stage["hr_coordinates"].required_artifacts == ("lr_shape_latent_key",)
    assert routes_by_stage["hr_shape_latent"].required_artifacts == (
        "hr_coordinate_key",
        "conditioning_key",
    )
    assert routes_by_stage["shape_decode"].required_artifacts == ("hr_shape_latent_key",)
    assert routes_by_stage["mesh_extract"].required_artifacts == ("shape_key",)
    assert routes_by_stage["mesh_postprocess"].required_artifacts == ("raw_mesh_key",)
    assert routes_by_stage["texture_latent"].required_artifacts == (
        "hr_shape_latent_key",
        "hr_coordinate_key",
        "conditioning_key",
    )
    assert routes_by_stage["texture_decode"].required_artifacts == ("texture_latent_key",)
    assert routes_by_stage["texture_bake"].required_artifacts == ("mesh_key", "texture_key")
    assert routes_by_stage["export"].required_artifacts == ("mesh_key", "texture_bake_key")
    assert production_model_role_ids() == (
        "dinov3_image_encoder",
        "sparse_structure_flow",
        "sparse_structure_decoder",
        "shape_flow_lr",
        "shape_flow_hr",
        "shape_decoder",
        "texture_flow",
        "texture_decoder",
    )


def test_production_loader_requests_preserve_route_identity_and_timing(tmp_path):
    from trellmlx.interleaved_production import (
        build_trellis_production_loader_requests,
        production_model_role_ids,
    )
    from trellmlx.stage_handle_loader import LoadedStageHandle, load_stage_handles

    plan = _single_job_plan(tmp_path)
    events = []

    def make_load(role_id):
        def load(runtime):
            events.append(("load", runtime.handle_id, runtime.requested_loader_route))
            return LoadedStageHandle(
                handle=f"handle://{role_id}",
                effective_loader_route="fixture",
                metadata={"weights_path": f"fixture://{role_id}"},
            )

        return load

    factories = {role_id: make_load(role_id) for role_id in production_model_role_ids()}
    requests = build_trellis_production_loader_requests(
        factories=factories,
        requested_loader_route="fixture",
    )

    assert [request.handle_id for request in requests] == list(production_model_role_ids())
    assert all(request.requested_loader_route == "fixture" for request in requests)

    report_path = tmp_path / "production-loader-report.json"
    report = load_stage_handles(plan, requests, report_path=report_path, run_id="production-loader")

    assert report.ok is True
    assert report.requested_handle_ids == production_model_role_ids()
    assert report.loaded_handle_ids == production_model_role_ids()
    assert events[0] == ("load", "dinov3_image_encoder", "fixture")
    assert report.load_reports[0].metadata["role"] == "dinov3_image_encoder"
    assert report.load_reports[0].metadata["effective_loader_route"] == "fixture"
    assert report.load_reports[0].elapsed_seconds >= 0.0

    persisted = json.loads(report_path.read_text())
    assert persisted["requested_handle_ids"] == list(production_model_role_ids())
    assert persisted["load_reports"][0]["metadata"]["weights_path"] == "fixture://dinov3_image_encoder"
    assert persisted["load_reports"][0]["elapsed_seconds"] >= 0.0


def test_production_stage_handlers_wire_model_and_no_model_stages(tmp_path):
    from trellmlx.interleaved_generation import GenerationStageResult, StageRunner, StageRunnerOutput
    from trellmlx.interleaved_production import (
        build_trellis_production_loader_context,
        build_trellis_production_stage_handlers,
    )
    from trellmlx.stage_handle_loader import LoadedStageHandle

    plan = _single_job_plan(
        tmp_path,
        stages=(
            "image_conditioning",
            "sparse_structure",
            "lr_shape_latent",
            "hr_coordinates",
            "hr_shape_latent",
            "shape_decode",
            "mesh_extract",
        ),
    )
    events = []

    def make_load(role_id):
        def load(runtime):
            return LoadedStageHandle(handle=f"{role_id}-handle", effective_loader_route="fixture")

        return load

    role_ids = (
        "dinov3_image_encoder",
        "sparse_structure_flow",
        "sparse_structure_decoder",
        "shape_flow_lr",
        "shape_flow_hr",
        "shape_decoder",
    )

    context_factory, context_closer = build_trellis_production_loader_context(
        factories={role_id: make_load(role_id) for role_id in role_ids},
        role_ids=role_ids,
        requested_loader_route="fixture",
        run_id="production-route",
    )

    def image_conditioning(runtime):
        events.append(
            (
                runtime.invocation.stage,
                tuple(runtime.handles),
                runtime.handle_metadata["dinov3_image_encoder"]["effective_loader_route"],
            )
        )
        return StageRunnerOutput(
            result=GenerationStageResult(runtime.invocation.stage, elapsed_seconds=0.01),
            artifacts={"conditioning_key": "conditioning://seed-101"},
        )

    def sparse_structure(runtime):
        events.append((runtime.invocation.stage, tuple(runtime.handles)))
        return StageRunnerOutput(
            result=GenerationStageResult(runtime.invocation.stage, elapsed_seconds=0.01),
            artifacts={"sparse_structure_key": "sparse://seed-101"},
        )

    def lr_shape_latent(runtime):
        events.append((runtime.invocation.stage, tuple(runtime.handles)))
        return StageRunnerOutput(
            result=GenerationStageResult(runtime.invocation.stage, elapsed_seconds=0.01),
            artifacts={"lr_shape_latent_key": "shape-lr://seed-101"},
        )

    def hr_coordinates(runtime):
        events.append(
            (
                runtime.invocation.stage,
                tuple(runtime.handles),
                runtime.handle_metadata["shape_decoder"]["consumer_stages"],
            )
        )
        return StageRunnerOutput(
            result=GenerationStageResult(runtime.invocation.stage, elapsed_seconds=0.02),
            artifacts={"hr_coordinate_key": "hr://seed-101"},
        )

    def hr_shape_latent(runtime):
        events.append((runtime.invocation.stage, tuple(runtime.handles)))
        return StageRunnerOutput(
            result=GenerationStageResult(runtime.invocation.stage, elapsed_seconds=0.01),
            artifacts={"hr_shape_latent_key": "shape-hr://seed-101"},
        )

    def shape_decode(runtime):
        events.append(
            (
                runtime.invocation.stage,
                tuple(runtime.handles),
                runtime.handle_metadata["shape_decoder"]["consumer_stages"],
            )
        )
        return StageRunnerOutput(
            result=GenerationStageResult(runtime.invocation.stage, elapsed_seconds=0.01),
            artifacts={"shape_key": "shape://seed-101"},
        )

    def mesh_extract(runtime):
        events.append((runtime.invocation.stage, tuple(runtime.handles), tuple(runtime.handle_metadata)))
        return StageRunnerOutput(
            result=GenerationStageResult(runtime.invocation.stage, elapsed_seconds=0.03),
            artifacts={"raw_mesh_key": "mesh://seed-101"},
        )

    handlers = build_trellis_production_stage_handlers(
        fixtures={
            "image_conditioning": image_conditioning,
            "sparse_structure": sparse_structure,
            "lr_shape_latent": lr_shape_latent,
            "hr_coordinates": hr_coordinates,
            "hr_shape_latent": hr_shape_latent,
            "shape_decode": shape_decode,
            "mesh_extract": mesh_extract,
        },
        stages=plan.stages,
    )

    result = StageRunner(
        plan,
        handlers=handlers,
        context_factory=context_factory,
        context_closer=context_closer,
    ).run()

    assert result.ok is True
    assert events == [
        ("image_conditioning", ("dinov3_image_encoder",), "fixture"),
        ("sparse_structure", ("sparse_structure_flow", "sparse_structure_decoder")),
        ("lr_shape_latent", ("shape_flow_lr",)),
        ("hr_coordinates", ("shape_decoder",), "hr_coordinates,shape_decode"),
        ("hr_shape_latent", ("shape_flow_hr",)),
        ("shape_decode", ("shape_decoder",), "hr_coordinates,shape_decode"),
        ("mesh_extract", (), ()),
    ]
    assert result.job_states["seed-101"].artifacts == {
        "conditioning_key": "conditioning://seed-101",
        "sparse_structure_key": "sparse://seed-101",
        "lr_shape_latent_key": "shape-lr://seed-101",
        "hr_coordinate_key": "hr://seed-101",
        "hr_shape_latent_key": "shape-hr://seed-101",
        "shape_key": "shape://seed-101",
        "raw_mesh_key": "mesh://seed-101",
    }


def test_production_stage_handlers_reject_missing_model_stage_artifacts(tmp_path):
    import pytest

    from trellmlx.interleaved_generation import GenerationStageInvocation, JobState, StageExecutionContext
    from trellmlx.interleaved_production import build_trellis_production_stage_handlers

    plan = _single_job_plan(tmp_path, stages=("sparse_structure",))

    def sparse_structure(runtime):
        raise AssertionError("fixture should not run without conditioning_key")

    handlers = build_trellis_production_stage_handlers(
        fixtures={"sparse_structure": sparse_structure},
        stages=plan.stages,
    )
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])
    context = StageExecutionContext(
        run_id="production-route",
        handles={
            "sparse_structure_flow": object(),
            "sparse_structure_decoder": object(),
        },
        handle_metadata={
            "sparse_structure_flow": _minimal_model_metadata("sparse_structure_flow", "sparse_structure"),
            "sparse_structure_decoder": _minimal_model_metadata(
                "sparse_structure_decoder",
                "sparse_structure",
            ),
        },
    )

    assert isinstance(invocation, GenerationStageInvocation)
    with pytest.raises(
        KeyError,
        match="missing required state artifact for sparse_structure: conditioning_key",
    ):
        handlers["sparse_structure"](invocation, state, context)


def test_production_stage_handlers_reject_shape_flow_without_conditioning(tmp_path):
    import pytest

    from trellmlx.interleaved_generation import JobState, StageExecutionContext
    from trellmlx.interleaved_production import build_trellis_production_stage_handlers

    plan = _single_job_plan(tmp_path, stages=("lr_shape_latent",))

    def lr_shape_latent(runtime):
        raise AssertionError("fixture should not run without conditioning_key")

    handlers = build_trellis_production_stage_handlers(
        fixtures={"lr_shape_latent": lr_shape_latent},
        stages=plan.stages,
    )
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0]).record_artifacts(
        {"sparse_structure_key": "sparse://seed-101"}
    )
    context = StageExecutionContext(
        run_id="production-route",
        handles={"shape_flow_lr": object()},
        handle_metadata={
            "shape_flow_lr": _minimal_model_metadata("shape_flow_lr", "lr_shape_latent"),
        },
    )

    with pytest.raises(
        KeyError,
        match="missing required state artifact for lr_shape_latent: conditioning_key",
    ):
        handlers["lr_shape_latent"](invocation, state, context)


def test_production_stage_handlers_reject_texture_latent_without_shape_inputs(tmp_path):
    import pytest

    from trellmlx.interleaved_generation import JobState, StageExecutionContext
    from trellmlx.interleaved_production import build_trellis_production_stage_handlers

    plan = _single_job_plan(tmp_path, stages=("texture_latent",))

    def texture_latent(runtime):
        raise AssertionError("fixture should not run without shape latent and coordinates")

    handlers = build_trellis_production_stage_handlers(
        fixtures={"texture_latent": texture_latent},
        stages=plan.stages,
    )
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0]).record_artifacts(
        {
            "mesh_key": "mesh://seed-101/postprocessed",
            "conditioning_key": "conditioning://seed-101",
        }
    )
    context = StageExecutionContext(
        run_id="production-route",
        handles={"texture_flow": object()},
        handle_metadata={
            "texture_flow": _minimal_model_metadata("texture_flow", "texture_latent"),
        },
    )

    with pytest.raises(
        KeyError,
        match=(
            "missing required state artifact for texture_latent: "
            "hr_shape_latent_key, hr_coordinate_key"
        ),
    ):
        handlers["texture_latent"](invocation, state, context)


def test_production_stage_handlers_reject_missing_no_model_stage_artifacts(tmp_path):
    import pytest

    from trellmlx.interleaved_generation import JobState, StageExecutionContext
    from trellmlx.interleaved_production import build_trellis_production_stage_handlers

    plan = _single_job_plan(tmp_path, stages=("mesh_postprocess",))

    def mesh_postprocess(runtime):
        raise AssertionError("fixture should not run without raw_mesh_key")

    handlers = build_trellis_production_stage_handlers(
        fixtures={"mesh_postprocess": mesh_postprocess},
        stages=plan.stages,
    )
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])
    context = StageExecutionContext(run_id="production-route")

    with pytest.raises(
        KeyError,
        match="missing required state artifact for mesh_postprocess: raw_mesh_key",
    ):
        handlers["mesh_postprocess"](invocation, state, context)

"""Contracts for no-generation TRELLIS production route wiring."""

import json


def _single_job_plan(tmp_path, *, stages=("image_conditioning",)):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    return InterleavedBatchPlan(jobs=(job,), stages=stages)


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
        stages=("image_conditioning", "hr_coordinates", "mesh_extract"),
    )
    events = []

    def load_dinov3(runtime):
        return LoadedStageHandle(handle="dinov3-handle", effective_loader_route="fixture")

    def load_shape_decoder(runtime):
        return LoadedStageHandle(handle="shape-decoder-handle", effective_loader_route="fixture")

    context_factory, context_closer = build_trellis_production_loader_context(
        factories={
            "dinov3_image_encoder": load_dinov3,
            "shape_decoder": load_shape_decoder,
        },
        role_ids=("dinov3_image_encoder", "shape_decoder"),
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

    def mesh_extract(runtime):
        events.append((runtime.invocation.stage, tuple(runtime.handles), tuple(runtime.handle_metadata)))
        return StageRunnerOutput(
            result=GenerationStageResult(runtime.invocation.stage, elapsed_seconds=0.03),
            artifacts={"raw_mesh_key": "mesh://seed-101"},
        )

    handlers = build_trellis_production_stage_handlers(
        fixtures={
            "image_conditioning": image_conditioning,
            "hr_coordinates": hr_coordinates,
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
        ("hr_coordinates", ("shape_decoder",), "hr_coordinates,shape_decode"),
        ("mesh_extract", (), ()),
    ]
    assert result.job_states["seed-101"].artifacts == {
        "conditioning_key": "conditioning://seed-101",
        "hr_coordinate_key": "hr://seed-101",
        "raw_mesh_key": "mesh://seed-101",
    }

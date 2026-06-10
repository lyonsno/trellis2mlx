"""Contracts for declarative TRELLIS model-handle role bundles."""

import json

import pytest


def _single_job_plan(tmp_path):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    return InterleavedBatchPlan(jobs=(job,), stages=("image_conditioning",))


def test_canonical_trellis_model_roles_are_stable_and_portable():
    from trellmlx.model_handle_roles import TRELLIS_MODEL_HANDLE_ROLES

    assert [role.handle_id for role in TRELLIS_MODEL_HANDLE_ROLES] == [
        "dinov3_image_encoder",
        "sparse_structure_flow",
        "sparse_structure_decoder",
        "shape_flow_lr",
        "shape_flow_hr",
        "shape_decoder",
        "texture_flow",
        "texture_decoder",
    ]
    assert [role.stage for role in TRELLIS_MODEL_HANDLE_ROLES] == [
        "image_conditioning",
        "sparse_structure",
        "sparse_structure",
        "lr_shape_latent",
        "hr_shape_latent",
        "shape_decode",
        "texture_latent",
        "texture_decode",
    ]
    assert [role.consumer_stages for role in TRELLIS_MODEL_HANDLE_ROLES] == [
        ("image_conditioning",),
        ("sparse_structure",),
        ("sparse_structure",),
        ("lr_shape_latent",),
        ("hr_shape_latent",),
        ("hr_coordinates", "shape_decode"),
        ("texture_latent",),
        ("texture_decode",),
    ]
    assert all(role.kind == "model" for role in TRELLIS_MODEL_HANDLE_ROLES)
    assert all(role.default_loader_route == "mlx" for role in TRELLIS_MODEL_HANDLE_ROLES)
    assert TRELLIS_MODEL_HANDLE_ROLES[0].metadata() == {
        "role": "dinov3_image_encoder",
        "stage": "image_conditioning",
        "model_family": "dinov3",
        "checkpoint": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }
    assert TRELLIS_MODEL_HANDLE_ROLES[1].metadata()["checkpoint"] == (
        "microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"
    )
    assert TRELLIS_MODEL_HANDLE_ROLES[5].metadata()["consumer_stages"] == "hr_coordinates,shape_decode"


def test_model_role_requests_require_factories_and_reject_duplicate_roles():
    from trellmlx.model_handle_roles import ModelHandleRole, build_model_role_requests

    role = ModelHandleRole(
        handle_id="dinov3_image_encoder",
        stage="image_conditioning",
        model_family="dinov3",
        checkpoint="fixture://dinov3",
    )

    with pytest.raises(ValueError, match="missing factory for model handle role: dinov3_image_encoder"):
        build_model_role_requests((role,), factories={})

    with pytest.raises(ValueError, match="duplicate model handle role: dinov3_image_encoder"):
        build_model_role_requests(
            (role, role),
            factories={"dinov3_image_encoder": lambda runtime: object()},
        )


def test_model_role_requests_preserve_route_identity_and_feed_stage_runner(tmp_path):
    from trellmlx.interleaved_generation import GenerationStageResult, StageRunner
    from trellmlx.model_handle_roles import build_trellis_model_role_requests
    from trellmlx.stage_handle_loader import LoadedStageHandle, build_stage_loader_context

    plan = _single_job_plan(tmp_path)
    handle = object()
    events = []

    def load_dinov3(runtime):
        events.append(("load", runtime.handle_id, runtime.requested_loader_route))
        return LoadedStageHandle(
            handle=handle,
            effective_loader_route="fixture",
            metadata={"weights_path": "fixture://dinov3"},
        )

    def close_dinov3(runtime, loaded_handle):
        events.append(("close", runtime.handle_id, loaded_handle is handle))

    requests = build_trellis_model_role_requests(
        factories={"dinov3_image_encoder": load_dinov3},
        closes={"dinov3_image_encoder": close_dinov3},
        role_ids=("dinov3_image_encoder",),
        requested_loader_route="fixture",
    )

    assert [request.handle_id for request in requests] == ["dinov3_image_encoder"]
    assert requests[0].kind == "model"
    assert requests[0].requested_loader_route == "fixture"
    assert requests[0].metadata == {
        "role": "dinov3_image_encoder",
        "stage": "image_conditioning",
        "model_family": "dinov3",
        "checkpoint": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }

    report_path = tmp_path / "model-role-report.json"
    context_factory, context_closer = build_stage_loader_context(
        requests,
        report_path=report_path,
        run_id="role-probe",
    )

    handler_calls = []

    def image_conditioning(invocation, state, context):
        handler_calls.append(
            (
                context.require_handle("dinov3_image_encoder") is handle,
                context.handle_metadata["dinov3_image_encoder"]["role"],
                context.handle_metadata["dinov3_image_encoder"]["effective_loader_route"],
            )
        )
        return GenerationStageResult(invocation.stage, elapsed_seconds=0.0)

    result = StageRunner(
        plan,
        handlers={"image_conditioning": image_conditioning},
        context_factory=context_factory,
        context_closer=context_closer,
    ).run()

    assert result.ok is True
    assert handler_calls == [(True, "dinov3_image_encoder", "fixture")]
    assert events == [
        ("load", "dinov3_image_encoder", "fixture"),
        ("close", "dinov3_image_encoder", True),
    ]

    persisted = json.loads(report_path.read_text())
    assert persisted["ok"] is True
    assert persisted["requested_handle_ids"] == ["dinov3_image_encoder"]
    assert persisted["load_reports"][0]["metadata"]["checkpoint"] == (
        "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    assert persisted["load_reports"][0]["metadata"]["weights_path"] == "fixture://dinov3"


def test_model_role_requests_accept_per_role_routes_and_metadata(tmp_path):
    from trellmlx.model_handle_roles import build_trellis_model_role_requests
    from trellmlx.stage_handle_loader import LoadedStageHandle, load_stage_handles

    plan = _single_job_plan(tmp_path)

    def load(runtime):
        return LoadedStageHandle(handle=object(), effective_loader_route="fixture-dino")

    requests = build_trellis_model_role_requests(
        factories={"dinov3_image_encoder": load},
        role_ids=("dinov3_image_encoder",),
        requested_loader_route={"dinov3_image_encoder": "fixture-dino"},
        metadata={"dinov3_image_encoder": {"precision": "fp16"}},
    )

    report = load_stage_handles(plan, requests, run_id="role-probe")

    assert report.ok is True
    assert report.load_reports[0].metadata["requested_loader_route"] == "fixture-dino"
    assert report.load_reports[0].metadata["precision"] == "fp16"


def test_model_role_requests_reject_canonical_metadata_overrides():
    from trellmlx.model_handle_roles import build_trellis_model_role_requests

    def load(runtime):
        return object()

    with pytest.raises(ValueError, match="model role metadata cannot override canonical keys: checkpoint, role"):
        build_trellis_model_role_requests(
            factories={"dinov3_image_encoder": load},
            role_ids=("dinov3_image_encoder",),
            metadata={
                "dinov3_image_encoder": {
                    "role": "wrong-role",
                    "checkpoint": "wrong-checkpoint",
                }
            },
        )

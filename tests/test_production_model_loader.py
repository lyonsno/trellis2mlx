"""Production model loader contracts behind interleaved TRELLIS routes."""


def _plan(tmp_path):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    return InterleavedBatchPlan(jobs=(job,), stages=("image_conditioning",))


class FakeModel:
    def __init__(self, role_id):
        self.role_id = role_id
        self.compiled = False

    def compile(self):
        self.compiled = True


def _fake_constructors(role_ids, events):
    return {
        role_id: (lambda role_id=role_id: events.append(("construct", role_id)) or FakeModel(role_id))
        for role_id in role_ids
    }


def test_production_model_loader_requests_defer_construction_until_context_open(tmp_path):
    from trellmlx.production_model_loader import build_trellis_production_model_loader_requests
    from trellmlx.stage_handle_loader import load_stage_handles

    role_ids = ("dinov3_image_encoder", "sparse_structure_flow", "shape_decoder")
    events = []

    def load_model_weights(model, checkpoint_path, *, verbose):
        events.append(("load", model.role_id, checkpoint_path, verbose))

    def load_dinov3_weights(model, checkpoint_path):
        events.append(("load-dino", model.role_id, checkpoint_path))
        return 412

    requests = build_trellis_production_model_loader_requests(
        role_ids=role_ids,
        constructors=_fake_constructors(role_ids, events),
        load_model_weights=load_model_weights,
        load_dinov3_weights=load_dinov3_weights,
        checkpoint_paths={
            "dinov3_image_encoder": "weights://dinov3",
            "sparse_structure_flow": "weights://ss-flow",
            "shape_decoder": "weights://shape-decoder",
        },
        requested_loader_route="mlx",
        verbose=False,
    )

    assert events == []
    assert [request.handle_id for request in requests] == list(role_ids)

    report = load_stage_handles(_plan(tmp_path), requests, run_id="production-loader")

    assert report.ok is True
    assert report.loaded_handle_ids == role_ids
    assert events == [
        ("construct", "dinov3_image_encoder"),
        ("load-dino", "dinov3_image_encoder", "weights://dinov3"),
        ("construct", "sparse_structure_flow"),
        ("load", "sparse_structure_flow", "weights://ss-flow", False),
        ("construct", "shape_decoder"),
        ("load", "shape_decoder", "weights://shape-decoder", False),
    ]
    first_report = report.load_reports[0]
    assert first_report.metadata["role"] == "dinov3_image_encoder"
    assert first_report.metadata["effective_loader_route"] == "mlx"
    assert first_report.metadata["weights_path"] == "weights://dinov3"
    assert first_report.metadata["loader_kind"] == "mlx_model"
    assert first_report.metadata["loaded_weight_arrays"] == 412


def test_production_model_loader_defaults_to_generate_style_cache_paths(tmp_path, monkeypatch):
    from trellmlx.production_model_loader import build_trellis_production_model_loader_requests
    from trellmlx.stage_handle_loader import load_stage_handles

    role_ids = ("dinov3_image_encoder", "sparse_structure_flow")
    events = []
    hf_cache_root = tmp_path / "hf-cache"
    dino_snapshot = (
        hf_cache_root
        / "models--facebook--dinov3-vitl16-pretrain-lvd1689m"
        / "snapshots"
        / "abc123"
    )
    dino_snapshot.mkdir(parents=True)

    def load_model_weights(model, checkpoint_path, *, verbose):
        events.append(("load", model.role_id, checkpoint_path, verbose))

    def load_dinov3_weights(model, checkpoint_path):
        events.append(("load-dino", model.role_id, checkpoint_path))

    monkeypatch.setattr(
        "trellmlx.production_model_loader._default_cleanup_model",
        lambda model: events.append(("cleanup", model.role_id)),
    )

    requests = build_trellis_production_model_loader_requests(
        role_ids=role_ids,
        constructors=_fake_constructors(role_ids, events),
        load_model_weights=load_model_weights,
        load_dinov3_weights=load_dinov3_weights,
        hf_cache_root=hf_cache_root,
        verbose=False,
    )

    report = load_stage_handles(_plan(tmp_path), requests, run_id="production-loader")

    assert report.ok is True
    assert events[:4] == [
        ("construct", "dinov3_image_encoder"),
        ("load-dino", "dinov3_image_encoder", str(dino_snapshot)),
        ("construct", "sparse_structure_flow"),
        (
            "load",
            "sparse_structure_flow",
            str(
                hf_cache_root
                / "models--microsoft--TRELLIS.2-4B"
                / "snapshots"
                / "af44b45f2e35a493886929c6d786e563ec68364d"
                / "ckpts"
                / "ss_flow_img_dit_1_3B_64_bf16.safetensors"
            ),
            False,
        ),
    ]
    assert report.load_reports[0].metadata["weights_path"] == str(dino_snapshot)


def test_production_model_loader_quantizes_and_compiles_flow_roles_only(tmp_path):
    from trellmlx.production_model_loader import (
        FLOW_MODEL_ROLE_IDS,
        build_trellis_production_model_loader_requests,
    )
    from trellmlx.stage_handle_loader import load_stage_handles

    role_ids = (
        "sparse_structure_flow",
        "sparse_structure_decoder",
        "shape_flow_lr",
        "shape_flow_hr",
        "shape_decoder",
        "texture_flow",
        "texture_decoder",
    )
    events = []

    def load_model_weights(model, checkpoint_path, *, verbose):
        events.append(("load", model.role_id))

    def quantize_model(model, *, bits):
        events.append(("quantize", model.role_id, bits))

    requests = build_trellis_production_model_loader_requests(
        role_ids=role_ids,
        constructors=_fake_constructors(role_ids, events),
        load_model_weights=load_model_weights,
        quantize_model=quantize_model,
        compile_models=True,
        quantize_bits=4,
        requested_loader_route="mlx",
    )

    report = load_stage_handles(_plan(tmp_path), requests, run_id="production-loader")

    assert report.ok is True
    assert FLOW_MODEL_ROLE_IDS == (
        "sparse_structure_flow",
        "shape_flow_lr",
        "shape_flow_hr",
        "texture_flow",
    )
    quantized = [event[1] for event in events if event[0] == "quantize"]
    assert tuple(quantized) == FLOW_MODEL_ROLE_IDS
    compiled = [
        load_report.handle_id
        for load_report in report.load_reports
        if load_report.metadata["compiled"] is True
    ]
    assert tuple(compiled) == FLOW_MODEL_ROLE_IDS
    not_compiled = [
        load_report.handle_id
        for load_report in report.load_reports
        if load_report.metadata["compiled"] is False
    ]
    assert tuple(not_compiled) == (
        "sparse_structure_decoder",
        "shape_decoder",
        "texture_decoder",
    )


def test_production_model_loader_uses_default_cleanup_without_explicit_closers(
    tmp_path,
    monkeypatch,
):
    from trellmlx.production_model_loader import build_trellis_production_model_loader_requests
    from trellmlx.stage_handle_loader import load_stage_handles

    role_ids = ("shape_flow_lr",)
    events = []

    def load_model_weights(model, checkpoint_path, *, verbose):
        events.append(("load", model.role_id, verbose))

    monkeypatch.setattr(
        "trellmlx.production_model_loader._default_cleanup_model",
        lambda model: events.append(("cleanup-default", model.role_id)),
    )

    requests = build_trellis_production_model_loader_requests(
        role_ids=role_ids,
        constructors=_fake_constructors(role_ids, events),
        load_model_weights=load_model_weights,
        verbose=True,
    )

    report = load_stage_handles(_plan(tmp_path), requests, run_id="production-loader")

    assert report.ok is True
    assert events == [
        ("construct", "shape_flow_lr"),
        ("load", "shape_flow_lr", True),
        ("cleanup-default", "shape_flow_lr"),
    ]


def test_production_model_loader_closer_uses_injected_cleanup(tmp_path):
    from trellmlx.production_model_loader import (
        build_trellis_production_model_loader_closers,
        build_trellis_production_model_loader_requests,
    )
    from trellmlx.stage_handle_loader import load_stage_handles

    role_ids = ("shape_flow_lr",)
    events = []

    def load_model_weights(model, checkpoint_path, *, verbose):
        events.append(("load", model.role_id))

    def cleanup_model(model):
        events.append(("cleanup", model.role_id))

    requests = build_trellis_production_model_loader_requests(
        role_ids=role_ids,
        constructors=_fake_constructors(role_ids, events),
        load_model_weights=load_model_weights,
        closes=build_trellis_production_model_loader_closers(cleanup_model=cleanup_model),
    )

    report = load_stage_handles(_plan(tmp_path), requests, run_id="production-loader")

    assert report.ok is True
    assert events == [
        ("construct", "shape_flow_lr"),
        ("load", "shape_flow_lr"),
        ("cleanup", "shape_flow_lr"),
    ]
    assert report.close_reports[0].handle_id == "shape_flow_lr"
    assert report.close_reports[0].metadata["requested_loader_route"] == "mlx"

"""No-heavy assembly contracts for production-shaped interleaved TRELLIS runs."""


def _single_job_plan(tmp_path, stages=None):
    from trellmlx.interleaved_generation import DEFAULT_STAGE_SEQUENCE, GenerationJob, InterleavedBatchPlan

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    return InterleavedBatchPlan(jobs=(job,), stages=stages or DEFAULT_STAGE_SEQUENCE)


def _fake_constructors(role_ids, events):
    class FakeModel:
        def __init__(self, role_id):
            self.role_id = role_id
            self.compiled = False

        def compile(self):
            self.compiled = True

    return {
        role_id: (lambda role_id=role_id: events.append(("construct", role_id)) or FakeModel(role_id))
        for role_id in role_ids
    }


def _fixture_outputs():
    return {
        "image_conditioning": {"conditioning_key": "conditioning://seed-101"},
        "sparse_structure": {"sparse_structure_key": "sparse://seed-101"},
        "lr_shape_latent": {"lr_shape_latent_key": "shape-lr://seed-101"},
        "hr_coordinates": {"hr_coordinate_key": "coords://seed-101"},
        "hr_shape_latent": {"hr_shape_latent_key": "shape-hr://seed-101"},
        "shape_decode": {"shape_key": "shape://seed-101"},
        "mesh_extract": {"raw_mesh_key": "raw-mesh://seed-101"},
        "mesh_postprocess": {"mesh_key": "mesh://seed-101"},
        "texture_latent": {"texture_latent_key": "texture-latent://seed-101"},
        "texture_decode": {"texture_key": "texture://seed-101"},
        "texture_bake": {"texture_bake_key": "texture-bake://seed-101"},
        "export": {"export_key": "export://seed-101"},
    }


def _fixtures(events):
    from trellmlx.interleaved_generation import GenerationStageResult, StageRunnerOutput

    outputs = _fixture_outputs()

    def make_fixture(stage):
        def fixture(runtime):
            events.append(("stage", stage, tuple(runtime.handles), runtime.context.run_id))
            return StageRunnerOutput(
                result=GenerationStageResult(stage, elapsed_seconds=0.01),
                artifacts=outputs[stage],
            )

        return fixture

    return {stage: make_fixture(stage) for stage in outputs}


def _expected_artifacts():
    artifacts = {}
    for stage_artifacts in _fixture_outputs().values():
        artifacts.update(stage_artifacts)
    return artifacts


def test_production_assembly_builds_lazy_full_runner_components(tmp_path):
    from trellmlx.interleaved_production import production_model_role_ids, production_stage_sequence
    from trellmlx.production_assembly import build_trellis_production_assembly

    events = []
    role_ids = production_model_role_ids()

    def load_model_weights(model, checkpoint_path, *, verbose):
        events.append(("load", model.role_id, checkpoint_path, verbose))

    def load_dinov3_weights(model, checkpoint_path):
        events.append(("load-dino", model.role_id, checkpoint_path))
        return 412

    def cleanup_model(model):
        events.append(("cleanup", model.role_id))

    assembly = build_trellis_production_assembly(
        fixtures=_fixtures(events),
        constructors=_fake_constructors(role_ids, events),
        checkpoint_paths={role_id: f"weights://{role_id}" for role_id in role_ids},
        load_model_weights=load_model_weights,
        load_dinov3_weights=load_dinov3_weights,
        cleanup_model=cleanup_model,
        requested_loader_route="fixture",
        run_id="production-assembly",
        verbose=False,
    )

    assert assembly.stages == production_stage_sequence()
    assert assembly.role_ids == role_ids
    assert [request.handle_id for request in assembly.loader_requests] == list(role_ids)
    assert set(assembly.handlers) == set(production_stage_sequence())
    assert events == []

    result = assembly.build_runner(_single_job_plan(tmp_path)).run()

    assert result.ok is True
    assert result.context_closed is True
    assert result.context.run_id == "production-assembly"
    assert result.job_states["seed-101"].artifacts == _expected_artifacts()
    assert events[:3] == [
        ("construct", "dinov3_image_encoder"),
        ("load-dino", "dinov3_image_encoder", "weights://dinov3_image_encoder"),
        ("construct", "sparse_structure_flow"),
    ]
    assert ("stage", "mesh_extract", (), "production-assembly") in events
    assert [event for event in events if event[0] == "cleanup"] == [
        ("cleanup", role_id) for role_id in reversed(role_ids)
    ]


def test_production_assembly_threads_report_path_run_id_and_overrides(tmp_path):
    import json

    from trellmlx.interleaved_production import production_model_role_ids
    from trellmlx.production_assembly import build_trellis_production_assembly

    events = []
    role_ids = production_model_role_ids()
    report_path = tmp_path / "production-loader-report.json"

    def load_model_weights(model, checkpoint_path, *, verbose):
        events.append(("load", model.role_id, checkpoint_path, verbose))

    def load_dinov3_weights(model, checkpoint_path):
        events.append(("load-dino", model.role_id, checkpoint_path))

    assembly = build_trellis_production_assembly(
        fixtures=_fixtures(events),
        constructors=_fake_constructors(role_ids, events),
        checkpoint_paths={"shape_flow_lr": "override://shape-flow-lr"},
        hf_cache_root=tmp_path / "hf-cache",
        load_model_weights=load_model_weights,
        load_dinov3_weights=load_dinov3_weights,
        cleanup_model=lambda model: events.append(("cleanup", model.role_id)),
        requested_loader_route={"shape_flow_lr": "fixture-lr", **{role_id: "fixture" for role_id in role_ids if role_id != "shape_flow_lr"}},
        report_path=report_path,
        run_id="assembly-report",
        verbose=True,
    )

    result = assembly.build_runner(_single_job_plan(tmp_path)).run()

    assert result.ok is True
    persisted = json.loads(report_path.read_text())
    assert persisted["run_id"] == "assembly-report"
    assert persisted["requested_handle_ids"] == list(role_ids)
    lr_report = next(
        load_report for load_report in persisted["load_reports"]
        if load_report["handle_id"] == "shape_flow_lr"
    )
    assert lr_report["metadata"]["requested_loader_route"] == "fixture-lr"
    assert lr_report["metadata"]["effective_loader_route"] == "fixture-lr"
    assert lr_report["metadata"]["weights_path"] == "override://shape-flow-lr"
    assert ("load", "shape_flow_lr", "override://shape-flow-lr", True) in events


def test_production_assembly_defaults_roles_to_selected_stages(tmp_path):
    from trellmlx.interleaved_production import production_model_role_ids
    from trellmlx.production_assembly import build_trellis_production_assembly

    events = []
    role_ids = production_model_role_ids()
    assembly = build_trellis_production_assembly(
        fixtures=_fixtures(events),
        stages=("mesh_extract",),
        constructors=_fake_constructors(role_ids, events),
        load_model_weights=lambda model, checkpoint_path, *, verbose: events.append(("load", model.role_id)),
        load_dinov3_weights=lambda model, checkpoint_path: events.append(("load-dino", model.role_id)),
        requested_loader_route="fixture",
        run_id="mesh-only",
    )

    assert assembly.role_ids == ()
    assert assembly.loader_requests == ()

    context = assembly.context_factory(_single_job_plan(tmp_path, stages=("mesh_extract",)))
    try:
        assert context.run_id == "mesh-only"
        assert context.handles == {}
        assert events == []
    finally:
        assembly.context_closer(context)


def test_production_assembly_exposes_run_id_for_batch_load_failure(tmp_path):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_interleaved_batch
    from trellmlx.interleaved_production import production_model_role_ids
    from trellmlx.production_assembly import build_trellis_production_assembly

    events = []
    role_ids = production_model_role_ids()
    report_path = tmp_path / "interleaved-report.json"
    assembly = build_trellis_production_assembly(
        fixtures=_fixtures(events),
        stages=("image_conditioning",),
        constructors=_fake_constructors(role_ids, events),
        load_model_weights=lambda model, checkpoint_path, *, verbose: None,
        load_dinov3_weights=lambda model, checkpoint_path: (_ for _ in ()).throw(RuntimeError("dino boom")),
        requested_loader_route="fixture",
        run_id="assembly-load-fail",
    )
    job = BatchJob(
        images=("subject.png",),
        seed=101,
        output_path=tmp_path / "seed-101.glb",
        resolution=512,
    )
    options = BatchRunOptions(
        max_concurrent=1,
        repo_root=".",
        report_path=report_path,
        command_line=("pytest", "production-assembly-load-failure"),
    )

    report = run_interleaved_batch(
        [job],
        options,
        stages=assembly.stages,
        handlers=assembly.handlers,
        context_factory=assembly.context_factory,
        context_closer=assembly.context_closer,
    )

    assert report.failure_phase == "context_load"
    assert report.context_run_id == "assembly-load-fail"

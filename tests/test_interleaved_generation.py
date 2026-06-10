"""Contracts for in-process stage-major generation interleaving."""

from pathlib import Path

import pytest


def test_generation_job_requires_explicit_conditioning_route(tmp_path):
    from trellmlx.interleaved_generation import GenerationJob

    with pytest.raises(ValueError, match="image or explicit random conditioning"):
        GenerationJob(job_id="missing", images=(), seed=1, output_path=tmp_path / "out.glb")

    random_job = GenerationJob(
        job_id="random",
        images=(),
        seed=1,
        output_path=tmp_path / "random.glb",
        random_conditioning=True,
    )

    assert random_job.conditioning_route == "random"


def test_stage_major_plan_groups_jobs_by_stage_before_advancing(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        InterleavedBatchPlan,
    )

    jobs = (
        GenerationJob(
            job_id="seed-101",
            images=("subject.png",),
            seed=101,
            output_path=tmp_path / "seed-101.glb",
        ),
        GenerationJob(
            job_id="seed-202",
            images=("subject.png",),
            seed=202,
            output_path=tmp_path / "seed-202.glb",
        ),
    )
    plan = InterleavedBatchPlan(
        jobs=jobs,
        stages=("image_conditioning", "sparse_structure", "mesh_export"),
    )

    assert [
        (invocation.stage, invocation.job_id, invocation.seed)
        for invocation in plan.iter_invocations()
    ] == [
        ("image_conditioning", "seed-101", 101),
        ("image_conditioning", "seed-202", 202),
        ("sparse_structure", "seed-101", 101),
        ("sparse_structure", "seed-202", 202),
        ("mesh_export", "seed-101", 101),
        ("mesh_export", "seed-202", 202),
    ]


def test_plan_rejects_duplicate_job_ids(tmp_path):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan

    jobs = (
        GenerationJob(
            job_id="dup",
            images=("a.png",),
            seed=1,
            output_path=tmp_path / "one.glb",
        ),
        GenerationJob(
            job_id="dup",
            images=("b.png",),
            seed=2,
            output_path=tmp_path / "two.glb",
        ),
    )

    with pytest.raises(ValueError, match="duplicate job_id"):
        InterleavedBatchPlan(jobs=jobs)


def test_job_trace_records_stage_timings_and_output_counts(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        GenerationStageResult,
        new_job_traces,
    )

    job = GenerationJob(
        job_id="seed-101",
        images=("subject.png",),
        seed=101,
        output_path=tmp_path / "seed-101.glb",
        resolution=512,
        texture_size=1024,
    )
    traces = new_job_traces((job,))
    trace = traces["seed-101"].record_stage(
        GenerationStageResult(
            stage="sparse_structure",
            elapsed_seconds=1.25,
            output_counts={"lr_voxels": 3072},
        )
    )

    assert trace.job_id == "seed-101"
    assert trace.seed == 101
    assert trace.output_path == str(tmp_path / "seed-101.glb")
    assert trace.stage_results[0].stage == "sparse_structure"
    assert trace.stage_results[0].output_counts == {"lr_voxels": 3072}
    assert trace.config["resolution"] == 512
    assert trace.config["texture_size"] == 1024


def test_generation_job_can_be_built_from_process_batch_job(tmp_path):
    from trellmlx.batch_inference import BatchJob
    from trellmlx.interleaved_generation import GenerationJob

    batch_job = BatchJob(
        images=("subject.png",),
        seed=101,
        output_path=tmp_path / "seed-101.glb",
        resolution=512,
        target_faces=50_000,
        no_cleanup=True,
    )

    generation_job = GenerationJob.from_batch_job(batch_job)

    assert generation_job.job_id == "seed-101"
    assert generation_job.images == ("subject.png",)
    assert generation_job.seed == 101
    assert generation_job.output_path == Path(tmp_path / "seed-101.glb")
    assert generation_job.resolution == 512
    assert generation_job.target_faces == 50_000
    assert generation_job.no_cleanup is True


def test_stage_seed_is_deterministic_per_job_and_stage():
    from trellmlx.interleaved_generation import derive_stage_seed

    assert derive_stage_seed(job_seed=101, stage_index=3) == derive_stage_seed(
        job_seed=101,
        stage_index=3,
    )
    assert derive_stage_seed(job_seed=101, stage_index=3) != derive_stage_seed(
        job_seed=202,
        stage_index=3,
    )
    assert derive_stage_seed(job_seed=101, stage_index=3) != derive_stage_seed(
        job_seed=101,
        stage_index=4,
    )
    assert 0 <= derive_stage_seed(job_seed=101, stage_index=3) < 2**32


def test_stage_runner_executes_stage_major_and_preserves_job_state(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        GenerationStageResult,
        InterleavedBatchPlan,
        StageRunner,
        StageRunnerOutput,
        derive_stage_seed,
    )

    jobs = (
        GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb"),
        GenerationJob("seed-202", ("subject.png",), 202, tmp_path / "seed-202.glb"),
    )
    plan = InterleavedBatchPlan(jobs=jobs, stages=("image_conditioning", "sparse_structure"))
    calls = []

    def image_conditioning(invocation, state, context):
        calls.append((invocation.stage, invocation.job_id, state.next_stage_index))
        return StageRunnerOutput(
            result=GenerationStageResult(invocation.stage, elapsed_seconds=0.01),
            artifacts={"conditioning_seed": invocation.stage_seed},
        )

    def sparse_structure(invocation, state, context):
        calls.append((invocation.stage, invocation.job_id, state.next_stage_index))
        assert state.artifacts["conditioning_seed"] == derive_stage_seed(
            job_seed=invocation.seed,
            stage_index=0,
        )
        return StageRunnerOutput(
            result=GenerationStageResult(
                invocation.stage,
                elapsed_seconds=0.02,
                output_counts={"tokens": 17},
            ),
            artifacts={"sparse_tokens": 17},
        )

    result = StageRunner(
        plan,
        handlers={
            "image_conditioning": image_conditioning,
            "sparse_structure": sparse_structure,
        },
    ).run()

    assert calls == [
        ("image_conditioning", "seed-101", 0),
        ("image_conditioning", "seed-202", 0),
        ("sparse_structure", "seed-101", 1),
        ("sparse_structure", "seed-202", 1),
    ]
    assert result.ok is True
    assert result.job_states["seed-101"].next_stage_index == 2
    assert result.job_states["seed-101"].artifacts["sparse_tokens"] == 17
    assert [stage.stage for stage in result.job_states["seed-202"].stage_results] == [
        "image_conditioning",
        "sparse_structure",
    ]


def test_stage_runner_captures_failure_phase_and_skips_failed_job(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        GenerationStageResult,
        InterleavedBatchPlan,
        StageRunner,
    )

    jobs = (
        GenerationJob("ok", ("subject.png",), 101, tmp_path / "ok.glb"),
        GenerationJob("bad", ("subject.png",), 202, tmp_path / "bad.glb"),
    )
    plan = InterleavedBatchPlan(jobs=jobs, stages=("sparse_structure", "export"))
    calls = []

    def sparse_structure(invocation, state, context):
        calls.append((invocation.stage, invocation.job_id))
        if invocation.job_id == "bad":
            return GenerationStageResult(
                invocation.stage,
                elapsed_seconds=0.01,
                failure_phase="sparse_structure:model_error",
            )
        return GenerationStageResult(invocation.stage, elapsed_seconds=0.01)

    def export(invocation, state, context):
        calls.append((invocation.stage, invocation.job_id))
        return GenerationStageResult(invocation.stage, elapsed_seconds=0.01)

    result = StageRunner(
        plan,
        handlers={"sparse_structure": sparse_structure, "export": export},
    ).run()

    assert calls == [
        ("sparse_structure", "ok"),
        ("sparse_structure", "bad"),
        ("export", "ok"),
    ]
    assert result.ok is False
    assert result.job_states["bad"].failure_phase == "sparse_structure:model_error"
    assert [stage.stage for stage in result.job_states["bad"].stage_results] == [
        "sparse_structure",
    ]


def test_stage_runner_requires_handlers_for_all_stages(tmp_path):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan, StageRunner

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    plan = InterleavedBatchPlan(jobs=(job,), stages=("image_conditioning", "export"))

    with pytest.raises(ValueError, match="missing stage handlers: export"):
        StageRunner(plan, handlers={"image_conditioning": lambda invocation, state, context: None})


def test_stage_state_rejects_nonportable_artifacts(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        GenerationStageResult,
        JobState,
        StageRunnerOutput,
    )

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")

    with pytest.raises(ValueError, match="artifact values must be bool, int, float, or str"):
        StageRunnerOutput(
            result=GenerationStageResult("stage", elapsed_seconds=0.0),
            artifacts={"tensor_like": [1, 2, 3]},
        )

    with pytest.raises(ValueError, match="artifact keys must be strings"):
        JobState.from_job(job).record_artifacts({7: "not-portable"})


def test_stage_runner_rejects_initial_states_outside_plan(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        GenerationStageResult,
        InterleavedBatchPlan,
        JobState,
        StageRunner,
    )

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    plan = InterleavedBatchPlan(jobs=(job,), stages=("stage",))
    runner = StageRunner(
        plan,
        handlers={
            "stage": lambda invocation, state, context: GenerationStageResult(
                invocation.stage,
                elapsed_seconds=0.0,
            )
        },
    )

    with pytest.raises(ValueError, match="missing initial state for job_id: seed-101"):
        runner.run(initial_states={})

    extra = JobState(
        job_id="unplanned-failed",
        seed=202,
        images=("subject.png",),
        output_path=str(tmp_path / "extra.glb"),
        config={},
        failure_phase="stale_failure",
    )

    with pytest.raises(ValueError, match="unexpected initial state job_id: unplanned-failed"):
        runner.run(initial_states={"seed-101": JobState.from_job(job), "unplanned-failed": extra})


def test_stage_execution_context_validates_metadata_and_requires_handles():
    from trellmlx.interleaved_generation import StageExecutionContext

    handle = object()
    context = StageExecutionContext(
        run_id="fixture-run",
        handles={"dinov3": handle},
        handle_metadata={"dinov3": {"kind": "fixture", "warm": True}},
    )

    assert context.require_handle("dinov3") is handle
    assert context.handle_ids == ("dinov3",)
    assert context.handle_metadata["dinov3"] == {"kind": "fixture", "warm": True}

    with pytest.raises(ValueError, match="artifact values must be bool, int, float, or str"):
        StageExecutionContext(
            run_id="bad-metadata",
            handles={"dinov3": handle},
            handle_metadata={"dinov3": {"tensor_like": [1, 2, 3]}},
        )

    with pytest.raises(KeyError, match="missing stage execution handle: sparse_structure"):
        context.require_handle("sparse_structure")


def test_stage_runner_passes_context_to_every_handler_and_closes_once(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        GenerationStageResult,
        InterleavedBatchPlan,
        StageExecutionContext,
        StageRunner,
        StageRunnerOutput,
    )

    jobs = (
        GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb"),
        GenerationJob("seed-202", ("subject.png",), 202, tmp_path / "seed-202.glb"),
    )
    plan = InterleavedBatchPlan(jobs=jobs, stages=("image_conditioning", "sparse_structure"))
    handle = object()
    events = []
    context_ids = []

    def open_context(open_plan):
        events.append(("open", tuple(job.job_id for job in open_plan.jobs)))
        return StageExecutionContext(
            run_id="fixture-run",
            handles={"dinov3": handle},
            handle_metadata={"dinov3": {"kind": "fixture"}},
        )

    def close_context(context):
        events.append(("close", context.run_id, context.handle_ids))

    def image_conditioning(invocation, state, context):
        assert context.require_handle("dinov3") is handle
        context_ids.append(id(context))
        return StageRunnerOutput(
            result=GenerationStageResult(invocation.stage, elapsed_seconds=0.01),
            artifacts={"context_seen": context.run_id},
        )

    def sparse_structure(invocation, state, context):
        assert state.artifacts["context_seen"] == "fixture-run"
        assert context.require_handle("dinov3") is handle
        context_ids.append(id(context))
        return GenerationStageResult(invocation.stage, elapsed_seconds=0.02)

    result = StageRunner(
        plan,
        handlers={
            "image_conditioning": image_conditioning,
            "sparse_structure": sparse_structure,
        },
        context_factory=open_context,
        context_closer=close_context,
    ).run()

    assert result.ok is True
    assert result.context.run_id == "fixture-run"
    assert result.context_closed is True
    assert len(set(context_ids)) == 1
    assert events == [
        ("open", ("seed-101", "seed-202")),
        ("close", "fixture-run", ("dinov3",)),
    ]


def test_stage_runner_closes_context_when_one_job_fails(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        GenerationStageResult,
        InterleavedBatchPlan,
        StageExecutionContext,
        StageRunner,
    )

    jobs = (
        GenerationJob("ok", ("subject.png",), 101, tmp_path / "ok.glb"),
        GenerationJob("bad", ("subject.png",), 202, tmp_path / "bad.glb"),
    )
    plan = InterleavedBatchPlan(jobs=jobs, stages=("sparse_structure", "export"))
    events = []

    def open_context(open_plan):
        events.append(("open", len(open_plan.jobs)))
        return StageExecutionContext(run_id="failure-run", handles={"shape": object()})

    def close_context(context):
        events.append(("close", context.run_id))

    def sparse_structure(invocation, state, context):
        context.require_handle("shape")
        if invocation.job_id == "bad":
            return GenerationStageResult(
                invocation.stage,
                elapsed_seconds=0.01,
                failure_phase="sparse_structure:model_error",
            )
        return GenerationStageResult(invocation.stage, elapsed_seconds=0.01)

    def export(invocation, state, context):
        context.require_handle("shape")
        return GenerationStageResult(invocation.stage, elapsed_seconds=0.01)

    result = StageRunner(
        plan,
        handlers={"sparse_structure": sparse_structure, "export": export},
        context_factory=open_context,
        context_closer=close_context,
    ).run()

    assert result.ok is False
    assert result.context_closed is True
    assert result.job_states["bad"].failure_phase == "sparse_structure:model_error"
    assert events == [("open", 2), ("close", "failure-run")]


def test_stage_runner_validates_initial_states_before_opening_context(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        GenerationStageResult,
        InterleavedBatchPlan,
        StageExecutionContext,
        StageRunner,
    )

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    plan = InterleavedBatchPlan(jobs=(job,), stages=("stage",))
    events = []

    def open_context(open_plan):
        events.append(("open", len(open_plan.jobs)))
        return StageExecutionContext(run_id="should-not-open")

    runner = StageRunner(
        plan,
        handlers={
            "stage": lambda invocation, state, context: GenerationStageResult(
                invocation.stage,
                elapsed_seconds=0.0,
            )
        },
        context_factory=open_context,
    )

    with pytest.raises(ValueError, match="missing initial state for job_id: seed-101"):
        runner.run(initial_states={})

    assert events == []


def test_stage_handle_specs_validate_identity_and_scalar_metadata():
    from trellmlx.interleaved_generation import StageHandleSpec, build_stage_context_factory

    spec = StageHandleSpec(
        handle_id="dinov3",
        kind="fixture",
        factory=lambda runtime: object(),
        metadata={"weights": "fixture", "quantized": False},
    )

    assert spec.handle_id == "dinov3"
    assert spec.kind == "fixture"
    assert spec.metadata == {"weights": "fixture", "quantized": False}

    with pytest.raises(ValueError, match="StageHandleSpec requires a handle_id"):
        StageHandleSpec(handle_id="", kind="fixture", factory=lambda runtime: object())

    with pytest.raises(ValueError, match="artifact values must be bool, int, float, or str"):
        StageHandleSpec(
            handle_id="bad",
            kind="fixture",
            factory=lambda runtime: object(),
            metadata={"tensor_like": [1, 2, 3]},
        )

    with pytest.raises(ValueError, match="metadata cannot use reserved report keys: kind, load_phase"):
        StageHandleSpec(
            handle_id="reserved",
            kind="fixture",
            factory=lambda runtime: object(),
            metadata={"kind": "caller-kind", "load_phase": "caller-load"},
        )

    with pytest.raises(ValueError, match="duplicate handle_id: dinov3"):
        build_stage_context_factory(
            [
                spec,
                StageHandleSpec("dinov3", "fixture", lambda runtime: object()),
            ]
        )


def test_stage_context_factory_loads_named_handles_once_and_reports_close_order(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        InterleavedBatchPlan,
        StageHandleSpec,
        build_stage_context_factory,
    )

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    plan = InterleavedBatchPlan(jobs=(job,), stages=("stage",))
    events = []
    handles = {"dinov3": object(), "sparse_structure": object()}

    def load_dinov3(runtime):
        events.append(("load", runtime.run_id, runtime.spec.handle_id, len(runtime.plan.jobs)))
        return handles["dinov3"]

    def load_sparse(runtime):
        events.append(("load", runtime.run_id, runtime.spec.handle_id, len(runtime.plan.jobs)))
        return handles["sparse_structure"]

    def close_handle(runtime, handle):
        events.append(("close", runtime.run_id, runtime.spec.handle_id, handle is handles[runtime.spec.handle_id]))

    context_factory, context_closer = build_stage_context_factory(
        [
            StageHandleSpec(
                "dinov3",
                "fixture",
                load_dinov3,
                close=close_handle,
                metadata={"weights": "fixture-dino"},
            ),
            StageHandleSpec(
                "sparse_structure",
                "fixture",
                load_sparse,
                close=close_handle,
                metadata={"weights": "fixture-ss"},
            ),
        ],
        run_id="fixture-run",
    )

    context = context_factory(plan)

    assert context.require_handle("dinov3") is handles["dinov3"]
    assert context.require_handle("sparse_structure") is handles["sparse_structure"]
    assert context.handle_ids == ("dinov3", "sparse_structure")
    assert context.handle_metadata == {
        "dinov3": {
            "kind": "fixture",
            "load_phase": "loaded",
            "weights": "fixture-dino",
        },
        "sparse_structure": {
            "kind": "fixture",
            "load_phase": "loaded",
            "weights": "fixture-ss",
        },
    }
    assert [report.handle_id for report in context.load_reports] == ["dinov3", "sparse_structure"]
    assert [report.load_phase for report in context.load_reports] == ["loaded", "loaded"]
    assert context.load_reports[0].metadata == {
        "kind": "fixture",
        "load_phase": "loaded",
        "weights": "fixture-dino",
    }

    context_closer(context)

    assert events == [
        ("load", "fixture-run", "dinov3", 1),
        ("load", "fixture-run", "sparse_structure", 1),
        ("close", "fixture-run", "sparse_structure", True),
        ("close", "fixture-run", "dinov3", True),
    ]
    assert [report.close_phase for report in context.close_reports] == ["closed", "closed"]
    assert [report.handle_id for report in context.close_reports] == ["sparse_structure", "dinov3"]


def test_stage_runner_context_factory_failure_reports_without_stage_calls(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        GenerationStageResult,
        InterleavedBatchPlan,
        StageHandleSpec,
        StageHandleLoadError,
        StageRunner,
        build_stage_context_factory,
    )

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    plan = InterleavedBatchPlan(jobs=(job,), stages=("stage",))
    calls = []

    def fail_load(runtime):
        raise RuntimeError("fixture loader exploded")

    context_factory, context_closer = build_stage_context_factory(
        [StageHandleSpec("dinov3", "fixture", fail_load)],
        run_id="failure-run",
    )
    runner = StageRunner(
        plan,
        handlers={
            "stage": lambda invocation, state, context: calls.append(invocation.job_id)
            or GenerationStageResult(invocation.stage, elapsed_seconds=0.0)
        },
        context_factory=context_factory,
        context_closer=context_closer,
    )

    with pytest.raises(StageHandleLoadError, match="failed to load stage handle dinov3"):
        runner.run()

    assert calls == []
    assert context_factory.last_load_reports[0].handle_id == "dinov3"
    assert context_factory.last_load_reports[0].load_phase == "load_error"
    assert "fixture loader exploded" in context_factory.last_load_reports[0].error


def test_stage_context_factory_closes_loaded_handles_after_later_load_failure(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        InterleavedBatchPlan,
        StageHandleLoadError,
        StageHandleSpec,
        build_stage_context_factory,
    )

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    plan = InterleavedBatchPlan(jobs=(job,), stages=("stage",))
    first_handle = object()
    events = []

    def load_first(runtime):
        events.append(("load", runtime.spec.handle_id))
        return first_handle

    def close_first(runtime, handle):
        events.append(("close", runtime.spec.handle_id, handle is first_handle))

    def load_second(runtime):
        events.append(("load", runtime.spec.handle_id))
        raise RuntimeError("second handle failed")

    context_factory, context_closer = build_stage_context_factory(
        [
            StageHandleSpec("first", "fixture", load_first, close=close_first),
            StageHandleSpec("second", "fixture", load_second),
        ],
        run_id="partial-failure",
    )

    with pytest.raises(StageHandleLoadError, match="failed to load stage handle second"):
        context_factory(plan)

    assert events == [
        ("load", "first"),
        ("load", "second"),
        ("close", "first", True),
    ]
    assert [report.load_phase for report in context_factory.last_load_reports] == [
        "loaded",
        "load_error",
    ]
    assert [report.handle_id for report in context_factory.last_close_reports] == ["first"]
    assert [report.close_phase for report in context_factory.last_close_reports] == ["closed"]
    assert context_closer.last_close_reports == ()


def test_stage_context_closer_reports_close_errors_and_continues(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        InterleavedBatchPlan,
        StageHandleCloseError,
        StageHandleSpec,
        build_stage_context_factory,
    )

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    plan = InterleavedBatchPlan(jobs=(job,), stages=("stage",))
    events = []
    handles = {"first": object(), "second": object()}

    def close_first(runtime, handle):
        events.append(("close", runtime.spec.handle_id, handle is handles["first"]))

    def close_second(runtime, handle):
        events.append(("close", runtime.spec.handle_id, handle is handles["second"]))
        raise RuntimeError("second close failed")

    context_factory, context_closer = build_stage_context_factory(
        [
            StageHandleSpec("first", "fixture", lambda runtime: handles["first"], close=close_first),
            StageHandleSpec("second", "fixture", lambda runtime: handles["second"], close=close_second),
        ],
        run_id="close-failure",
    )
    context = context_factory(plan)

    with pytest.raises(StageHandleCloseError, match="failed to close stage handle second"):
        context_closer(context)

    assert events == [
        ("close", "second", True),
        ("close", "first", True),
    ]
    assert [report.handle_id for report in context.close_reports] == ["second", "first"]
    assert [report.close_phase for report in context.close_reports] == ["close_error", "closed"]
    assert "second close failed" in context.close_reports[0].error
    assert context_closer.last_close_reports == context.close_reports

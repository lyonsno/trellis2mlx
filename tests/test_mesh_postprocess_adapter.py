"""No-generation mesh postprocess stage adapter contract."""

import pytest


def _plan(tmp_path, *, stages=("mesh_postprocess",)):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    return InterleavedBatchPlan(jobs=(job,), stages=stages)


def test_mesh_postprocess_adapter_records_scalar_mesh_facts_and_runtime_object(tmp_path):
    from trellmlx.interleaved_generation import JobState, StageExecutionContext
    from trellmlx.mesh_postprocess_adapter import (
        MeshPostprocessFixtureResult,
        build_mesh_postprocess_stage_handler,
    )

    plan = _plan(tmp_path)
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0]).record_artifacts({"raw_mesh_key": "mesh://seed-101/raw"})
    raw_mesh = object()
    processed_mesh = object()
    calls = []

    def postprocess(runtime):
        calls.append(
            (
                runtime.invocation.job_id,
                runtime.input_mesh_key,
                runtime.mesh_object is raw_mesh,
                runtime.state.config["target_faces"],
            )
        )
        return MeshPostprocessFixtureResult(
            mesh_key="mesh://seed-101/postprocess",
            vertices=76_296,
            faces=145_414,
            elapsed_seconds=0.42,
            mesh_object=processed_mesh,
        )

    context = StageExecutionContext(
        run_id="mesh-postprocess-probe",
        runtime_objects={"mesh://seed-101/raw": raw_mesh},
    )
    output = build_mesh_postprocess_stage_handler(fixture=postprocess)(invocation, state, context)

    assert output.result.stage == "mesh_postprocess"
    assert output.result.elapsed_seconds == 0.42
    assert output.result.output_counts == {"vertices": 76_296, "faces": 145_414}
    assert output.artifacts == {
        "raw_mesh_key": "mesh://seed-101/raw",
        "mesh_key": "mesh://seed-101/postprocess",
        "mesh_postprocess_route": "fixture",
        "mesh_target_faces": 200_000,
        "mesh_no_cleanup": False,
        "mesh_keep_largest": False,
        "mesh_vertices": 76_296,
        "mesh_faces": 145_414,
    }
    assert context.require_runtime_object("mesh://seed-101/postprocess") is processed_mesh
    assert "mesh_object" not in output.artifacts
    assert calls == [("seed-101", "mesh://seed-101/raw", True, 200_000)]


def test_mesh_postprocess_adapter_uses_stage_runner_initial_state(tmp_path):
    from trellmlx.interleaved_generation import JobState, StageExecutionContext, StageRunner
    from trellmlx.mesh_postprocess_adapter import (
        MeshPostprocessFixtureResult,
        build_mesh_postprocess_stage_handler,
    )

    plan = _plan(tmp_path)
    raw_mesh = object()
    processed_mesh = object()
    context = StageExecutionContext(
        run_id="mesh-postprocess-probe",
        runtime_objects={"mesh://seed-101/raw": raw_mesh},
    )
    initial_state = JobState.from_job(plan.jobs[0]).record_artifacts(
        {"raw_mesh_key": "mesh://seed-101/raw"}
    )

    def postprocess(runtime):
        return MeshPostprocessFixtureResult(
            mesh_key="mesh://seed-101/postprocess",
            vertices=12,
            faces=34,
            mesh_object=processed_mesh,
        )

    result = StageRunner(
        plan,
        handlers={"mesh_postprocess": build_mesh_postprocess_stage_handler(fixture=postprocess)},
        context_factory=lambda plan: context,
    ).run(initial_states={"seed-101": initial_state})

    assert result.ok is True
    artifacts = result.job_states["seed-101"].artifacts
    assert artifacts["raw_mesh_key"] == "mesh://seed-101/raw"
    assert artifacts["mesh_key"] == "mesh://seed-101/postprocess"
    assert artifacts["mesh_vertices"] == 12
    assert artifacts["mesh_faces"] == 34
    assert "mesh_object" not in artifacts
    assert result.context.require_runtime_object("mesh://seed-101/postprocess") is processed_mesh


def test_mesh_postprocess_adapter_rejects_missing_raw_mesh_key_before_fixture(tmp_path):
    from trellmlx.interleaved_generation import JobState, StageExecutionContext
    from trellmlx.mesh_postprocess_adapter import build_mesh_postprocess_stage_handler

    plan = _plan(tmp_path)
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0])
    called = False

    def postprocess(runtime):
        nonlocal called
        called = True
        raise AssertionError("fixture should not run without raw_mesh_key")

    handler = build_mesh_postprocess_stage_handler(fixture=postprocess)

    with pytest.raises(KeyError, match="missing required state artifact for mesh_postprocess: raw_mesh_key"):
        handler(invocation, state, StageExecutionContext(run_id="mesh-postprocess-probe"))
    assert called is False


def test_mesh_postprocess_adapter_rejects_duplicate_output_mesh_key(tmp_path):
    from trellmlx.interleaved_generation import JobState, StageExecutionContext
    from trellmlx.mesh_postprocess_adapter import (
        MeshPostprocessFixtureResult,
        build_mesh_postprocess_stage_handler,
    )

    plan = _plan(tmp_path)
    invocation = next(plan.iter_invocations())
    state = JobState.from_job(plan.jobs[0]).record_artifacts({"raw_mesh_key": "mesh://seed-101/raw"})
    raw_mesh = object()
    existing_mesh = object()
    context = StageExecutionContext(
        run_id="mesh-postprocess-probe",
        runtime_objects={
            "mesh://seed-101/raw": raw_mesh,
            "mesh://seed-101/postprocess": existing_mesh,
        },
    )

    def postprocess(runtime):
        return MeshPostprocessFixtureResult(
            mesh_key="mesh://seed-101/postprocess",
            vertices=12,
            faces=34,
            mesh_object=object(),
        )

    handler = build_mesh_postprocess_stage_handler(fixture=postprocess)

    with pytest.raises(
        ValueError,
        match="mesh postprocess key collision for mesh://seed-101/postprocess on job seed-101",
    ):
        handler(invocation, state, context)

    assert context.require_runtime_object("mesh://seed-101/postprocess") is existing_mesh


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("vertices", True),
        ("vertices", 12.5),
        ("faces", True),
        ("faces", 34.5),
    ),
)
def test_mesh_postprocess_fixture_result_rejects_non_integer_counts(field, value):
    from trellmlx.mesh_postprocess_adapter import MeshPostprocessFixtureResult

    kwargs = {
        "mesh_key": "mesh://seed-101/postprocess",
        "vertices": 12,
        "faces": 34,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        MeshPostprocessFixtureResult(**kwargs)

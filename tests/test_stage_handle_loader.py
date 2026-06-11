import json

import pytest


def _single_job_plan(tmp_path):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan

    job = GenerationJob("seed-101", ("subject.png",), 101, tmp_path / "seed-101.glb")
    return InterleavedBatchPlan(jobs=(job,), stages=("image_conditioning",))


def test_load_stage_handles_records_effective_loader_route_and_writes_report(tmp_path):
    from trellmlx.stage_handle_loader import (
        LoadedStageHandle,
        StageHandleLoaderRequest,
        load_stage_handles,
    )

    plan = _single_job_plan(tmp_path)
    handle = object()
    events = []

    def load_dinov3(runtime):
        events.append(("load", runtime.stage_runtime.run_id, runtime.handle_id, len(runtime.stage_runtime.plan.jobs)))
        return LoadedStageHandle(
            handle=handle,
            effective_loader_route="fixture",
            metadata={"weights_path": "fixture://dinov3"},
        )

    def close_dinov3(runtime, loaded_handle):
        events.append(("close", runtime.stage_runtime.run_id, runtime.handle_id, loaded_handle is handle))

    report_path = tmp_path / "handle-report.json"
    report = load_stage_handles(
        plan,
        [
            StageHandleLoaderRequest(
                handle_id="dinov3",
                kind="model",
                requested_loader_route="fixture",
                factory=load_dinov3,
                close=close_dinov3,
                metadata={"model_family": "dinov3"},
            )
        ],
        report_path=report_path,
        run_id="handle-probe",
    )

    assert report.ok is True
    assert report.schema == "trellis2mlx.stage_handle_loader_report.v1"
    assert report.loaded_handle_ids == ("dinov3",)
    assert events == [
        ("load", "handle-probe", "dinov3", 1),
        ("close", "handle-probe", "dinov3", True),
    ]
    assert report.load_reports[0].metadata == {
        "kind": "model",
        "load_phase": "loaded",
        "requested_loader_route": "fixture",
        "model_family": "dinov3",
        "effective_loader_route": "fixture",
        "weights_path": "fixture://dinov3",
    }
    assert report.load_reports[0].elapsed_seconds >= 0.0
    assert report.close_reports[0].metadata == {
        "kind": "model",
        "close_phase": "closed",
        "requested_loader_route": "fixture",
        "model_family": "dinov3",
    }
    assert report.close_reports[0].elapsed_seconds >= 0.0

    persisted = json.loads(report_path.read_text())
    assert persisted["schema"] == "trellis2mlx.stage_handle_loader_report.v1"
    assert persisted["ok"] is True
    assert persisted["loaded_handle_ids"] == ["dinov3"]
    assert persisted["load_reports"][0]["elapsed_seconds"] >= 0.0
    assert persisted["close_reports"][0]["elapsed_seconds"] >= 0.0
    assert persisted["load_reports"][0]["metadata"]["effective_loader_route"] == "fixture"


def test_load_stage_handles_writes_failure_report_on_loader_route_mismatch(tmp_path):
    from trellmlx.stage_handle_loader import (
        LoadedStageHandle,
        StageHandleLoaderRequest,
        load_stage_handles,
    )

    plan = _single_job_plan(tmp_path)
    handle = object()
    events = []

    def load_with_fallback(runtime):
        events.append(("load", runtime.handle_id))
        return LoadedStageHandle(
            handle=handle,
            effective_loader_route="fallback",
            metadata={"weights_path": "fallback://dinov3"},
        )

    def close_fallback(runtime, loaded_handle):
        events.append(("close", runtime.handle_id, loaded_handle is handle))

    report_path = tmp_path / "handle-report.json"
    report = load_stage_handles(
        plan,
        [
            StageHandleLoaderRequest(
                handle_id="dinov3",
                kind="model",
                requested_loader_route="mlx",
                factory=load_with_fallback,
                close=close_fallback,
            )
        ],
        report_path=report_path,
        run_id="handle-probe",
    )

    assert report.ok is False
    assert report.failure_phase == "load"
    assert "effective loader route mismatch for dinov3" in report.error
    assert report.loaded_handle_ids == ()
    assert [load_report.load_phase for load_report in report.load_reports] == ["load_error"]
    assert report.load_reports[0].elapsed_seconds >= 0.0
    assert events == [
        ("load", "dinov3"),
        ("close", "dinov3", True),
    ]
    assert [close_report.close_phase for close_report in report.close_reports] == ["closed"]
    assert report.close_reports[0].elapsed_seconds >= 0.0

    persisted = json.loads(report_path.read_text())
    assert persisted["ok"] is False
    assert persisted["failure_phase"] == "load"
    assert "effective loader route mismatch for dinov3" in persisted["error"]
    assert persisted["load_reports"][0]["elapsed_seconds"] >= 0.0
    assert persisted["close_reports"][0]["elapsed_seconds"] >= 0.0
    assert persisted["load_reports"][0]["metadata"] == {
        "kind": "model",
        "load_phase": "load_error",
        "requested_loader_route": "mlx",
        "effective_loader_route": "fallback",
    }
    assert persisted["close_reports"][0]["metadata"] == {
        "kind": "model",
        "close_phase": "closed",
        "requested_loader_route": "mlx",
    }


def test_load_stage_handles_writes_setup_failure_report_before_factory_runs(tmp_path):
    from trellmlx.stage_handle_loader import StageHandleLoaderRequest, load_stage_handles

    plan = _single_job_plan(tmp_path)
    calls = []

    def load(runtime):
        calls.append(runtime.handle_id)
        return object()

    report_path = tmp_path / "handle-report.json"
    report = load_stage_handles(
        plan,
        [
            StageHandleLoaderRequest("dinov3", "model", "fixture", load),
            StageHandleLoaderRequest("dinov3", "model", "fixture", load),
        ],
        report_path=report_path,
        run_id="handle-probe",
    )

    assert calls == []
    assert report.ok is False
    assert report.failure_phase == "setup"
    assert report.requested_handle_ids == ("dinov3", "dinov3")
    assert report.loaded_handle_ids == ()
    assert report.load_reports == ()
    assert report.close_reports == ()
    assert "duplicate handle_id: dinov3" in report.error

    persisted = json.loads(report_path.read_text())
    assert persisted["ok"] is False
    assert persisted["failure_phase"] == "setup"
    assert persisted["requested_handle_ids"] == ["dinov3", "dinov3"]
    assert "duplicate handle_id: dinov3" in persisted["error"]


def test_loader_context_factory_feeds_stage_runner_and_writes_close_report(tmp_path):
    from trellmlx.interleaved_generation import GenerationStageResult, StageRunner
    from trellmlx.stage_handle_loader import (
        LoadedStageHandle,
        StageHandleLoaderRequest,
        build_stage_loader_context,
    )

    plan = _single_job_plan(tmp_path)
    handle = object()
    events = []

    def load_dinov3(runtime):
        events.append(("load", runtime.stage_runtime.run_id, runtime.handle_id))
        return LoadedStageHandle(
            handle=handle,
            effective_loader_route="fixture",
            metadata={"weights_path": "fixture://dinov3"},
        )

    def close_dinov3(runtime, loaded_handle):
        events.append(("close", runtime.stage_runtime.run_id, runtime.handle_id, loaded_handle is handle))

    report_path = tmp_path / "runner-handle-report.json"
    context_factory, context_closer = build_stage_loader_context(
        [
            StageHandleLoaderRequest(
                handle_id="dinov3",
                kind="model",
                requested_loader_route="fixture",
                factory=load_dinov3,
                close=close_dinov3,
                metadata={"model_family": "dinov3"},
            )
        ],
        report_path=report_path,
        run_id="runner-probe",
    )

    handler_calls = []

    def image_conditioning(invocation, state, context):
        handler_calls.append(
            (
                invocation.job_id,
                context.require_handle("dinov3") is handle,
                context.handle_metadata["dinov3"]["effective_loader_route"],
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
    assert result.context_closed is True
    assert handler_calls == [("seed-101", True, "fixture")]
    assert events == [
        ("load", "runner-probe", "dinov3"),
        ("close", "runner-probe", "dinov3", True),
    ]

    persisted = json.loads(report_path.read_text())
    assert persisted["ok"] is True
    assert persisted["run_id"] == "runner-probe"
    assert persisted["requested_handle_ids"] == ["dinov3"]
    assert persisted["loaded_handle_ids"] == ["dinov3"]
    assert persisted["load_reports"][0]["metadata"]["effective_loader_route"] == "fixture"
    assert persisted["close_reports"][0]["close_phase"] == "closed"


def test_loader_context_factory_writes_load_failure_report_before_stage_handlers(tmp_path):
    from trellmlx.interleaved_generation import GenerationStageResult, StageHandleLoadError, StageRunner
    from trellmlx.stage_handle_loader import StageHandleLoaderRequest, build_stage_loader_context

    plan = _single_job_plan(tmp_path)
    handler_calls = []

    def fail_load(runtime):
        raise RuntimeError("fixture loader exploded")

    report_path = tmp_path / "runner-handle-report.json"
    context_factory, context_closer = build_stage_loader_context(
        [
            StageHandleLoaderRequest(
                handle_id="dinov3",
                kind="model",
                requested_loader_route="fixture",
                factory=fail_load,
            )
        ],
        report_path=report_path,
        run_id="runner-probe",
    )

    runner = StageRunner(
        plan,
        handlers={
            "image_conditioning": lambda invocation, state, context: handler_calls.append(invocation.job_id)
            or GenerationStageResult(invocation.stage, elapsed_seconds=0.0)
        },
        context_factory=context_factory,
        context_closer=context_closer,
    )

    with pytest.raises(StageHandleLoadError, match="failed to load stage handle dinov3"):
        runner.run()

    assert handler_calls == []
    persisted = json.loads(report_path.read_text())
    assert persisted["ok"] is False
    assert persisted["failure_phase"] == "load"
    assert persisted["requested_handle_ids"] == ["dinov3"]
    assert persisted["loaded_handle_ids"] == []
    assert persisted["load_reports"][0]["handle_id"] == "dinov3"
    assert persisted["load_reports"][0]["load_phase"] == "load_error"
    assert "fixture loader exploded" in persisted["error"]


def test_loader_context_closer_writes_close_failure_report(tmp_path):
    from trellmlx.interleaved_generation import GenerationStageResult, StageHandleCloseError, StageRunner
    from trellmlx.stage_handle_loader import (
        LoadedStageHandle,
        StageHandleLoaderRequest,
        build_stage_loader_context,
    )

    plan = _single_job_plan(tmp_path)
    handle = object()

    def load_dinov3(runtime):
        return LoadedStageHandle(handle=handle, effective_loader_route="fixture")

    def close_dinov3(runtime, loaded_handle):
        raise RuntimeError("fixture close exploded")

    report_path = tmp_path / "runner-handle-report.json"
    context_factory, context_closer = build_stage_loader_context(
        [
            StageHandleLoaderRequest(
                handle_id="dinov3",
                kind="model",
                requested_loader_route="fixture",
                factory=load_dinov3,
                close=close_dinov3,
            )
        ],
        report_path=report_path,
        run_id="runner-probe",
    )

    runner = StageRunner(
        plan,
        handlers={
            "image_conditioning": lambda invocation, state, context: GenerationStageResult(
                invocation.stage,
                elapsed_seconds=0.0,
            )
        },
        context_factory=context_factory,
        context_closer=context_closer,
    )

    with pytest.raises(StageHandleCloseError, match="failed to close stage handle dinov3"):
        runner.run()

    persisted = json.loads(report_path.read_text())
    assert persisted["ok"] is False
    assert persisted["failure_phase"] == "close"
    assert persisted["loaded_handle_ids"] == ["dinov3"]
    assert persisted["close_reports"][0]["close_phase"] == "close_error"
    assert "fixture close exploded" in persisted["error"]

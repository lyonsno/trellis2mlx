import json


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
    assert report.close_reports[0].metadata == {
        "kind": "model",
        "close_phase": "closed",
        "requested_loader_route": "fixture",
        "model_family": "dinov3",
    }

    persisted = json.loads(report_path.read_text())
    assert persisted["schema"] == "trellis2mlx.stage_handle_loader_report.v1"
    assert persisted["ok"] is True
    assert persisted["loaded_handle_ids"] == ["dinov3"]
    assert persisted["load_reports"][0]["metadata"]["effective_loader_route"] == "fixture"


def test_load_stage_handles_writes_failure_report_on_loader_route_mismatch(tmp_path):
    from trellmlx.stage_handle_loader import (
        LoadedStageHandle,
        StageHandleLoaderRequest,
        load_stage_handles,
    )

    plan = _single_job_plan(tmp_path)

    def load_with_fallback(runtime):
        return LoadedStageHandle(
            handle=object(),
            effective_loader_route="fallback",
            metadata={"weights_path": "fallback://dinov3"},
        )

    report_path = tmp_path / "handle-report.json"
    report = load_stage_handles(
        plan,
        [
            StageHandleLoaderRequest(
                handle_id="dinov3",
                kind="model",
                requested_loader_route="mlx",
                factory=load_with_fallback,
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
    assert report.close_reports == ()

    persisted = json.loads(report_path.read_text())
    assert persisted["ok"] is False
    assert persisted["failure_phase"] == "load"
    assert "effective loader route mismatch for dinov3" in persisted["error"]
    assert persisted["load_reports"][0]["metadata"] == {
        "kind": "model",
        "load_phase": "load_error",
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

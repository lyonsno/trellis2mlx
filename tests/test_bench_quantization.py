"""Quantization benchmark harness contracts."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path("scripts/bench_quantization.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("bench_quantization", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_variants_maps_labels_to_bits():
    bench = _load_module()

    assert bench.parse_variants("fp16,int8,int4") == [("fp16", 0), ("int8", 8), ("int4", 4)]
    assert bench.parse_variants("int4") == [("int4", 4)]

    with pytest.raises(ValueError, match="unknown quantization variant"):
        bench.parse_variants("fp16,q4")


def test_initial_report_records_route_identity(tmp_path):
    bench = _load_module()

    asset = tmp_path / "image.png"
    asset.write_bytes(b"png")
    args = bench.parse_args([
        "--image",
        str(asset),
        "--report",
        str(tmp_path / "report.json"),
        "--checkpoint-root",
        str(tmp_path / "ckpts"),
        "--variants",
        "fp16,int4",
        "--stages",
        "ss-flow",
        "--slat-tokens",
        "128",
        "--warmup-steps",
        "1",
        "--timed-steps",
        "2",
    ])

    report = bench.initial_report(args, command_line=["bench", "--image", str(asset)])

    assert report["schema"] == "trellis2mlx.quant_bench.v1"
    assert report["status"] == "running"
    assert report["asset"]["path"] == str(asset)
    assert report["asset"]["exists"] is True
    assert report["settings"]["variants"] == ["fp16", "int4"]
    assert report["settings"]["variant_bits"] == {"fp16": 0, "int4": 4}
    assert report["settings"]["stages"] == ["ss-flow"]
    assert report["settings"]["slat_tokens"] == 128
    assert report["command_line"] == ["bench", "--image", str(asset)]
    assert report["checkpoints"]["root"] == str(tmp_path / "ckpts")
    assert report["checkpoints"]["files"]["ss-flow"]["path"].endswith(
        "ss_flow_img_dit_1_3B_64_bf16.safetensors"
    )
    assert report["checkpoints"]["files"]["ss-flow"]["exists"] is False
    assert "repo" in report
    assert "host" in report


def test_failure_report_is_written_before_primary_output(tmp_path):
    bench = _load_module()

    report = {
        "schema": "trellis2mlx.quant_bench.v1",
        "status": "running",
        "results": [],
    }
    report_path = tmp_path / "report.json"

    bench.mark_failure(
        report,
        report_path,
        phase="load_model",
        error="missing checkpoint",
        extra={"stage": "ss-flow", "variant": "int4"},
    )

    persisted = json.loads(report_path.read_text())
    assert persisted["status"] == "failed"
    assert persisted["phase"] == "load_model"
    assert persisted["error"] == "missing checkpoint"
    assert persisted["last_trustworthy_evidence"]["report_existed_before_write"] is False
    assert persisted["stage"] == "ss-flow"
    assert persisted["variant"] == "int4"

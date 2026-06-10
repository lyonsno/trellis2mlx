"""Pixal3D route-control harness contracts."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path("generate_pixal3d.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_pixal3d", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_route_report_proves_projected_conditioning(tmp_path):
    gen = _load_module()

    args = gen.parse_args([
        "--smoke-route",
        "--report",
        str(tmp_path / "route.json"),
        "--grid-resolution",
        "2",
        "--image-size",
        "32",
        "--patch-size",
        "16",
        "--context-channels",
        "8",
    ])

    report = gen.run_smoke_route(args, command_line=["generate_pixal3d.py", "--smoke-route"])

    assert report["schema"] == "trellis2mlx.pixal3d_route.v1"
    assert report["status"] == "ok"
    assert report["route"]["requested"] == "pixal3d-proj"
    assert report["route"]["effective"] == "pixal3d-proj"
    assert report["route"]["pixal3d_projected_conditioning"] is True
    assert report["route"]["context_keys"] == ["global", "proj"]
    assert report["route"]["model_classes"]["ss_flow"] == "Pixal3DSparseStructureFlowModel"
    assert report["route"]["model_classes"]["slat_flow"] == "Pixal3DSLatFlowModel"
    assert report["route"]["projected_shape"] == [1, 8, 8]
    assert report["smoke"]["ss_flow_output_shape"] == [1, 8, 2, 2, 2]
    assert report["smoke"]["slat_flow_output_shape"] == [8, 32]
    assert report["route"]["fallback_detected"] is False


def test_route_validation_rejects_global_only_fallback():
    gen = _load_module()

    route = {
        "requested": "pixal3d-proj",
        "effective": "trellis2mlx-global",
        "context_keys": ["global"],
        "model_classes": {
            "ss_flow": "SparseStructureFlowModel",
            "slat_flow": "SLatFlowModel",
        },
        "projected_shape": None,
    }

    with pytest.raises(RuntimeError, match="projected conditioning"):
        gen.validate_effective_route(route)


def test_cli_smoke_writes_report(tmp_path):
    gen = _load_module()

    report_path = tmp_path / "route.json"
    status = gen.main([
        "--smoke-route",
        "--report",
        str(report_path),
        "--grid-resolution",
        "2",
        "--image-size",
        "32",
        "--patch-size",
        "16",
        "--context-channels",
        "8",
    ])

    persisted = json.loads(report_path.read_text())
    assert status == 0
    assert persisted["status"] == "ok"
    assert persisted["route"]["effective"] == "pixal3d-proj"
    assert persisted["last_trustworthy_evidence"]["phase"] == "smoke_route"


def test_existing_report_requires_explicit_overwrite(tmp_path):
    gen = _load_module()

    report_path = tmp_path / "route.json"
    report_path.write_text("{}\n")

    with pytest.raises(FileExistsError, match="already exists"):
        gen.ensure_report_writable(report_path, overwrite=False)

    gen.ensure_report_writable(report_path, overwrite=True)

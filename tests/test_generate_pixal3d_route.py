"""Pixal3D route-control harness contracts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


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


def test_checkpoint_inventory_counts_projection_and_remapped_keys(tmp_path):
    gen = _load_module()

    checkpoint = tmp_path / "pixal-mini.safetensors"
    save_file(
        {
            "blocks.0.cross_attn.to_q.weight": np.zeros((64, 64), dtype=np.float32),
            "blocks.0.cross_attn.proj_linear.weight": np.zeros((64, 8), dtype=np.float32),
            "blocks.0.cross_attn.proj_linear.bias": np.zeros((64,), dtype=np.float32),
            "blocks.0.cross_attn.to_out.bias": np.zeros((64,), dtype=np.float32),
            "unrelated.weight": np.zeros((2, 2), dtype=np.float32),
        },
        checkpoint,
    )
    args = gen.parse_args([
        "--smoke-route",
        "--checkpoint",
        str(checkpoint),
        "--grid-resolution",
        "2",
        "--image-size",
        "32",
        "--patch-size",
        "16",
        "--context-channels",
        "8",
    ])

    models = gen.build_smoke_models(args)
    inventory = gen.inspect_checkpoint(checkpoint, models)

    assert inventory["projection_keys_present"] is True
    assert inventory["projection_key_count"] == 2
    assert inventory["plain_cross_attn_key_count"] == 2
    assert inventory["plain_cross_attn_remapped_count"] == 2
    assert inventory["wrapped_cross_attn_key_count"] == 0
    assert inventory["matched_by_name"] == 4
    assert inventory["matched_by_shape"] == 4
    assert inventory["shape_mismatch_count"] == 0
    assert inventory["skipped"] == 1


def test_checkpoint_without_projection_keys_fails_before_ok_report(tmp_path):
    gen = _load_module()

    checkpoint = tmp_path / "vanilla.safetensors"
    report_path = tmp_path / "route.json"
    save_file(
        {
            "blocks.0.cross_attn.to_q.weight": np.zeros((64, 64), dtype=np.float32),
            "blocks.0.cross_attn.to_out.bias": np.zeros((64,), dtype=np.float32),
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="projection keys"):
        gen.main([
            "--smoke-route",
            "--checkpoint",
            str(checkpoint),
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
    assert persisted["status"] == "failed"
    assert persisted["phase"] == "checkpoint_inventory"
    assert persisted["checkpoint"]["projection_keys_present"] is False
    assert persisted["last_trustworthy_evidence"]["phase"] == "checkpoint_inventory"


def test_existing_report_requires_explicit_overwrite(tmp_path):
    gen = _load_module()

    report_path = tmp_path / "route.json"
    report_path.write_text("{}\n")

    with pytest.raises(FileExistsError, match="already exists"):
        gen.ensure_report_writable(report_path, overwrite=False)

    gen.ensure_report_writable(report_path, overwrite=True)

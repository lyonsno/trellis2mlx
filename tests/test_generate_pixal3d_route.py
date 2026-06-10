"""Pixal3D route-control harness contracts."""

import importlib.util
import json
from pathlib import Path

import mlx.nn as nn
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file


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


def test_architecture_profile_matches_full_size_checkpoint_shapes(tmp_path):
    gen = _load_module()

    config = tmp_path / "ss_flow.json"
    config.write_text(json.dumps({
        "name": "SparseStructureFlowModel",
        "args": {
            "resolution": 16,
            "in_channels": 8,
            "out_channels": 8,
            "model_channels": 1536,
            "cond_channels": 1024,
            "num_blocks": 30,
            "num_heads": 12,
            "mlp_ratio": 5.3334,
            "image_attn_mode": "proj",
        },
    }))
    checkpoint = tmp_path / "pixal-profile.safetensors"
    save_file(
        {
            "blocks.0.cross_attn.cross_attn_block.to_q.weight": np.zeros((1536, 1536), dtype=np.float32),
            "blocks.0.cross_attn.cross_attn_block.to_kv.weight": np.zeros((3072, 1024), dtype=np.float32),
            "blocks.0.cross_attn.proj_linear.weight": np.zeros((1536, 1024), dtype=np.float32),
            "blocks.0.self_attn.to_qkv.weight": np.zeros((4608, 1536), dtype=np.float32),
            "input_layer.weight": np.zeros((1536, 8), dtype=np.float32),
            "out_layer.weight": np.zeros((8, 1536), dtype=np.float32),
        },
        checkpoint,
    )

    profile = gen.profile_checkpoint_architecture(checkpoint, config)

    assert profile["mode"] == "config-header-no-allocation"
    assert profile["config"]["name"] == "SparseStructureFlowModel"
    assert profile["config"]["model_channels"] == 1536
    assert profile["config"]["cond_channels"] == 1024
    assert profile["config"]["proj_in_channels"] == 1024
    assert profile["expected_shape_count"] >= 690
    assert profile["matched_shape_count"] == 6
    assert profile["shape_mismatch_count"] == 0
    assert profile["projection_shape_match_count"] == 1
    assert profile["wrapped_cross_attn_shape_match_count"] == 2
    assert profile["allocates_model"] is False


def test_architecture_profile_reports_shape_mismatches(tmp_path):
    gen = _load_module()

    config = tmp_path / "slat_flow.json"
    config.write_text(json.dumps({
        "name": "ElasticSLatFlowModel",
        "args": {
            "resolution": 32,
            "in_channels": 32,
            "out_channels": 32,
            "model_channels": 1536,
            "cond_channels": 1024,
            "num_blocks": 30,
            "num_heads": 12,
            "mlp_ratio": 5.3334,
            "image_attn_mode": "proj",
            "proj_in_channels": 2048,
        },
    }))
    checkpoint = tmp_path / "bad-profile.safetensors"
    save_file(
        {
            "blocks.0.cross_attn.proj_linear.weight": np.zeros((1536, 1024), dtype=np.float32),
        },
        checkpoint,
    )

    profile = gen.profile_checkpoint_architecture(checkpoint, config)

    assert profile["projection_shape_match_count"] == 0
    assert profile["shape_mismatch_count"] == 1
    assert profile["shape_mismatch_samples"][0]["expected_shape"] == [1536, 2048]
    assert profile["shape_mismatch_samples"][0]["checkpoint_shape"] == [1536, 1024]


def test_sparse_structure_quantized_assignment_loads_sentinel_then_quantizes(tmp_path):
    gen = _load_module()

    checkpoint = tmp_path / "ss-quant.safetensors"
    sentinel = (np.arange(192 * 64, dtype=np.float32).reshape(192, 64) / 1000.0)
    save_file(
        {
            "blocks.0.self_attn.to_qkv.weight": sentinel,
            "blocks.0.self_attn.to_qkv.bias": np.zeros((192,), dtype=np.float32),
            "blocks.0.cross_attn.proj_linear.weight": np.ones((64, 8), dtype=np.float32),
            "blocks.0.cross_attn.proj_linear.bias": np.zeros((64,), dtype=np.float32),
        },
        checkpoint,
    )

    report = gen.quantized_sparse_structure_assignment_smoke(
        checkpoint,
        bits=4,
        group_size=64,
        grid_resolution=2,
        context_channels=8,
        sentinel_key="blocks.0.self_attn.to_qkv.weight",
        sentinel_expected=sentinel,
    )

    assert report["stage"] == "sparse-structure"
    assert report["quantization"]["requested_bits"] == 4
    assert report["quantization"]["effective_bits"] == 4
    assert report["quantization"]["order"] == "load_fp_then_quantize_in_memory"
    assert report["assignment"]["sentinel_key"] == "blocks.0.self_attn.to_qkv.weight"
    assert report["assignment"]["sentinel_assigned_before_quantize"] is True
    assert report["assignment"]["matched_by_shape"] == 4
    assert "blocks.0.self_attn.to_qkv" in report["quantization"]["quantized_module_names"]
    assert report["quantization"]["quantized_module_count"] > 0
    assert report["quantization"]["packed_weight_dtypes"]["blocks.0.self_attn.to_qkv"] == "uint32"
    assert report["smoke"]["ss_flow_output_shape"] == [1, 8, 2, 2, 2]


def test_sparse_structure_quantized_assignment_rejects_missing_sentinel(tmp_path):
    gen = _load_module()

    checkpoint = tmp_path / "ss-missing.safetensors"
    save_file(
        {
            "blocks.0.cross_attn.proj_linear.weight": np.ones((64, 8), dtype=np.float32),
            "blocks.0.cross_attn.proj_linear.bias": np.zeros((64,), dtype=np.float32),
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="sentinel"):
        gen.quantized_sparse_structure_assignment_smoke(
            checkpoint,
            bits=4,
            group_size=64,
            grid_resolution=2,
            context_channels=8,
            sentinel_key="blocks.0.self_attn.to_qkv.weight",
            sentinel_expected=np.zeros((192, 64), dtype=np.float32),
        )


def test_chunked_sparse_structure_quantization_reports_real_packed_tensors(tmp_path):
    gen = _load_module()

    config = tmp_path / "ss-flow.json"
    config.write_text(json.dumps({
        "name": "SparseStructureFlowModel",
        "args": {
            "resolution": 16,
            "in_channels": 8,
            "out_channels": 8,
            "model_channels": 128,
            "cond_channels": 64,
            "num_blocks": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "image_attn_mode": "proj",
            "proj_in_channels": 64,
        },
    }))
    checkpoint = tmp_path / "ss-chunked.safetensors"
    save_file(
        {
            "blocks.0.self_attn.to_qkv.weight": np.arange(384 * 128, dtype=np.float32).reshape(384, 128),
            "input_layer.weight": np.ones((128, 8), dtype=np.float32),
            "out_layer.weight": np.ones((8, 128), dtype=np.float32),
            "blocks.0.self_attn.to_qkv.bias": np.zeros((384,), dtype=np.float32),
            "rope_phases": np.zeros((4, 2), dtype=np.float32),
        },
        checkpoint,
    )

    report = gen.chunked_quantize_sparse_structure_checkpoint(
        checkpoint,
        config,
        bits=4,
        group_size=64,
    )

    assert report["mode"] == "chunked-header-guarded-mx-quantize"
    assert report["allocates_full_model"] is False
    assert report["loads_all_tensors_at_once"] is False
    assert report["profile"]["shape_mismatch_count"] == 0
    assert report["quantized_weight_count"] == 1
    assert report["eligible_weight_count"] == 1
    assert report["skipped_weight_count"] == 2
    assert report["nonweight_key_count"] == 1
    assert report["extra_checkpoint_key_count"] == 1
    assert report["original_weight_bytes"] == 384 * 128 * 4
    assert report["packed_weight_bytes"] == 384 * 16 * 4
    assert report["scale_bytes"] == 384 * 2 * 4
    assert report["biases_bytes"] == 384 * 2 * 4
    assert report["quantized_payload_bytes"] == (384 * 16 * 4) + (384 * 2 * 4) + (384 * 2 * 4)
    assert report["peak_materialized_tensor_bytes"] == 384 * 128 * 4
    assert report["sample_quantized_weights"][0]["key"] == "blocks.0.self_attn.to_qkv.weight"
    assert report["sample_quantized_weights"][0]["packed_shape"] == [384, 16]
    assert report["sample_quantized_weights"][0]["scale_shape"] == [384, 2]
    assert report["sample_quantized_weights"][0]["packed_dtype"] == "uint32"
    assert report["sample_skipped_weights"][0]["key"] == "input_layer.weight"
    assert report["sample_skipped_weights"][0]["reason"] == "below_group_size"
    assert {item["key"] for item in report["sample_skipped_weights"]} == {"input_layer.weight", "out_layer.weight"}


def test_packed_sparse_structure_artifact_exports_and_loads_effective_runtime_weights(tmp_path):
    gen = _load_module()

    config = tmp_path / "ss-flow.json"
    config.write_text(json.dumps({
        "name": "SparseStructureFlowModel",
        "args": {
            "resolution": 16,
            "in_channels": 8,
            "out_channels": 8,
            "model_channels": 128,
            "cond_channels": 64,
            "num_blocks": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "image_attn_mode": "proj",
            "proj_in_channels": 64,
        },
    }))
    checkpoint = tmp_path / "ss-export.safetensors"
    save_file(
        {
            "blocks.0.self_attn.to_qkv.weight": np.arange(384 * 128, dtype=np.float32).reshape(384, 128),
            "blocks.0.self_attn.to_qkv.bias": np.arange(384, dtype=np.float32),
            "input_layer.weight": np.ones((128, 8), dtype=np.float32),
            "out_layer.weight": np.ones((8, 128), dtype=np.float32),
            "rope_phases": np.zeros((4, 2), dtype=np.float32),
        },
        checkpoint,
    )
    artifact_dir = tmp_path / "packed-ss"

    export = gen.export_packed_sparse_structure_artifact(
        checkpoint,
        config,
        artifact_dir,
        bits=4,
        group_size=64,
    )
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    tensors = load_file(artifact_dir / "model.safetensors")

    assert export["schema"] == "trellis2mlx.pixal3d_sparse_quant_artifact.v1"
    assert export["artifact"]["manifest"] == str(artifact_dir / "manifest.json")
    assert export["artifact"]["tensor_file"] == str(artifact_dir / "model.safetensors")
    assert manifest["quantization"]["bits"] == 4
    assert manifest["quantization"]["group_size"] == 64
    assert manifest["quantized_weight_count"] == 1
    assert "blocks.0.self_attn.to_qkv" in manifest["quantized_module_names"]
    assert tensors["blocks.0.self_attn.to_qkv.weight"].dtype == np.uint32
    assert tensors["blocks.0.self_attn.to_qkv.weight"].shape == (384, 16)
    assert tensors["blocks.0.self_attn.to_qkv.scales"].shape == (384, 2)
    assert tensors["blocks.0.self_attn.to_qkv.biases"].shape == (384, 2)
    assert tensors["blocks.0.self_attn.to_qkv.bias"].shape == (384,)
    assert tensors["input_layer.weight"].shape == (128, 8)
    assert tensors["out_layer.weight"].shape == (8, 128)
    assert "rope_phases" not in tensors
    assert "out_layer" not in manifest["quantized_module_names"]

    model = gen.sparse_structure_model_from_config(config)
    load = gen.load_packed_sparse_structure_artifact(model, artifact_dir)
    module = model.blocks[0].self_attn.to_qkv

    assert load["effective_route"] == "packed-quantized-sparse-structure"
    assert load["loaded_quantized_module_count"] == 1
    assert isinstance(module, nn.QuantizedLinear)
    assert str(module.weight.dtype).replace("mlx.core.", "") == "uint32"
    assert np.array_equal(np.array(module.weight), tensors["blocks.0.self_attn.to_qkv.weight"])


def test_cli_packed_sparse_structure_export_marks_last_evidence_phase(tmp_path):
    gen = _load_module()

    config = tmp_path / "ss-flow-cli.json"
    config.write_text(json.dumps({
        "name": "SparseStructureFlowModel",
        "args": {
            "resolution": 16,
            "in_channels": 8,
            "out_channels": 8,
            "model_channels": 64,
            "cond_channels": 64,
            "num_blocks": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "image_attn_mode": "proj",
            "proj_in_channels": 64,
        },
    }))
    checkpoint = tmp_path / "ss-cli.safetensors"
    save_file(
        {
            "blocks.0.self_attn.to_qkv.weight": np.ones((192, 64), dtype=np.float32),
            "blocks.0.cross_attn.proj_linear.weight": np.ones((64, 64), dtype=np.float32),
            "blocks.0.cross_attn.proj_linear.bias": np.zeros((64,), dtype=np.float32),
        },
        checkpoint,
    )
    report_path = tmp_path / "route.json"
    artifact_dir = tmp_path / "packed-cli"

    status = gen.main([
        "--smoke-route",
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-config",
        str(config),
        "--export-packed-sparse-structure",
        str(artifact_dir),
        "--report",
        str(report_path),
        "--grid-resolution",
        "2",
        "--image-size",
        "32",
        "--patch-size",
        "16",
        "--context-channels",
        "64",
    ])

    persisted = json.loads(report_path.read_text())
    assert status == 0
    assert persisted["status"] == "ok"
    assert persisted["packed_artifact_export"]["quantized_weight_count"] == 2
    assert persisted["last_trustworthy_evidence"]["phase"] == "packed_artifact_export"


def test_packed_sparse_structure_stage_smoke_routes_sampler_through_artifact(tmp_path):
    gen = _load_module()

    config = tmp_path / "ss-stage.json"
    config.write_text(json.dumps({
        "name": "SparseStructureFlowModel",
        "args": {
            "resolution": 2,
            "in_channels": 8,
            "out_channels": 8,
            "model_channels": 128,
            "cond_channels": 64,
            "num_blocks": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "image_attn_mode": "proj",
            "proj_in_channels": 64,
        },
    }))
    checkpoint = tmp_path / "ss-stage.safetensors"
    save_file(
        {
            "blocks.0.self_attn.to_qkv.weight": np.ones((384, 128), dtype=np.float32),
            "blocks.0.cross_attn.proj_linear.weight": np.ones((128, 64), dtype=np.float32),
            "blocks.0.cross_attn.proj_linear.bias": np.zeros((128,), dtype=np.float32),
            "input_layer.weight": np.ones((128, 8), dtype=np.float32),
            "out_layer.weight": np.ones((8, 128), dtype=np.float32),
        },
        checkpoint,
    )
    artifact_dir = tmp_path / "packed-stage"
    gen.export_packed_sparse_structure_artifact(
        checkpoint,
        config,
        artifact_dir,
        bits=4,
        group_size=64,
    )

    report = gen.run_packed_sparse_structure_stage_smoke(
        artifact_dir,
        config,
        steps=1,
        seed=7,
    )

    assert report["stage"] == "sparse-structure"
    assert report["route"]["requested"] == "packed-quantized-sparse-structure"
    assert report["route"]["effective"] == "packed-quantized-sparse-structure"
    assert report["route"]["fp_checkpoint_loaded"] is False
    assert report["load"]["loaded_quantized_module_count"] == 2
    assert report["sampler"]["steps"] == 1
    assert report["sampler"]["guidance_strength"] == 1.0
    assert report["inputs"]["context_proj_shape"] == [1, 8, 64]
    assert report["output"]["shape"] == [1, 8, 2, 2, 2]
    assert report["sample_modules"]["blocks.0.cross_attn.proj_linear"]["class"] == "QuantizedLinear"
    assert report["sample_modules"]["blocks.0.cross_attn.proj_linear"]["weight_dtype"] == "uint32"
    assert report["sample_modules"]["out_layer"]["class"] == "Linear"


def test_packed_sparse_structure_stage_smoke_rejects_artifact_without_quantized_modules(tmp_path):
    gen = _load_module()
    from safetensors.numpy import save_file as save_safetensors

    config = tmp_path / "ss-stage-empty.json"
    config.write_text(json.dumps({
        "name": "SparseStructureFlowModel",
        "args": {
            "resolution": 2,
            "in_channels": 8,
            "out_channels": 8,
            "model_channels": 64,
            "cond_channels": 64,
            "num_blocks": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "image_attn_mode": "proj",
            "proj_in_channels": 64,
        },
    }))
    artifact_dir = tmp_path / "bad-packed-stage"
    artifact_dir.mkdir()
    save_safetensors({"input_layer.weight": np.ones((64, 8), dtype=np.float32)}, artifact_dir / "model.safetensors")
    (artifact_dir / "manifest.json").write_text(json.dumps({
        "schema": "trellis2mlx.pixal3d_sparse_quant_artifact.v1",
        "stage": "sparse-structure",
        "artifact": {"tensor_file": str(artifact_dir / "model.safetensors")},
        "quantization": {"bits": 4, "group_size": 64},
        "quantized_module_names": [],
    }))

    with pytest.raises(RuntimeError, match="no quantized module names"):
        gen.run_packed_sparse_structure_stage_smoke(
            artifact_dir,
            config,
            steps=1,
            seed=7,
        )


def test_existing_report_requires_explicit_overwrite(tmp_path):
    gen = _load_module()

    report_path = tmp_path / "route.json"
    report_path.write_text("{}\n")

    with pytest.raises(FileExistsError, match="already exists"):
        gen.ensure_report_writable(report_path, overwrite=False)

    gen.ensure_report_writable(report_path, overwrite=True)

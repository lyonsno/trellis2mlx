import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.decoder_level1_trace_contract import (
    REQUIRED_ARRAYS,
    decoder_level1_trace_input_sha256,
    load_decoder_level1_trace,
    validate_decoder_level1_trace,
    write_decoder_level1_trace_npz,
)

LEVEL1_HASH_BOUNDARY_NAMES = tuple(
    f"level1_block{index}_output" for index in range(16)
) + (
    "level1_upsample_subdiv_logits",
    "level1_upsample_norm1",
    "level1_upsample_silu1",
    "level1_upsample_conv1",
    "level2_child_coords",
    "level1_upsample_h_c2s",
    "level1_upsample_skip_c2s",
    "level1_upsample_skip_repeated",
    "level1_upsample_norm2",
    "level1_upsample_silu2",
    "level1_upsample_conv2",
    "level1_upsample_output",
) + tuple(
    f"level2_block{index}_output" for index in range(8)
) + (
    "level2_upsample_subdiv_logits",
    "level2_upsample_norm1",
    "level2_upsample_silu1",
    "level2_upsample_conv1",
    "level3_child_coords",
    "level2_upsample_h_c2s",
    "level2_upsample_skip_c2s",
    "level2_upsample_skip_repeated",
    "level2_upsample_norm2",
    "level2_upsample_silu2",
    "level2_upsample_conv2",
    "level2_upsample_output",
)


def _hash_boundary(name, values):
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(name.encode("ascii") + b"\0")
    digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
    digest.update(
        ",".join(str(value) for value in contiguous.shape).encode("ascii")
        + b"\0"
    )
    digest.update(contiguous.tobytes())
    return {
        "name": name,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": digest.hexdigest(),
    }


def _valid_hash_boundary_arrays(
    child_rows=5,
    level2_rows=7,
    level3_rows=9,
):
    arrays = {
        f"level1_block{index}_output": np.full(
            (child_rows, 512),
            index,
            dtype=np.float16,
        )
        for index in range(16)
    }
    arrays.update(
        {
            "level1_upsample_subdiv_logits": np.zeros(
                (child_rows, 8),
                dtype=np.float16,
            ),
            "level1_upsample_norm1": np.zeros(
                (child_rows, 512),
                dtype=np.float16,
            ),
            "level1_upsample_silu1": np.zeros(
                (child_rows, 512),
                dtype=np.float16,
            ),
            "level1_upsample_conv1": np.zeros(
                (child_rows, 2048),
                dtype=np.float16,
            ),
            "level2_child_coords": np.arange(
                level2_rows * 4,
                dtype=np.int32,
            ).reshape(level2_rows, 4),
            "level1_upsample_h_c2s": np.zeros(
                (level2_rows, 256),
                dtype=np.float16,
            ),
            "level1_upsample_skip_c2s": np.zeros(
                (level2_rows, 64),
                dtype=np.float16,
            ),
            "level1_upsample_skip_repeated": np.zeros(
                (level2_rows, 256),
                dtype=np.float16,
            ),
            "level1_upsample_norm2": np.zeros(
                (level2_rows, 256),
                dtype=np.float16,
            ),
            "level1_upsample_silu2": np.zeros(
                (level2_rows, 256),
                dtype=np.float16,
            ),
            "level1_upsample_conv2": np.zeros(
                (level2_rows, 256),
                dtype=np.float16,
            ),
            "level1_upsample_output": np.zeros(
                (level2_rows, 256),
                dtype=np.float16,
            ),
        }
    )
    arrays.update(
        {
            f"level2_block{index}_output": np.full(
                (level2_rows, 256),
                index,
                dtype=np.float16,
            )
            for index in range(8)
        }
    )
    arrays.update(
        {
            "level2_upsample_subdiv_logits": np.zeros(
                (level2_rows, 8),
                dtype=np.float16,
            ),
            "level2_upsample_norm1": np.zeros(
                (level2_rows, 256),
                dtype=np.float16,
            ),
            "level2_upsample_silu1": np.zeros(
                (level2_rows, 256),
                dtype=np.float16,
            ),
            "level2_upsample_conv1": np.zeros(
                (level2_rows, 1024),
                dtype=np.float16,
            ),
            "level3_child_coords": np.arange(
                level3_rows * 4,
                dtype=np.int32,
            ).reshape(level3_rows, 4),
            "level2_upsample_h_c2s": np.zeros(
                (level3_rows, 128),
                dtype=np.float16,
            ),
            "level2_upsample_skip_c2s": np.zeros(
                (level3_rows, 32),
                dtype=np.float16,
            ),
            "level2_upsample_skip_repeated": np.zeros(
                (level3_rows, 128),
                dtype=np.float16,
            ),
            "level2_upsample_norm2": np.zeros(
                (level3_rows, 128),
                dtype=np.float16,
            ),
            "level2_upsample_silu2": np.zeros(
                (level3_rows, 128),
                dtype=np.float16,
            ),
            "level2_upsample_conv2": np.zeros(
                (level3_rows, 128),
                dtype=np.float16,
            ),
            "level2_upsample_output": np.zeros(
                (level3_rows, 128),
                dtype=np.float16,
            ),
        }
    )
    return arrays


def _valid_hash_ledger():
    arrays = _valid_hash_boundary_arrays()
    return {
        "schema": "trellis2mlx.decoder_level1_hash_ledger.v2",
        "entries": [
            _hash_boundary(name, arrays[name])
            for name in LEVEL1_HASH_BOUNDARY_NAMES
        ],
    }


def _valid_trace():
    parent_coords = np.array(
        [[0, 1, 2, 3], [0, 5, 6, 7]],
        dtype=np.int32,
    )
    parent_rows = parent_coords.shape[0]
    level0_output = np.arange(
        parent_rows * 1024,
        dtype=np.float16,
    ).reshape(parent_rows, 1024)
    subdiv_logits = np.full((parent_rows, 8), -1, dtype=np.float16)
    subdiv_logits[0, [0, 3, 7]] = 1
    subdiv_logits[1, [1, 6]] = 1
    mask = subdiv_logits > 0
    parent_indices, child_indices = np.nonzero(mask)
    child_coords = parent_coords[parent_indices].copy()
    child_coords[:, 1:] *= 2
    child_coords[:, 1] += child_indices % 2
    child_coords[:, 2] += (child_indices // 2) % 2
    child_coords[:, 3] += child_indices // 4
    child_rows = child_coords.shape[0]
    conv1 = np.arange(
        parent_rows * 4096,
        dtype=np.float16,
    ).reshape(parent_rows, 4096)
    h_c2s = conv1.reshape(parent_rows, 8, 512)[mask]
    skip_c2s = level0_output.reshape(parent_rows, 8, 128)[mask]
    skip_repeated = np.repeat(skip_c2s, 4, axis=1)

    arrays = {
        "parent_coords": parent_coords,
        "child_coords": child_coords,
        "level0_output": level0_output,
        "upsample_subdiv_logits": subdiv_logits,
        "upsample_norm1": np.zeros((parent_rows, 1024), dtype=np.float16),
        "upsample_silu1": np.zeros((parent_rows, 1024), dtype=np.float16),
        "upsample_conv1": conv1,
        "upsample_h_c2s": h_c2s,
        "upsample_skip_c2s": skip_c2s,
        "upsample_skip_repeated": skip_repeated,
        "upsample_norm2": np.zeros((child_rows, 512), dtype=np.float16),
        "upsample_silu2": np.zeros((child_rows, 512), dtype=np.float16),
        "upsample_conv2": np.zeros((child_rows, 512), dtype=np.float16),
        "upsample_output": np.zeros((child_rows, 512), dtype=np.float16),
        "level1_block0_conv": np.zeros((child_rows, 512), dtype=np.float16),
        "level1_block0_norm": np.zeros((child_rows, 512), dtype=np.float16),
        "level1_block0_mlp_fc1": np.zeros((child_rows, 2048), dtype=np.float16),
        "level1_block0_silu": np.zeros((child_rows, 2048), dtype=np.float16),
        "level1_block0_mlp_fc2": np.zeros((child_rows, 512), dtype=np.float16),
        "level1_block0_output": np.zeros((child_rows, 512), dtype=np.float16),
    }
    return arrays


def test_level1_trace_contract_round_trips_exactly(tmp_path):
    arrays = _valid_trace()
    output = tmp_path / "trace.npz"

    report = write_decoder_level1_trace_npz(output, arrays)
    reopened = load_decoder_level1_trace(output)

    assert report["reopened_exact"] is True
    assert report["child_expansion_exact"] is True
    assert tuple(reopened) == REQUIRED_ARRAYS
    for name in REQUIRED_ARRAYS:
        np.testing.assert_array_equal(reopened[name], arrays[name])


def test_level1_trace_contract_rejects_child_coordinate_reordering():
    arrays = _valid_trace()
    arrays["child_coords"] = arrays["child_coords"][::-1].copy()

    with pytest.raises(ValueError, match="parent-major"):
        validate_decoder_level1_trace(arrays)


def test_level1_trace_contract_rejects_mislabeled_channel_slice():
    arrays = _valid_trace()
    arrays["upsample_h_c2s"] = arrays["upsample_h_c2s"].copy()
    arrays["upsample_h_c2s"][0, 0] += np.float16(1)

    with pytest.raises(ValueError, match="conv1 channel slices"):
        validate_decoder_level1_trace(arrays)


def test_level1_trace_contract_rejects_wrong_skip_repeat_order():
    arrays = _valid_trace()
    arrays["upsample_skip_repeated"] = np.tile(
        arrays["upsample_skip_c2s"],
        (1, 4),
    )

    with pytest.raises(ValueError, match="source repeat order"):
        validate_decoder_level1_trace(arrays)


def test_level1_trace_contract_rejects_missing_or_extra_arrays():
    missing = _valid_trace()
    missing.pop("level1_block0_silu")
    with pytest.raises(KeyError, match="level1_block0_silu"):
        validate_decoder_level1_trace(missing)

    extra = _valid_trace()
    extra["cached_output"] = extra["level1_block0_output"]
    with pytest.raises(KeyError, match="cached_output"):
        validate_decoder_level1_trace(extra)


def test_level1_trace_input_identity_binds_parent_values_and_coords():
    arrays = _valid_trace()
    baseline = decoder_level1_trace_input_sha256(
        arrays["level0_output"],
        arrays["parent_coords"],
    )
    changed_values = arrays["level0_output"].copy()
    changed_values[0, 0] += np.float16(1)
    changed_coords = arrays["parent_coords"].copy()
    changed_coords[0, 1] += 1

    assert baseline != decoder_level1_trace_input_sha256(
        changed_values,
        arrays["parent_coords"],
    )
    assert baseline != decoder_level1_trace_input_sha256(
        arrays["level0_output"],
        changed_coords,
    )


def test_level1_hash_ledger_contract_rejects_partial_and_binds_boundary_bytes():
    from scripts.decoder_level1_trace_contract import (
        build_decoder_level1_hash_ledger,
        validate_decoder_level1_hash_ledger,
    )

    arrays = _valid_hash_boundary_arrays()
    ledger = build_decoder_level1_hash_ledger(arrays)
    assert validate_decoder_level1_hash_ledger(ledger) == ledger
    assert [entry["name"] for entry in ledger["entries"]] == list(
        LEVEL1_HASH_BOUNDARY_NAMES
    )

    changed = _valid_hash_boundary_arrays()
    changed["level2_block3_output"][0, 0] += np.float16(1)
    changed_ledger = build_decoder_level1_hash_ledger(changed)
    changed_index = LEVEL1_HASH_BOUNDARY_NAMES.index(
        "level2_block3_output"
    )
    assert (
        changed_ledger["entries"][changed_index]["sha256"]
        != ledger["entries"][changed_index]["sha256"]
    )

    partial = {
        **ledger,
        "entries": ledger["entries"][:-1],
    }
    with pytest.raises(ValueError, match="ordered boundaries"):
        validate_decoder_level1_hash_ledger(partial)


def _write_comparison_inputs(tmp_path, source_arrays, local_arrays):
    source_path = tmp_path / "source.npz"
    local_path = tmp_path / "local.npz"
    write_decoder_level1_trace_npz(source_path, source_arrays)
    write_decoder_level1_trace_npz(local_path, local_arrays)
    input_identity = decoder_level1_trace_input_sha256(
        source_arrays["level0_output"],
        source_arrays["parent_coords"],
    )
    parent_path = tmp_path / "level0-parent.npz"
    parent_path.write_bytes(b"level-zero-parent")
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    silu_path = tmp_path / "decoder-silu-lut.npz"
    silu_path.write_bytes(b"authenticated-silu")
    silu_sha = hashlib.sha256(silu_path.read_bytes()).hexdigest()
    validation = {
        "reopened_exact": True,
        "child_expansion_exact": True,
    }
    hash_ledger = _valid_hash_ledger()
    source_report = tmp_path / "source.json"
    source_report.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.source_cuda_shape_slat_grid_decode.v1",
                "status": "done",
                "effective_route": {
                    "route": "official-source-cuda-shape-decoder-level1-trace",
                    "device_type": "cuda",
                    "cuda_device": "Tesla T4",
                    "sparse_conv_backend": "none",
                    "decoder_state_only": False,
                    "decoder_level0_trace": False,
                    "decoder_level1_trace": True,
                    "raw_meshes": False,
                    "post_fill_holes_snapshots": False,
                    "mesh_conversion": False,
                    "one_model_load": True,
                },
                "decoder_trace_artifacts": [
                    {
                        "path": str(source_path),
                        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        "input_tensor_sha256": input_identity,
                        "status": "written",
                        "validation": validation,
                        "hash_ledger": hash_ledger,
                    }
                ],
            }
        )
    )
    local_report = tmp_path / "local.json"
    local_report.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.decoder_level1_trace_run.v1",
                "status": "done",
                "effective_route": {
                    "route": "mlx-shape-decoder-level1-trace",
                    "device_type": "metal",
                    "device": "Device(gpu, 0)",
                    "decoder_linear_backend": "turing_fda",
                    "sparse_conv_matmul_backend": "turing_fda",
                    "decoder_layernorm": {
                        "backend": "mlx-fast-layer-norm",
                        "algorithm": "mlx-fast-layer-norm",
                        "experimental": False,
                    },
                    "decoder_silu": {
                        "backend": "cuda-turing-t4-fp16-lut",
                        "algorithm": "exhaustive-fp16-bit-pattern-output-lookup",
                        "experimental": True,
                        "cuda_architecture": "sm_75",
                        "cuda_device_anchor": "Tesla T4",
                        "cuda_source_operation": "torch.nn.functional.silu",
                        "cuda_source_version": "torch-2.10.0+cu128",
                        "authenticated_contract": {
                            "input_dtype": "float16",
                            "output_dtype": "float16",
                            "domain": "all-65536-bit-patterns",
                        },
                        "output_lut_artifact_path": str(silu_path),
                        "output_lut_artifact_sha256_attested": silu_sha,
                        "output_lut_artifact_sha256_effective": silu_sha,
                    },
                    "parent_state": {
                        "path": str(parent_path),
                        "sha256": parent_sha,
                        "input_tensor_sha256": input_identity,
                    },
                },
                "input_tensor_sha256": input_identity,
                "parent_trace": {
                    "path": str(parent_path),
                    "sha256": parent_sha,
                    "input_tensor_sha256": input_identity,
                },
                "primary": {
                    "path": str(local_path),
                    "sha256": hashlib.sha256(local_path.read_bytes()).hexdigest(),
                    "status": "written",
                    "validation": validation,
                    "hash_ledger": hash_ledger,
                },
            }
        )
    )
    return source_path, source_report, local_path, local_report


def _valid_full_decoder_hash_ledger(
    *,
    first_value=0,
    fork_name=None,
):
    from scripts.decoder_full_hash_ledger_contract import (
        FULL_DECODER_HASH_BOUNDARY_NAMES,
        build_decoder_full_hash_ledger,
    )

    level3_rows = 9
    level4_rows = 11
    shapes = {
        "level2_upsample_output": (level3_rows, 128),
        "level3_block0_conv": (level3_rows, 128),
        "level3_block0_norm": (level3_rows, 128),
        "level3_block0_mlp_fc1": (level3_rows, 512),
        "level3_block0_silu": (level3_rows, 512),
        "level3_block0_mlp_fc2": (level3_rows, 128),
        "level3_block0_output": (level3_rows, 128),
        "level3_block1_output": (level3_rows, 128),
        "level3_block2_output": (level3_rows, 128),
        "level3_block3_output": (level3_rows, 128),
        "level3_upsample_subdiv_logits": (level3_rows, 8),
        "level3_upsample_norm1": (level3_rows, 128),
        "level3_upsample_silu1": (level3_rows, 128),
        "level3_upsample_conv1": (level3_rows, 512),
        "level4_child_coords": (level4_rows, 4),
        "level3_upsample_h_c2s": (level4_rows, 64),
        "level3_upsample_skip_c2s": (level4_rows, 16),
        "level3_upsample_skip_repeated": (level4_rows, 64),
        "level3_upsample_norm2": (level4_rows, 64),
        "level3_upsample_silu2": (level4_rows, 64),
        "level3_upsample_conv2": (level4_rows, 64),
        "level3_upsample_output": (level4_rows, 64),
        "decoder_final_layernorm": (level4_rows, 64),
        "decoder_output": (level4_rows, 7),
    }
    boundaries = {}
    for name in FULL_DECODER_HASH_BOUNDARY_NAMES:
        dtype = np.int32 if name == "level4_child_coords" else (
            np.float32
            if name in {"decoder_final_layernorm", "decoder_output"}
            else np.float16
        )
        value = first_value if name == "level2_upsample_output" else 0
        if name == fork_name:
            value = 1
        boundaries[name] = np.full(shapes[name], value, dtype=dtype)
    return build_decoder_full_hash_ledger(boundaries)


def _write_full_decoder_comparison_inputs(
    tmp_path,
    *,
    source_full_ledger=None,
    local_full_ledger=None,
):
    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    source_full_ledger = (
        source_full_ledger or _valid_full_decoder_hash_ledger()
    )
    local_full_ledger = (
        local_full_ledger or _valid_full_decoder_hash_ledger()
    )

    source = json.loads(source_report.read_text())
    source["requested_route"] = {
        "route": "official-source-cuda-shape-decoder-full-hash-ledger",
        "full_decoder_hash_ledger": True,
        "decoder_output_head_backend": "torch-sparse-linear-fp32",
    }
    source["effective_route"]["route"] = (
        "official-source-cuda-shape-decoder-full-hash-ledger"
    )
    source["effective_route"]["full_decoder_hash_ledger"] = True
    source["effective_route"]["decoder_output_head_backend"] = (
        "torch-sparse-linear-fp32"
    )
    source["decoder_trace_artifacts"][0][
        "full_decoder_hash_ledger"
    ] = source_full_ledger
    source_report.write_text(json.dumps(source))

    local = json.loads(local_report.read_text())
    local["requested_route"] = {
        "route": "mlx-shape-decoder-level1-trace",
        "device_type": "metal",
        "decoder_linear_backend": "turing_fda",
        "sparse_conv_matmul_backend": "turing_fda",
        "decoder_layernorm_backend": "cuda-welford-turing-t4",
        "decoder_silu_backend": "cuda-turing-t4-fp16-lut",
        "full_decoder_hash_ledger": True,
        "decoder_output_head_backend": "mlx-native-fp32",
    }
    lut = tmp_path / "full-decoder-turing-rsqrt.npz"
    np.savez_compressed(
        lut,
        normalized_delta=np.zeros((1 << 24,), dtype=np.int8),
    )
    local["effective_route"].update(_exact_layernorm_route(lut))
    local["effective_route"]["full_decoder_hash_ledger"] = True
    local["effective_route"]["decoder_output_head_backend"] = (
        "mlx-native-fp32"
    )
    local["primary"]["full_decoder_hash_ledger"] = local_full_ledger
    local_report.write_text(json.dumps(local))
    return source_path, source_report, local_path, local_report


def test_full_decoder_comparator_names_first_surviving_boundary(tmp_path):
    from scripts.compare_decoder_full_hash_ledgers import (
        compare_decoder_full_hash_reports,
    )

    local_ledger = _valid_full_decoder_hash_ledger(
        fork_name="level3_block0_norm"
    )
    source_path, source_report, local_path, local_report = (
        _write_full_decoder_comparison_inputs(
            tmp_path,
            local_full_ledger=local_ledger,
        )
    )

    comparison = compare_decoder_full_hash_reports(
        source_path=source_path,
        source_report_path=source_report,
        local_path=local_path,
        local_report_path=local_report,
    )

    assert comparison["parent_exact"] is True
    assert comparison["first_nonexact_boundary"] == "level3_block0_norm"
    assert comparison["baseline"]["first_nonexact_hash_boundary"] is None


def test_full_decoder_comparator_rejects_missing_ledger(tmp_path):
    from scripts.compare_decoder_full_hash_ledgers import (
        compare_decoder_full_hash_reports,
    )

    source_path, source_report, local_path, local_report = (
        _write_full_decoder_comparison_inputs(tmp_path)
    )
    report = json.loads(local_report.read_text())
    report["primary"].pop("full_decoder_hash_ledger")
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="full-decoder hash ledger"):
        compare_decoder_full_hash_reports(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_full_decoder_comparator_rejects_terminal_head_route_lie(tmp_path):
    from scripts.compare_decoder_full_hash_ledgers import (
        compare_decoder_full_hash_reports,
    )

    source_path, source_report, local_path, local_report = (
        _write_full_decoder_comparison_inputs(tmp_path)
    )
    report = json.loads(local_report.read_text())
    report["effective_route"]["decoder_output_head_backend"] = "turing_fda"
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="terminal output-head route"):
        compare_decoder_full_hash_reports(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_full_decoder_comparator_rejects_native_layernorm_route(tmp_path):
    from scripts.compare_decoder_full_hash_ledgers import (
        compare_decoder_full_hash_reports,
    )

    source_path, source_report, local_path, local_report = (
        _write_full_decoder_comparison_inputs(tmp_path)
    )
    report = json.loads(local_report.read_text())
    report["requested_route"]["decoder_layernorm_backend"] = (
        "mlx-fast-layer-norm"
    )
    report["effective_route"]["decoder_layernorm"] = {
        "backend": "mlx-fast-layer-norm",
        "algorithm": "mlx-fast-layer-norm",
        "experimental": False,
    }
    report["effective_route"]["decoder_layernorm_lut"] = None
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="full-decoder LayerNorm route"):
        compare_decoder_full_hash_reports(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_full_decoder_comparator_rejects_missing_width128_affine_contract(
    tmp_path,
):
    from scripts.compare_decoder_full_hash_ledgers import (
        compare_decoder_full_hash_reports,
    )

    source_path, source_report, local_path, local_report = (
        _write_full_decoder_comparison_inputs(tmp_path)
    )
    report = json.loads(local_report.read_text())
    contracts = report["effective_route"]["decoder_layernorm"][
        "authenticated_contracts"
    ]
    contracts.pop(5)
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="width-128 affine"):
        compare_decoder_full_hash_reports(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_full_decoder_comparator_rejects_missing_width64_nonaffine_contract(
    tmp_path,
):
    from scripts.compare_decoder_full_hash_ledgers import (
        compare_decoder_full_hash_reports,
    )

    source_path, source_report, local_path, local_report = (
        _write_full_decoder_comparison_inputs(tmp_path)
    )
    report = json.loads(local_report.read_text())
    contracts = report["effective_route"]["decoder_layernorm"][
        "authenticated_contracts"
    ]
    contracts.pop()
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="width-64 non-affine"):
        compare_decoder_full_hash_reports(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_full_decoder_comparator_rejects_detached_self_consistent_parent(
    tmp_path,
):
    from scripts.compare_decoder_full_hash_ledgers import (
        compare_decoder_full_hash_reports,
    )

    detached = _valid_full_decoder_hash_ledger(first_value=1)
    source_path, source_report, local_path, local_report = (
        _write_full_decoder_comparison_inputs(
            tmp_path,
            source_full_ledger=detached,
            local_full_ledger=detached,
        )
    )

    with pytest.raises(ValueError, match="authenticated level-two output"):
        compare_decoder_full_hash_reports(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_full_decoder_comparator_failed_run_replaces_stale_output(tmp_path):
    from scripts.compare_decoder_full_hash_ledgers import main

    source_path, source_report, local_path, local_report = (
        _write_full_decoder_comparison_inputs(tmp_path)
    )
    report = json.loads(local_report.read_text())
    report["effective_route"]["full_decoder_hash_ledger"] = False
    local_report.write_text(json.dumps(report))
    output = tmp_path / "comparison.json"
    output.write_text('{"status":"done","stale":true}\n')

    rc = main(
        [
            "--source",
            str(source_path),
            "--source-report",
            str(source_report),
            "--local",
            str(local_path),
            "--local-report",
            str(local_report),
            "--output",
            str(output),
        ]
    )

    failed = json.loads(output.read_text())
    assert rc == 1
    assert failed["status"] == "failed"
    assert failed["failure_phase"] == "comparison"
    assert "stale" not in failed


def test_full_decoder_comparator_refuses_output_input_collision(tmp_path):
    from scripts.compare_decoder_full_hash_ledgers import main

    source_path, source_report, local_path, local_report = (
        _write_full_decoder_comparison_inputs(tmp_path)
    )
    source_before = source_path.read_bytes()

    rc = main(
        [
            "--source",
            str(source_path),
            "--source-report",
            str(source_report),
            "--local",
            str(local_path),
            "--local-report",
            str(local_report),
            "--output",
            str(source_path),
        ]
    )

    assert rc == 1
    assert source_path.read_bytes() == source_before
    failure = Path(str(source_path) + ".failure.json")
    report = json.loads(failure.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"


def test_level1_comparator_rejects_missing_hash_ledger(tmp_path):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    report = json.loads(local_report.read_text())
    report["primary"].pop("hash_ledger")
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="hash ledger"):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_level1_comparator_names_first_nonexact_hash_boundary(tmp_path):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    report = json.loads(local_report.read_text())
    report["primary"]["hash_ledger"]["entries"][7]["sha256"] = "f" * 64
    local_report.write_text(json.dumps(report))

    comparison = compare_level1_traces(
        source_path=source_path,
        source_report_path=source_report,
        local_path=local_path,
        local_report_path=local_report,
    )

    assert (
        comparison["first_nonexact_hash_boundary"]
        == "level1_block7_output"
    )


def test_level1_comparator_rejects_ledger_detached_from_primary_trace(tmp_path):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    report = json.loads(local_report.read_text())
    report["primary"]["hash_ledger"]["entries"][0]["sha256"] = "f" * 64
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="does not match primary"):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def _exact_layernorm_route(lut):
    with np.load(lut, allow_pickle=False) as archive:
        normalized_delta = np.asarray(archive["normalized_delta"])
    lut_sha = hashlib.sha256(lut.read_bytes()).hexdigest()
    content_sha = hashlib.sha256(
        np.ascontiguousarray(normalized_delta).tobytes()
    ).hexdigest()
    return {
        "decoder_layernorm": {
            "backend": "cuda-welford-turing-t4",
            "algorithm": (
                "pytorch-2.10-vectorized-layernorm-128-thread-welford-"
                "turing-rsqrt-on-metal"
            ),
            "experimental": True,
            "cuda_source_tag": "pytorch-v2.10.0",
            "cuda_source_kernel": "vectorized_layer_norm_kernel",
            "cuda_architecture": "sm_75",
            "cuda_device_anchor": "Tesla T4",
            "cuda_rsqrt_bit_exact_for_configured_lut": True,
            "authenticated_contract": {
                "input_dtype": "float16",
                "parameter_dtype": "float16",
                "hidden_width": 1024,
                "affine": True,
            },
            "authenticated_contracts": [
                {
                    "input_dtype": "float16",
                    "parameter_dtype": "float16",
                    "hidden_width": 1024,
                    "affine": True,
                    "reduction": {
                        "threads": 128,
                        "warps": 4,
                        "vector_width": 4,
                        "values_per_thread": 8,
                        "accumulator_dtype": "float32",
                    },
                },
                {
                    "input_dtype": "float16",
                    "parameter_dtype": "float16",
                    "hidden_width": 512,
                    "affine": True,
                    "reduction": {
                        "threads": 128,
                        "warps": 4,
                        "vector_width": 4,
                        "values_per_thread": 4,
                        "accumulator_dtype": "float32",
                    },
                },
                {
                    "input_dtype": "float16",
                    "hidden_width": 512,
                    "affine": False,
                    "reduction": {
                        "threads": 128,
                        "warps": 4,
                        "vector_width": 4,
                        "values_per_thread": 4,
                        "accumulator_dtype": "float32",
                    },
                },
                {
                    "input_dtype": "float16",
                    "parameter_dtype": "float16",
                    "hidden_width": 256,
                    "affine": True,
                    "reduction": {
                        "threads": 128,
                        "warps": 4,
                        "vector_width": 4,
                        "active_values_per_thread": 4,
                        "average_values_per_launched_thread": 2,
                        "active_vector_threads": 64,
                        "inactive_vector_threads": 64,
                        "accumulator_dtype": "float32",
                    },
                },
                {
                    "input_dtype": "float16",
                    "hidden_width": 256,
                    "affine": False,
                    "reduction": {
                        "threads": 128,
                        "warps": 4,
                        "vector_width": 4,
                        "active_values_per_thread": 4,
                        "average_values_per_launched_thread": 2,
                        "active_vector_threads": 64,
                        "inactive_vector_threads": 64,
                        "accumulator_dtype": "float32",
                    },
                },
                {
                    "input_dtype": "float16",
                    "parameter_dtype": "float16",
                    "hidden_width": 128,
                    "affine": True,
                    "reduction": {
                        "threads": 128,
                        "warps": 4,
                        "vector_width": 4,
                        "active_values_per_thread": 4,
                        "average_values_per_launched_thread": 1,
                        "active_vector_threads": 32,
                        "inactive_vector_threads": 96,
                        "accumulator_dtype": "float32",
                    },
                },
                {
                    "input_dtype": "float16",
                    "hidden_width": 128,
                    "affine": False,
                    "reduction": {
                        "threads": 128,
                        "warps": 4,
                        "vector_width": 4,
                        "active_values_per_thread": 4,
                        "average_values_per_launched_thread": 1,
                        "active_vector_threads": 32,
                        "inactive_vector_threads": 96,
                        "accumulator_dtype": "float32",
                    },
                },
                {
                    "input_dtype": "float16",
                    "hidden_width": 64,
                    "affine": False,
                    "reduction": {
                        "threads": 128,
                        "warps": 4,
                        "vector_width": 4,
                        "active_values_per_thread": 4,
                        "average_values_per_launched_thread": 0.5,
                        "active_vector_threads": 16,
                        "inactive_vector_threads": 112,
                        "accumulator_dtype": "float32",
                    },
                },
                {
                    "input_dtype": "float32",
                    "hidden_width": 64,
                    "affine": False,
                    "eps": 1e-5,
                    "consumer": "shape-decoder-terminal",
                    "reduction": {
                        "threads": 128,
                        "warps": 4,
                        "vector_width": 4,
                        "active_values_per_thread": 4,
                        "average_values_per_launched_thread": 0.5,
                        "active_vector_threads": 16,
                        "inactive_vector_threads": 112,
                        "accumulator_dtype": "float32",
                    },
                },
            ],
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "values_per_thread": 8,
                "accumulator_dtype": "float32",
            },
            "rsqrt": "Turing MUFU.RSQ normalized signed-ULP LUT",
            "turing_rsqrt_lut_artifact_sha256_attested": lut_sha,
            "turing_rsqrt_lut_content_sha256": content_sha,
            "turing_rsqrt_lut_entries": 1 << 24,
        },
        "decoder_layernorm_lut": {
            "path": str(lut),
            "sha256": lut_sha,
            "normalized_delta_sha256": content_sha,
            "entries": 1 << 24,
            "dtype": "int8",
        },
    }


def test_level1_comparator_accepts_file_bound_exact_layernorm_route(tmp_path):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    lut = tmp_path / "turing-rsqrt.npz"
    np.savez_compressed(
        lut,
        normalized_delta=np.zeros((1 << 24,), dtype=np.int8),
    )
    lut_sha = hashlib.sha256(lut.read_bytes()).hexdigest()
    report = json.loads(local_report.read_text())
    report["effective_route"].update(_exact_layernorm_route(lut))
    local_report.write_text(json.dumps(report))

    comparison = compare_level1_traces(
        source_path=source_path,
        source_report_path=source_report,
        local_path=local_path,
        local_report_path=local_report,
    )

    assert comparison["first_nonexact_boundary"] is None
    assert (
        comparison["artifacts"]["local"]["effective_route"][
            "decoder_layernorm_lut"
        ]["sha256"]
        == lut_sha
    )


@pytest.mark.parametrize(
    ("contract_index", "contract_kind"),
    [
        (5, "width-128 affine"),
        (6, "width-128 non-affine"),
        (7, "width-64 non-affine"),
        (8, "terminal float32 width-64 non-affine"),
    ],
)
def test_level1_comparator_rejects_missing_late_layernorm_contract(
    tmp_path,
    contract_index,
    contract_kind,
):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    lut = tmp_path / "turing-rsqrt.npz"
    np.savez_compressed(
        lut,
        normalized_delta=np.zeros((1 << 24,), dtype=np.int8),
    )
    exact_route = _exact_layernorm_route(lut)
    exact_route["decoder_layernorm"]["authenticated_contracts"].pop(
        contract_index
    )
    report = json.loads(local_report.read_text())
    report["effective_route"].update(exact_route)
    local_report.write_text(json.dumps(report))

    with pytest.raises(
        ValueError,
        match="width-64 non-affine contract",
    ):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


@pytest.mark.parametrize(
    ("contract_index", "contract_kind"),
    [(3, "affine"), (4, "non-affine")],
)
def test_level1_comparator_rejects_missing_width256_layernorm_contract(
    tmp_path,
    contract_index,
    contract_kind,
):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    lut = tmp_path / "turing-rsqrt.npz"
    np.savez_compressed(
        lut,
        normalized_delta=np.zeros((1 << 24,), dtype=np.int8),
    )
    exact_route = _exact_layernorm_route(lut)
    exact_route["decoder_layernorm"]["authenticated_contracts"].pop(
        contract_index
    )
    report = json.loads(local_report.read_text())
    report["effective_route"].update(exact_route)
    local_report.write_text(json.dumps(report))

    with pytest.raises(
        ValueError,
        match="authenticated width-256 affine or non-affine contract",
    ):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_level1_comparator_rejects_substituted_exact_layernorm_artifact(tmp_path):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    lut = tmp_path / "turing-rsqrt.npz"
    np.savez_compressed(
        lut,
        normalized_delta=np.zeros((1 << 24,), dtype=np.int8),
    )
    report = json.loads(local_report.read_text())
    report["effective_route"].update(_exact_layernorm_route(lut))
    local_report.write_text(json.dumps(report))
    lut.write_bytes(b"substituted")

    with pytest.raises(
        ValueError,
        match="LayerNorm rsqrt artifact bytes do not match identity",
    ):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_level1_comparator_rejects_truthfully_hashed_malformed_layernorm_artifact(
    tmp_path,
):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    lut = tmp_path / "turing-rsqrt.npz"
    np.savez_compressed(
        lut,
        normalized_delta=np.zeros((1 << 24,), dtype=np.int8),
    )
    exact_route = _exact_layernorm_route(lut)
    lut.write_bytes(b"not an npz normalized_delta payload")
    lut_sha = hashlib.sha256(lut.read_bytes()).hexdigest()
    invented_payload_sha = "0" * 64
    report = json.loads(local_report.read_text())
    report["effective_route"].update(
        {
            "decoder_layernorm": {
                **exact_route["decoder_layernorm"],
                "turing_rsqrt_lut_artifact_sha256_attested": lut_sha,
                "turing_rsqrt_lut_content_sha256": invented_payload_sha,
            },
            "decoder_layernorm_lut": {
                "path": str(lut),
                "sha256": lut_sha,
                "normalized_delta_sha256": invented_payload_sha,
                "entries": 1 << 24,
                "dtype": "int8",
            },
        }
    )
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="LayerNorm rsqrt payload"):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_level1_comparator_rejects_requested_exact_effective_native_layernorm(
    tmp_path,
):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    report = json.loads(local_report.read_text())
    report["requested_route"] = {
        "route": "mlx-shape-decoder-level1-trace",
        "device_type": "metal",
        "decoder_linear_backend": "turing_fda",
        "sparse_conv_matmul_backend": "turing_fda",
        "decoder_layernorm_backend": "cuda-welford-turing-t4",
        "decoder_silu_backend": "cuda-turing-t4-fp16-lut",
        "parent_state": "externally-captured-level0-trace",
    }
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="requested/effective decoder LayerNorm"):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_level1_comparator_rejects_requested_effective_concrete_device_mismatch(
    tmp_path,
):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    report = json.loads(local_report.read_text())
    effective = report["effective_route"]
    report["requested_route"] = {
        "route": effective["route"],
        "device_type": effective["device_type"],
        "device": "Device(cpu, 0)",
        "decoder_linear_backend": effective["decoder_linear_backend"],
        "sparse_conv_matmul_backend": effective["sparse_conv_matmul_backend"],
        "decoder_layernorm_backend": effective["decoder_layernorm"]["backend"],
        "decoder_silu_backend": effective["decoder_silu"]["backend"],
        "parent_state": "externally-captured-level0-trace",
    }
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="requested/effective concrete device"):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_local_level1_trace_turing_rsqrt_loader_rejects_malformed_or_substituted(
    tmp_path,
):
    from scripts.run_mlx_decoder_level1_trace import _load_turing_rsqrt_lut

    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, normalized_delta=np.zeros((16,), dtype=np.int8))
    malformed_sha = hashlib.sha256(malformed.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=r"int8\[16777216\]"):
        _load_turing_rsqrt_lut(malformed, malformed_sha)

    with pytest.raises(ValueError, match="rsqrt LUT digest mismatch"):
        _load_turing_rsqrt_lut(malformed, "0" * 64)


def test_level1_comparator_reports_hidden_internal_boundary_first(tmp_path):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    source = _valid_trace()
    local = {name: values.copy() for name, values in source.items()}
    local["level1_block0_silu"][0, 0] += np.float16(1)
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, source, local)
    )

    comparison = compare_level1_traces(
        source_path=source_path,
        source_report_path=source_report,
        local_path=local_path,
        local_report_path=local_report,
    )

    assert comparison["first_nonexact_boundary"] == "level1_block0_silu"
    assert comparison["stages"]["level1_block0_silu"]["nonzero_count"] == 1
    assert comparison["stages"]["level1_block0_output"]["nonzero_count"] == 0


def test_level1_comparator_rejects_wrong_effective_route(tmp_path):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    source = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, source, source)
    )
    report = json.loads(local_report.read_text())
    report["effective_route"]["route"] = "mlx-shape-decoder-level0-trace-fp16"
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="local trace route field"):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_level1_comparator_rejects_under_authenticated_local_route(tmp_path):
    from scripts.compare_decoder_level1_traces import compare_level1_traces

    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    report = json.loads(local_report.read_text())
    report["effective_route"] = {
        "route": "mlx-shape-decoder-level1-trace",
        "device_type": "cpu",
        "device": "cpu",
        "decoder_linear_backend": "native",
        "sparse_conv_matmul_backend": "native",
    }
    report["primary"]["status"] = "not_written"
    local_report.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="local trace route field|primary status"):
        compare_level1_traces(
            source_path=source_path,
            source_report_path=source_report,
            local_path=local_path,
            local_report_path=local_report,
        )


def test_level1_comparator_failed_run_replaces_stale_output_with_phase_report(
    tmp_path,
):
    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    report = json.loads(local_report.read_text())
    report["effective_route"]["route"] = "wrong-local-route"
    local_report.write_text(json.dumps(report))
    output = tmp_path / "comparison.json"
    output.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.decoder_level1_trace_comparison.v1",
                "status": "done",
                "first_nonexact_boundary": "stale_boundary",
            }
        )
    )
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/compare_decoder_level1_traces.py"),
            "--source",
            str(source_path),
            "--source-report",
            str(source_report),
            "--local",
            str(local_path),
            "--local-report",
            str(local_report),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    failed = json.loads(output.read_text())
    assert completed.returncode != 0
    assert failed["status"] == "failed"
    assert failed["failure_phase"] == "comparison"
    assert failed["last_trustworthy_phase"] == "request_validation"
    assert failed["first_nonexact_boundary"] is None


def test_level1_comparator_collision_preserves_input_and_uses_failure_sibling(
    tmp_path,
):
    arrays = _valid_trace()
    source_path, source_report, local_path, local_report = (
        _write_comparison_inputs(tmp_path, arrays, arrays)
    )
    original_report = local_report.read_bytes()
    fallback = local_report.with_name(local_report.name + ".failure.json")
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/compare_decoder_level1_traces.py"),
            "--source",
            str(source_path),
            "--source-report",
            str(source_report),
            "--local",
            str(local_path),
            "--local-report",
            str(local_report),
            "--output",
            str(local_report),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert local_report.read_bytes() == original_report
    failed = json.loads(fallback.read_text())
    assert failed["status"] == "failed"
    assert failed["failure_phase"] == "request_validation"
    assert failed["requested"]["output"] == str(local_report)
    assert failed["effective_output"] == str(fallback)


def test_level1_trace_script_entry_points_resolve_repo_contract_module():
    repo_root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "scripts/run_mlx_decoder_level1_trace.py",
        "scripts/compare_decoder_level1_traces.py",
        "scripts/source_cuda_postcond_full_decode_timing.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(repo_root / relative_path), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            f"{relative_path} direct entry failed:\n{completed.stderr}"
        )


def test_local_level1_trace_failure_writes_phase_report_and_invalidates_stale_primary(
    tmp_path,
):
    from scripts.run_mlx_decoder_level1_trace import main

    missing_parent = tmp_path / "missing-level0.npz"
    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    silu_lut = tmp_path / "silu.npz"
    silu_lut.write_bytes(b"silu")
    rsqrt_lut = tmp_path / "rsqrt.npz"
    rsqrt_lut.write_bytes(b"rsqrt")
    output_npz = tmp_path / "trace.npz"
    output_npz.write_bytes(b"stale-primary")
    output_json = tmp_path / "trace.json"

    rc = main(
        [
            "--level0-trace",
            str(missing_parent),
            "--expected-level0-trace-sha256",
            "a" * 64,
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--decoder-silu-lut",
            str(silu_lut),
            "--expected-decoder-silu-lut-sha256",
            hashlib.sha256(silu_lut.read_bytes()).hexdigest(),
            "--turing-rsqrt-lut",
            str(rsqrt_lut),
            "--expected-turing-rsqrt-lut-sha256",
            hashlib.sha256(rsqrt_lut.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text())
    assert rc == 1
    assert report["schema"] == "trellis2mlx.decoder_level1_trace_run.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "parent_trace_validation"
    assert report["last_trustworthy_phase"] == "request_validation"
    assert report["effective_route"] is None
    assert report["stale_primary_invalidated"] is True
    assert report["primary"]["status"] == "not_written"
    assert not output_npz.exists()


def test_local_level1_trace_rejects_stale_parent_digest_before_model_load(
    tmp_path,
):
    from scripts.run_mlx_decoder_level1_trace import main

    parent = tmp_path / "level0.npz"
    rows = 2
    np.savez(
        parent,
        coords=np.array(
            [[0, 1, 2, 3], [0, 4, 5, 6]],
            dtype=np.int32,
        ),
        block3_output=np.ones((rows, 1024), dtype=np.float16),
    )

    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    silu_lut = tmp_path / "silu.npz"
    silu_lut.write_bytes(b"silu")
    rsqrt_lut = tmp_path / "rsqrt.npz"
    rsqrt_lut.write_bytes(b"rsqrt")
    output_npz = tmp_path / "trace.npz"
    output_json = tmp_path / "trace.json"

    rc = main(
        [
            "--level0-trace",
            str(parent),
            "--expected-level0-trace-sha256",
            "0" * 64,
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--decoder-silu-lut",
            str(silu_lut),
            "--expected-decoder-silu-lut-sha256",
            hashlib.sha256(silu_lut.read_bytes()).hexdigest(),
            "--turing-rsqrt-lut",
            str(rsqrt_lut),
            "--expected-turing-rsqrt-lut-sha256",
            hashlib.sha256(rsqrt_lut.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text())
    assert rc == 1
    assert report["failure_phase"] == "parent_trace_validation"
    assert "level-zero trace digest mismatch" in report["error"]
    assert report["last_trustworthy_phase"] == "request_validation"
    assert not output_npz.exists()


def test_local_level1_trace_report_collision_preserves_parent_and_uses_sibling(
    tmp_path,
):
    from scripts.run_mlx_decoder_level1_trace import main

    parent = tmp_path / "level0.npz"
    np.savez(
        parent,
        coords=np.array([[0, 1, 2, 3]], dtype=np.int32),
        block3_output=np.ones((1, 1024), dtype=np.float16),
    )
    original_parent = parent.read_bytes()
    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    silu_lut = tmp_path / "silu.npz"
    silu_lut.write_bytes(b"silu")
    rsqrt_lut = tmp_path / "rsqrt.npz"
    rsqrt_lut.write_bytes(b"rsqrt")
    output_npz = tmp_path / "trace.npz"
    output_npz.write_bytes(b"stale-primary")
    fallback = parent.with_name(parent.name + ".failure.json")

    rc = main(
        [
            "--level0-trace",
            str(parent),
            "--expected-level0-trace-sha256",
            hashlib.sha256(original_parent).hexdigest(),
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--decoder-silu-lut",
            str(silu_lut),
            "--expected-decoder-silu-lut-sha256",
            hashlib.sha256(silu_lut.read_bytes()).hexdigest(),
            "--turing-rsqrt-lut",
            str(rsqrt_lut),
            "--expected-turing-rsqrt-lut-sha256",
            hashlib.sha256(rsqrt_lut.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(parent),
        ]
    )

    assert rc == 1
    assert parent.read_bytes() == original_parent
    failed = json.loads(fallback.read_text())
    assert failed["status"] == "failed"
    assert failed["failure_phase"] == "request_validation"
    assert failed["requested_report_path"] == str(parent)
    assert failed["effective_report_path"] == str(fallback)
    assert failed["stale_primary_invalidated"] is True
    assert not output_npz.exists()


def test_local_level1_trace_primary_report_alias_invalidates_stale_primary(
    tmp_path,
):
    from scripts.run_mlx_decoder_level1_trace import main

    parent = tmp_path / "level0.npz"
    np.savez(
        parent,
        coords=np.array([[0, 1, 2, 3]], dtype=np.int32),
        block3_output=np.ones((1, 1024), dtype=np.float16),
    )
    original_parent = parent.read_bytes()
    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    silu_lut = tmp_path / "silu.npz"
    silu_lut.write_bytes(b"silu")
    rsqrt_lut = tmp_path / "rsqrt.npz"
    rsqrt_lut.write_bytes(b"rsqrt")
    output_npz = tmp_path / "trace.npz"
    output_npz.write_bytes(b"stale-primary")
    fallback = output_npz.with_name(output_npz.name + ".failure.json")

    rc = main(
        [
            "--level0-trace",
            str(parent),
            "--expected-level0-trace-sha256",
            hashlib.sha256(original_parent).hexdigest(),
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--decoder-silu-lut",
            str(silu_lut),
            "--expected-decoder-silu-lut-sha256",
            hashlib.sha256(silu_lut.read_bytes()).hexdigest(),
            "--turing-rsqrt-lut",
            str(rsqrt_lut),
            "--expected-turing-rsqrt-lut-sha256",
            hashlib.sha256(rsqrt_lut.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_npz),
        ]
    )

    assert rc == 1
    assert parent.read_bytes() == original_parent
    assert not output_npz.exists()
    failed = json.loads(fallback.read_text())
    assert failed["status"] == "failed"
    assert failed["failure_phase"] == "request_validation"
    assert failed["requested_report_path"] == str(output_npz)
    assert failed["effective_report_path"] == str(fallback)
    assert failed["stale_primary_invalidated"] is True

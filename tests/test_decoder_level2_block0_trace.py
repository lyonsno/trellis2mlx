import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.decoder_level1_trace_contract import (
    LEVEL1_HASH_BOUNDARY_NAMES,
    LEVEL1_HASH_LEDGER_SCHEMA,
    _expected_hash_ledger_specs,
    build_decoder_level1_hash_ledger,
)
from scripts.decoder_level2_block0_trace_contract import (
    CHILD_ARRAY_NAMES,
    LEVEL2_BLOCK0_NORM_BOUNDARY_ROUTE,
    PARENT_COORD_BOUNDARY,
    PARENT_FEATURE_BOUNDARY,
    compare_decoder_level2_block0_traces,
    decoder_boundary_hash_entry,
    load_decoder_level2_block0_trace,
    write_decoder_level2_block0_trace_npz,
)


PARENT_COMMIT = "f382af6000d77e48ce105fe7084fa90096ed2a44"
WIDTH256_AFFINE_CONTRACT = {
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
}
DECODER_LAYERNORM_CONTRACTS = [
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
    copy.deepcopy(WIDTH256_AFFINE_CONTRACT),
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
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parent_boundaries():
    specs = _expected_hash_ledger_specs(2, 3, 4)
    boundaries = {}
    for index, name in enumerate(LEVEL1_HASH_BOUNDARY_NAMES):
        shape, dtype = specs[name]
        boundaries[name] = np.full(shape, index % 5, dtype=dtype)
    boundaries[PARENT_COORD_BOUNDARY] = np.array(
        [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]],
        dtype=np.int32,
    )
    boundaries[PARENT_FEATURE_BOUNDARY] = np.arange(
        3 * 256,
        dtype=np.float16,
    ).reshape(3, 256)
    return boundaries


def _child_arrays(parent_boundaries=None):
    parent_boundaries = parent_boundaries or _parent_boundaries()
    rows = parent_boundaries[PARENT_COORD_BOUNDARY].shape[0]
    features = parent_boundaries[PARENT_FEATURE_BOUNDARY].copy()
    return {
        PARENT_COORD_BOUNDARY: parent_boundaries[PARENT_COORD_BOUNDARY].copy(),
        PARENT_FEATURE_BOUNDARY: features,
        "level2_block0_conv": features + np.float16(1),
        "level2_block0_norm": features + np.float16(2),
        "level2_block0_mlp_fc1": np.tile(features, (1, 4)),
        "level2_block0_silu": np.tile(features + np.float16(3), (1, 4)),
        "level2_block0_mlp_fc2": features + np.float16(4),
        "level2_block0_output": features + np.float16(5),
    }


def _local_child_arrays(parent_boundaries=None):
    arrays = _child_arrays(parent_boundaries)
    arrays["level2_block0_norm"][0, 0] += np.float16(0.125)
    arrays["level2_block0_mlp_fc1"][0, 0] += np.float16(0.125)
    arrays["level2_block0_silu"][0, 0] += np.float16(0.125)
    arrays["level2_block0_mlp_fc2"][0, 0] += np.float16(0.125)
    arrays["level2_block0_output"][0, 0] += np.float16(0.125)
    return arrays


def _write_parent_receipts(tmp_path: Path, parent_boundaries=None):
    parent_boundaries = parent_boundaries or _parent_boundaries()
    source_boundaries = copy.deepcopy(parent_boundaries)
    local_boundaries = copy.deepcopy(parent_boundaries)
    source_boundaries["level2_block0_output"] = _child_arrays(
        parent_boundaries
    )["level2_block0_output"]
    local_boundaries["level2_block0_output"] = _local_child_arrays(
        parent_boundaries
    )["level2_block0_output"]
    source_ledger = build_decoder_level1_hash_ledger(source_boundaries)
    local_ledger = build_decoder_level1_hash_ledger(local_boundaries)
    source_primary = tmp_path / "source-parent.npz"
    local_primary = tmp_path / "local-parent.npz"
    np.savez(source_primary, marker=np.array([1], dtype=np.int8))
    np.savez(local_primary, marker=np.array([2], dtype=np.int8))
    source_primary_sha = _sha256(source_primary)
    local_primary_sha = _sha256(local_primary)

    source_report = tmp_path / "source-parent.json"
    source_payload = {
        "schema": "trellis2mlx.source_cuda_shape_slat_grid_decode.v1",
        "status": "done",
        "effective_route": {
            "route": "official-source-cuda-shape-decoder-level1-trace",
            "device_type": "cuda",
            "cuda_device": "Tesla T4",
            "decoder_level1_trace": True,
            "sparse_conv_backend": "none",
        },
        "decoder_trace_artifacts": [
            {
                "path": source_primary.name,
                "status": "written",
                "sha256": source_primary_sha,
                "validation": {
                    "reopened_exact": True,
                    "child_expansion_exact": True,
                },
                "hash_ledger": source_ledger,
            }
        ],
    }
    _write_json(source_report, source_payload)

    local_report = tmp_path / "local-parent.json"
    local_route = {
        "route": "mlx-shape-decoder-level1-trace",
        "device_type": "metal",
        "device": "Device(gpu, 0)",
        "decoder_linear_backend": "turing_fda",
        "sparse_conv_matmul_backend": "turing_fda",
        "decoder_layernorm": {"backend": "cuda-welford-turing-t4"},
        "decoder_silu": {"backend": "cuda-turing-t4-fp16-lut"},
    }
    _write_json(
        local_report,
        {
            "schema": "trellis2mlx.decoder_level1_trace_run.v1",
            "status": "done",
            "effective_route": local_route,
            "primary": {
                "path": str(local_primary),
                "status": "written",
                "sha256": local_primary_sha,
                "validation": {
                    "reopened_exact": True,
                    "child_expansion_exact": True,
                },
                "hash_ledger": local_ledger,
            },
        },
    )

    comparison = tmp_path / "parent-comparison.json"
    comparison_rows = []
    for source_entry, local_entry in zip(
        source_ledger["entries"],
        local_ledger["entries"],
    ):
        comparison_rows.append(
            {
                "name": source_entry["name"],
                "dtype": source_entry["dtype"],
                "shape": source_entry["shape"],
                "source_sha256": source_entry["sha256"],
                "local_sha256": local_entry["sha256"],
                "exact": source_entry["sha256"] == local_entry["sha256"],
            }
        )
    _write_json(
        comparison,
        {
            "schema": "trellis2mlx.decoder_level1_trace_comparison.v1",
            "status": "done",
            "first_nonexact_boundary": None,
            "first_nonexact_hash_boundary": "level2_block0_output",
            "artifacts": {
                "source": {
                    "path": str(source_primary),
                    "sha256": source_primary_sha,
                    "effective_route": source_payload["effective_route"],
                },
                "local": {
                    "path": str(local_primary),
                    "sha256": local_primary_sha,
                    "effective_route": local_route,
                },
            },
            "hash_boundaries": comparison_rows,
        },
    )

    manifest = tmp_path / "command-manifest.json"
    _write_json(
        manifest,
        {
            "schema": "gpu-greenroom.command.v1",
            "repo_root": "/private/tmp/parent",
            "route_identity": "test-parent",
            "argv": [
                "python",
                "scripts/run_mlx_decoder_level1_trace.py",
                "--expected-repo-commit",
                PARENT_COMMIT,
            ],
        },
    )
    receipt = {
        "schema": "trellis2mlx.decoder_level2_block0_parent_receipt.v1",
        "parent_object_commit": PARENT_COMMIT,
        "parent_contract_schema": LEVEL1_HASH_LEDGER_SCHEMA,
        "boundary_names": {
            "features": PARENT_FEATURE_BOUNDARY,
            "coordinates": PARENT_COORD_BOUNDARY,
        },
        "source_parent_report": {
            "path": str(source_report),
            "sha256": _sha256(source_report),
        },
        "local_parent_report": {
            "path": str(local_report),
            "sha256": _sha256(local_report),
        },
        "local_command_manifest": {
            "path": str(manifest),
            "sha256": _sha256(manifest),
        },
        "parent_strict_comparison": {
            "path": str(comparison),
            "sha256": _sha256(comparison),
        },
    }
    return receipt


def _child_route(label, *, turing_rsqrt_lut=None):
    if label == "source":
        return {
            "route": "official-source-cuda-shape-decoder-level2-block0-trace",
            "device_type": "cuda",
            "cuda_device": "Tesla T4",
            "decoder_level2_block0_trace": True,
            "sparse_conv_backend": "none",
        }
    assert turing_rsqrt_lut is not None
    with np.load(turing_rsqrt_lut, allow_pickle=False) as archive:
        normalized_delta = np.asarray(archive["normalized_delta"])
    lut_sha256 = _sha256(turing_rsqrt_lut)
    content_sha256 = hashlib.sha256(
        np.ascontiguousarray(normalized_delta).tobytes()
    ).hexdigest()
    return {
        "route": "mlx-shape-decoder-level2-block0-trace",
        "device_type": "metal",
        "device": "Device(gpu, 0)",
        "decoder_linear_backend": "turing_fda",
        "sparse_conv_matmul_backend": "turing_fda",
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
            "authenticated_contracts": copy.deepcopy(
                DECODER_LAYERNORM_CONTRACTS
            ),
            "reduction": copy.deepcopy(
                DECODER_LAYERNORM_CONTRACTS[0]["reduction"]
            ),
            "rsqrt": "Turing MUFU.RSQ normalized signed-ULP LUT",
            "turing_rsqrt_lut_artifact_sha256_attested": lut_sha256,
            "turing_rsqrt_lut_content_sha256": content_sha256,
            "turing_rsqrt_lut_entries": 1 << 24,
        },
        "decoder_layernorm_lut": {
            "path": str(turing_rsqrt_lut),
            "sha256": lut_sha256,
            "normalized_delta_sha256": content_sha256,
            "entries": 1 << 24,
            "dtype": "int8",
        },
        "decoder_silu": {"backend": "cuda-turing-t4-fp16-lut"},
        "boundary_routes": {
            "level2_block0_norm": dict(LEVEL2_BLOCK0_NORM_BOUNDARY_ROUTE)
        },
    }


def _write_child(tmp_path, label, arrays, *, manual_equal=True, route=None):
    primary = tmp_path / f"{label}-child.npz"
    validation = write_decoder_level2_block0_trace_npz(primary, arrays)
    report = tmp_path / f"{label}-child.json"
    turing_rsqrt_lut = None
    if label == "local" and route is None:
        turing_rsqrt_lut = tmp_path / "child-turing-rsqrt.npz"
        if not turing_rsqrt_lut.exists():
            np.savez_compressed(
                turing_rsqrt_lut,
                normalized_delta=np.zeros((1 << 24,), dtype=np.int8),
            )
    effective_route = route or _child_route(
        label,
        turing_rsqrt_lut=turing_rsqrt_lut,
    )
    primary_row = {
        "path": str(primary),
        "status": "written",
        "sha256": _sha256(primary),
        "validation": validation,
    }
    equality = {
        "features": manual_equal,
        "coordinates": manual_equal,
    }
    if label == "source":
        primary_row["manual_natural_equality"] = equality
        payload = {
            "schema": "trellis2mlx.source_cuda_shape_slat_grid_decode.v1",
            "status": "done",
            "requested_route": {
                "route": effective_route["route"],
                "decoder_level2_block0_trace": True,
                "raw_meshes": False,
                "mesh_conversion": False,
            },
            "effective_route": effective_route,
            "decoder_trace_artifacts": [primary_row],
        }
    else:
        payload = {
            "schema": "trellis2mlx.decoder_level2_block0_trace_run.v1",
            "status": "done",
            "requested_route": {
                "route": effective_route["route"],
                "device_type": "metal",
                "decoder_linear_backend": "turing_fda",
                "sparse_conv_matmul_backend": "turing_fda",
                "decoder_layernorm_backend": "cuda-welford-turing-t4",
                "decoder_silu_backend": "cuda-turing-t4-fp16-lut",
                "boundary_routes": effective_route["boundary_routes"],
            },
            "effective_route": effective_route,
            "manual_natural_equality": equality,
            "primary": primary_row,
        }
    _write_json(report, payload)
    return primary, report


def _valid_comparison_inputs(tmp_path):
    parents = _parent_boundaries()
    receipt = _write_parent_receipts(tmp_path, parents)
    source_primary, source_report = _write_child(
        tmp_path,
        "source",
        _child_arrays(parents),
    )
    local_arrays = _local_child_arrays(parents)
    local_primary, local_report = _write_child(
        tmp_path,
        "local",
        local_arrays,
    )
    return receipt, source_primary, source_report, local_primary, local_report


def _local_lut_identity(inputs):
    local_report = Path(inputs[4])
    report = json.loads(local_report.read_text())
    lut = report["effective_route"]["decoder_layernorm_lut"]
    path = Path(lut["path"])
    if not path.is_absolute():
        path = local_report.parent / path
    return path.resolve(), lut["sha256"]


def _compare(
    inputs,
    *,
    expected_lut_sha256=None,
    parent_fork_disposition="historical-fork",
):
    receipt, source_primary, source_report, local_primary, local_report = inputs
    receipt_path = Path(receipt["source_parent_report"]["path"]).with_name(
        "parent-receipt.json"
    )
    _write_json(receipt_path, receipt)
    lut_path, reported_lut_sha256 = _local_lut_identity(inputs)
    return compare_decoder_level2_block0_traces(
        source_path=source_primary,
        source_report_path=source_report,
        local_path=local_primary,
        local_report_path=local_report,
        parent_receipt_path=receipt_path,
        expected_parent_receipt_sha256=_sha256(receipt_path),
        turing_rsqrt_lut_path=lut_path,
        expected_turing_rsqrt_lut_sha256=(
            expected_lut_sha256 or reported_lut_sha256
        ),
        parent_fork_disposition=parent_fork_disposition,
    )


def test_block0_child_contract_accepts_parent_composed_witness(tmp_path):
    inputs = _valid_comparison_inputs(tmp_path)
    result = _compare(inputs)
    assert result["status"] == "done"
    assert result["first_nonexact_boundary"] == "level2_block0_norm"
    assert result["parent_receipt"]["parent_object_commit"] == PARENT_COMMIT
    assert result["parent_receipt"]["receipt_file"] == {
        "path": str(
            Path(inputs[0]["source_parent_report"]["path"])
            .with_name("parent-receipt.json")
            .resolve()
        ),
        "sha256": _sha256(
            Path(inputs[0]["source_parent_report"]["path"]).with_name(
                "parent-receipt.json"
            )
        ),
    }
    assert result["child_input_identity"]["source"] == result[
        "child_input_identity"
    ]["local"]


@pytest.mark.parametrize("mutation", ["shift", "reorder", "feature_bit"])
def test_block0_child_rejects_source_local_equal_input_detached_from_parent(
    tmp_path,
    mutation,
):
    parents = _parent_boundaries()
    receipt = _write_parent_receipts(tmp_path, parents)
    source_arrays = _child_arrays(parents)
    local_arrays = _local_child_arrays(parents)
    if mutation == "shift":
        source_arrays[PARENT_COORD_BOUNDARY][:, 1:] += 1000
        local_arrays[PARENT_COORD_BOUNDARY][:, 1:] += 1000
    elif mutation == "reorder":
        order = np.array([2, 0, 1])
        for arrays in (source_arrays, local_arrays):
            arrays[PARENT_COORD_BOUNDARY] = arrays[PARENT_COORD_BOUNDARY][order]
            arrays[PARENT_FEATURE_BOUNDARY] = arrays[
                PARENT_FEATURE_BOUNDARY
            ][order]
    else:
        for arrays in (source_arrays, local_arrays):
            bits = arrays[PARENT_FEATURE_BOUNDARY].view(np.uint16)
            bits[0, 0] ^= np.uint16(1)
    source = _write_child(tmp_path, "source", source_arrays)
    local = _write_child(tmp_path, "local", local_arrays)
    with pytest.raises(ValueError, match="parent boundary hash"):
        _compare((receipt, source[0], source[1], local[0], local[1]))


def test_block0_child_rejects_stale_parent_report_bytes(tmp_path):
    inputs = _valid_comparison_inputs(tmp_path)
    source_report = Path(inputs[0]["source_parent_report"]["path"])
    source_report.write_text(source_report.read_text() + " ")
    with pytest.raises(ValueError, match="source parent report SHA256"):
        _compare(inputs)


def test_block0_comparator_requires_caller_bound_parent_receipt_sha256():
    from scripts.compare_decoder_level2_block0_traces import build_parser

    actions = {
        option: action
        for action in build_parser()._actions
        for option in action.option_strings
    }

    assert actions["--expected-parent-receipt-sha256"].required is True
    assert actions["--turing-rsqrt-lut"].required is True
    assert actions["--expected-turing-rsqrt-lut-sha256"].required is True
    disposition = actions["--parent-fork-disposition"]
    assert disposition.required is True
    assert disposition.choices == (
        "historical-fork",
        "corrected-child-exact-to-source",
    )


def test_block0_comparator_rejects_foreign_internally_coherent_receipt(
    tmp_path,
):
    from scripts.compare_decoder_level2_block0_traces import main

    inputs = _valid_comparison_inputs(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    _write_json(receipt_path, inputs[0])
    trusted_sha256 = _sha256(receipt_path)

    manifest_path = Path(inputs[0]["local_command_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["foreign_but_internally_coherent"] = True
    _write_json(manifest_path, manifest)
    inputs[0]["local_command_manifest"]["sha256"] = _sha256(manifest_path)
    _write_json(receipt_path, inputs[0])
    lut_path, lut_sha256 = _local_lut_identity(inputs)

    output = tmp_path / "comparison.json"
    rc = main(
        [
            "--source",
            str(inputs[1]),
            "--source-report",
            str(inputs[2]),
            "--local",
            str(inputs[3]),
            "--local-report",
            str(inputs[4]),
            "--parent-receipt",
            str(receipt_path),
            "--expected-parent-receipt-sha256",
            trusted_sha256,
            "--turing-rsqrt-lut",
            str(lut_path),
            "--expected-turing-rsqrt-lut-sha256",
            lut_sha256,
            "--parent-fork-disposition",
            "historical-fork",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["failure_phase"] == "parent_receipt_authentication"
    assert "parent receipt SHA256 mismatch" in report["error"]


@pytest.mark.parametrize("label", ["source", "local"])
def test_block0_child_rejects_missing_parent_primary(tmp_path, label):
    inputs = _valid_comparison_inputs(tmp_path)
    (tmp_path / f"{label}-parent.npz").unlink()

    with pytest.raises(ValueError, match=f"{label} parent primary.*missing"):
        _compare(inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "not_written"),
        ("validation", None),
        ("reopened_exact", False),
        ("child_expansion_exact", False),
    ],
)
def test_block0_child_rejects_incomplete_local_parent_primary_validation(
    tmp_path,
    field,
    value,
):
    inputs = _valid_comparison_inputs(tmp_path)
    receipt = inputs[0]
    local_report_path = Path(receipt["local_parent_report"]["path"])
    local_report = json.loads(local_report_path.read_text())
    primary = local_report["primary"]
    if field == "status":
        primary["status"] = value
    elif field == "validation":
        primary.pop("validation")
    else:
        primary["validation"][field] = value
    _write_json(local_report_path, local_report)
    receipt["local_parent_report"]["sha256"] = _sha256(local_report_path)

    with pytest.raises(ValueError, match="local parent primary validation"):
        _compare(inputs)


@pytest.mark.parametrize(
    "mutation",
    ["exact_true", "equal_hashes", "detached_hash"],
)
def test_block0_child_rejects_false_parent_fork_row(tmp_path, mutation):
    inputs = _valid_comparison_inputs(tmp_path)
    receipt = inputs[0]
    comparison_path = Path(receipt["parent_strict_comparison"]["path"])
    comparison = json.loads(comparison_path.read_text())
    fork = next(
        row
        for row in comparison["hash_boundaries"]
        if row["name"] == "level2_block0_output"
    )
    if mutation == "exact_true":
        fork["exact"] = True
    elif mutation == "equal_hashes":
        fork["source_sha256"] = fork["local_sha256"]
        fork["exact"] = True
    else:
        fork["source_sha256"] = "0" * 64
    _write_json(comparison_path, comparison)
    receipt["parent_strict_comparison"]["sha256"] = _sha256(comparison_path)

    with pytest.raises(ValueError, match="parent fork boundary"):
        _compare(inputs)


def test_block0_child_rejects_wrong_parent_commit(tmp_path):
    inputs = _valid_comparison_inputs(tmp_path)
    inputs[0]["parent_object_commit"] = "0" * 40
    with pytest.raises(ValueError, match="parent object commit"):
        _compare(inputs)


def test_block0_child_rejects_earlier_parent_fork(tmp_path):
    inputs = _valid_comparison_inputs(tmp_path)
    comparison_path = Path(
        inputs[0]["parent_strict_comparison"]["path"]
    )
    comparison = json.loads(comparison_path.read_text())
    comparison["first_nonexact_hash_boundary"] = PARENT_FEATURE_BOUNDARY
    _write_json(comparison_path, comparison)
    inputs[0]["parent_strict_comparison"]["sha256"] = _sha256(comparison_path)
    with pytest.raises(ValueError, match="first nonexact hash boundary"):
        _compare(inputs)


def test_block0_child_rejects_raw_hash_domain_and_boundary_rename(tmp_path):
    inputs = _valid_comparison_inputs(tmp_path)
    receipt = inputs[0]
    receipt["boundary_names"]["features"] = "level2_block0_input"
    with pytest.raises(ValueError, match="boundary names"):
        _compare(inputs)

    receipt["boundary_names"]["features"] = PARENT_FEATURE_BOUNDARY
    source_path = inputs[1]
    arrays = load_decoder_level2_block0_trace(source_path)
    raw = hashlib.sha256(
        np.ascontiguousarray(arrays[PARENT_FEATURE_BOUNDARY]).tobytes()
    ).hexdigest()
    assert raw != decoder_boundary_hash_entry(
        PARENT_FEATURE_BOUNDARY,
        arrays[PARENT_FEATURE_BOUNDARY],
    )["sha256"]


def test_block0_child_rejects_manual_natural_or_route_lie(tmp_path):
    inputs = _valid_comparison_inputs(tmp_path)
    local_report_path = inputs[4]
    report = json.loads(local_report_path.read_text())
    report["manual_natural_equality"]["features"] = False
    _write_json(local_report_path, report)
    with pytest.raises(ValueError, match="manual.*natural"):
        _compare(inputs)

    report["manual_natural_equality"]["features"] = True
    report["effective_route"]["boundary_routes"].pop("level2_block0_norm")
    _write_json(local_report_path, report)
    with pytest.raises(ValueError, match="level2_block0_norm"):
        _compare(inputs)


def test_block0_child_rejects_missing_affine_width256_layernorm_contract(
    tmp_path,
):
    inputs = _valid_comparison_inputs(tmp_path)
    local_report_path = inputs[4]
    report = json.loads(local_report_path.read_text())
    report["effective_route"]["decoder_layernorm"][
        "authenticated_contracts"
    ].clear()
    _write_json(local_report_path, report)

    with pytest.raises(ValueError, match="affine width-256"):
        _compare(inputs)


def test_block0_child_rejects_missing_width128_layernorm_contract(
    tmp_path,
):
    inputs = _valid_comparison_inputs(tmp_path)
    local_report_path = inputs[4]
    report = json.loads(local_report_path.read_text())
    report["effective_route"]["decoder_layernorm"][
        "authenticated_contracts"
    ].pop()
    _write_json(local_report_path, report)

    with pytest.raises(ValueError, match="incomplete authenticated contract ledger"):
        _compare(inputs)


def test_block0_child_rejects_missing_width64_layernorm_contract(
    tmp_path,
):
    inputs = _valid_comparison_inputs(tmp_path)
    local_report_path = inputs[4]
    report = json.loads(local_report_path.read_text())
    report["effective_route"]["decoder_layernorm"][
        "authenticated_contracts"
    ].pop()
    _write_json(local_report_path, report)

    with pytest.raises(ValueError, match="incomplete authenticated contract ledger"):
        _compare(inputs)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("algorithm", "impersonating-layernorm"),
        ("cuda_source_kernel", None),
        ("experimental", False),
    ],
)
def test_block0_child_rejects_malformed_global_layernorm_identity(
    tmp_path,
    field,
    replacement,
):
    inputs = _valid_comparison_inputs(tmp_path)
    local_report_path = inputs[4]
    report = json.loads(local_report_path.read_text())
    layernorm = report["effective_route"]["decoder_layernorm"]
    if replacement is None:
        layernorm.pop(field)
    else:
        layernorm[field] = replacement
    _write_json(local_report_path, report)

    with pytest.raises(ValueError, match="global LayerNorm identity"):
        _compare(inputs)


def test_block0_child_rejects_self_consistent_substituted_turing_lut(
    tmp_path,
):
    inputs = _valid_comparison_inputs(tmp_path)
    lut_path, trusted_sha256 = _local_lut_identity(inputs)
    with np.load(lut_path, allow_pickle=False) as archive:
        normalized_delta = np.asarray(archive["normalized_delta"]).copy()
    normalized_delta[0] = np.int8(1)
    np.savez_compressed(lut_path, normalized_delta=normalized_delta)
    substituted_sha256 = _sha256(lut_path)
    content_sha256 = hashlib.sha256(
        np.ascontiguousarray(normalized_delta).tobytes()
    ).hexdigest()
    report = json.loads(inputs[4].read_text())
    layernorm = report["effective_route"]["decoder_layernorm"]
    lut = report["effective_route"]["decoder_layernorm_lut"]
    layernorm["turing_rsqrt_lut_artifact_sha256_attested"] = (
        substituted_sha256
    )
    layernorm["turing_rsqrt_lut_content_sha256"] = content_sha256
    lut["sha256"] = substituted_sha256
    lut["normalized_delta_sha256"] = content_sha256
    _write_json(inputs[4], report)

    with pytest.raises(ValueError, match="artifact SHA256 mismatch"):
        _compare(inputs, expected_lut_sha256=trusted_sha256)


def test_block0_child_rejects_truthfully_hashed_malformed_turing_lut(
    tmp_path,
):
    inputs = _valid_comparison_inputs(tmp_path)
    lut_path, _ = _local_lut_identity(inputs)
    normalized_delta = np.zeros((4,), dtype=np.int8)
    np.savez_compressed(lut_path, normalized_delta=normalized_delta)
    lut_sha256 = _sha256(lut_path)
    content_sha256 = hashlib.sha256(normalized_delta.tobytes()).hexdigest()
    report = json.loads(inputs[4].read_text())
    layernorm = report["effective_route"]["decoder_layernorm"]
    lut = report["effective_route"]["decoder_layernorm_lut"]
    layernorm["turing_rsqrt_lut_artifact_sha256_attested"] = lut_sha256
    layernorm["turing_rsqrt_lut_content_sha256"] = content_sha256
    lut["sha256"] = lut_sha256
    lut["normalized_delta_sha256"] = content_sha256
    _write_json(inputs[4], report)

    with pytest.raises(ValueError, match="payload schema mismatch"):
        _compare(inputs, expected_lut_sha256=lut_sha256)


def test_block0_child_rejects_source_local_input_disagreement(tmp_path):
    inputs = _valid_comparison_inputs(tmp_path)
    local_arrays = load_decoder_level2_block0_trace(inputs[3])
    local_arrays[PARENT_FEATURE_BOUNDARY][0, 0] += np.float16(1)
    write_decoder_level2_block0_trace_npz(inputs[3], local_arrays)
    local_report = json.loads(inputs[4].read_text())
    local_report["primary"]["sha256"] = _sha256(inputs[3])
    _write_json(inputs[4], local_report)
    with pytest.raises(ValueError, match="source and local child inputs"):
        _compare(inputs)


def test_block0_child_rejects_output_detached_from_parent_fork(tmp_path):
    inputs = _valid_comparison_inputs(tmp_path)
    source_arrays = load_decoder_level2_block0_trace(inputs[1])
    source_arrays["level2_block0_output"][0, 0] += np.float16(1)
    write_decoder_level2_block0_trace_npz(inputs[1], source_arrays)
    source_report = json.loads(inputs[2].read_text())
    source_report["decoder_trace_artifacts"][0]["sha256"] = _sha256(inputs[1])
    _write_json(inputs[2], source_report)
    with pytest.raises(ValueError, match="block0 output parent boundary hash"):
        _compare(inputs)


def _corrected_comparison_inputs(tmp_path):
    parents = _parent_boundaries()
    receipt = _write_parent_receipts(tmp_path, parents)
    exact_arrays = _child_arrays(parents)
    source_primary, source_report = _write_child(
        tmp_path,
        "source",
        exact_arrays,
    )
    local_primary, local_report = _write_child(
        tmp_path,
        "local",
        exact_arrays,
    )
    return receipt, source_primary, source_report, local_primary, local_report


def test_block0_child_accepts_exact_correction_against_historical_fork(
    tmp_path,
):
    result = _compare(
        _corrected_comparison_inputs(tmp_path),
        parent_fork_disposition="corrected-child-exact-to-source",
    )

    disposition = result["parent_fork_disposition"]
    assert result["status"] == "done"
    assert result["first_nonexact_boundary"] is None
    assert disposition["requested"] == "corrected-child-exact-to-source"
    assert disposition["effective"] == "corrected-child-exact-to-source"
    assert disposition["historical"]["source"]["sha256"] != disposition[
        "historical"
    ]["local"]["sha256"]
    assert disposition["current"]["source"] == disposition["current"]["local"]
    assert disposition["current"]["local"] == disposition["historical"][
        "source"
    ]
    assert disposition["current"]["local"] != disposition["historical"][
        "local"
    ]
    assert disposition["all_block_boundaries_exact_to_source"] is True


def test_block0_child_correction_rejects_arbitrary_new_output(tmp_path):
    inputs = _corrected_comparison_inputs(tmp_path)
    local_arrays = load_decoder_level2_block0_trace(inputs[3])
    local_arrays["level2_block0_output"][0, 0] += np.float16(1)
    write_decoder_level2_block0_trace_npz(inputs[3], local_arrays)
    local_report = json.loads(inputs[4].read_text())
    local_report["primary"]["sha256"] = _sha256(inputs[3])
    _write_json(inputs[4], local_report)

    with pytest.raises(
        ValueError,
        match="corrected local block0 output does not match source",
    ):
        _compare(
            inputs,
            parent_fork_disposition="corrected-child-exact-to-source",
        )


def test_block0_child_correction_rejects_hidden_intermediate_divergence(
    tmp_path,
):
    inputs = _corrected_comparison_inputs(tmp_path)
    local_arrays = load_decoder_level2_block0_trace(inputs[3])
    local_arrays["level2_block0_norm"][0, 0] += np.float16(1)
    write_decoder_level2_block0_trace_npz(inputs[3], local_arrays)
    local_report = json.loads(inputs[4].read_text())
    local_report["primary"]["sha256"] = _sha256(inputs[3])
    _write_json(inputs[4], local_report)

    with pytest.raises(
        ValueError,
        match="corrected child trace is not exact to source",
    ):
        _compare(
            inputs,
            parent_fork_disposition="corrected-child-exact-to-source",
        )


def test_block0_child_npz_rejects_extra_or_missing_arrays(tmp_path):
    arrays = _child_arrays()
    arrays["unexpected"] = np.zeros((1,), dtype=np.float16)
    with pytest.raises(KeyError, match="unexpected"):
        write_decoder_level2_block0_trace_npz(tmp_path / "extra.npz", arrays)
    arrays.pop("unexpected")
    arrays.pop(CHILD_ARRAY_NAMES[-1])
    with pytest.raises(KeyError, match="missing"):
        write_decoder_level2_block0_trace_npz(tmp_path / "missing.npz", arrays)


def test_actual_decoder_reports_authenticated_affine_width256_norm():
    from scripts.run_mlx_decoder_level2_block0_trace import (
        _level2_block0_norm_route_identity,
    )
    from trellmlx.models.shape_slat_decoder import SLatDecoder

    decoder = SLatDecoder(out_channels=7, pred_subdiv=True, use_fp16=True)

    assert _level2_block0_norm_route_identity(decoder) == (
        LEVEL2_BLOCK0_NORM_BOUNDARY_ROUTE
    )


def test_block0_comparator_preserves_colliding_input_and_writes_failure(tmp_path):
    from scripts.compare_decoder_level2_block0_traces import main

    inputs = _valid_comparison_inputs(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    _write_json(receipt_path, inputs[0])
    source_before = inputs[1].read_bytes()
    lut_path, lut_sha256 = _local_lut_identity(inputs)

    rc = main(
        [
            "--source",
            str(inputs[1]),
            "--source-report",
            str(inputs[2]),
            "--local",
            str(inputs[3]),
            "--local-report",
            str(inputs[4]),
            "--parent-receipt",
            str(receipt_path),
            "--expected-parent-receipt-sha256",
            _sha256(receipt_path),
            "--turing-rsqrt-lut",
            str(lut_path),
            "--expected-turing-rsqrt-lut-sha256",
            lut_sha256,
            "--parent-fork-disposition",
            "historical-fork",
            "--output",
            str(inputs[1]),
        ]
    )

    failure = inputs[1].with_name(inputs[1].name + ".failure.json")
    assert rc == 1
    assert inputs[1].read_bytes() == source_before
    assert json.loads(failure.read_text())["failure_phase"] == (
        "request_validation"
    )


def test_block0_comparator_preserves_colliding_turing_lut_and_writes_failure(
    tmp_path,
):
    from scripts.compare_decoder_level2_block0_traces import main

    inputs = _valid_comparison_inputs(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    _write_json(receipt_path, inputs[0])
    lut_path, lut_sha256 = _local_lut_identity(inputs)
    lut_before = lut_path.read_bytes()

    rc = main(
        [
            "--source",
            str(inputs[1]),
            "--source-report",
            str(inputs[2]),
            "--local",
            str(inputs[3]),
            "--local-report",
            str(inputs[4]),
            "--parent-receipt",
            str(receipt_path),
            "--expected-parent-receipt-sha256",
            _sha256(receipt_path),
            "--turing-rsqrt-lut",
            str(lut_path),
            "--expected-turing-rsqrt-lut-sha256",
            lut_sha256,
            "--parent-fork-disposition",
            "historical-fork",
            "--output",
            str(lut_path),
        ]
    )

    failure = lut_path.with_name(lut_path.name + ".failure.json")
    assert rc == 1
    assert lut_path.read_bytes() == lut_before
    assert json.loads(failure.read_text())["failure_phase"] == (
        "request_validation"
    )


def test_block0_comparator_replaces_stale_output_with_durable_failure(tmp_path):
    from scripts.compare_decoder_level2_block0_traces import main

    inputs = _valid_comparison_inputs(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    inputs[0]["source_parent_report"]["sha256"] = "0" * 64
    _write_json(receipt_path, inputs[0])
    receipt_sha256 = _sha256(receipt_path)
    output = tmp_path / "comparison.json"
    _write_json(output, {"status": "done", "stale": True})
    lut_path, lut_sha256 = _local_lut_identity(inputs)

    rc = main(
        [
            "--source",
            str(inputs[1]),
            "--source-report",
            str(inputs[2]),
            "--local",
            str(inputs[3]),
            "--local-report",
            str(inputs[4]),
            "--parent-receipt",
            str(receipt_path),
            "--expected-parent-receipt-sha256",
            receipt_sha256,
            "--turing-rsqrt-lut",
            str(lut_path),
            "--expected-turing-rsqrt-lut-sha256",
            lut_sha256,
            "--parent-fork-disposition",
            "historical-fork",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["failure_phase"] == "comparison"
    assert report["first_nonexact_boundary"] is None

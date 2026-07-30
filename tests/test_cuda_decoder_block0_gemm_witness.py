import hashlib
import json
import sys

import numpy as np
import pytest


def _route_identity(*, rows=2, channels=4):
    return {
        "schema": "trellis2mlx.decoder_block0_gemm_input.v1",
        "operation": "shape_decoder.level0.block0.conv.center_gemm",
        "source_control": "alpha-1_beta-1",
        "kernel_index": 13,
        "logical_matrix_shape": [rows, channels, channels],
        "source_trace_sha256": "a" * 64,
        "local_fp16_trace_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "comparison_sha256": "d" * 64,
        "input_tensor_sha256": "e" * 64,
        "source_effective_route": {
            "route": "official-source-cuda-shape-decoder-level0-trace",
            "device_type": "cuda",
            "cuda_device": "Tesla T4",
            "sparse_conv_backend": "none",
            "decoder_level0_trace": True,
            "mesh_conversion": False,
            "raw_meshes": False,
        },
        "local_effective_route": {
            "route": "mlx-shape-decoder-level0-trace-fp16",
            "device_type": "metal",
            "decoder_precision": "fp16",
        },
    }


def _witness_arrays(*, singleton=True, duplicate_target=False):
    coords = np.asarray(
        [[0, 1, 1, 1], [0, 8, 8, 8]],
        dtype=np.int32,
    )
    if not singleton:
        coords[1] = np.asarray([0, 1, 1, 2], dtype=np.int32)
    if duplicate_target:
        coords[0] = coords[1]
    return {
        "coords": coords,
        "torso_input": np.ones((2, 4), dtype=np.float16),
        "center_weight": np.eye(4, dtype=np.float16),
        "bias": np.zeros(4, dtype=np.float16),
        "source_trace_row": np.ones(4, dtype=np.float16),
        "local_trace_row": np.ones(4, dtype=np.float16),
        "row_index": np.asarray(1, dtype=np.int32),
        "route_identity_json": np.asarray(
            json.dumps(_route_identity())
        ),
    }


def test_gemm_witness_requires_a_true_singleton_target():
    from scripts.cuda_decoder_block0_gemm_witness import (
        validate_witness_arrays,
    )

    with pytest.raises(ValueError, match="exactly one active neighbor"):
        validate_witness_arrays(
            _witness_arrays(singleton=False),
            expected_rows=2,
            channels=4,
            expected_row=1,
        )


def test_gemm_witness_rejects_duplicate_target_coordinate():
    from scripts.cuda_decoder_block0_gemm_witness import (
        validate_witness_arrays,
    )

    with pytest.raises(ValueError, match="duplicate coordinates"):
        validate_witness_arrays(
            _witness_arrays(duplicate_target=True),
            expected_rows=2,
            channels=4,
            expected_row=1,
        )


def test_gemm_witness_rejects_unbound_route_identity():
    from scripts.cuda_decoder_block0_gemm_witness import (
        validate_witness_arrays,
    )

    arrays = _witness_arrays()
    arrays["route_identity_json"] = np.asarray(
        json.dumps({"route": "test-singleton"})
    )

    with pytest.raises(ValueError, match="route_identity_json"):
        validate_witness_arrays(
            arrays,
            expected_rows=2,
            channels=4,
            expected_row=1,
        )


class _FakeTensor:
    def __init__(self, value, *, dtype):
        self.value = np.asarray(value, dtype=dtype)

    def to(self, *args, device=None, dtype=None):
        if args:
            device = args[0]
        del device
        target = self.value.dtype if dtype is None else dtype
        return _FakeTensor(self.value, dtype=target)

    def __matmul__(self, other):
        return _FakeTensor(self.value @ other.value, dtype=self.value.dtype)

    def __getitem__(self, index):
        return _FakeTensor(self.value[index], dtype=self.value.dtype)

    def __add__(self, other):
        return _FakeTensor(self.value + other.value, dtype=self.value.dtype)

    def detach(self):
        return self

    def numpy(self):
        return self.value


def test_fp32_variant_adds_bias_before_final_fp16_rounding():
    from scripts.cuda_decoder_block0_gemm_witness import (
        _fp32_gemm_bias_row,
    )

    x = _FakeTensor([[0.5001500248908997]], dtype=np.float32)
    weight = _FakeTensor([[1.0]], dtype=np.float32)
    bias = _FakeTensor([-0.01000213623046875], dtype=np.float16)

    result = _fp32_gemm_bias_row(
        x,
        weight,
        bias,
        row_index=0,
        float32_dtype=np.float32,
        float16_dtype=np.float16,
    )

    assert result.value.dtype == np.float16
    assert result.value.item() == np.float16(0.490234375)


def test_cuda_collection_preserves_variant_dtype_for_validation():
    from scripts.cuda_decoder_block0_gemm_witness import (
        _to_cpu_numpy_preserve_dtype,
    )

    result = _to_cpu_numpy_preserve_dtype(
        _FakeTensor([1.0], dtype=np.float32)
    )

    assert result.dtype == np.float32


def test_gemm_witness_authenticates_default_cuda_against_source_trace():
    from scripts.cuda_decoder_block0_gemm_witness import analyze_outputs

    source = np.asarray([1.0, 2.0], dtype=np.float16)
    local = np.asarray([1.0, 2.5], dtype=np.float16)
    outputs = {
        "cuda_default_full": source.copy(),
        "cuda_default_m1": local.copy(),
        "cuda_no_reduced_full": source.copy(),
        "cuda_fp32_full": source.copy(),
    }

    report = analyze_outputs(
        outputs=outputs,
        source_trace_row=source,
        local_trace_row=local,
    )

    assert report["self_authentication"]["default_full_exact_source"] is True
    assert report["variants"]["cuda_default_m1"]["vs_source"]["nonzero"] == 1
    assert report["variants"]["cuda_default_full"]["vs_local"]["nonzero"] == 1

    outputs["cuda_default_full"] = local.copy()
    with pytest.raises(ValueError, match="does not reproduce source trace"):
        analyze_outputs(
            outputs=outputs,
            source_trace_row=source,
            local_trace_row=local,
        )


def test_gemm_witness_rejects_cuda_output_dtype_substitution():
    from scripts.cuda_decoder_block0_gemm_witness import analyze_outputs

    source = np.asarray([1.0, 2.0], dtype=np.float16)
    outputs = {
        name: source.copy()
        for name in (
            "cuda_default_full",
            "cuda_default_m1",
            "cuda_no_reduced_full",
            "cuda_fp32_full",
        )
    }
    outputs["cuda_default_m1"] = outputs["cuda_default_m1"].astype(np.float32)

    with pytest.raises(ValueError, match="cuda_default_m1.*dtype"):
        analyze_outputs(
            outputs=outputs,
            source_trace_row=source,
            local_trace_row=source,
        )


def test_gemm_witness_rejects_wrong_digest_before_torch_and_clears_primary(
    monkeypatch,
    tmp_path,
):
    from scripts import cuda_decoder_block0_gemm_witness as witness

    input_path = tmp_path / "witness.npz"
    np.savez(input_path, **_witness_arrays())
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_decoder_block0_gemm_witness.py",
            "--witness",
            str(input_path),
            "--expected-witness-sha256",
            "0" * 64,
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
            "--expected-rows",
            "2",
            "--channels",
            "4",
            "--expected-row",
            "1",
        ],
    )

    assert witness.main() == 1

    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "input_validation"
    assert report["primary_output"]["exists"] is False
    assert "witness sha256 mismatch" in report["error"]
    assert "torch" not in sys.modules
    assert not output_npz.exists()
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() != "0" * 64

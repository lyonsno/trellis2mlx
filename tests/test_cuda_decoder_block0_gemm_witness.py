import hashlib
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

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


def test_gemm_witness_replaces_unsupported_strict_policy_with_direct_cublas():
    from scripts.cuda_decoder_block0_gemm_witness import VARIANT_NAMES

    assert "cuda_no_reduced_no_splitk_full" not in VARIANT_NAMES
    assert "cublas_default_tensor_op_full" in VARIANT_NAMES


def test_legacy_cublas_algorithm_ids_are_uncapped_complete():
    from scripts.cuda_decoder_block0_gemm_witness import (
        LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS,
    )

    assert LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS == (
        *range(0, 24),
        *range(100, 116),
    )


def test_cublas_gemm_ex_maps_row_major_product_to_column_major_call():
    from scripts.cuda_decoder_block0_gemm_witness import (
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        _invoke_cublas_gemm_ex,
    )

    class PointerTensor:
        def __init__(self, shape, pointer):
            self.shape = shape
            self._pointer = pointer

        def data_ptr(self):
            return self._pointer

        def is_contiguous(self):
            return True

    calls = []

    def gemm_ex(*args):
        calls.append(args)
        return 0

    status = _invoke_cublas_gemm_ex(
        gemm_ex,
        handle=44,
        x=PointerTensor((7697, 1024), 11),
        weight=PointerTensor((1024, 1024), 22),
        output=PointerTensor((7697, 1024), 33),
        algorithm_id=CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )

    assert status == 0
    assert len(calls) == 1
    call = calls[0]
    integer = lambda value: value.value
    pointer = lambda value: value.value
    assert pointer(call[0]) == 44
    assert (integer(call[1]), integer(call[2])) == (0, 0)
    assert (integer(call[3]), integer(call[4]), integer(call[5])) == (
        1024,
        7697,
        1024,
    )
    assert pointer(call[7]) == 22
    assert (integer(call[8]), integer(call[9])) == (2, 1024)
    assert pointer(call[10]) == 11
    assert (integer(call[11]), integer(call[12])) == (2, 1024)
    assert pointer(call[14]) == 33
    assert (integer(call[15]), integer(call[16])) == (2, 1024)
    assert integer(call[17]) == 0
    assert integer(call[18]) == CUBLAS_GEMM_DEFAULT_TENSOR_OP


def test_cublas_gemm_ex_can_publish_fp32_output_without_changing_inputs():
    from scripts.cuda_decoder_block0_gemm_witness import (
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        CUDA_R_32F,
        _invoke_cublas_gemm_ex,
    )

    class PointerTensor:
        def __init__(self, shape, pointer):
            self.shape = shape
            self._pointer = pointer

        def data_ptr(self):
            return self._pointer

        def is_contiguous(self):
            return True

    calls = []

    def gemm_ex(*args):
        calls.append(args)
        return 0

    status = _invoke_cublas_gemm_ex(
        gemm_ex,
        handle=44,
        x=PointerTensor((7697, 1024), 11),
        weight=PointerTensor((1024, 1024), 22),
        output=PointerTensor((7697, 1024), 33),
        output_cuda_type=CUDA_R_32F,
        algorithm_id=CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )

    assert status == 0
    assert len(calls) == 1
    call = calls[0]
    integer = lambda value: value.value
    assert (integer(call[8]), integer(call[11])) == (2, 2)
    assert integer(call[15]) == CUDA_R_32F
    assert integer(call[17]) == CUDA_R_32F


def test_sm75_compiler_evidence_requires_wmma_ptx_and_hmma_1688_sass():
    from scripts import cuda_decoder_block0_gemm_witness as witness

    classify = getattr(witness, "classify_sm75_compiler_evidence", None)
    assert callable(classify), "SM75 compiler classifier is not implemented"

    target_symbol = "_Z25sm75_wmma_gemm_kernelPK6__halfS1_Pfiii"
    evidence = classify(
        ptx="\n".join(
            [
                "Fatbin ptx code:",
                "arch = sm_75",
                ".target sm_75",
                f".visible .entry {target_symbol}(",
                ")",
                "{",
                (
                    "wmma.mma.sync.aligned.row.row.m16n16k16"
                    ".f32.f16.f16.f32"
                ),
                "}",
            ]
        ),
        sass="\n".join(
            [
                "Fatbin elf code:",
                "arch = sm_75",
                "code for sm_75",
                f"Function : {target_symbol}",
                "/*03d0*/ HMMA.1688.F32 R8, R12, R20, R8;",
            ]
        ),
    )
    assert evidence["effective_ptx_architecture"] == "sm_75"
    assert evidence["effective_cubin_architecture"] == "sm_75"
    assert evidence["ptx_target_symbol"] == target_symbol
    assert evidence["sass_target_symbol"] == target_symbol
    assert evidence["ptx_wmma_m16n16k16_count"] == 1
    assert evidence["sass_hmma_1688_count"] == 1

    unrelated_symbol = "_Z21unrelated_wmma_kernelv"
    with pytest.raises(ValueError, match="target.*PTX"):
        classify(
            ptx="\n".join(
                [
                    "arch = sm_75",
                    ".target sm_75",
                    f".visible .entry {unrelated_symbol}()",
                    "{",
                    (
                        "wmma.mma.sync.aligned.row.row.m16n16k16"
                        ".f32.f16.f16.f32"
                    ),
                    "}",
                ]
            ),
            sass="\n".join(
                [
                    "arch = sm_75",
                    "code for sm_75",
                    f"Function : {unrelated_symbol}",
                    "/*03d0*/ HMMA.1688.F32 R8, R12, R20, R8;",
                ]
            ),
        )

    with pytest.raises(ValueError, match="sm_75.*SASS"):
        classify(
            ptx="\n".join(
                [
                    "arch = sm_75",
                    ".target sm_75",
                    f".visible .entry {target_symbol}()",
                    "{",
                    (
                        "wmma.mma.sync.aligned.row.row.m16n16k16"
                        ".f32.f16.f16.f32"
                    ),
                    "}",
                ]
            ),
            sass="\n".join(
                [
                    "arch = sm_80",
                    "code for sm_80",
                    f"Function : {target_symbol}",
                    "/*03d0*/ HMMA.1688.F32 R8, R12, R20, R8;",
                ]
            ),
        )


def test_sm75_probe_arrays_require_exact_input_window_bytes():
    from scripts import cuda_decoder_block0_gemm_witness as witness

    validate = getattr(witness, "validate_sm75_mma_probe_arrays", None)
    assert callable(validate), "SM75 probe-array validation is not implemented"

    arrays = {
        "wmma_input_window": np.zeros((16, 32), dtype=np.float16),
        "cublas_tensor_fp16_unbiased_row": np.zeros(4, dtype=np.float16),
        "cublas_regular_fp16_unbiased_row": np.zeros(4, dtype=np.float16),
        "cublas_tensor_fp32_row": np.zeros(4, dtype=np.float32),
        "cublas_regular_fp32_row": np.zeros(4, dtype=np.float32),
        "wmma_fp32_row": np.zeros(4, dtype=np.float32),
    }
    validated = validate(arrays, reduction=32, channels=4)
    assert np.array_equal(validated["wmma_input_window"], arrays["wmma_input_window"])

    missing = dict(arrays)
    del missing["wmma_input_window"]
    with pytest.raises(ValueError, match="wmma_input_window"):
        validate(missing, reduction=32, channels=4)

    wrong_dtype = dict(arrays)
    wrong_dtype["wmma_input_window"] = np.zeros((16, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="wmma_input_window.*float16"):
        validate(wrong_dtype, reduction=32, channels=4)


def test_sm75_analysis_authenticates_tensor_and_regular_prebias_rows():
    from scripts import cuda_decoder_block0_gemm_witness as witness

    analyze = getattr(witness, "analyze_sm75_mma_outputs", None)
    assert callable(analyze), "SM75 MMA analysis is not implemented"

    bias = np.asarray([0.5, -0.25], dtype=np.float16)
    tensor = np.asarray([1.0, 2.0], dtype=np.float16)
    regular = np.asarray([1.5, 1.0], dtype=np.float16)
    outputs = {
        "cublas_tensor_fp16_unbiased_row": tensor,
        "cublas_regular_fp16_unbiased_row": regular,
        "cublas_tensor_fp32_row": tensor.astype(np.float32),
        "cublas_regular_fp32_row": regular.astype(np.float32),
        "wmma_fp32_row": tensor.astype(np.float32),
    }
    report = analyze(
        outputs=outputs,
        bias=bias,
        source_trace_row=(tensor + bias).astype(np.float16),
        local_trace_row=(regular + bias).astype(np.float16),
    )

    assert report["self_authentication"] == {
        "tensor_fp16_plus_bias_exact_source": True,
        "regular_fp16_plus_bias_exact_local": True,
    }
    assert report["localization"] == "sm75_wmma_chain_exact_cublas_tensor"
    assert report["wmma_fp32_vs_cublas_tensor_fp32"]["nonzero"] == 0

    outputs["cublas_regular_fp16_unbiased_row"] = tensor
    with pytest.raises(ValueError, match="regular.*local"):
        analyze(
            outputs=outputs,
            bias=bias,
            source_trace_row=(tensor + bias).astype(np.float16),
            local_trace_row=(regular + bias).astype(np.float16),
        )


def test_sm75_analysis_preserves_mainloop_or_epilogue_discriminator():
    from scripts import cuda_decoder_block0_gemm_witness as witness

    analyze = getattr(witness, "analyze_sm75_mma_outputs", None)
    assert callable(analyze), "SM75 MMA analysis is not implemented"

    tensor = np.asarray([1.0, 2.0], dtype=np.float16)
    regular = np.asarray([1.0, 2.5], dtype=np.float16)
    outputs = {
        "cublas_tensor_fp16_unbiased_row": tensor,
        "cublas_regular_fp16_unbiased_row": regular,
        "cublas_tensor_fp32_row": tensor.astype(np.float32),
        "cublas_regular_fp32_row": regular.astype(np.float32),
        "wmma_fp32_row": regular.astype(np.float32),
    }
    report = analyze(
        outputs=outputs,
        bias=np.zeros(2, dtype=np.float16),
        source_trace_row=tensor,
        local_trace_row=regular,
    )

    assert report["localization"] == "cublas_mainloop_or_epilogue_required"
    assert report["wmma_fp16_vs_cublas_tensor_fp16"]["nonzero"] == 1
    assert report["wmma_fp16_vs_cublas_regular_fp16"]["nonzero"] == 0


def test_cublas_sweep_records_every_status_and_full_matrix_metric():
    from scripts.cuda_decoder_block0_gemm_witness import (
        CUBLAS_STATUS_NOT_SUPPORTED,
        _collect_cublas_algorithm_results,
    )

    calls = []
    candidates = {
        0: (0, np.asarray([1.0, 2.0], dtype=np.float16)),
        1: (CUBLAS_STATUS_NOT_SUPPORTED, None),
        100: (0, np.asarray([1.0, 2.5], dtype=np.float16)),
    }

    def run_algorithm(algorithm_id):
        calls.append(algorithm_id)
        return candidates[algorithm_id]

    def summarize_success(candidate):
        reference = np.asarray([1.0, 2.0], dtype=np.float16)
        local = np.asarray([1.0, 2.5], dtype=np.float16)
        metric = {
            "nonzero_vs_default_full": int(
                np.count_nonzero(candidate != reference)
            ),
            "max_abs_vs_default_full": float(
                np.max(np.abs(candidate.astype(np.float32) - reference))
            ),
            "nonzero_vs_local_full": int(
                np.count_nonzero(candidate != local)
            ),
            "max_abs_vs_local_full": float(
                np.max(np.abs(candidate.astype(np.float32) - local))
            ),
        }
        return metric, candidate

    arrays, report = _collect_cublas_algorithm_results(
        algorithm_ids=(0, 1, 100),
        run_algorithm=run_algorithm,
        summarize_success=summarize_success,
        channels=2,
    )

    assert calls == [0, 1, 100]
    np.testing.assert_array_equal(
        arrays["cublas_explicit_algorithm_ids"],
        np.asarray([0, 1, 100], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        arrays["cublas_explicit_statuses"],
        np.asarray([0, CUBLAS_STATUS_NOT_SUPPORTED, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        arrays["cublas_supported_algorithm_ids"],
        np.asarray([0, 100], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        arrays["cublas_supported_rows"],
        np.asarray([[1.0, 2.0], [1.0, 2.5]], dtype=np.float16),
    )
    np.testing.assert_array_equal(
        arrays["cublas_supported_nonzero_vs_default_full"],
        np.asarray([0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        arrays["cublas_supported_nonzero_vs_local_full"],
        np.asarray([1, 0], dtype=np.int64),
    )
    assert report["exact_match_algorithm_ids"] == [0]
    assert report["exact_local_match_algorithm_ids"] == [100]
    assert report["unsupported_algorithm_ids"] == [1]
    assert [entry["algorithm_id"] for entry in report["algorithms"]] == [
        0,
        1,
        100,
    ]


def test_reduction_policy_assignment_records_effective_splitk():
    from scripts.cuda_decoder_block0_gemm_witness import (
        _set_reduction_policy,
    )

    class FakeMatmulBackend:
        def __init__(self):
            object.__setattr__(self, "allow_reduced", True)
            object.__setattr__(self, "allow_splitk", True)
            object.__setattr__(self, "assignments", [])

        def __getattr__(self, name):
            if name == "allow_fp16_reduced_precision_reduction":
                return self.allow_reduced
            if name == "allow_fp16_reduced_precision_reduction_split_k":
                return self.allow_splitk
            raise AttributeError(name)

        def __setattr__(self, name, value):
            if name != "allow_fp16_reduced_precision_reduction":
                object.__setattr__(self, name, value)
                return
            if isinstance(value, bool):
                parsed = (value, True)
            else:
                parsed = tuple(value)
            self.assignments.append(parsed)
            object.__setattr__(self, "allow_reduced", parsed[0])
            object.__setattr__(self, "allow_splitk", parsed[1])

    backend = FakeMatmulBackend()

    identity = _set_reduction_policy(
        backend,
        allow_reduced_precision=False,
        allow_splitk=False,
    )

    assert backend.assignments == [(False, False)]
    assert identity == {
        "requested": {
            "allow_reduced_precision": False,
            "allow_splitk": False,
        },
        "effective": {
            "allow_reduced_precision": False,
            "allow_splitk": False,
        },
    }


def test_cuda_runtime_records_requested_and_effective_policy_per_variant(
    monkeypatch,
):
    from scripts import cuda_decoder_block0_gemm_witness as witness

    class FakeMatmulBackend:
        def __init__(self):
            object.__setattr__(self, "allow_reduced", True)
            object.__setattr__(self, "allow_splitk", True)

        def __getattr__(self, name):
            if name == "allow_fp16_reduced_precision_reduction":
                return self.allow_reduced
            if name == "allow_fp16_reduced_precision_reduction_split_k":
                return self.allow_splitk
            raise AttributeError(name)

        def __setattr__(self, name, value):
            if name != "allow_fp16_reduced_precision_reduction":
                object.__setattr__(self, name, value)
                return
            parsed = (value, True) if isinstance(value, bool) else tuple(value)
            object.__setattr__(self, "allow_reduced", parsed[0])
            object.__setattr__(self, "allow_splitk", parsed[1])

    backend = FakeMatmulBackend()
    fake_torch = SimpleNamespace(
        __version__=witness.EXPECTED_TORCH,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=backend)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: witness.EXPECTED_DEVICE,
            synchronize=lambda: None,
        ),
        float16=np.float16,
        float32=np.float32,
        from_numpy=lambda value: _FakeTensor(value, dtype=value.dtype),
        no_grad=nullcontext,
    )
    monkeypatch.setattr(
        witness,
        "_run_cublas_algorithm_sweep",
        lambda **kwargs: (
            {
                "cublas_default_tensor_op_full": (
                    kwargs["default_product"][kwargs["row_index"]]
                    + kwargs["bias"]
                ).numpy(),
                "cublas_explicit_algorithm_ids": np.asarray(
                    [0],
                    dtype=np.int32,
                ),
            },
            {"default_tensor_op_exact_pytorch_full": True},
        ),
    )
    monkeypatch.setattr(
        witness,
        "_run_sm75_mma_probe",
        lambda **kwargs: ({}, {"route": "test-sm75-wmma"}),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    arrays = _witness_arrays()
    local_full_product = (
        arrays["torso_input"] @ arrays["center_weight"]
    ).astype(np.float16)
    _, runtime = witness._run_cuda(arrays, local_full_product)

    expected = {
        "cuda_default_full": {
            "requested": {
                "allow_reduced_precision": True,
                "allow_splitk": True,
            },
            "effective": {
                "allow_reduced_precision": True,
                "allow_splitk": True,
            },
        },
        "cuda_default_m1": {
            "requested": {
                "allow_reduced_precision": True,
                "allow_splitk": True,
            },
            "effective": {
                "allow_reduced_precision": True,
                "allow_splitk": True,
            },
        },
        "cuda_no_reduced_full": {
            "requested": {
                "allow_reduced_precision": False,
                "allow_splitk": True,
            },
            "effective": {
                "allow_reduced_precision": False,
                "allow_splitk": True,
            },
        },
        "cublas_default_tensor_op_full": {
            "requested": {
                "route": "legacy_cublasGemmEx",
                "algorithm_id": 99,
            },
            "effective": {
                "route": "legacy_cublasGemmEx",
                "algorithm_id": 99,
                "exact_pytorch_default_full": True,
            },
        },
        "cuda_fp32_full": {
            "requested": {
                "allow_reduced_precision": True,
                "allow_splitk": True,
            },
            "effective": {
                "allow_reduced_precision": True,
                "allow_splitk": True,
            },
        },
    }
    assert runtime["variant_reduction_policies"] == expected


def test_gemm_witness_authenticates_default_cuda_against_source_trace():
    from scripts.cuda_decoder_block0_gemm_witness import analyze_outputs

    source = np.asarray([1.0, 2.0], dtype=np.float16)
    local = np.asarray([1.0, 2.5], dtype=np.float16)
    outputs = {
        "cuda_default_full": source.copy(),
        "cuda_default_m1": local.copy(),
        "cuda_no_reduced_full": source.copy(),
        "cublas_default_tensor_op_full": source.copy(),
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
            "cublas_default_tensor_op_full",
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
    local_product_path = tmp_path / "local-product.npz"
    np.savez(
        local_product_path,
        best_product=np.ones((2, 4), dtype=np.float16),
    )
    local_product_digest = hashlib.sha256(
        local_product_path.read_bytes()
    ).hexdigest()
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
            "--local-full-product",
            str(local_product_path),
            "--expected-local-full-product-sha256",
            local_product_digest,
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


def test_gemm_witness_invalidates_stale_outputs_before_hard_interrupt(
    monkeypatch,
    tmp_path,
):
    from scripts import cuda_decoder_block0_gemm_witness as witness

    input_path = tmp_path / "witness.npz"
    np.savez(input_path, **_witness_arrays())
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    local_product_path = tmp_path / "local-product.npz"
    np.savez(
        local_product_path,
        best_product=np.ones((2, 4), dtype=np.float16),
    )
    local_product_digest = hashlib.sha256(
        local_product_path.read_bytes()
    ).hexdigest()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_json.write_text('{"status":"done"}\n')
    output_npz.write_bytes(b"stale")
    monkeypatch.setattr(
        witness,
        "_run_cuda",
        lambda arrays, local: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        witness.main(
            [
                "--witness",
                str(input_path),
                "--expected-witness-sha256",
                digest,
                "--local-full-product",
                str(local_product_path),
                "--expected-local-full-product-sha256",
                local_product_digest,
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
            ]
        )

    assert not output_json.exists()
    assert not output_npz.exists()


def test_gemm_witness_removes_primary_on_postpublication_hard_interrupt(
    monkeypatch,
    tmp_path,
):
    from scripts import cuda_decoder_block0_gemm_witness as witness

    input_path = tmp_path / "witness.npz"
    np.savez(input_path, **_witness_arrays())
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    local_product_path = tmp_path / "local-product.npz"
    np.savez(
        local_product_path,
        best_product=np.ones((2, 4), dtype=np.float16),
    )
    local_product_digest = hashlib.sha256(
        local_product_path.read_bytes()
    ).hexdigest()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    source_row = _witness_arrays()["source_trace_row"]
    outputs = {
        name: source_row.copy()
        for name in witness.VARIANT_NAMES
    }
    outputs.update(
        {
            "wmma_input_window": np.zeros((16, 4), dtype=np.float16),
            "cublas_tensor_fp16_unbiased_row": source_row.copy(),
            "cublas_regular_fp16_unbiased_row": source_row.copy(),
            "cublas_tensor_fp32_row": source_row.astype(np.float32),
            "cublas_regular_fp32_row": source_row.astype(np.float32),
            "wmma_fp32_row": source_row.astype(np.float32),
        }
    )
    monkeypatch.setattr(
        witness,
        "_run_cuda",
        lambda arrays, local: (outputs, {}),
    )
    load_npz = witness._load_npz

    def interrupt_output_reopen(path):
        if path == output_npz:
            raise KeyboardInterrupt("interrupt during output reopen")
        return load_npz(path)

    monkeypatch.setattr(witness, "_load_npz", interrupt_output_reopen)

    with pytest.raises(KeyboardInterrupt, match="output reopen"):
        witness.main(
            [
                "--witness",
                str(input_path),
                "--expected-witness-sha256",
                digest,
                "--local-full-product",
                str(local_product_path),
                "--expected-local-full-product-sha256",
                local_product_digest,
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
            ]
        )

    assert not output_json.exists()
    assert not output_npz.exists()


def test_gemm_witness_rejects_wrong_local_product_digest_before_torch(
    monkeypatch,
    tmp_path,
):
    from scripts import cuda_decoder_block0_gemm_witness as witness

    input_path = tmp_path / "witness.npz"
    np.savez(input_path, **_witness_arrays())
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    local_product_path = tmp_path / "local-product.npz"
    np.savez(
        local_product_path,
        best_product=np.ones((2, 4), dtype=np.float16),
    )
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert witness.main(
        [
            "--witness",
            str(input_path),
            "--expected-witness-sha256",
            digest,
            "--local-full-product",
            str(local_product_path),
            "--expected-local-full-product-sha256",
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
        ]
    ) == 1

    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "input_validation"
    assert "local full product sha256 mismatch" in report["error"]
    assert "torch" not in sys.modules
    assert not output_npz.exists()

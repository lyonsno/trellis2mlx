from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_cuda_route_identity_rejects_non_cuda_effective_device():
    from scripts.source_cuda_sparse_block_trace import build_route_identity

    with pytest.raises(ValueError, match="CUDA"):
        build_route_identity(
            effective_device_type="cpu",
            source_steps=Path("/inputs/source_cuda_steps.npz"),
            conditioning=Path("/inputs/conditioning.npz"),
            checkpoint=Path("/inputs/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
            source_root=Path("/work/source"),
            branch="pos",
            step_index=2,
            block_indices=(4,),
            sparse_conv_backend="none",
            sparse_attn_backend="sdpa",
        )


def test_cuda_route_identity_records_effective_route_and_inputs():
    from scripts.source_cuda_sparse_block_trace import build_route_identity

    identity = build_route_identity(
        effective_device_type="cuda",
        source_steps=Path("/inputs/source_cuda_steps.npz"),
        conditioning=Path("/inputs/conditioning.npz"),
        checkpoint=Path("/inputs/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
        source_root=Path("/work/source"),
        branch="neg",
        step_index=6,
        block_indices=(3, 4),
        sparse_conv_backend="none",
        sparse_attn_backend="sdpa",
    )

    assert identity["requested_route"] == "source-cuda-sparse-block-trace"
    assert identity["effective_route"] == "official-trellis2-source-cuda-sparse-flow-block-trace"
    assert identity["effective_device_type"] == "cuda"
    assert identity["branch"] == "neg"
    assert identity["step_index"] == 6
    assert identity["block_indices"] == [3, 4]
    assert identity["source_steps"] == "/inputs/source_cuda_steps.npz"
    assert identity["conditioning"] == "/inputs/conditioning.npz"
    assert identity["forced_env"] == {
        "SPARSE_CONV_BACKEND": "none",
        "SPARSE_ATTN_BACKEND": "sdpa",
        "ATTN_BACKEND": "sdpa",
    }
    assert "not a captured-MLX-block-input replay" in identity["forbidden_inferences"]


def test_select_branch_conditioning_rejects_unknown_branch():
    from scripts.source_cuda_sparse_block_trace import select_branch_conditioning_key

    assert select_branch_conditioning_key("pos") == "cond"
    assert select_branch_conditioning_key("neg") == "neg_cond"
    with pytest.raises(ValueError, match="branch"):
        select_branch_conditioning_key("cfg")


def test_split_block_modulation_adds_share_mod_block_offset():
    from scripts.source_cuda_sparse_block_trace import split_block_modulation

    class Tensorish:
        def __init__(self, array):
            self.array = np.asarray(array, dtype=np.float32)
            self.dtype = self.array.dtype

        def type(self, _dtype):
            return self

        def chunk(self, count, dim):
            assert count == 6
            assert dim == 1
            return tuple(Tensorish(part) for part in np.split(self.array, count, axis=dim))

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

        def __add__(self, other):
            return Tensorish(self.array + other.array)

    block = SimpleNamespace(
        share_mod=True,
        modulation=Tensorish(np.arange(12, dtype=np.float32).reshape(1, 12)),
    )
    shared = Tensorish(np.ones((1, 12), dtype=np.float32))

    pieces = split_block_modulation(block, shared)

    assert len(pieces) == 6
    np.testing.assert_array_equal(pieces[0].array, np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_array_equal(pieces[-1].array, np.array([[11.0, 12.0]], dtype=np.float32))


def test_parse_trace_names_supports_compact_alias_and_rejects_unknown_names():
    from scripts.source_cuda_sparse_block_trace import TRACE_NAMES, parse_trace_names

    assert parse_trace_names(None) == TRACE_NAMES
    assert parse_trace_names("all") == TRACE_NAMES
    assert parse_trace_names("compact") == ("input", "after_self", "after_cross", "after_mlp")
    assert parse_trace_names("input, after_self") == ("input", "after_self")
    with pytest.raises(ValueError, match="unknown trace"):
        parse_trace_names("input,not_a_real_trace")


def test_parse_step_indices_supports_csv_and_rejects_duplicates_or_bounds():
    from scripts.source_cuda_sparse_block_trace import parse_step_indices

    assert parse_step_indices(None, step_index=2, steps=8) == (2,)
    assert parse_step_indices("0, 2,7", step_index=99, steps=8) == (0, 2, 7)
    with pytest.raises(ValueError, match="duplicates"):
        parse_step_indices("2,2", step_index=0, steps=8)
    with pytest.raises(ValueError, match="outside"):
        parse_step_indices("8", step_index=0, steps=8)


def test_cuda_route_identity_records_multi_step_indices():
    from scripts.source_cuda_sparse_block_trace import build_route_identity

    identity = build_route_identity(
        effective_device_type="cuda",
        source_steps=Path("/inputs/source_cuda_steps.npz"),
        conditioning=Path("/inputs/conditioning.npz"),
        checkpoint=Path("/inputs/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
        source_root=Path("/work/source"),
        branch="pos",
        step_index=2,
        step_indices=(0, 1, 2),
        block_indices=(0,),
        sparse_conv_backend="none",
        sparse_attn_backend="sdpa",
    )

    assert identity["step_index"] == 2
    assert identity["step_indices"] == [0, 1, 2]


def test_trace_array_key_step_qualifies_multi_step_outputs_only():
    from scripts.source_cuda_sparse_block_trace import trace_array_key

    assert (
        trace_array_key("pos", step_index=2, block_index=0, name="norm1", multistep=False)
        == "pos_block0_norm1"
    )
    assert (
        trace_array_key("pos", step_index=2, block_index=0, name="norm1", multistep=True)
        == "step2_pos_block0_norm1"
    )


def test_saved_source_comparison_marks_branch_only_sample_next_as_non_route_identity():
    from scripts.source_cuda_sparse_block_trace import compare_saved_source_outputs

    arrays = {
        "pos_final_output": np.array([[[[[2.0]]]]], dtype=np.float32),
    }
    optional_source = {
        "pred_pos": np.array([[[[[[2.0]]]]]], dtype=np.float32),
        "sample_next": np.array([[[[[[0.25]]]]]], dtype=np.float32),
    }
    sample_in = np.array([[[[[1.0]]]]], dtype=np.float32)

    report = compare_saved_source_outputs(
        branch="pos",
        arrays=arrays,
        optional_source=optional_source,
        step_index=0,
        sample_in_np=sample_in,
        t=0.5,
        t_prev=0.25,
    )

    assert "sample_next_from_pred_vs_source_steps_sample_next" not in report
    assert "pos_branch_only_sample_next_from_pred" in arrays
    branch_report = report["branch_only_sample_next_vs_source_steps_sample_next"]
    assert branch_report["route_identity_evidence"] is False
    assert branch_report["comparison_class"] == "branch_only_euler_vs_saved_cfg_or_scheduler_sample_next"
    np.testing.assert_allclose(arrays["pos_branch_only_sample_next_from_pred"], 0.5)

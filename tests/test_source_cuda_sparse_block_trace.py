from pathlib import Path

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

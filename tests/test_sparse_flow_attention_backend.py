from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import pytest


@pytest.fixture(autouse=True)
def _restore_sparse_attention_backend():
    from trellmlx.sparse_flow_attention import (
        DEFAULT_BACKEND,
        configure_sparse_flow_attention_backend,
    )

    configure_sparse_flow_attention_backend(DEFAULT_BACKEND)
    yield
    configure_sparse_flow_attention_backend(DEFAULT_BACKEND)


def test_sparse_source_attention_route_is_independent_from_global_env(
    monkeypatch,
):
    from trellmlx import sparse_flow_attention

    calls = []

    def explicit_route(q, k, v, mask=None, **route):
        calls.append(route)
        return mx.zeros(q.shape, dtype=q.dtype)

    monkeypatch.setenv("TRELLIS2MLX_ATTENTION_BACKEND", "fast")
    monkeypatch.setattr(
        sparse_flow_attention,
        "scaled_dot_product_attention_for_backend",
        explicit_route,
    )
    sparse_flow_attention.configure_sparse_flow_attention_backend(
        sparse_flow_attention.SOURCE_CUDA_MATH_TURING_T4_BACKEND
    )
    q = mx.ones((1, 1, 2, 4), dtype=mx.bfloat16)

    actual = sparse_flow_attention.scaled_dot_product_attention(q, q, q)
    mx.eval(actual)

    assert calls == [
        {
            "backend": "source-cuda-self",
            "softmax_backend": "source-cuda-turing",
            "value_backend": "source-cuda-sequential",
        }
    ]


def test_sparse_structure_model_consumes_sparse_attention_route():
    from trellmlx.models import sparse_structure_flow

    assert (
        sparse_structure_flow.scaled_dot_product_attention.__module__
        == "trellmlx.sparse_flow_attention"
    )


def test_sparse_source_attention_identity_names_authenticated_schedule():
    from trellmlx.sparse_flow_attention import (
        SOURCE_CUDA_MATH_TURING_T4_BACKEND,
        configure_sparse_flow_attention_backend,
        sparse_flow_attention_backend_identity,
    )

    configure_sparse_flow_attention_backend(
        SOURCE_CUDA_MATH_TURING_T4_BACKEND
    )
    identity = sparse_flow_attention_backend_identity()

    assert identity["backend"] == SOURCE_CUDA_MATH_TURING_T4_BACKEND
    assert identity["self_attention_width"] == 4096
    assert identity["softmax_threads"] == 1024
    assert identity["softmax_registers_per_thread"] == 4
    assert identity["source_runtime"] == "torch-2.10.0+cu128"


def test_generate_exposes_distinct_sparse_flow_attention_selector():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "generate.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--sparse-flow-attention-backend" in result.stdout

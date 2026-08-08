import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


class _Parser:
    def error(self, message):
        raise argparse.ArgumentError(None, message)


def _args(**overrides):
    values = {
        "decoder_linear_backend": "native",
        "decoder_sparse_conv_matmul_backend": "native",
        "decoder_layernorm_backend": "mlx-fast-layer-norm",
        "decoder_silu_backend": "mlx-native",
        "decoder_silu_lut": None,
        "expected_decoder_silu_lut_sha256": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_exact_decoder_route_enrolls_every_authenticated_backend(monkeypatch):
    import generate

    layernorm_calls = []
    silu_calls = []
    monkeypatch.setattr(
        generate,
        "configure_decoder_layernorm_backend",
        lambda *args, **kwargs: layernorm_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        generate,
        "configure_decoder_silu_backend",
        lambda *args, **kwargs: silu_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        generate,
        "decoder_layernorm_backend_identity",
        lambda: {"backend": "cuda-welford-turing-t4"},
    )
    monkeypatch.setattr(
        generate,
        "decoder_silu_backend_identity",
        lambda: {"backend": "cuda-turing-t4-fp16-lut"},
    )
    monkeypatch.delenv("TRELLIS2MLX_DECODER_LINEAR_BACKEND", raising=False)
    monkeypatch.delenv(
        "TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND", raising=False
    )
    digest = "d" * 64
    silu_digest = "e" * 64
    turing_rsqrt_lut = object()

    route = generate._configure_decoder_route(
        _args(
            decoder_linear_backend="turing_fda",
            decoder_sparse_conv_matmul_backend="turing_fda",
            decoder_layernorm_backend="cuda-welford-turing-t4",
            decoder_silu_backend="cuda-turing-t4-fp16-lut",
            decoder_silu_lut="/evidence/silu.npz",
            expected_decoder_silu_lut_sha256=silu_digest,
        ),
        _Parser(),
        turing_rsqrt_lut=turing_rsqrt_lut,
        turing_rsqrt_lut_sha256=digest,
    )

    assert os.environ["TRELLIS2MLX_DECODER_LINEAR_BACKEND"] == "turing_fda"
    assert os.environ["TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND"] == "turing_fda"
    assert layernorm_calls == [
        (
            ("cuda-welford-turing-t4",),
            {
                "turing_rsqrt_delta_lut": turing_rsqrt_lut,
                "turing_rsqrt_lut_artifact_sha256_attested": digest,
            },
        )
    ]
    assert silu_calls == [
        (
            ("cuda-turing-t4-fp16-lut",),
            {
                "output_lut_artifact_path": "/evidence/silu.npz",
                "output_lut_artifact_sha256_attested": silu_digest,
            },
        )
    ]
    assert route == {
        "decoder_linear_backend": "turing_fda",
        "sparse_conv_matmul_backend": "turing_fda",
        "decoder_layernorm": {"backend": "cuda-welford-turing-t4"},
        "decoder_silu": {"backend": "cuda-turing-t4-fp16-lut"},
        "decoder_output_head_backend": "mlx-native-fp32",
    }


def test_exact_decoder_layernorm_requires_loaded_rsqrt_evidence():
    import generate

    with pytest.raises(argparse.ArgumentError, match="requires the Turing rsqrt"):
        generate._configure_decoder_route(
            _args(decoder_layernorm_backend="cuda-welford-turing-t4"),
            _Parser(),
            turing_rsqrt_lut=None,
            turing_rsqrt_lut_sha256=None,
        )


def test_exact_decoder_silu_requires_both_artifact_coordinates():
    import generate

    with pytest.raises(
        argparse.ArgumentError,
        match="requires --decoder-silu-lut and "
        "--expected-decoder-silu-lut-sha256",
    ):
        generate._configure_decoder_route(
            _args(decoder_silu_backend="cuda-turing-t4-fp16-lut"),
            _Parser(),
            turing_rsqrt_lut=None,
            turing_rsqrt_lut_sha256=None,
        )


def test_native_decoder_silu_rejects_unconsumed_artifact_identity():
    import generate

    with pytest.raises(
        argparse.ArgumentError,
        match="only apply to cuda-turing-t4-fp16-lut",
    ):
        generate._configure_decoder_route(
            _args(
                decoder_silu_lut="/evidence/silu.npz",
                expected_decoder_silu_lut_sha256="e" * 64,
            ),
            _Parser(),
            turing_rsqrt_lut=None,
            turing_rsqrt_lut_sha256=None,
        )


def test_decoder_consumers_record_effective_decoder_route():
    source = (Path(__file__).parents[1] / "generate.py").read_text()

    assert source.count("decoder_route_json=np.array(") == 3


def test_generate_cli_exposes_all_decoder_route_coordinates():
    source = (Path(__file__).parents[1] / "generate.py").read_text()

    for option in (
        "--decoder-linear-backend",
        "--decoder-sparse-conv-matmul-backend",
        "--decoder-layernorm-backend",
        "--decoder-silu-backend",
        "--decoder-silu-lut",
        "--expected-decoder-silu-lut-sha256",
    ):
        assert source.count(f'"{option}"') >= 1

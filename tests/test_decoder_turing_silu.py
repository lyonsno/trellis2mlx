from __future__ import annotations

import hashlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


FIXTURES = Path(__file__).with_name("fixtures")
LUT_ARTIFACT_SHA256 = (
    "f159f871c22483bdf7d299ad8a31822b4e84dea0862d4feae2fcfd262e59d622"
)


def _load_lut():
    with np.load(
        FIXTURES / "source_cuda_t4_silu_fp16_lut.npz",
        allow_pickle=False,
    ) as fixture:
        input_bits = np.asarray(fixture["input_bits"])
        output_bits = np.asarray(fixture["output_bits"])
    assert np.array_equal(input_bits, np.arange(1 << 16, dtype=np.uint16))
    return input_bits, output_bits


@pytest.fixture(autouse=True)
def _restore_decoder_silu_backend():
    from trellmlx.decoder_turing_silu import (
        DEFAULT_BACKEND,
        configure_decoder_silu_backend,
    )

    configure_decoder_silu_backend(DEFAULT_BACKEND)
    yield
    configure_decoder_silu_backend(DEFAULT_BACKEND)


def test_turing_silu_lut_matches_all_finite_fp16_inputs():
    from trellmlx.decoder_turing_silu import turing_silu_fp16

    input_bits, output_bits = _load_lut()
    finite = np.isfinite(input_bits.view(np.float16))
    values = input_bits[finite].view(np.float16)
    expected = output_bits[finite]

    actual = turing_silu_fp16(
        mx.array(values),
        mx.array(output_bits),
    )
    mx.eval(actual)

    np.testing.assert_array_equal(np.asarray(actual).view(np.uint16), expected)


def test_turing_silu_backend_requires_attested_lut_and_reports_identity():
    from trellmlx.decoder_turing_silu import (
        CUDA_TURING_T4_LUT_BACKEND,
        configure_decoder_silu_backend,
        decoder_silu_backend_identity,
    )

    with pytest.raises(ValueError, match="requires an explicit output LUT"):
        configure_decoder_silu_backend(CUDA_TURING_T4_LUT_BACKEND)

    artifact_path = FIXTURES / "source_cuda_t4_silu_fp16_lut.npz"
    _, output_bits = _load_lut()
    configure_decoder_silu_backend(
        CUDA_TURING_T4_LUT_BACKEND,
        output_lut_artifact_path=artifact_path,
        output_lut_artifact_sha256_attested=LUT_ARTIFACT_SHA256,
    )

    identity = decoder_silu_backend_identity()
    assert identity["backend"] == CUDA_TURING_T4_LUT_BACKEND
    assert identity["cuda_architecture"] == "sm_75"
    assert identity["authenticated_contract"] == {
        "input_dtype": "float16",
        "output_dtype": "float16",
        "domain": "all-65536-bit-patterns",
    }
    assert identity["output_lut_artifact_sha256_attested"] == LUT_ARTIFACT_SHA256
    assert identity["output_lut_artifact_sha256_effective"] == LUT_ARTIFACT_SHA256
    assert identity["output_lut_artifact_path"] == str(artifact_path.resolve())
    assert identity["output_lut_content_sha256"] == hashlib.sha256(
        output_bits.tobytes()
    ).hexdigest()


def test_turing_silu_backend_rejects_substituted_artifact_bytes(tmp_path):
    from trellmlx.decoder_turing_silu import (
        CUDA_TURING_T4_LUT_BACKEND,
        configure_decoder_silu_backend,
    )

    input_bits, output_bits = _load_lut()
    substituted = output_bits.copy()
    substituted[0x3555] ^= np.uint16(1)
    artifact = tmp_path / "substituted.npz"
    np.savez(artifact, input_bits=input_bits, output_bits=substituted)
    effective_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert effective_sha256 != LUT_ARTIFACT_SHA256

    with pytest.raises(ValueError, match="artifact SHA256 mismatch"):
        configure_decoder_silu_backend(
            CUDA_TURING_T4_LUT_BACKEND,
            output_lut_artifact_path=artifact,
            output_lut_artifact_sha256_attested=LUT_ARTIFACT_SHA256,
        )


def test_decoder_silu_consumer_dispatches_configured_backend():
    from trellmlx.decoder_turing_silu import (
        CUDA_TURING_T4_LUT_BACKEND,
        configure_decoder_silu_backend,
    )
    from trellmlx.models.shape_slat_decoder import _decoder_silu

    input_bits, output_bits = _load_lut()
    selected = input_bits[np.array([0, 1, 0x3555, 0xB955, 0x7BFF])]
    configure_decoder_silu_backend(
        CUDA_TURING_T4_LUT_BACKEND,
        output_lut_artifact_path=FIXTURES / "source_cuda_t4_silu_fp16_lut.npz",
        output_lut_artifact_sha256_attested=LUT_ARTIFACT_SHA256,
    )

    actual = _decoder_silu(mx.array(selected.view(np.float16)))
    mx.eval(actual)

    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16),
        output_bits[selected],
    )

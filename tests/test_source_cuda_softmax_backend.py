import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "source_cuda_softmax_rows_7697.npz"
)
CROSS_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "source_cuda_softmax_rows_1029.npz"
)
BOUNDARY_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "source_cuda_softmax_boundary_row_7697.npz"
)
UPPER_BOUNDARY_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "source_cuda_softmax_upper_boundary_row_7697.npz"
)
NEGATIVE_REDUCED_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "source_cuda_softmax_negative_reduced_row_7697.npz"
)
EXPECTED_KEYS = {
    "selected_rows",
    "row_tokens",
    "row_heads",
    "scores_fp32",
    "probs_fp32",
    "exponents_fp32",
    "thread_sums_fp32",
    "warp_sums_fp32",
    "row_maxes_fp32",
    "row_sums_fp32",
    "route_identity_json",
}
SOURCE_ORACLE_SHA256 = (
    "7658bc6ef52e8c9ab42c388a76242785"
    "7f229e5b7f4ea28592c5d213b965d8e5"
)
SOURCE_ORACLE_SCRIPT_SHA256 = (
    "044725b9bfee3e43695def211cba81d71"
    "1f98b8577528476cf01990500b80a22"
)
SOURCE_ORACLE_CUDA_SHA256 = (
    "c338b2dbbec7e3adc7f88a6fb07d3df7"
    "74fea39c7162c59c9dd42c7830966464"
)
CROSS_FIXTURE_SHA256 = (
    "2f8e3e9c6d932128680b8b5283d5e63a"
    "806be6790ff1f55792445388e64e8e80"
)
BOUNDARY_FIXTURE_SHA256 = {
    BOUNDARY_FIXTURE.name: (
        "95f0ecbe9f7b70dcfd9d031d3aaa43fc"
        "87e5462fb9d26fa39d39e27d7b9f212e"
    ),
    UPPER_BOUNDARY_FIXTURE.name: (
        "d5ce291250b9cfea1595e8df173f9a6d"
        "7cb8cc299f4b1facbeee670df44764dc"
    ),
    NEGATIVE_REDUCED_FIXTURE.name: (
        "a5afb150ba14c11423cd8a7f0c62ad9d"
        "170abde1810d21fce2edabb5ba9290ed"
    ),
}


def _assert_boundary_fixture_digest(path: Path) -> None:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        BOUNDARY_FIXTURE_SHA256[path.name]
    )


def _load_fixture():
    with np.load(FIXTURE, allow_pickle=False) as loaded:
        assert set(loaded.files) == EXPECTED_KEYS
        return {name: loaded[name].copy() for name in loaded.files}


def test_fixture_preserves_authenticated_mlx_softmax_failure():
    fixture = _load_fixture()
    identity = json.loads(str(fixture["route_identity_json"].item()))

    assert identity == {
        "schema": "trellis2mlx.source_cuda_softmax_fixture.v1",
        "source_stage_sha256": (
            "5b972dbec717213bc9cc5fb01e99d3346"
            "c813bff4b78b9733f9c355f7451e51c"
        ),
        "source_oracle_sha256": (
            "7658bc6ef52e8c9ab42c388a76242785"
            "7f229e5b7f4ea28592c5d213b965d8e5"
        ),
        "source_oracle_script_sha256": (
            "044725b9bfee3e43695def211cba81d71"
            "1f98b8577528476cf01990500b80a22"
        ),
        "selected_rows": [182, 1059, 1261, 3821],
        "selection": [
            "max_abs",
            "max_nonzero",
            "min_nonzero",
            "last_control",
        ],
        "width": 7697,
    }
    assert fixture["scores_fp32"].shape == (4, 7697)
    assert fixture["scores_fp32"].dtype == np.float32
    assert fixture["probs_fp32"].shape == (4, 7697)
    assert fixture["probs_fp32"].dtype == np.float32
    assert fixture["exponents_fp32"].shape == (4, 7697)
    assert fixture["thread_sums_fp32"].shape == (4, 1024)
    assert fixture["warp_sums_fp32"].shape == (4, 32)

    actual = mx.softmax(mx.array(fixture["scores_fp32"]), axis=-1)
    mx.eval(actual)
    delta = np.abs(np.asarray(actual) - fixture["probs_fp32"])

    assert np.count_nonzero(delta, axis=1).tolist() == [
        7287,
        7697,
        1414,
        5343,
    ]
    assert delta.max(axis=1).tolist() == [
        2.980232238769531e-07,
        7.275957614183426e-11,
        2.9103830456733704e-11,
        2.3283064365386963e-10,
    ]


def test_source_cuda_long_row_softmax_matches_authenticated_fixture():
    from trellmlx.modules import attention

    fixture = _load_fixture()
    actual = attention._source_cuda_long_row_softmax(
        mx.array(fixture["scores_fp32"])
    )
    mx.eval(actual)

    assert actual.shape == (4, 7697)
    assert actual.dtype == mx.float32
    assert np.array_equal(np.asarray(actual), fixture["probs_fp32"])


def test_source_cuda_warp_softmax_matches_authenticated_cross_fixture():
    from trellmlx.modules import attention

    assert hashlib.sha256(CROSS_FIXTURE.read_bytes()).hexdigest() == (
        CROSS_FIXTURE_SHA256
    )
    with np.load(CROSS_FIXTURE, allow_pickle=False) as loaded:
        assert set(loaded.files) == {
            "scores_fp32",
            "probs_fp32",
            "route_identity_json",
        }
        scores = loaded["scores_fp32"].copy()
        expected = loaded["probs_fp32"].copy()
        identity = json.loads(str(loaded["route_identity_json"].item()))

    assert identity == {
        "schema": "trellis2mlx.source_cuda_softmax_fixture.v1",
        "source_stage_sha256": (
            "2170a74d970da09aa242853cf0f3e4c9"
            "52eaeac0150005e89eb7e8fb0dc74bb9"
        ),
        "source_report_sha256": (
            "d4472d4b5c49f324b49e4dce1ba774e"
            "548345b4cd3e9121d4c44161faf0b1282"
        ),
        "witness_sha256": (
            "2068dd5db21e26ec1cf50f6a83558bc"
            "eadd04d9205aa96ffc87618755b1f27c2"
        ),
        "selection_sha256": (
            "e9aee9373852d0ca28e4ffe7f2319ef"
            "c9864642577ddedfcd1ca40cf49e00316"
        ),
        "selected_rows": [654, 3, 1662, 2242],
        "query_tokens": [2319, 15, 6035, 7696],
        "heads": [6, 10, 8, 11],
        "selection": [
            "max_abs",
            "max_nonzero",
            "min_nonzero",
            "last_selected",
        ],
        "width": 1029,
        "cuda_schedule": {
            "next_power_of_two": 2048,
            "warp_size": 32,
            "warp_iterations": 64,
            "warps_per_block": 4,
            "reduction": "xor",
        },
    }
    assert scores.shape == (4, 1029)
    assert expected.shape == scores.shape

    actual = attention._source_cuda_long_row_softmax(mx.array(scores))
    mx.eval(actual)

    assert np.array_equal(np.asarray(actual), expected)


def test_source_cuda_softmax_matches_directed_rounding_boundary_row():
    from trellmlx.modules import attention

    _assert_boundary_fixture_digest(BOUNDARY_FIXTURE)
    with np.load(BOUNDARY_FIXTURE, allow_pickle=False) as loaded:
        assert set(loaded.files) == {
            "scores_fp32",
            "probs_fp32",
            "route_identity_json",
        }
        scores = loaded["scores_fp32"].copy()
        expected = loaded["probs_fp32"].copy()
        identity = json.loads(str(loaded["route_identity_json"].item()))

    assert identity == {
        "schema": "trellis2mlx.source_cuda_softmax_boundary_fixture.v1",
        "source_stage_sha256": (
            "5b972dbec717213bc9cc5fb01e99d3346"
            "c813bff4b78b9733f9c355f7451e51c"
        ),
        "source_oracle_sha256": SOURCE_ORACLE_SHA256,
        "source_oracle_script_sha256": SOURCE_ORACLE_SCRIPT_SHA256,
        "source_oracle_cuda_sha256": SOURCE_ORACLE_CUDA_SHA256,
        "selected_row": 36,
        "query_token": 1123,
        "head": 6,
        "width": 7697,
        "selection": (
            "first complete-census mismatch under pre-fix Metal "
            "directed-rounding approximation"
        ),
    }
    assert scores.shape == (1, 7697)
    assert expected.shape == scores.shape

    actual = attention._source_cuda_long_row_softmax(mx.array(scores))
    mx.eval(actual)

    assert np.array_equal(np.asarray(actual), expected)


def test_source_cuda_softmax_extreme_negative_inputs_underflow_to_exact_zero():
    from trellmlx.modules import attention

    scores = np.full((3, 7697), -90.0, dtype=np.float32)
    scores[1, :] = -1000.0
    scores[2, :] = -1.0e20
    scores[:, 0] = 0.0
    expected = np.zeros_like(scores)
    expected[:, 0] = 1.0

    actual = attention._source_cuda_long_row_softmax(mx.array(scores))
    mx.eval(actual)

    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_source_cuda_softmax_matches_upper_primary_boundary_row():
    from trellmlx.modules import attention

    _assert_boundary_fixture_digest(UPPER_BOUNDARY_FIXTURE)
    with np.load(UPPER_BOUNDARY_FIXTURE, allow_pickle=False) as loaded:
        scores = loaded["scores_fp32"].copy()
        expected = loaded["probs_fp32"].copy()
        identity = json.loads(str(loaded["route_identity_json"].item()))

    assert identity == {
        "schema": "trellis2mlx.source_cuda_softmax_boundary_fixture.v1",
        "source_stage_sha256": (
            "5b972dbec717213bc9cc5fb01e99d3346"
            "c813bff4b78b9733f9c355f7451e51c"
        ),
        "source_oracle_sha256": SOURCE_ORACLE_SHA256,
        "source_oracle_script_sha256": SOURCE_ORACLE_SCRIPT_SHA256,
        "source_oracle_cuda_sha256": SOURCE_ORACLE_CUDA_SHA256,
        "selected_row": 2036,
        "query_token": 1816,
        "head": 2,
        "width": 7697,
        "selection": (
            "first complete-census row exposing a reduced EX2 input "
            "at the upper primary boundary"
        ),
    }

    actual = attention._source_cuda_long_row_softmax(mx.array(scores))
    mx.eval(actual)

    assert np.array_equal(np.asarray(actual), expected)


def test_source_cuda_softmax_matches_negative_reduced_boundary_row():
    from trellmlx.modules import attention

    _assert_boundary_fixture_digest(NEGATIVE_REDUCED_FIXTURE)
    with np.load(NEGATIVE_REDUCED_FIXTURE, allow_pickle=False) as loaded:
        scores = loaded["scores_fp32"].copy()
        expected = loaded["probs_fp32"].copy()
        identity = json.loads(str(loaded["route_identity_json"].item()))

    assert identity == {
        "schema": "trellis2mlx.source_cuda_softmax_boundary_fixture.v1",
        "source_stage_sha256": (
            "5b972dbec717213bc9cc5fb01e99d3346"
            "c813bff4b78b9733f9c355f7451e51c"
        ),
        "source_oracle_sha256": SOURCE_ORACLE_SHA256,
        "source_oracle_script_sha256": SOURCE_ORACLE_SCRIPT_SHA256,
        "source_oracle_cuda_sha256": SOURCE_ORACLE_CUDA_SHA256,
        "selected_rows": [98, 76],
        "query_tokens": [3339, 491],
        "heads": [1, 6],
        "width": 7697,
        "selection": [
            (
                "first post-directed-rounding residual row with a "
                "negative reduced EX2 argument"
            ),
            (
                "first residual row whose negative reduced EX2 "
                "argument rounds to exactly one"
            ),
        ],
    }
    assert scores.shape == (2, 7697)
    assert expected.shape == scores.shape

    actual = attention._source_cuda_long_row_softmax(mx.array(scores))
    mx.eval(actual)

    assert np.array_equal(np.asarray(actual), expected)

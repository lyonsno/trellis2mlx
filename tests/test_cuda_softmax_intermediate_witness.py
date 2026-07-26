import json
import os

import numpy as np
import pytest


def _valid_intermediates():
    scores = np.array(
        [[0.0, -1.0, -2.0], [-2.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    probs = np.array(
        [
            [0.66524094, 0.24472848, 0.09003057],
            [0.09003057, 0.66524094, 0.24472848],
        ],
        dtype=np.float32,
    )
    exponents = np.array(
        [
            [1.0, 0.36787945, 0.13533528],
            [0.13533528, 1.0, 0.36787945],
        ],
        dtype=np.float32,
    )
    thread_sums = np.zeros((2, 32), dtype=np.float32)
    thread_sums[:, :3] = exponents
    warp_sums = exponents.sum(axis=1, dtype=np.float32).reshape(2, 1)
    row_maxes = np.zeros((2,), dtype=np.float32)
    row_sums = warp_sums.reshape(2)
    return {
        "scores": scores,
        "persisted_probs": probs,
        "live_native_probs": probs.copy(),
        "custom_probs": probs.copy(),
        "captured_probs": probs.copy(),
        "exponents": exponents,
        "thread_sums": thread_sums,
        "warp_sums": warp_sums,
        "row_maxes": row_maxes,
        "row_sums": row_sums,
    }


def test_requested_stage_identity_requires_exact_lowercase_sha256():
    from scripts.cuda_softmax_intermediate_witness import (
        requested_stage_identity,
    )

    digest = "a" * 64
    assert requested_stage_identity(digest) == {
        "sha256": digest,
        "rows": 3822,
        "width": 7697,
        "dtype": "float32",
    }
    for invalid in (None, "", "A" * 64, "a" * 63, "g" * 64):
        with pytest.raises(ValueError, match="stage sha256"):
            requested_stage_identity(invalid)


def test_oracle_rejects_substituted_persisted_native_probabilities():
    from scripts.cuda_softmax_intermediate_witness import (
        analyze_softmax_oracle,
    )

    payload = _valid_intermediates()
    payload["persisted_probs"] = payload["persisted_probs"].copy()
    payload["persisted_probs"][0, 0] = np.nextafter(
        payload["persisted_probs"][0, 0],
        np.float32(0.0),
    )

    with pytest.raises(
        ValueError,
        match="persisted probabilities do not reproduce live native softmax",
    ):
        analyze_softmax_oracle(
            **payload,
            expected_shape=(2, 3),
            expected_threads=32,
            expected_warps=1,
        )


def test_oracle_rejects_nonexact_custom_softmax():
    from scripts.cuda_softmax_intermediate_witness import (
        analyze_softmax_oracle,
    )

    payload = _valid_intermediates()
    payload["custom_probs"] = payload["custom_probs"].copy()
    payload["custom_probs"][1, 2] = np.nextafter(
        payload["custom_probs"][1, 2],
        np.float32(1.0),
    )

    with pytest.raises(
        ValueError,
        match="custom softmax does not reproduce live native softmax",
    ):
        analyze_softmax_oracle(
            **payload,
            expected_shape=(2, 3),
            expected_threads=32,
            expected_warps=1,
        )


def test_oracle_rejects_partial_intermediate_geometry():
    from scripts.cuda_softmax_intermediate_witness import (
        analyze_softmax_oracle,
    )

    payload = _valid_intermediates()
    payload["thread_sums"] = payload["thread_sums"][:, :31]

    with pytest.raises(ValueError, match="thread_sums shape"):
        analyze_softmax_oracle(
            **payload,
            expected_shape=(2, 3),
            expected_threads=32,
            expected_warps=1,
        )


def test_oracle_rejects_position_wrong_exponents_with_consistent_sums():
    from scripts.cuda_softmax_intermediate_witness import (
        _warp_reduce_sum,
        analyze_softmax_oracle,
    )

    payload = _valid_intermediates()
    payload["exponents"] = payload["exponents"].copy()
    payload["exponents"][0, [1, 2]] = payload["exponents"][0, [2, 1]]
    payload["exponents"][1, [0, 2]] = payload["exponents"][1, [2, 0]]
    payload["thread_sums"] = np.zeros((2, 32), dtype=np.float32)
    payload["thread_sums"][:, :3] = payload["exponents"]
    payload["warp_sums"] = _warp_reduce_sum(
        payload["thread_sums"].reshape(2, 1, 32)
    )
    padded_warps = np.zeros((2, 32), dtype=np.float32)
    padded_warps[:, :1] = payload["warp_sums"]
    payload["row_sums"] = _warp_reduce_sum(padded_warps)

    with pytest.raises(
        ValueError,
        match="captured exponent positions do not reproduce probabilities",
    ):
        analyze_softmax_oracle(
            **payload,
            expected_shape=(2, 3),
            expected_threads=32,
            expected_warps=1,
        )


def test_oracle_accepts_complete_production_reduction_geometry():
    from scripts.cuda_softmax_intermediate_witness import (
        _warp_reduce_sum,
        analyze_softmax_oracle,
    )

    payload = _valid_intermediates()
    payload["thread_sums"] = np.zeros((2, 1024), dtype=np.float32)
    payload["thread_sums"][:, :3] = payload["exponents"]
    payload["warp_sums"] = _warp_reduce_sum(
        payload["thread_sums"].reshape(2, 32, 32)
    )
    payload["row_sums"] = _warp_reduce_sum(payload["warp_sums"])
    source_tree_probs = np.float32(
        payload["exponents"] / payload["row_sums"][:, None]
    )
    for name in (
        "persisted_probs",
        "live_native_probs",
        "custom_probs",
        "captured_probs",
    ):
        payload[name] = source_tree_probs.copy()

    analysis = analyze_softmax_oracle(
        **payload,
        expected_shape=(2, 3),
    )

    assert all(analysis["self_authentication"].values())


@pytest.mark.parametrize("colliding_output", ["json", "npz"])
def test_output_paths_cannot_alias_protected_stage(
    tmp_path, colliding_output
):
    from scripts import cuda_softmax_intermediate_witness as witness

    stage_path = tmp_path / "stage.npz"
    stage_bytes = b"protected-stage-bytes"
    stage_path.write_bytes(stage_bytes)
    digest = __import__("hashlib").sha256(stage_bytes).hexdigest()
    requested_json = tmp_path / "report.json"
    requested_npz = tmp_path / "oracle.npz"
    if colliding_output == "json":
        requested_json = stage_path
    else:
        requested_npz = stage_path

    exit_code = witness.main(
        [
            "--stage-npz",
            str(stage_path),
            "--expected-stage-sha256",
            digest,
            "--output-json",
            str(requested_json),
            "--output-npz",
            str(requested_npz),
        ]
    )

    assert exit_code == 1
    assert stage_path.read_bytes() == stage_bytes
    fallback = tmp_path / "stage.npz.softmax-oracle-failure.json"
    report_path = fallback if colliding_output == "json" else requested_json
    report = json.loads(report_path.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "request_received"
    expected_status = (
        "protected_input" if colliding_output == "npz" else "missing"
    )
    assert report["primary_output_status"] == expected_status


@pytest.mark.parametrize("colliding_output", ["json", "npz"])
def test_hard_linked_output_paths_cannot_alias_protected_stage(
    tmp_path, colliding_output
):
    from scripts import cuda_softmax_intermediate_witness as witness

    stage_path = tmp_path / "stage.npz"
    stage_bytes = b"protected-hard-link-stage-bytes"
    stage_path.write_bytes(stage_bytes)
    digest = __import__("hashlib").sha256(stage_bytes).hexdigest()
    alias_path = tmp_path / f"alias.{colliding_output}"
    os.link(stage_path, alias_path)
    requested_json = tmp_path / "report.json"
    requested_npz = tmp_path / "oracle.npz"
    if colliding_output == "json":
        requested_json = alias_path
    else:
        requested_npz = alias_path

    exit_code = witness.main(
        [
            "--stage-npz",
            str(stage_path),
            "--expected-stage-sha256",
            digest,
            "--output-json",
            str(requested_json),
            "--output-npz",
            str(requested_npz),
        ]
    )

    assert exit_code == 1
    assert stage_path.read_bytes() == stage_bytes
    fallback = tmp_path / "stage.npz.softmax-oracle-failure.json"
    report_path = fallback if colliding_output == "json" else requested_json
    report = json.loads(report_path.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "request_received"
    expected_status = (
        "protected_input" if colliding_output == "npz" else "missing"
    )
    assert report["primary_output_status"] == expected_status


def test_failure_report_survives_before_primary_output(tmp_path):
    from scripts import cuda_softmax_intermediate_witness as witness

    report_path = tmp_path / "report.json"
    output_path = tmp_path / "oracle.npz"

    exit_code = witness.main(
        [
            "--stage-npz",
            str(tmp_path / "missing.npz"),
            "--expected-stage-sha256",
            "b" * 64,
            "--output-json",
            str(report_path),
            "--output-npz",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert report_path.exists()
    assert not output_path.exists()
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "stage_load"
    assert report["last_trustworthy_phase"] == "request_validated"
    assert report["primary_output_status"] == "missing"

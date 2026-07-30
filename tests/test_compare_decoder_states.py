import json

import numpy as np
import pytest


def _upsample(coords, logits):
    parent, child = np.nonzero(logits > 0)
    counts = np.bincount(parent, minlength=len(coords))
    output = np.repeat(coords, counts, axis=0).astype(np.int32, copy=True)
    output[:, 1:] *= 2
    output[:, 1] += child % 2
    output[:, 2] += (child // 2) % 2
    output[:, 3] += child // 4
    return output


def _write_decoder_pair(tmp_path):
    initial = np.array(
        [[0, 1, 2, 3], [0, 4, 5, 6]],
        dtype=np.int32,
    )
    source_arrays = {}
    candidate_arrays = {}
    source_coords = initial
    candidate_coords = initial
    for level in range(4):
        source_logits = np.full((len(source_coords), 8), -1.0, dtype=np.float16)
        source_logits[:, 0] = np.float16(1.0)
        candidate_logits = np.full(
            (len(candidate_coords), 8),
            -1.0,
            dtype=np.float16,
        )
        candidate_logits[:, 0] = np.float16(1.0)
        if level == 0:
            source_logits[1, 0] = np.float16(-0.25)
            source_logits[1, 7] = np.float16(0.25)
            candidate_logits[1, 0] = np.float16(-0.2501)
            candidate_logits[1, 6] = np.float16(0.25)

        source_arrays[f"shape_subs_{level}"] = source_logits
        source_arrays[f"shape_subs_{level}_coords"] = source_coords
        candidate_arrays[f"shape_subs_{level}"] = candidate_logits
        source_coords = _upsample(source_coords, source_logits)
        candidate_coords = _upsample(candidate_coords, candidate_logits)

    source_arrays["coords"] = source_coords
    source_arrays["feats"] = np.ones((len(source_coords), 7), dtype=np.float32)
    candidate_arrays["coords"] = candidate_coords
    candidate_arrays["feats"] = np.full(
        (len(candidate_coords), 7),
        1.25,
        dtype=np.float32,
    )

    source = tmp_path / "source.npz"
    candidate = tmp_path / "candidate.npz"
    candidate_input = tmp_path / "candidate-input.npz"
    np.savez(source, **source_arrays)
    np.savez(candidate, **candidate_arrays)
    np.savez(
        candidate_input,
        coords=initial,
        feats=np.ones((len(initial), 32), dtype=np.float32),
    )
    return source, candidate, candidate_input


def _rewrite_npz(path, transform):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    transform(arrays)
    np.savez(path, **arrays)


def _cast_candidate_subdivision_logits(candidate, dtype):
    _rewrite_npz(
        candidate,
        lambda arrays: arrays.update(
            {
                f"shape_subs_{level}": arrays[f"shape_subs_{level}"].astype(
                    dtype
                )
                for level in range(4)
            }
        ),
    )


def test_compare_decoder_states_locates_numeric_threshold_and_support_forks(
    tmp_path,
):
    from scripts.compare_decoder_states import compare_decoder_states

    source, candidate, candidate_input = _write_decoder_pair(tmp_path)

    report = compare_decoder_states(source, candidate, candidate_input)

    assert report["schema"] == "trellis2mlx.decoder_state_comparison.v1"
    assert report["first_numeric_fork_level"] == 0
    assert report["first_threshold_fork_level"] == 0
    assert report["first_support_fork_after_level"] == 0
    assert report["levels"][0]["input_support"]["jaccard"] == 1.0
    assert report["levels"][0]["input_support"]["exact_order_match"] is True
    assert report["levels"][0]["logits"]["common_shape"] == [2, 8]
    assert report["levels"][0]["threshold_decisions"]["mismatched_count"] == 2
    assert report["levels"][0]["next_support"]["reference_only_count"] == 1
    assert report["levels"][0]["next_support"]["candidate_only_count"] == 1
    assert report["final"]["coords"]["jaccard"] < 1.0
    assert report["final"]["features"]["max_abs_diff"] == 0.25


def test_compare_decoder_states_accepts_source_compatible_candidate_fp16_logits(
    tmp_path,
):
    from scripts.compare_decoder_states import compare_decoder_states

    source, candidate, candidate_input = _write_decoder_pair(tmp_path)
    _cast_candidate_subdivision_logits(candidate, np.float16)

    report = compare_decoder_states(source, candidate, candidate_input)

    assert report["levels"][0]["logits"]["candidate_dtype"] == "float16"


def test_compare_decoder_states_rejects_candidate_fp32_subdivision_logits(
    tmp_path,
):
    from scripts.compare_decoder_states import compare_decoder_states

    source, candidate, candidate_input = _write_decoder_pair(tmp_path)
    _cast_candidate_subdivision_logits(candidate, np.float32)

    with pytest.raises(
        ValueError,
        match="candidate subdivision level 0 logits must have dtype float16",
    ):
        compare_decoder_states(source, candidate, candidate_input)


def test_compare_decoder_states_rejects_missing_source_level_coordinates(tmp_path):
    from scripts.compare_decoder_states import compare_decoder_states

    source, candidate, candidate_input = _write_decoder_pair(tmp_path)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {
            name: archive[name]
            for name in archive.files
            if name != "shape_subs_2_coords"
        }
    np.savez(source, **arrays)

    with pytest.raises(
        KeyError,
        match="source artifact missing required arrays: shape_subs_2_coords",
    ):
        compare_decoder_states(source, candidate, candidate_input)


def test_compare_decoder_states_rejects_candidate_row_support_substitution(
    tmp_path,
):
    from scripts.compare_decoder_states import compare_decoder_states

    source, candidate, candidate_input = _write_decoder_pair(tmp_path)
    with np.load(candidate, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["shape_subs_1"] = np.concatenate(
        [
            arrays["shape_subs_1"],
            np.zeros((1, 8), dtype=np.float16),
        ],
        axis=0,
    )
    np.savez(candidate, **arrays)

    with pytest.raises(
        ValueError,
        match="candidate subdivision level 1 row count does not match reconstructed support",
    ):
        compare_decoder_states(source, candidate, candidate_input)


def test_compare_decoder_states_does_not_mislabel_input_support_as_numeric_or_threshold(
    tmp_path,
):
    from scripts.compare_decoder_states import compare_decoder_states

    source, candidate, candidate_input = _write_decoder_pair(tmp_path)
    shifted_input = np.array(
        [[0, 1, 2, 3], [0, 7, 8, 9]],
        dtype=np.int32,
    )
    np.savez(
        candidate_input,
        coords=shifted_input,
        feats=np.ones((len(shifted_input), 32), dtype=np.float32),
    )
    with np.load(candidate, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    shifted_coords = shifted_input
    for level in range(4):
        shifted_coords = _upsample(
            shifted_coords,
            arrays[f"shape_subs_{level}"],
        )
    arrays["coords"] = shifted_coords
    np.savez(candidate, **arrays)

    report = compare_decoder_states(source, candidate, candidate_input)

    assert report["first_numeric_fork_level"] is None
    assert report["first_threshold_fork_level"] is None
    assert report["first_support_fork_after_level"] == 0


def test_compare_decoder_states_cli_writes_report(tmp_path):
    from scripts.compare_decoder_states import main

    source, candidate, candidate_input = _write_decoder_pair(tmp_path)
    output = tmp_path / "comparison.json"

    rc = main(
        [
            "--source",
            str(source),
            "--candidate",
            str(candidate),
            "--candidate-input-slat",
            str(candidate_input),
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    report = json.loads(output.read_text())
    assert report["first_support_fork_after_level"] == 0
    assert report["source"]["sha256"]
    assert report["candidate"]["sha256"]
    assert report["candidate_input_slat"]["sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda arrays: arrays.pop("feats"),
            "candidate input SLat artifact missing required arrays: feats",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "feats", arrays["feats"].astype(np.float64)
            ),
            "candidate input SLat feats must have dtype float32",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "feats", np.ones((len(arrays["coords"]), 31), dtype=np.float32)
            ),
            r"candidate input SLat feats must have shape \[2, 32\]",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "feats", arrays["feats"][:-1]
            ),
            r"candidate input SLat feats must have shape \[2, 32\]",
        ),
        (
            lambda arrays: arrays["feats"].__setitem__((0, 0), np.nan),
            "candidate input SLat feats contains non-finite values",
        ),
    ],
    ids=[
        "coords-only",
        "wrong-dtype",
        "wrong-width",
        "row-mismatch",
        "non-finite",
    ],
)
def test_compare_decoder_states_rejects_incomplete_candidate_input_before_report(
    tmp_path,
    mutation,
    message,
):
    from scripts.compare_decoder_states import main

    source, candidate, candidate_input = _write_decoder_pair(tmp_path)
    _rewrite_npz(candidate_input, mutation)
    output = tmp_path / "comparison.json"

    with pytest.raises((KeyError, ValueError), match=message):
        main(
            [
                "--source",
                str(source),
                "--candidate",
                str(candidate),
                "--candidate-input-slat",
                str(candidate_input),
                "--output",
                str(output),
            ]
        )

    assert not output.exists()

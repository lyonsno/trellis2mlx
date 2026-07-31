from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.decoder_full_hash_ledger_contract import (
    FULL_DECODER_HASH_BOUNDARY_NAMES,
    FULL_DECODER_HASH_LEDGER_SCHEMA,
    build_decoder_full_hash_ledger,
    compare_decoder_full_hash_ledgers,
    validate_decoder_full_hash_ledger,
)


def _boundaries(
    *,
    level3_rows: int = 3,
    level4_rows: int = 5,
) -> dict[str, np.ndarray]:
    boundaries: dict[str, np.ndarray] = {
        "level2_upsample_output": np.zeros(
            (level3_rows, 128),
            dtype=np.float16,
        ),
        "level3_block0_conv": np.ones(
            (level3_rows, 128),
            dtype=np.float16,
        ),
        "level3_block0_norm": np.full(
            (level3_rows, 128),
            2,
            dtype=np.float16,
        ),
        "level3_block0_mlp_fc1": np.zeros(
            (level3_rows, 512),
            dtype=np.float16,
        ),
        "level3_block0_silu": np.ones(
            (level3_rows, 512),
            dtype=np.float16,
        ),
        "level3_block0_mlp_fc2": np.full(
            (level3_rows, 128),
            3,
            dtype=np.float16,
        ),
        "level3_block0_output": np.full(
            (level3_rows, 128),
            4,
            dtype=np.float16,
        ),
        "level3_block1_output": np.full(
            (level3_rows, 128),
            5,
            dtype=np.float16,
        ),
        "level3_block2_output": np.full(
            (level3_rows, 128),
            6,
            dtype=np.float16,
        ),
        "level3_block3_output": np.full(
            (level3_rows, 128),
            7,
            dtype=np.float16,
        ),
        "level3_upsample_subdiv_logits": np.zeros(
            (level3_rows, 8),
            dtype=np.float16,
        ),
        "level3_upsample_norm1": np.zeros(
            (level3_rows, 128),
            dtype=np.float16,
        ),
        "level3_upsample_silu1": np.zeros(
            (level3_rows, 128),
            dtype=np.float16,
        ),
        "level3_upsample_conv1": np.zeros(
            (level3_rows, 512),
            dtype=np.float16,
        ),
        "level4_child_coords": np.arange(
            level4_rows * 4,
            dtype=np.int32,
        ).reshape(level4_rows, 4),
        "level3_upsample_h_c2s": np.zeros(
            (level4_rows, 64),
            dtype=np.float16,
        ),
        "level3_upsample_skip_c2s": np.zeros(
            (level4_rows, 16),
            dtype=np.float16,
        ),
        "level3_upsample_skip_repeated": np.zeros(
            (level4_rows, 64),
            dtype=np.float16,
        ),
        "level3_upsample_norm2": np.zeros(
            (level4_rows, 64),
            dtype=np.float16,
        ),
        "level3_upsample_silu2": np.zeros(
            (level4_rows, 64),
            dtype=np.float16,
        ),
        "level3_upsample_conv2": np.zeros(
            (level4_rows, 64),
            dtype=np.float16,
        ),
        "level3_upsample_output": np.zeros(
            (level4_rows, 64),
            dtype=np.float16,
        ),
        "decoder_final_layernorm": np.zeros(
            (level4_rows, 64),
            dtype=np.float32,
        ),
        "decoder_output": np.zeros(
            (level4_rows, 7),
            dtype=np.float32,
        ),
    }
    assert tuple(boundaries) == FULL_DECODER_HASH_BOUNDARY_NAMES
    return boundaries


def test_full_decoder_hash_ledger_binds_exact_order_shapes_dtypes_and_bytes():
    boundaries = _boundaries()
    ledger = build_decoder_full_hash_ledger(boundaries)

    assert ledger["schema"] == FULL_DECODER_HASH_LEDGER_SCHEMA
    assert [entry["name"] for entry in ledger["entries"]] == list(
        FULL_DECODER_HASH_BOUNDARY_NAMES
    )
    assert validate_decoder_full_hash_ledger(ledger) == ledger

    changed = dict(boundaries)
    changed["level3_block0_norm"] = changed["level3_block0_norm"].copy()
    changed["level3_block0_norm"][0, 0] += np.float16(0.125)
    changed_ledger = build_decoder_full_hash_ledger(changed)
    index = FULL_DECODER_HASH_BOUNDARY_NAMES.index("level3_block0_norm")
    assert (
        changed_ledger["entries"][index]["sha256"]
        != ledger["entries"][index]["sha256"]
    )


def test_full_decoder_hash_ledger_rejects_partial_wrong_shape_and_wrong_dtype():
    ledger = build_decoder_full_hash_ledger(_boundaries())

    partial = copy.deepcopy(ledger)
    partial["entries"].pop()
    with pytest.raises(ValueError, match="exact ordered boundaries"):
        validate_decoder_full_hash_ledger(partial)

    wrong_shape = copy.deepcopy(ledger)
    index = FULL_DECODER_HASH_BOUNDARY_NAMES.index(
        "level3_upsample_conv1"
    )
    wrong_shape["entries"][index]["shape"] = [3, 128]
    with pytest.raises(ValueError, match="level3_upsample_conv1 shape"):
        validate_decoder_full_hash_ledger(wrong_shape)

    wrong_dtype = copy.deepcopy(ledger)
    index = FULL_DECODER_HASH_BOUNDARY_NAMES.index("decoder_output")
    wrong_dtype["entries"][index]["dtype"] = "float16"
    with pytest.raises(ValueError, match="decoder_output dtype"):
        validate_decoder_full_hash_ledger(wrong_dtype)


def test_full_decoder_hash_comparison_binds_exact_parent_and_names_first_fork():
    source = build_decoder_full_hash_ledger(_boundaries())
    changed = _boundaries()
    changed["level3_block0_norm"] = changed["level3_block0_norm"].copy()
    changed["level3_block0_norm"][0, 0] += np.float16(0.125)
    local = build_decoder_full_hash_ledger(changed)
    parent = source["entries"][0]

    comparison = compare_decoder_full_hash_ledgers(
        source,
        local,
        source_parent_entry=parent,
        local_parent_entry=parent,
    )

    assert comparison["status"] == "done"
    assert comparison["first_nonexact_boundary"] == "level3_block0_norm"
    assert comparison["parent_exact"] is True
    assert comparison["boundaries"][0]["exact"] is True


def test_full_decoder_hash_comparison_rejects_detached_or_nonexact_parent():
    source = build_decoder_full_hash_ledger(_boundaries())
    local = build_decoder_full_hash_ledger(_boundaries())
    detached = dict(source["entries"][0])
    detached["sha256"] = "f" * 64

    with pytest.raises(ValueError, match="source full ledger parent"):
        compare_decoder_full_hash_ledgers(
            source,
            local,
            source_parent_entry=detached,
            local_parent_entry=local["entries"][0],
        )

    with pytest.raises(ValueError, match="local full ledger parent"):
        compare_decoder_full_hash_ledgers(
            source,
            local,
            source_parent_entry=source["entries"][0],
            local_parent_entry=detached,
        )

    detached_boundaries = _boundaries()
    detached_boundaries["level2_upsample_output"] = detached_boundaries[
        "level2_upsample_output"
    ].copy()
    detached_boundaries["level2_upsample_output"][0, 0] += np.float16(1)
    detached_local = build_decoder_full_hash_ledger(detached_boundaries)
    with pytest.raises(ValueError, match="parent hashes differ"):
        compare_decoder_full_hash_ledgers(
            source,
            detached_local,
            source_parent_entry=source["entries"][0],
            local_parent_entry=detached_local["entries"][0],
        )


def test_terminal_output_head_bypasses_turing_fda_and_preserves_fp32(
    monkeypatch,
):
    import mlx.core as mx

    from trellmlx import decoder_level1_trace

    class OutputHead:
        def __init__(self):
            self.weight = mx.ones((7, 64), dtype=mx.float32)
            self.bias = mx.zeros((7,), dtype=mx.float32)
            self.calls = 0

        def __call__(self, values):
            self.calls += 1
            return values @ self.weight.T + self.bias

    class Decoder:
        output_layer = OutputHead()

    def forbidden_turing_fda(*args, **kwargs):
        raise AssertionError("terminal fp32 head reached Turing FDA")

    monkeypatch.setattr(
        decoder_level1_trace,
        "_decoder_linear",
        forbidden_turing_fda,
    )
    values = mx.ones((3, 64), dtype=mx.float32)

    output = decoder_level1_trace._decoder_output_head(Decoder(), values)
    mx.eval(output)

    assert Decoder.output_layer.calls == 1
    assert output.dtype == mx.float32
    assert output.shape == (3, 7)

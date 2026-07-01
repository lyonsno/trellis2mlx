import json

import pytest

from trellmlx.checkpoint_yield import (
    CHECKPOINT_YIELD_EXIT_CODE,
    CHECKPOINT_YIELD_SCHEMA,
    CheckpointYieldRequested,
    maybe_checkpoint_yield,
)


def test_checkpoint_yield_ignores_missing_stop_file(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    stop_file = tmp_path / "stop"

    assert maybe_checkpoint_yield(
        stop_file=stop_file,
        checkpoint_dir=checkpoint_dir,
        completed_stage="mesh_raw",
        next_stage="texture",
        output_path=tmp_path / "mesh.glb",
    ) is None
    assert not (checkpoint_dir / "_control" / "checkpoint_yield.json").exists()


def test_checkpoint_yield_requires_checkpoint_dir_when_stop_file_exists(tmp_path):
    stop_file = tmp_path / "stop"
    stop_file.write_text("pause\n")

    with pytest.raises(ValueError, match="checkpoint_dir is required"):
        maybe_checkpoint_yield(
            stop_file=stop_file,
            checkpoint_dir=None,
            completed_stage="mesh_raw",
        )


def test_checkpoint_yield_writes_receipt_and_raises_distinct_exit(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    stop_file = tmp_path / "stop"
    stop_file.write_text("pause\n")
    output_path = tmp_path / "mesh.glb"

    with pytest.raises(CheckpointYieldRequested) as exc_info:
        maybe_checkpoint_yield(
            stop_file=stop_file,
            checkpoint_dir=checkpoint_dir,
            completed_stage="texture",
            next_stage="texture_bake",
            output_path=output_path,
        )

    assert exc_info.value.code == CHECKPOINT_YIELD_EXIT_CODE
    receipt_path = checkpoint_dir / "_control" / "checkpoint_yield.json"
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema"] == CHECKPOINT_YIELD_SCHEMA
    assert receipt["status"] == "paused_at_checkpoint"
    assert receipt["completed_stage"] == "texture"
    assert receipt["next_stage"] == "texture_bake"
    assert receipt["checkpoint_dir"] == str(checkpoint_dir)
    assert receipt["stop_file"] == str(stop_file)
    assert receipt["output_path"] == str(output_path)
    assert receipt["exit_code"] == CHECKPOINT_YIELD_EXIT_CODE
    assert receipt["receipt_path"] == str(receipt_path)
    assert receipt["resume_command_hint"] == [
        "python",
        "generate.py",
        "--resume",
        str(checkpoint_dir),
        "--output",
        str(output_path),
    ]
    assert exc_info.value.receipt == receipt


def test_checkpoint_yield_can_record_checkpoint_without_resume_support(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    stop_file = tmp_path / "stop"
    stop_file.write_text("pause\n")

    with pytest.raises(CheckpointYieldRequested):
        maybe_checkpoint_yield(
            stop_file=stop_file,
            checkpoint_dir=checkpoint_dir,
            completed_stage="mesh_raw",
            next_stage="texture",
            resume_supported=False,
            resume_blocker="mesh_raw checkpoint exists, but mesh-only resume is not implemented",
        )

    receipt = json.loads((checkpoint_dir / "_control" / "checkpoint_yield.json").read_text())
    assert receipt["resume_supported"] is False
    assert "resume_command_hint" not in receipt
    assert receipt["resume_blocker"] == "mesh_raw checkpoint exists, but mesh-only resume is not implemented"


def test_checkpoint_yield_receipt_does_not_register_as_pipeline_stage(tmp_path):
    from trellmlx.checkpoint import list_checkpoints

    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    stop_file = tmp_path / "stop"
    stop_file.write_text("pause\n")

    with pytest.raises(CheckpointYieldRequested):
        maybe_checkpoint_yield(
            stop_file=stop_file,
            checkpoint_dir=checkpoint_dir,
            completed_stage="mesh_raw",
            next_stage="texture",
        )

    assert "checkpoint_yield" not in list_checkpoints(str(checkpoint_dir))

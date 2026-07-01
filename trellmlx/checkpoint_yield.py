"""Cooperative checkpoint-yield support for long TRELLIS generation runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_YIELD_SCHEMA = "trellis2mlx.checkpoint_yield.v1"
CHECKPOINT_YIELD_STATUS = "paused_at_checkpoint"
CHECKPOINT_YIELD_EXIT_CODE = 75


class CheckpointYieldRequested(SystemExit):
    """Raised after a durable checkpoint boundary when a stop file is present."""

    def __init__(self, receipt: dict[str, Any], exit_code: int = CHECKPOINT_YIELD_EXIT_CODE):
        super().__init__(exit_code)
        self.receipt = receipt


def maybe_checkpoint_yield(
    *,
    stop_file: str | os.PathLike[str] | None,
    checkpoint_dir: str | os.PathLike[str] | None,
    completed_stage: str,
    next_stage: str | None = None,
    output_path: str | os.PathLike[str] | None = None,
    resume_supported: bool = True,
    resume_blocker: str | None = None,
    exit_code: int = CHECKPOINT_YIELD_EXIT_CODE,
) -> None:
    """Write a yield receipt and exit if ``stop_file`` exists.

    This function is deliberately cooperative: callers invoke it only after
    they have finished writing a durable checkpoint. It never interrupts model
    execution in the middle of an MLX pass.
    """
    if stop_file is None:
        return None

    stop_path = Path(stop_file)
    if not stop_path.exists():
        return None

    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when checkpoint yield is requested")

    ckpt_path = Path(checkpoint_dir)
    receipt_path = ckpt_path / "_control" / "checkpoint_yield.json"
    receipt = build_checkpoint_yield_receipt(
        completed_stage=completed_stage,
        checkpoint_dir=ckpt_path,
        stop_file=stop_path,
        receipt_path=receipt_path,
        next_stage=next_stage,
        output_path=output_path,
        resume_supported=resume_supported,
        resume_blocker=resume_blocker,
        exit_code=exit_code,
    )
    write_checkpoint_yield_receipt(receipt_path, receipt)
    raise CheckpointYieldRequested(receipt, exit_code=exit_code)


def build_checkpoint_yield_receipt(
    *,
    completed_stage: str,
    checkpoint_dir: str | os.PathLike[str],
    stop_file: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    next_stage: str | None = None,
    output_path: str | os.PathLike[str] | None = None,
    resume_supported: bool = True,
    resume_blocker: str | None = None,
    exit_code: int = CHECKPOINT_YIELD_EXIT_CODE,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    stop_file = Path(stop_file)
    receipt_path = Path(receipt_path)

    receipt: dict[str, Any] = {
        "schema": CHECKPOINT_YIELD_SCHEMA,
        "status": CHECKPOINT_YIELD_STATUS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_stage": completed_stage,
        "next_stage": next_stage,
        "checkpoint_dir": str(checkpoint_dir),
        "stop_file": str(stop_file),
        "receipt_path": str(receipt_path),
        "exit_code": exit_code,
        "pause_semantics": "cooperative_checkpoint_and_exit",
        "resume_supported": resume_supported,
    }
    if resume_blocker is not None:
        receipt["resume_blocker"] = resume_blocker
    if resume_supported:
        receipt["resume_command_hint"] = [
            "python",
            "generate.py",
            "--resume",
            str(checkpoint_dir),
        ]
    if output_path is not None:
        receipt["output_path"] = str(output_path)
        if resume_supported:
            receipt["resume_command_hint"].extend(["--output", str(output_path)])
    return receipt


def write_checkpoint_yield_receipt(
    receipt_path: str | os.PathLike[str],
    receipt: dict[str, Any],
) -> None:
    path = Path(receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)

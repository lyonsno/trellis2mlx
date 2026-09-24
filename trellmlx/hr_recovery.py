"""Exact high-resolution flow replay after a failed TRELLIS generation.

This is deliberately a stage boundary, not a claim that every earlier pipeline
stage can resume. The pre-flow input and each completed Euler step are saved
before the next expensive step begins.
"""

import json
import hashlib
import os
from pathlib import Path
import uuid

import mlx.core as mx
import numpy as np

from trellmlx.checkpoint import has_checkpoint, load_checkpoint, save_checkpoint
from trellmlx.samplers import flow_euler_sample


def _input_identity(checkpoint_dir):
    """Bind every flow step to the exact completed HR input manifest."""
    if not has_checkpoint(checkpoint_dir, "hr_flow_input"):
        raise ValueError("hr_flow_input checkpoint invalid or missing")
    path = Path(checkpoint_dir) / "hr_flow_input.complete.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _memory_reading():
    readings = {}
    for name in ("get_active_memory", "get_cache_memory", "get_peak_memory"):
        fn = getattr(mx, name, None)
        if fn is not None:
            try:
                readings[name] = int(fn())
            except Exception:
                readings[name] = None
    return readings


def _write_failure(checkpoint_dir, *, state, error):
    control = Path(checkpoint_dir) / "_control"
    control.mkdir(parents=True, exist_ok=True)
    target = control / "failure.json"
    temporary = control / f"failure.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema": 1,
        "stage": "hr_flow",
        "last_complete_step": state["last_complete_step"],
        "active_step": state["active_step"],
        "active_phase": state["active_phase"],
        "error_type": type(error).__name__,
        "error": str(error),
        "memory_bytes": _memory_reading(),
    }
    try:
        with temporary.open("w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _sample(model, sample, cond, neg_cond, coords, checkpoint_dir, sampler,
            *, start_step_index, state, input_identity):
    def on_phase(index, phase):
        state["active_step"] = index
        state["active_phase"] = phase

    def on_complete(index, value):
        state["active_phase"] = "saving_step_checkpoint"
        save_checkpoint(
            checkpoint_dir, f"hr_flow_step_{index:03d}",
            sample_next=np.array(value), step_index=index,
            input_identity=input_identity,
        )
        state["last_complete_step"] = index
        state["active_phase"] = "step_complete"

    try:
        return flow_euler_sample(
            model, sample, cond, neg_cond, coords=coords, verbose=False,
            start_step_index=start_step_index, on_phase=on_phase,
            on_step_complete=on_complete, **sampler,
        )
    except Exception as error:
        try:
            _write_failure(checkpoint_dir, state=state, error=error)
        except Exception as report_error:
            error.add_note(f"Could not write HR failure report: {report_error}")
        raise


def run_hr_flow(model, noise, cond, neg_cond, coords, *, checkpoint_dir,
                sampler, runtime, quant_coords=None, mesh_grid_size=None,
                replay_provenance=None):
    """Save exact input before sampling; do not infer it later from the seed."""
    state = {"last_complete_step": -1, "active_step": None,
             "active_phase": "saving_input"}
    try:
        noise_np = np.array(noise)
        cond_np = np.array(cond)
        neg_cond_np = np.array(neg_cond)
        coords_np = np.array(coords)
        payload = {
            "noise": noise_np, "cond": cond_np,
            "neg_cond": neg_cond_np, "coords": coords_np,
            "sampler_json": json.dumps(sampler, sort_keys=True),
            "runtime_json": json.dumps(runtime, sort_keys=True),
        }
        if quant_coords is not None:
            payload["quant_coords"] = np.asarray(quant_coords)
        if mesh_grid_size is not None:
            payload["mesh_grid_size"] = int(mesh_grid_size)
        if replay_provenance is not None:
            payload["replay_provenance_json"] = json.dumps(
                replay_provenance, sort_keys=True
            )
        save_checkpoint(checkpoint_dir, "hr_flow_input", **payload)
        input_identity = _input_identity(checkpoint_dir)
    except Exception as error:
        try:
            _write_failure(checkpoint_dir, state=state, error=error)
        except Exception as report_error:
            error.add_note(f"Could not write HR failure report: {report_error}")
        raise
    # Consume the materialized bytes that were just saved, not a lazy random
    # expression whose value could be evaluated differently after a restart.
    return _sample(
        model, mx.array(noise_np), mx.array(cond_np), mx.array(neg_cond_np),
        mx.array(coords_np), checkpoint_dir, sampler,
        start_step_index=0, state=state, input_identity=input_identity,
    )


def replay_hr_input(model, source_dir, destination_dir, *, runtime):
    """Start a new trajectory from verified saved inputs under a changed route.

    The caller must claim an empty destination first. No completed source step
    is copied or consumed; this is never labeled exact continuation.
    """
    if Path(source_dir).resolve() == Path(destination_dir).resolve():
        raise ValueError("HR input replay requires a separate destination directory")
    source_identity = _input_identity(source_dir)
    data = load_checkpoint(source_dir, "hr_flow_input")
    if _input_identity(source_dir) != source_identity:
        raise RuntimeError("source HR input changed while loading")
    sampler = json.loads(str(data["sampler_json"]))
    source_runtime = json.loads(str(data["runtime_json"]))
    result = run_hr_flow(
        model, mx.array(data["noise"]), mx.array(data["cond"]),
        mx.array(data["neg_cond"]), mx.array(data["coords"]),
        checkpoint_dir=destination_dir, sampler=sampler, runtime=runtime,
        quant_coords=data.get("quant_coords"),
        mesh_grid_size=data.get("mesh_grid_size"),
        replay_provenance={
            "kind": "new_trajectory_from_saved_input",
            "source_input_sha256": source_identity,
            "source_runtime": source_runtime,
            "new_runtime": runtime,
        },
    )
    if _input_identity(source_dir) != source_identity:
        raise RuntimeError("source HR input changed during replay")
    return result


def resume_hr_flow(model, checkpoint_dir, *, expected_runtime):
    """Resume from the last verified step, checking the effective runtime."""
    if not has_checkpoint(checkpoint_dir, "hr_flow_input"):
        raise ValueError("hr_flow_input checkpoint invalid or missing")
    data = load_checkpoint(checkpoint_dir, "hr_flow_input")
    input_identity = _input_identity(checkpoint_dir)
    runtime = json.loads(str(data["runtime_json"]))
    if runtime != expected_runtime:
        raise ValueError(
            f"HR resume runtime mismatch: saved={runtime}, current={expected_runtime}"
        )
    sampler = json.loads(str(data["sampler_json"]))
    steps = int(sampler["steps"])
    noise = np.asarray(data["noise"])
    coords = np.asarray(data["coords"])
    if noise.ndim != 2 or coords.shape != (len(noise), 3):
        raise ValueError("HR input has inconsistent noise/coordinate shapes")
    next_step = 0
    sample = noise
    for index in range(steps):
        stage = f"hr_flow_step_{index:03d}"
        stage_root = Path(checkpoint_dir)
        stage_path = stage_root / f"{stage}.npz"
        manifest_path = stage_root / f"{stage}.complete.json"
        pending_path = stage_root / f"{stage}.pending"
        if not stage_path.exists() and not manifest_path.exists():
            # A pending write is not a completed step. Any later completed
            # checkpoint would be stale after replaying from this gap.
            if any(
                (stage_root / f"hr_flow_step_{later:03d}{suffix}").exists()
                for later in range(index + 1, steps)
                for suffix in (".npz", ".json", ".complete.json", ".pending")
            ):
                raise ValueError(f"HR step checkpoints are noncontiguous after {stage}")
            break
        if pending_path.exists():
            if any(
                (stage_root / f"hr_flow_step_{later:03d}{suffix}").exists()
                for later in range(index + 1, steps)
                for suffix in (".npz", ".json", ".complete.json", ".pending")
            ):
                raise ValueError(f"HR step checkpoints are noncontiguous after {stage}")
            break
        if not has_checkpoint(checkpoint_dir, stage):
            raise ValueError(f"HR step checkpoint {stage} invalid")
        step = load_checkpoint(checkpoint_dir, stage)
        if step.get("input_identity") != input_identity:
            raise ValueError(f"HR step checkpoint {stage} belongs to a different HR input")
        if int(step["step_index"]) != index:
            raise ValueError(f"HR step checkpoint {stage} has wrong index")
        sample = np.asarray(step["sample_next"])
        if sample.shape != noise.shape:
            raise ValueError(f"HR step checkpoint {stage} has wrong sample shape")
        next_step = index + 1
    if next_step == steps:
        return mx.array(sample)
    state = {"last_complete_step": next_step - 1, "active_step": None,
             "active_phase": "resume"}
    return _sample(
        model, mx.array(sample), mx.array(data["cond"]),
        mx.array(data["neg_cond"]), mx.array(coords), checkpoint_dir,
        sampler, start_step_index=next_step, state=state,
        input_identity=input_identity,
    )

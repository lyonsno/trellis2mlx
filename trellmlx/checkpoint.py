"""Save and load pipeline checkpoints for replay without re-running inference.

Saves intermediate representations at stage boundaries so that mesh cleanup,
simplification, texture baking, and export can be re-run with different
settings without repeating the expensive flow model inference stages.

Usage:
    # Save during generation:
    python generate.py --image ball.png --save-checkpoints /tmp/ball-ckpt/

    # Replay from mesh stage with different settings:
    python generate.py --resume /tmp/ball-ckpt/ --keep-largest --target-faces 1M
"""

import json
import os
import hashlib
import uuid

import numpy as np


def save_checkpoint(checkpoint_dir: str, stage: str, **arrays):
    """Save arrays at a pipeline stage boundary.

    Args:
        checkpoint_dir: Directory to save into (created if needed).
        stage: Stage name (used as filename prefix).
        **arrays: Named numpy arrays or scalars to save.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    npz_path = os.path.join(checkpoint_dir, f"{stage}.npz")
    meta_path = os.path.join(checkpoint_dir, f"{stage}.json")
    pending_path = os.path.join(checkpoint_dir, f"{stage}.pending")
    complete_path = os.path.join(checkpoint_dir, f"{stage}.complete.json")

    # Separate scalars/metadata from arrays
    metadata = {}
    np_arrays = {}
    for key, val in arrays.items():
        if isinstance(val, (int, float, str)):
            metadata[key] = val
        elif isinstance(val, np.ndarray):
            np_arrays[key] = val
        elif isinstance(val, list):
            # Subdivision masks: list of arrays
            for i, arr in enumerate(val):
                np_arrays[f"{key}_{i}"] = np.asarray(arr)
            metadata[f"{key}_count"] = len(val)
        else:
            # Try converting to numpy
            try:
                np_arrays[key] = np.array(val)
            except Exception:
                metadata[key] = str(val)

    # A pending marker invalidates both old and partially replaced stage files.
    # The completion manifest is published only after every payload is durable.
    with open(pending_path, "w") as f:
        f.write("pending\n")
        f.flush()
        os.fsync(f.fileno())
    temp_paths = []
    try:
        replacements = {}
        if np_arrays:
            temp_npz = f"{npz_path}.{uuid.uuid4().hex}.tmp"
            temp_paths.append(temp_npz)
            with open(temp_npz, "wb") as f:
                np.savez_compressed(f, **np_arrays)
                f.flush()
                os.fsync(f.fileno())
            replacements[npz_path] = temp_npz
        if metadata:
            temp_meta = f"{meta_path}.{uuid.uuid4().hex}.tmp"
            temp_paths.append(temp_meta)
            with open(temp_meta, "w") as f:
                json.dump(metadata, f)
                f.flush()
                os.fsync(f.fileno())
            replacements[meta_path] = temp_meta
        for final_path in (npz_path, meta_path):
            if final_path in replacements:
                os.replace(replacements[final_path], final_path)
            elif os.path.exists(final_path):
                os.remove(final_path)
        files = {}
        for path in (npz_path, meta_path):
            if os.path.exists(path):
                files[os.path.basename(path)] = _file_sha256(path)
        temp_complete = f"{complete_path}.{uuid.uuid4().hex}.tmp"
        temp_paths.append(temp_complete)
        with open(temp_complete, "w") as f:
            json.dump({"schema": 1, "files": files}, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_complete, complete_path)
        os.remove(pending_path)
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)

    total_bytes = sum(a.nbytes for a in np_arrays.values())
    print(f"  Checkpoint saved: {stage} ({total_bytes / 1e6:.1f} MB)", flush=True)


def load_checkpoint(checkpoint_dir: str, stage: str):
    """Load arrays from a pipeline stage checkpoint.

    Returns:
        dict of array name → numpy array, plus metadata from JSON.
    """
    if not has_checkpoint(checkpoint_dir, stage):
        raise ValueError(f"checkpoint {stage!r} invalid or missing in {checkpoint_dir}")
    result = {}

    # Load arrays
    npz_path = os.path.join(checkpoint_dir, f"{stage}.npz")
    if os.path.exists(npz_path):
        with np.load(npz_path) as data:
            for key in data.files:
                result[key] = data[key]

    # Load metadata
    meta_path = os.path.join(checkpoint_dir, f"{stage}.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            result.update(json.load(f))

    # Reconstruct lists (subdivision masks)
    for key in list(result.keys()):
        if key.endswith("_count") and isinstance(result[key], int):
            base = key[:-6]  # strip _count
            count = result.pop(key)
            result[base] = [result.pop(f"{base}_{i}") for i in range(count)]

    return result


def has_checkpoint(checkpoint_dir: str, stage: str) -> bool:
    """Check if a checkpoint exists for a stage."""
    if os.path.exists(os.path.join(checkpoint_dir, f"{stage}.pending")):
        return False
    complete_path = os.path.join(checkpoint_dir, f"{stage}.complete.json")
    if os.path.exists(complete_path):
        try:
            with open(complete_path) as f:
                manifest = json.load(f)
            files = manifest["files"]
            if manifest.get("schema") != 1 or not files:
                return False
            actual = {
                name for name in (f"{stage}.npz", f"{stage}.json")
                if os.path.exists(os.path.join(checkpoint_dir, name))
            }
            if set(files) != actual:
                return False
            return all(
                name in (f"{stage}.npz", f"{stage}.json")
                and _file_sha256(os.path.join(checkpoint_dir, name)) == digest
                for name, digest in files.items()
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False
    # Existing checkpoints predate completion manifests. They remain readable,
    # but the resume planner must not mistake them for crash-safe stage inputs.
    return (os.path.exists(os.path.join(checkpoint_dir, f"{stage}.npz"))
            or os.path.exists(os.path.join(checkpoint_dir, f"{stage}.json")))


def is_verified_checkpoint(checkpoint_dir: str, stage: str) -> bool:
    """Whether a stage has a valid completion manifest, unlike legacy files."""
    return (os.path.exists(os.path.join(checkpoint_dir, f"{stage}.complete.json"))
            and has_checkpoint(checkpoint_dir, stage))


def select_resume_stage(checkpoint_dir: str) -> str:
    """Choose an implemented boundary or fail; never restart inference silently."""
    if has_checkpoint(checkpoint_dir, "mesh_raw") and has_checkpoint(checkpoint_dir, "texture"):
        return "finalize"
    if is_verified_checkpoint(checkpoint_dir, "hr_flow_input"):
        return "hr_flow_input"
    available = list_checkpoints(checkpoint_dir)
    raise ValueError(
        "no supported resume boundary in "
        f"{checkpoint_dir}; available stages: {available}. "
        "This run will not restart from the image or random conditioning."
    )


def prepare_new_checkpoint_dir(checkpoint_dir: str) -> None:
    """Claim one fresh run directory; existing stage files cannot be mixed in."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    existing = os.listdir(checkpoint_dir)
    if existing:
        raise ValueError(
            f"checkpoint directory already contains checkpoint data or an "
            f"active run: {checkpoint_dir} ({sorted(existing)[:4]})"
        )
    claim = os.path.join(checkpoint_dir, ".run-claimed")
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError(
            f"checkpoint directory already contains checkpoint data or an "
            f"active run: {checkpoint_dir}"
        ) from error
    with os.fdopen(fd, "w") as stream:
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())


def claimed_by_this_process(checkpoint_dir: str) -> bool:
    claim = os.path.join(checkpoint_dir, ".run-claimed")
    try:
        with open(claim) as stream:
            return stream.read().strip() == f"pid={os.getpid()}"
    except OSError:
        return False


def write_pipeline_failure(checkpoint_dir: str, error: Exception, *, argv,
                           traceback_text: str | None = None) -> None:
    """Publish a failure receipt even when the primary output was never made."""
    control = os.path.join(checkpoint_dir, "_control")
    os.makedirs(control, exist_ok=True)
    target = os.path.join(control, "pipeline_failure.json")
    temporary = f"{target}.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema": 1,
        "status": "failed",
        "last_trustworthy_stages": list_checkpoints(checkpoint_dir),
        "error_type": type(error).__name__,
        "error": str(error),
        "argv": list(argv),
        "traceback": traceback_text,
    }
    try:
        with open(temporary, "w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _write_current_attempt(checkpoint_dir, {
            "schema": 1,
            "status": "failed",
            "failure_receipt": "pipeline_failure.json",
            "last_trustworthy_stages": payload["last_trustworthy_stages"],
        })
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def write_hr_recovery_status(checkpoint_dir: str) -> None:
    """Record that a prior failed attempt yielded shape_slat, not a mesh."""
    if not is_verified_checkpoint(checkpoint_dir, "shape_slat"):
        raise ValueError("cannot mark HR recovery without verified shape_slat")
    _write_current_attempt(checkpoint_dir, {
        "schema": 1,
        "status": "hr_shape_recovered_only",
        "produced_stage": "shape_slat",
        "produced_mesh": False,
        "prior_failure_receipts": [
            name for name in ("failure.json", "pipeline_failure.json")
            if os.path.exists(os.path.join(checkpoint_dir, "_control", name))
        ],
    })


def _write_current_attempt(checkpoint_dir: str, payload: dict) -> None:
    control = os.path.join(checkpoint_dir, "_control")
    os.makedirs(control, exist_ok=True)
    target = os.path.join(control, "current_attempt.json")
    temporary = f"{target}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary, "w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoints(checkpoint_dir: str):
    """Print summary of all checkpoints in a directory."""
    stages = list_checkpoints(checkpoint_dir)
    if not stages:
        print(f"No checkpoints in {checkpoint_dir}")
        return

    print(f"Checkpoints in {checkpoint_dir}:")
    for stage in stages:
        data = load_checkpoint(checkpoint_dir, stage)
        parts = []
        for key, val in sorted(data.items()):
            if isinstance(val, np.ndarray):
                size_mb = val.nbytes / 1e6
                parts.append(f"{key}: {val.shape} {val.dtype} ({size_mb:.1f} MB)")
            elif isinstance(val, list):
                parts.append(f"{key}: list[{len(val)}]")
            else:
                parts.append(f"{key}: {val}")

        # Compute file size on disk
        npz_path = os.path.join(checkpoint_dir, f"{stage}.npz")
        disk_mb = os.path.getsize(npz_path) / 1e6 if os.path.exists(npz_path) else 0

        print(f"\n  {stage} ({disk_mb:.1f} MB on disk):")
        for p in parts:
            print(f"    {p}")

        # PBR summary for texture checkpoints
        if "tex_np" in data:
            tex = data["tex_np"]
            print(f"    PBR: RGB [{tex[:,:3].min():.2f}, {tex[:,:3].max():.2f}] "
                  f"metallic [{tex[:,3].min():.2f}, {tex[:,3].max():.2f}] "
                  f"roughness [{tex[:,4].min():.2f}, {tex[:,4].max():.2f}]")


def list_checkpoints(checkpoint_dir: str) -> list[str]:
    """List available checkpoint stages."""
    if not os.path.isdir(checkpoint_dir):
        return []
    stages = set()
    for f in os.listdir(checkpoint_dir):
        if f.endswith(".complete.json"):
            continue
        if f.endswith(".npz"):
            stages.add(f[:-4])
        elif f.endswith(".json"):
            stages.add(f[:-5])
    return sorted(stage for stage in stages if has_checkpoint(checkpoint_dir, stage))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3 or sys.argv[1] != "inspect":
        print("Usage: python -m trellmlx.checkpoint inspect DIR")
        sys.exit(1)
    inspect_checkpoints(sys.argv[2])

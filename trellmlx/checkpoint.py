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

import numpy as np


def save_checkpoint(checkpoint_dir: str, stage: str, **arrays):
    """Save arrays at a pipeline stage boundary.

    Args:
        checkpoint_dir: Directory to save into (created if needed).
        stage: Stage name (used as filename prefix).
        **arrays: Named numpy arrays or scalars to save.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

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
                if isinstance(arr, np.ndarray):
                    np_arrays[f"{key}_{i}"] = arr
            metadata[f"{key}_count"] = len(val)
        else:
            # Try converting to numpy
            try:
                np_arrays[key] = np.array(val)
            except Exception:
                metadata[key] = str(val)

    # Save arrays
    if np_arrays:
        np.savez_compressed(
            os.path.join(checkpoint_dir, f"{stage}.npz"),
            **np_arrays,
        )

    # Save metadata
    if metadata:
        meta_path = os.path.join(checkpoint_dir, f"{stage}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

    total_bytes = sum(a.nbytes for a in np_arrays.values())
    print(f"  Checkpoint saved: {stage} ({total_bytes / 1e6:.1f} MB)", flush=True)


def load_checkpoint(checkpoint_dir: str, stage: str):
    """Load arrays from a pipeline stage checkpoint.

    Returns:
        dict of array name → numpy array, plus metadata from JSON.
    """
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
    return (os.path.exists(os.path.join(checkpoint_dir, f"{stage}.npz"))
            or os.path.exists(os.path.join(checkpoint_dir, f"{stage}.json")))


def list_checkpoints(checkpoint_dir: str) -> list[str]:
    """List available checkpoint stages."""
    if not os.path.isdir(checkpoint_dir):
        return []
    stages = set()
    for f in os.listdir(checkpoint_dir):
        if f.endswith(".npz"):
            stages.add(f[:-4])
    return sorted(stages)

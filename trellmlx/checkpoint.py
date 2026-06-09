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
        if f.endswith(".npz"):
            stages.add(f[:-4])
    return sorted(stages)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3 or sys.argv[1] != "inspect":
        print("Usage: python -m trellmlx.checkpoint inspect DIR")
        sys.exit(1)
    inspect_checkpoints(sys.argv[2])

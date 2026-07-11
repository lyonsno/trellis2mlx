"""Decode saved sparse-flow state into sparse support coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SupportCoords:
    coords: np.ndarray
    coords_3d: np.ndarray
    logits_shape_zyx: tuple[int, int, int]
    effective_logits: np.ndarray
    effective_logits_shape_zyx: tuple[int, int, int]
    mode: str

    @property
    def positive_count(self) -> int:
        return int(self.coords_3d.shape[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--steps", type=Path, help="Sparse-flow steps .npz to decode.")
    source.add_argument("--logits", type=Path, help="Already decoded occupancy logits .npz.")
    parser.add_argument("--decoder-checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--decode-array", default="sample_next")
    parser.add_argument("--decode-step", type=int, default=-1)
    parser.add_argument("--lr-resolution", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.logits:
        logits = load_logits(args.logits)
        logit_source = {
            "kind": "provided_logits",
            "logits": artifact_identity(args.logits),
        }
    else:
        if args.decoder_checkpoint is None:
            raise SystemExit("--decoder-checkpoint is required with --steps")
        sample = load_step_sample(args.steps, args.decode_array, args.decode_step)
        logits = decode_step_logits(sample, args.decoder_checkpoint)
        logit_source = {
            "kind": "decoded_sparse_flow_steps",
            "steps": artifact_identity(args.steps),
            "decoder_checkpoint": artifact_identity(args.decoder_checkpoint),
            "decode_array": args.decode_array,
            "decode_step": int(args.decode_step),
            "sample_shape": [int(v) for v in sample.shape],
        }

    support = coords_from_logits(logits, lr_resolution=args.lr_resolution)
    np.savez(output_dir / "sparse_coords.npz", coords=support.coords, coords_3d=support.coords_3d)
    np.savez(
        output_dir / "decoder_logits.npz",
        logits=np.asarray(logits, dtype=np.float32),
        effective_logits=support.effective_logits.astype(np.float32),
    )
    report = build_report(
        output_dir=output_dir,
        support=support,
        logit_source=logit_source,
        lr_resolution=args.lr_resolution,
    )
    write_json(output_dir / "decode_sparse_flow_support_report.json", report)
    print(json.dumps(console_summary(report), sort_keys=True))
    return 0


def coords_from_logits(logits: np.ndarray, *, lr_resolution: int = 32) -> SupportCoords:
    grid = squeeze_logits(np.asarray(logits, dtype=np.float32))
    if grid.shape == (lr_resolution, lr_resolution, lr_resolution):
        effective = grid
        mode = "raw"
    else:
        effective = block_max_grid(grid, lr_resolution)
        mode = "block-max"
    coords_3d = np.argwhere(effective > 0).astype(np.int32)
    if coords_3d.size:
        batch = np.zeros((coords_3d.shape[0], 1), dtype=np.int32)
        coords = np.concatenate([batch, coords_3d], axis=1)
    else:
        coords = np.zeros((0, 4), dtype=np.int32)
    return SupportCoords(
        coords=coords,
        coords_3d=coords_3d,
        logits_shape_zyx=tuple(int(v) for v in grid.shape),
        effective_logits=effective,
        effective_logits_shape_zyx=tuple(int(v) for v in effective.shape),
        mode=mode,
    )


def load_step_sample(steps_path: Path, array_name: str, step_index: int) -> np.ndarray:
    with np.load(steps_path) as data:
        if array_name not in data:
            raise KeyError(f"{steps_path} missing array {array_name!r}")
        steps = np.asarray(data[array_name], dtype=np.float32)
    if steps.ndim == 6:
        sample = steps[step_index]
    elif steps.ndim == 5:
        if step_index not in (-1, 0):
            raise IndexError(f"{array_name} has no step axis; only step 0/-1 is valid")
        sample = steps
    else:
        raise ValueError(f"{array_name} must have shape [S,B,C,D,H,W] or [B,C,D,H,W]; got {steps.shape}")
    if sample.ndim != 5:
        raise ValueError(f"decoded sample must have shape [B,C,D,H,W]; got {sample.shape}")
    return sample


def decode_step_logits(sample: np.ndarray, checkpoint_path: Path) -> np.ndarray:
    import mlx.core as mx

    from trellmlx.models.sparse_structure_decoder import SparseStructureDecoder
    from trellmlx.weight_loader import load_weights

    model = SparseStructureDecoder()
    load_weights(model, str(checkpoint_path), verbose=False)
    logits = model(mx.array(sample.astype(np.float32, copy=False)))
    mx.eval(logits)
    return np.asarray(logits, dtype=np.float32)


def load_logits(path: Path) -> np.ndarray:
    with np.load(path) as data:
        for key in ("logits", "decoded", "occupancy_logits"):
            if key in data:
                return np.asarray(data[key], dtype=np.float32)
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]], dtype=np.float32)
    raise KeyError(f"{path} missing logits array")


def squeeze_logits(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 5:
        if logits.shape[0] != 1 or logits.shape[1] != 1:
            raise ValueError(f"rank-5 logits must be [1,1,Z,Y,X]; got {logits.shape}")
        return logits[0, 0]
    if logits.ndim == 4:
        if logits.shape[0] != 1:
            raise ValueError(f"rank-4 logits must be [1,Z,Y,X]; got {logits.shape}")
        return logits[0]
    if logits.ndim == 3:
        return logits
    raise ValueError(f"logits must be rank 3, 4, or 5; got {logits.shape}")


def block_max_grid(grid: np.ndarray, lr_resolution: int) -> np.ndarray:
    if grid.ndim != 3:
        raise ValueError(f"block-max logits must be rank 3; got {grid.shape}")
    if any(dim % lr_resolution for dim in grid.shape):
        raise ValueError(f"logit shape {grid.shape} is not divisible by lr_resolution={lr_resolution}")
    factors = [dim // lr_resolution for dim in grid.shape]
    reshaped = grid.reshape(
        lr_resolution,
        factors[0],
        lr_resolution,
        factors[1],
        lr_resolution,
        factors[2],
    )
    return reshaped.max(axis=(1, 3, 5))


def build_report(
    *,
    output_dir: Path,
    support: SupportCoords,
    logit_source: dict[str, Any],
    lr_resolution: int,
) -> dict[str, Any]:
    return {
        "schema": "trellis2mlx.decode_sparse_flow_support.v1",
        "output_dir": str(output_dir),
        "logit_source": logit_source,
        "logit_grid": {
            "mode": support.mode,
            "logits_shape_zyx": [int(v) for v in support.logits_shape_zyx],
            "effective_shape_zyx": [int(v) for v in support.effective_logits_shape_zyx],
            "lr_resolution": int(lr_resolution),
        },
        "support": {
            "coord_count": support.positive_count,
            "bounds_zyx": bounds(support.coords_3d),
        },
        "outputs": {
            "sparse_coords": artifact_identity(output_dir / "sparse_coords.npz"),
            "decoder_logits": artifact_identity(output_dir / "decoder_logits.npz"),
        },
    }


def bounds(coords_3d: np.ndarray) -> dict[str, list[int] | None]:
    if coords_3d.size == 0:
        return {"min": None, "max": None}
    return {
        "min": [int(v) for v in coords_3d.min(axis=0).tolist()],
        "max": [int(v) for v in coords_3d.max(axis=0).tolist()],
    }


def artifact_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    identity: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        identity.update({"size": int(stat.st_size), "sha256": sha256(path)})
    return identity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": report["schema"],
        "coord_count": report["support"]["coord_count"],
        "logit_source": report["logit_source"]["kind"],
        "logit_grid": report["logit_grid"]["mode"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

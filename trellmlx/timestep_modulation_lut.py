"""Authenticated source-CUDA timestep modulation replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


SCHEMA = "trellis2mlx.source_cuda_timestep_modulation_lut.v1"
SOURCE_REPORT_SCHEMA = "trellis2mlx.cuda_timestep_modulation_witness.v1"
CANONICAL_STEP_INDICES = np.arange(8, dtype=np.int32)
CANONICAL_TIMESTEP_BITS = np.asarray(
    [
        0x447A0000,
        0x446EA2E9,
        0x44610000,
        0x44505555,
        0x443B8000,
        0x4420B6DB,
        0x43FA0000,
        0x43960000,
    ],
    dtype=np.uint32,
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA256")
    return value


def _require_string_list(value, *, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) for item in value)
    ):
        raise ValueError(f"{name} must be a nonempty string list")
    return value


@dataclass(frozen=True)
class SourceCudaTimestepModulationLut:
    npz_path: Path
    report_path: Path
    npz_sha256: str
    report_sha256: str
    source_checkpoint_sha256: str
    timestep_bits: np.ndarray
    modulation_bfloat16_bits: np.ndarray

    def lookup_numpy(self, timestep_float32: np.ndarray) -> np.ndarray:
        timesteps = np.asarray(timestep_float32)
        if timesteps.ndim == 0:
            timesteps = timesteps.reshape(1)
        if timesteps.dtype != np.float32 or timesteps.ndim != 1:
            raise ValueError(
                "source-CUDA timestep modulation lookup requires float32 [B] timesteps"
            )
        rows = []
        for timestep_bit in timesteps.view(np.uint32):
            matches = np.flatnonzero(self.timestep_bits == timestep_bit)
            if len(matches) != 1:
                raise ValueError(
                    "source-CUDA timestep modulation lookup received a timestep "
                    f"outside the canonical eight-step schedule: 0x{int(timestep_bit):08x}"
                )
            rows.append(self.modulation_bfloat16_bits[int(matches[0])])
        bits = np.stack(rows, axis=0).astype(np.uint32, copy=False)
        return (bits << np.uint32(16)).view(np.float32)

    def lookup_mlx(self, timestep, dtype):
        import mlx.core as mx

        values = self.lookup_numpy(np.asarray(timestep, dtype=np.float32))
        return mx.array(values).astype(dtype)

    def report_identity(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "route_identity_evidence": True,
            "route": "source-cuda-t4-canonical-shared-adaln-lut",
            "npz_path": str(self.npz_path),
            "npz_sha256_effective": self.npz_sha256,
            "report_path": str(self.report_path),
            "report_sha256_effective": self.report_sha256,
            "source_checkpoint_sha256_effective": (
                self.source_checkpoint_sha256
            ),
            "step_indices": CANONICAL_STEP_INDICES.tolist(),
            "timestep_float32_bits": [
                f"0x{int(value):08x}" for value in self.timestep_bits
            ],
            "modulation_shape": list(self.modulation_bfloat16_bits.shape),
        }


def load_source_cuda_timestep_modulation_lut(
    *,
    npz_path: str | Path,
    report_path: str | Path,
    expected_npz_sha256: str,
    expected_report_sha256: str,
    expected_source_checkpoint_sha256: str,
) -> SourceCudaTimestepModulationLut:
    npz_path = Path(npz_path)
    report_path = Path(report_path)
    expected_npz_sha256 = _require_sha256(
        expected_npz_sha256,
        name="expected NPZ SHA256",
    )
    expected_report_sha256 = _require_sha256(
        expected_report_sha256,
        name="expected report SHA256",
    )
    expected_source_checkpoint_sha256 = _require_sha256(
        expected_source_checkpoint_sha256,
        name="expected source checkpoint SHA256",
    )

    effective_npz_sha256 = _sha256_file(npz_path)
    if effective_npz_sha256 != expected_npz_sha256:
        raise ValueError(
            "source-CUDA timestep modulation NPZ SHA256 mismatch: "
            f"expected {expected_npz_sha256}, got {effective_npz_sha256}"
        )
    effective_report_sha256 = _sha256_file(report_path)
    if effective_report_sha256 != expected_report_sha256:
        raise ValueError(
            "source-CUDA timestep modulation report SHA256 mismatch: "
            f"expected {expected_report_sha256}, got {effective_report_sha256}"
        )

    report = json.loads(report_path.read_text())
    if report.get("schema") != SOURCE_REPORT_SCHEMA or report.get("status") != "done":
        raise ValueError(
            "source-CUDA timestep modulation report is not an admitted done witness"
        )
    route = report.get("effective_route")
    if route != {
        "cuda_device": "Tesla T4",
        "device_type": "cuda",
        "torch": "2.10.0+cu128",
    }:
        raise ValueError(
            "source-CUDA timestep modulation report has an unsupported effective route"
        )
    inputs = report.get("inputs")
    effective_checkpoint = (
        inputs.get("source_checkpoint_sha256_effective")
        if isinstance(inputs, dict)
        else None
    )
    if effective_checkpoint != expected_source_checkpoint_sha256:
        raise ValueError(
            "source-CUDA timestep modulation checkpoint SHA256 mismatch: "
            f"expected {expected_source_checkpoint_sha256}, got {effective_checkpoint}"
        )
    primary = report.get("primary_output")
    if not isinstance(primary, dict):
        raise ValueError(
            "source-CUDA timestep modulation report omits primary output identity"
        )
    if (
        primary.get("sha256") != effective_npz_sha256
        or primary.get("size_bytes") != npz_path.stat().st_size
        or Path(str(primary.get("path", ""))).name != npz_path.name
    ):
        raise ValueError(
            "source-CUDA timestep modulation report does not bind the effective NPZ"
        )

    expected_schedule_bits = [
        f"0x{int(value):08x}" for value in CANONICAL_TIMESTEP_BITS
    ]
    schedule = report.get("schedule_identity")
    if not isinstance(schedule, dict):
        raise ValueError(
            "source-CUDA timestep modulation report omits schedule identity"
        )
    if (
        schedule.get("step_indices_effective")
        != CANONICAL_STEP_INDICES.tolist()
        or schedule.get("step_indices_expected")
        != CANONICAL_STEP_INDICES.tolist()
        or _require_string_list(
            schedule.get("timestep_float32_bits_effective"),
            name="effective timestep bits",
        )
        != expected_schedule_bits
        or _require_string_list(
            schedule.get("timestep_float32_bits_expected"),
            name="expected timestep bits",
        )
        != expected_schedule_bits
    ):
        raise ValueError(
            "source-CUDA timestep modulation report does not bind the "
            "canonical eight-step schedule"
        )

    with np.load(npz_path, allow_pickle=False) as source:
        required = {
            "step_indices",
            "timestep_float32",
            "source_modulation_bfloat16_bits",
        }
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(
                f"source-CUDA timestep modulation NPZ missing arrays: {missing}"
            )
        step_indices = np.asarray(source["step_indices"])
        timesteps = np.asarray(source["timestep_float32"])
        modulation_bits = np.asarray(
            source["source_modulation_bfloat16_bits"]
        )

    if (
        step_indices.dtype != np.int32
        or step_indices.shape != (8,)
        or not np.array_equal(step_indices, CANONICAL_STEP_INDICES)
        or timesteps.dtype != np.float32
        or timesteps.shape != (8,)
        or not np.array_equal(
            timesteps.view(np.uint32),
            CANONICAL_TIMESTEP_BITS,
        )
    ):
        raise ValueError(
            "source-CUDA timestep modulation NPZ must use the canonical "
            "eight-step schedule"
        )
    if modulation_bits.dtype != np.uint16 or modulation_bits.shape != (8, 9216):
        raise ValueError(
            "source-CUDA modulation bits must have dtype uint16 and shape [8,9216]"
        )

    timestep_bits = timesteps.view(np.uint32).copy()
    modulation_bits = modulation_bits.copy()
    timestep_bits.flags.writeable = False
    modulation_bits.flags.writeable = False
    return SourceCudaTimestepModulationLut(
        npz_path=npz_path,
        report_path=report_path,
        npz_sha256=effective_npz_sha256,
        report_sha256=effective_report_sha256,
        source_checkpoint_sha256=effective_checkpoint,
        timestep_bits=timestep_bits,
        modulation_bfloat16_bits=modulation_bits,
    )

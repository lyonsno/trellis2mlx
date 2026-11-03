#!/usr/bin/env python3
"""Compare the first 1024 HR shape-flow prediction and step-1 Euler update on identical retained inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np

MODEL_REPO = "microsoft/TRELLIS.2-4B"
MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
TRELLIS_REVISION = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
STEM = "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16"
WEIGHT_HASHES = {
    ".json": "7ebd6d367393f52fee9e92c2b3d727a9026a338549c6a6fa84c7ea122140ffdd",
    ".safetensors": "07cd0596f634c5adc1890023d16023afc5eed02fb84b22bb23aff5bf0030fbbd",
}
NATIVE_PROFILES = (
    "fp16-control",
    "production-bf16",
    "bf16-f32-modulation",
    "bf16-mlx-sum",
)
ACCEPTED_TORCH_CUDA_VERSIONS = ("2.11.0+cu128", "2.11.0+cu130")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
SAMPLER_PARAMS = {
    "steps": 12,
    "guidance_strength": 7.5,
    "guidance_rescale": 0.5,
    "guidance_interval": (0.6, 1.0),
    "rescale_t": 3.0,
    "sigma_min": 1e-5,
}
STEP1_T = np.float32(1.0)
STEP1_T_PREV = np.float32(33.0 / 34.0)
STEP1_DT = np.float32(1.0 / 34.0)
SHAPE_SLAT_MEAN = np.array(
    [
        0.781296,
        0.018091,
        -0.495192,
        -0.558457,
        1.06053,
        0.093252,
        1.518149,
        -0.933218,
        -0.732996,
        2.604095,
        -0.118341,
        -2.143904,
        0.495076,
        -2.179512,
        -2.130751,
        -0.996944,
        0.261421,
        -2.217463,
        1.260067,
        -0.150213,
        3.790713,
        1.481266,
        -1.046058,
        -1.523667,
        -0.059621,
        2.22078,
        1.621212,
        0.87723,
        0.567247,
        -3.175944,
        -3.186688,
        1.578665,
    ],
    dtype=np.float32,
)
SHAPE_SLAT_STD = np.array(
    [
        5.972266,
        4.706852,
        5.44501,
        5.209927,
        5.32022,
        4.547237,
        5.020802,
        5.444004,
        5.226681,
        5.683095,
        4.831436,
        5.286469,
        5.652043,
        5.367606,
        5.525084,
        4.730578,
        4.805265,
        5.124013,
        5.530808,
        5.619001,
        5.10393,
        5.41767,
        5.269677,
        5.547194,
        5.634698,
        5.235274,
        6.110351,
        5.511298,
        6.237273,
        4.879207,
        5.347008,
        5.405691,
    ],
    dtype=np.float32,
)

FP16_CONTROL_CLAIM = (
    "First-step 1024 HR shape-flow prediction diagnostic only: same retained "
    "24578-row support/noise, same 4101x1024 conditioning and zero negative "
    "conditioning, same t=1000, and explicit float16 block computation because "
    "Tesla T4 refuses BF16 xFormers BlockDiagonalMask. Not native BF16 production "
    "parity, not full 12-step trajectory evidence, and not hosted SourceCUDA "
    "random-draw reproduction."
)
SHAPE_TRAJECTORY_CLAIM = (
    "Same-input 12-step 1024 HR shape-flow Euler trajectory trace on retained "
    "24578-row support/noise and 4101x1024 conditioning: explicit float16 block "
    "computation on T4 CUDA / MLX fp16-control, or native BF16 on MLX profiles. "
    "Not T4 CUDA BF16 equivalence and not hosted SourceCUDA random-draw reproduction."
)
PRODUCTION_BF16_CLAIM = (
    "Native production BF16 first-step 1024 HR shape-flow prediction control on "
    "the identical retained 24578-row support/noise and 4101x1024 conditioning at "
    "t=1000. Useful for quantifying the MLX BF16 vs FP16 delta and verifying exact "
    "equivalence to retained hr_shape_euler_01.npz; not a T4 CUDA BF16 equivalence "
    "claim."
)
BF16_CONTROL_CLAIM = (
    "Native BF16 first-step 1024 HR shape-flow prediction isolation control on the "
    "identical retained 24578-row support/noise and 4101x1024 conditioning at "
    "t=1000: block Linear weights remain bfloat16 while only the named "
    "sub-operation is varied. Diagnostic attribution only; not a production "
    "default change or T4 CUDA BF16 equivalence claim."
)

STEP1_MATRIX_KEYS = (
    "sample_in",
    "pred_pos",
    "pred_neg",
    "pred_cfg",
    "x0_pos",
    "x0_cfg",
    "x0_rescaled",
    "x0_after_rescale",
    "pred_final",
    "sample_next",
)
STEP1_SCALAR_KEYS = (
    "std_pos",
    "std_cfg",
    "ratio_raw",
    "std_ratio",
    "ratio_effective",
    "t",
    "t_prev",
)


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report_tmp_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_suffix(p.suffix + ".tmp")


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_tmp_path(p)
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(p)


def validate_creatable_parent(path: Path, label: str) -> None:
    for anc in (path.parent, *path.parent.parents):
        if anc.exists() or anc.is_symlink():
            if not anc.is_dir():
                raise ValueError(f"ancestor of {label} is not a directory: {anc}")


def _resolve_refusal_base_dir(parent: Path) -> tuple[bool, Path]:
    try:
        for anc in (parent, *parent.parents):
            if (anc.exists() or anc.is_symlink()) and not anc.is_dir():
                for candidate_dir in anc.parents:
                    if candidate_dir.exists() and candidate_dir.is_dir():
                        return False, candidate_dir
                return False, Path(tempfile.gettempdir())
        return True, parent
    except OSError:
        return False, Path(tempfile.gettempdir())


def choose_safe_refusal_report_path(
    requested_path: str | Path,
    protected_paths: set[Path],
) -> Path:
    req = Path(requested_path)
    use_parent, base_dir = _resolve_refusal_base_dir(req.parent)
    if use_parent:
        req_tmp = report_tmp_path(req)
        req_pair = {req.resolve(), req_tmp.resolve()}
        if not (req_pair & protected_paths) and not (
            req.exists()
            or req.is_symlink()
            or req_tmp.exists()
            or req_tmp.is_symlink()
        ):
            return req
    base_name = req.name or "report.json"
    for idx in range(1, 10000):
        candidate = base_dir / f"{base_name}.refusal-{idx}.json"
        candidate_tmp = report_tmp_path(candidate)
        resolved_pair = {candidate.resolve(), candidate_tmp.resolve()}
        if resolved_pair & protected_paths:
            continue
        if (
            candidate.exists()
            or candidate.is_symlink()
            or candidate_tmp.exists()
            or candidate_tmp.is_symlink()
        ):
            continue
        return candidate
    raise RuntimeError("unable to allocate safe refusal report path")


def admit_witness_output_paths(
    output_json: Path,
    output_npz: Path,
    protected_inputs: list[Path],
    *,
    mode: str = "first-prediction",
) -> dict[str, Path]:
    report_p = output_json.resolve()
    report_tmp = report_tmp_path(report_p).resolve()
    npz_p = output_npz.resolve()
    npz_tmp = npz_p.with_name(f"{npz_p.stem}.tmp{npz_p.suffix}").resolve()
    pause_p = (report_p.parent / "pause.json").resolve()
    pause_tmp = report_tmp_path(pause_p).resolve()

    mutation_paths = {
        "output_json": report_p,
        "output_json_tmp": report_tmp,
        "output_npz": npz_p,
        "output_npz_tmp": npz_tmp,
        "pause_json": pause_p,
        "pause_json_tmp": pause_tmp,
    }
    if mode == "shape-trajectory":
        mutation_paths["steps_dir"] = (report_p.parent / "steps").resolve()
    if len(set(mutation_paths.values())) != len(mutation_paths):
        raise ValueError("output and temporary mutation paths must be pairwise distinct")

    protected_resolved = {p.resolve() for p in protected_inputs}
    for label, mp in mutation_paths.items():
        if mp in protected_resolved:
            raise ValueError(f"{label} must not alias protected input path: {mp}")
        if mp.exists() or mp.is_symlink():
            raise ValueError(f"refusing to overwrite preexisting path for {label}: {mp}")
        validate_creatable_parent(mp, label)
    return mutation_paths


def xformers_pip_install_argv(
    torch_version: str,
    python_executable: str = sys.executable,
) -> list[str]:
    match = re.search(r"\+(cu\d+)$", torch_version or "")
    if match is None:
        raise ValueError(
            f"torch version must carry an explicit +cuXYZ CUDA tag, got {torch_version!r}"
        )
    cu_tag = match.group(1)
    return [
        python_executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--index-url",
        f"https://download.pytorch.org/whl/{cu_tag}",
        "xformers==0.0.35",
    ]


def verify_native_source(
    source_root: Path,
    requested_revision: str | None,
) -> dict[str, Any]:
    root = source_root.resolve()
    info: dict[str, Any] = {
        "source_root": str(root),
        "requested_revision": requested_revision,
        "observed_revision": None,
        "tree_clean": None,
    }
    if not requested_revision or not HEX40_RE.match(requested_revision):
        raise ValueError(
            "--native-revision must be a 40-character lowercase hex commit SHA for mlx backend"
        )
    if not root.is_dir():
        raise ValueError(f"native source_root does not exist: {root}")
    observed = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    info["observed_revision"] = observed
    diff_proc = subprocess.run(
        ["git", "-C", str(root), "diff", "--exit-code", "HEAD"],
        capture_output=True,
        text=True,
    )
    status_proc = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    clean = diff_proc.returncode == 0 and status_proc.stdout.strip() == ""
    info["tree_clean"] = clean
    if observed != requested_revision:
        raise ValueError(
            f"native revision differs: {observed} != {requested_revision}"
        )
    if not clean:
        raise ValueError("native tracked source tree is dirty")
    return info


def claim_ceiling_for(
    backend: str, native_profile: str, *, mode: str = "first-prediction"
) -> str:
    if mode == "shape-trajectory":
        return SHAPE_TRAJECTORY_CLAIM
    if backend == "mlx" and native_profile == "production-bf16":
        return PRODUCTION_BF16_CLAIM
    if backend == "mlx" and native_profile in {
        "bf16-f32-modulation",
        "bf16-mlx-sum",
    }:
        return BF16_CONTROL_CLAIM
    return FP16_CONTROL_CLAIM


def denormalize_shape_slat(sample: np.ndarray) -> np.ndarray:
    """Apply TRELLIS.2 pipeline.json shape_slat_normalization (sample * std + mean)."""
    s = np.asarray(sample, dtype=np.float32)
    return np.ascontiguousarray(s * SHAPE_SLAT_STD + SHAPE_SLAT_MEAN, dtype=np.float32)


def validate_output(
    pred: np.ndarray, coords: np.ndarray, expected_coords: np.ndarray
) -> None:
    if pred.dtype != np.float32 or coords.dtype != np.int32:
        raise ValueError(f"unexpected dtypes: pred={pred.dtype} coords={coords.dtype}")
    if (
        expected_coords.ndim != 2
        or expected_coords.shape[0] <= 0
        or expected_coords.shape[1] != 4
    ):
        raise ValueError(f"invalid expected_coords shape: {expected_coords.shape}")
    if pred.shape != (expected_coords.shape[0], 32):
        raise ValueError(f"prediction shape mismatch: {pred.shape}")
    if coords.shape != expected_coords.shape or not np.array_equal(
        coords, expected_coords
    ):
        raise ValueError("output coordinates do not match input support order")
    if not np.isfinite(pred).all():
        raise ValueError("nonfinite values in prediction")


def sparse_cfg_rescale_std_fp32(x0: np.ndarray) -> np.ndarray:
    """Reproduce SparseTensor.std's row-tree and serial segment reduction order in FP32."""
    x0 = np.ascontiguousarray(x0, dtype=np.float32)
    if x0.ndim != 2 or x0.shape[0] <= 0 or x0.shape[1] != 32:
        raise ValueError(f"sparse_cfg_rescale_std_fp32 expects [N,32], got {x0.shape}")
    values = x0.copy()
    squares = (values * values).astype(np.float32)
    for offset in (16, 8, 4, 2, 1):
        values[:, :offset] = (
            values[:, :offset] + values[:, offset : 2 * offset]
        ).astype(np.float32)
        squares[:, :offset] = (
            squares[:, :offset] + squares[:, offset : 2 * offset]
        ).astype(np.float32)
    inv_channels = np.float32(1.0 / 32.0)
    row_means = (values[:, 0] * inv_channels).astype(np.float32)
    row_mean2s = (squares[:, 0] * inv_channels).astype(np.float32)
    mean_sum = np.float32(0.0)
    mean2_sum = np.float32(0.0)
    for idx in range(x0.shape[0]):
        mean_sum = np.float32(mean_sum + row_means[idx])
        mean2_sum = np.float32(mean2_sum + row_mean2s[idx])
    inv_rows = np.float32(x0.shape[0])
    mean = np.float32(mean_sum / inv_rows)
    mean2 = np.float32(mean2_sum / inv_rows)
    variance = np.float32(mean2 - np.float32(mean * mean))
    return np.sqrt(variance).astype(np.float32).reshape(())


def compute_euler_step(
    sample_in: np.ndarray,
    pred_pos: np.ndarray,
    pred_neg: np.ndarray,
    coords: np.ndarray,
    *,
    t: float = 1.0,
    t_prev: float = 33.0 / 34.0,
    guided: bool = True,
) -> dict[str, np.ndarray]:
    sample = np.ascontiguousarray(sample_in, dtype=np.float32)
    pos = np.ascontiguousarray(pred_pos, dtype=np.float32)
    neg = np.ascontiguousarray(pred_neg, dtype=np.float32) if guided else pos.copy()
    coords_i32 = np.ascontiguousarray(coords, dtype=np.int32)
    t_val = float(t)
    t_prev_val = float(t_prev)
    dt = np.float32(t_val - t_prev_val)
    sigma_min = float(SAMPLER_PARAMS["sigma_min"])
    one_minus_sigma = np.float32(1.0 - sigma_min)
    coeff_val = sigma_min + (1.0 - sigma_min) * t_val
    coeff = np.float32(coeff_val)
    inv_coeff = np.float32(1.0 / coeff_val)

    x0_pos = (one_minus_sigma * sample - coeff * pos).astype(np.float32)
    std_pos = sparse_cfg_rescale_std_fp32(x0_pos)
    if guided:
        g = np.float32(SAMPLER_PARAMS["guidance_strength"])
        r = np.float32(SAMPLER_PARAMS["guidance_rescale"])
        pred_cfg = (g * pos + (np.float32(1.0) - g) * neg).astype(np.float32)
        x0_cfg = (one_minus_sigma * sample - coeff * pred_cfg).astype(np.float32)
        std_cfg = sparse_cfg_rescale_std_fp32(x0_cfg)
        ratio = (
            np.float32(std_pos / std_cfg).reshape(())
            if float(std_cfg) > 0.0
            else np.float32(1.0).reshape(())
        )
        x0_rescaled = (x0_cfg * ratio).astype(np.float32)
        x0_after_rescale = (
            r * x0_rescaled + (np.float32(1.0) - r) * x0_cfg
        ).astype(np.float32)
        pred_final = ((one_minus_sigma * sample - x0_after_rescale) * inv_coeff).astype(
            np.float32
        )
    else:
        pred_cfg = pos.copy()
        x0_cfg = x0_pos.copy()
        std_cfg = std_pos.copy()
        ratio = np.float32(1.0).reshape(())
        x0_rescaled = x0_pos.copy()
        x0_after_rescale = x0_pos.copy()
        pred_final = pos.copy()

    sample_next = (sample - dt * pred_final).astype(np.float32)
    return {
        "coords": coords_i32,
        "sample_in": sample,
        "pred_pos": pos,
        "pred_neg": neg,
        "pred_cfg": pred_cfg,
        "x0_pos": x0_pos,
        "x0_cfg": x0_cfg,
        "std_pos": std_pos,
        "std_cfg": std_cfg,
        "ratio_raw": ratio,
        "std_ratio": ratio,
        "ratio_effective": ratio,
        "x0_rescaled": x0_rescaled,
        "x0_after_rescale": x0_after_rescale,
        "pred_final": pred_final,
        "sample_next": sample_next,
        "t": np.float32(t_val).reshape(()),
        "t_prev": np.float32(t_prev_val).reshape(()),
    }


def compute_step1_euler(
    sample_in: np.ndarray,
    pred_pos: np.ndarray,
    pred_neg: np.ndarray,
    coords: np.ndarray,
) -> dict[str, np.ndarray]:
    return compute_euler_step(
        sample_in,
        pred_pos,
        pred_neg,
        coords,
        t=1.0,
        t_prev=33.0 / 34.0,
        guided=True,
    )


class ShapeTrajectoryTrace:
    """Keep each actual HR shape-flow step input/output, including raw corrupted inputs/outputs before rejection."""

    def __init__(
        self,
        root: Path,
        coords: np.ndarray,
        noise: np.ndarray,
        steps: int,
        rescale_t: float,
        report: dict[str, Any],
        save: Any,
    ) -> None:
        self.root = Path(root)
        if self.root.exists() or self.root.is_symlink():
            raise ValueError("fresh trajectory directory required")
        self.root.mkdir(parents=True)
        self.coords = np.ascontiguousarray(coords, dtype=np.int32)
        self.noise = np.ascontiguousarray(noise, dtype=np.float32)
        self.steps = int(steps)
        self.rescale_t = float(rescale_t)
        self.report = report
        self.save = save
        t_raw = np.linspace(1.0, 0.0, self.steps + 1)
        self.schedule = (
            self.rescale_t * t_raw / (1.0 + (self.rescale_t - 1.0) * t_raw)
        ).tolist()
        g_low, g_high = SAMPLER_PARAMS["guidance_interval"]
        self.guided_flags = [
            bool(g_low <= float(self.schedule[i]) <= g_high)
            for i in range(self.steps)
        ]
        self.expected_forwards = sum(2 if g else 1 for g in self.guided_flags)
        self.index = 0
        self.pending: dict[str, Any] | None = None
        self._active_branch: str | None = None
        self._prev_sample_next: np.ndarray | None = None
        self._transition_exact: bool | None = None
        self.report["trajectory"] = {
            "steps": self.steps,
            "rescale_t": self.rescale_t,
            "sigma_min": float(SAMPLER_PARAMS["sigma_min"]),
            "guidance_strength": float(SAMPLER_PARAMS["guidance_strength"]),
            "guidance_rescale": float(SAMPLER_PARAMS["guidance_rescale"]),
            "guidance_interval": list(SAMPLER_PARAMS["guidance_interval"]),
            "schedule": self.schedule,
            "guided_steps": int(sum(self.guided_flags)),
            "expected_forwards": int(self.expected_forwards),
            "completed_steps": 0,
            "records": [],
        }
        self._save()

    def _save(self) -> None:
        if callable(self.save):
            self.save()

    def _write(self) -> Path:
        assert self.pending is not None
        path = self.root / f"step-{self.index:03d}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as f:
            np.savez(f, **self.pending)
        temporary.replace(path)
        self.report["trajectory"]["active_record"] = {
            "path": str(path),
            "sha256": digest(path),
        }
        self._save()
        return path

    def begin(
        self,
        sample: np.ndarray,
        t: np.ndarray,
        coords: np.ndarray,
        *,
        branch: str = "pos",
    ) -> None:
        if self.index >= self.steps or self._active_branch is not None:
            raise ValueError("extra trajectory forward")
        guided = self.guided_flags[self.index]
        t_sched = float(self.schedule[self.index])
        t_prev_sched = float(self.schedule[self.index + 1])
        expected_t = np.array([1000.0 * t_sched], dtype=np.float32)

        if self.pending is None:
            if branch != "pos":
                raise ValueError(
                    f"expected pos branch first at step {self.index}, got {branch!r}"
                )
            self.report["phase"] = f"trajectory_step_{self.index}_pos_input"
            self.pending = {
                "sample_in": np.asarray(sample),
                "t_model": np.asarray(t),
                "coords": np.asarray(coords),
                "t": np.float32(t_sched).reshape(()),
                "t_prev": np.float32(t_prev_sched).reshape(()),
            }
            self._write()
            validate_output(
                self.pending["sample_in"], self.pending["coords"], self.coords
            )
            if (
                self.pending["t_model"].dtype != np.float32
                or not np.array_equal(self.pending["t_model"], expected_t)
            ):
                raise ValueError("trajectory timestep differs")
            if self.index == 0:
                if not np.array_equal(self.pending["sample_in"], self.noise):
                    raise ValueError("trajectory initial noise differs")
                self._transition_exact = True
            else:
                self._transition_exact = bool(
                    self._prev_sample_next is not None
                    and np.array_equal(
                        self.pending["sample_in"], self._prev_sample_next
                    )
                )
            self._active_branch = "pos"
            self.report["phase"] = f"trajectory_step_{self.index}_pos_forward"
            self._save()
        else:
            if (
                not guided
                or branch != "neg"
                or "pred_pos" not in self.pending
                or "pred_neg" in self.pending
            ):
                raise ValueError("extra trajectory forward")
            self.report["phase"] = f"trajectory_step_{self.index}_neg_input"
            self.pending["sample_in_neg"] = np.asarray(sample)
            self.pending["t_model_neg"] = np.asarray(t)
            self.pending["coords_neg_in"] = np.asarray(coords)
            self._write()
            validate_output(
                self.pending["sample_in_neg"],
                self.pending["coords_neg_in"],
                self.coords,
            )
            if not np.array_equal(
                self.pending["sample_in_neg"], self.pending["sample_in"]
            ):
                raise ValueError("neg branch sample_in differs from pos branch")
            if (
                self.pending["t_model_neg"].dtype != np.float32
                or not np.array_equal(self.pending["t_model_neg"], expected_t)
            ):
                raise ValueError("trajectory timestep differs")
            self._active_branch = "neg"
            self.report["phase"] = f"trajectory_step_{self.index}_neg_forward"
            self._save()

    def end(self, prediction: np.ndarray, coords: np.ndarray) -> None:
        if self.pending is None or self._active_branch is None:
            raise ValueError("trajectory output without input")
        branch = self._active_branch
        self._active_branch = None
        guided = self.guided_flags[self.index]
        t_sched = float(self.schedule[self.index])
        t_prev_sched = float(self.schedule[self.index + 1])

        if branch == "pos":
            self.pending["pred_pos"] = np.asarray(prediction)
            self.pending["pred_pos_coords"] = np.asarray(coords)
            if not guided:
                self.pending["pred_neg"] = np.asarray(prediction)
                self.pending["pred_neg_coords"] = np.asarray(coords)
            self._write()
            validate_output(
                self.pending["pred_pos"],
                self.pending["pred_pos_coords"],
                self.coords,
            )
            if guided:
                self.report["last_trustworthy_phase"] = (
                    f"trajectory_step_{self.index}_pos_prediction"
                )
                self._save()
                return
        else:
            self.pending["pred_neg"] = np.asarray(prediction)
            self.pending["pred_neg_coords"] = np.asarray(coords)
            self._write()
            validate_output(
                self.pending["pred_neg"],
                self.pending["pred_neg_coords"],
                self.coords,
            )

        step_bundle = compute_euler_step(
            self.pending["sample_in"],
            self.pending["pred_pos"],
            self.pending["pred_neg"],
            self.coords,
            t=t_sched,
            t_prev=t_prev_sched,
            guided=guided,
        )
        self.pending = step_bundle
        path = self._write()
        for key in STEP1_MATRIX_KEYS:
            validate_output(step_bundle[key], step_bundle["coords"], self.coords)
        for key in STEP1_SCALAR_KEYS:
            arr = np.asarray(step_bundle[key])
            if arr.dtype != np.float32 or arr.shape != () or not np.isfinite(arr).all():
                raise ValueError(f"invalid step scalar {key}: {arr}")
        for key in ("std_pos", "std_cfg", "ratio_raw", "std_ratio", "ratio_effective"):
            if float(step_bundle[key]) <= 0.0:
                raise ValueError(
                    f"non-positive guidance scalar {key}: {float(step_bundle[key])}"
                )
        self.report["trajectory"]["records"].append(
            {
                "index": self.index,
                "t": float(step_bundle["t"]),
                "t_prev": float(step_bundle["t_prev"]),
                "guided": guided,
                "path": str(path),
                "sha256": digest(path),
                "sampler_transition_exact": bool(self._transition_exact),
                "std_pos": float(step_bundle["std_pos"]),
                "std_cfg": float(step_bundle["std_cfg"]),
                "std_ratio": float(step_bundle["std_ratio"]),
            }
        )
        self._prev_sample_next = step_bundle["sample_next"].copy()
        self.index += 1
        self.pending = None
        self.report["trajectory"]["completed_steps"] = self.index
        self.report["last_trustworthy_phase"] = (
            f"trajectory_prediction_{self.index - 1}"
        )
        self._save()

    def finish(
        self,
        sample: np.ndarray,
        coords: np.ndarray,
        output_path: Path | None = None,
    ) -> None:
        self.report["phase"] = "trajectory_final_capture"
        self.report["trajectory"]["final_sample_validated"] = False
        self._save()
        path = (
            Path(output_path)
            if output_path is not None
            else self.root.parent / "sample.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        sample_arr = np.asarray(sample)
        coords_arr = np.asarray(coords)
        feats_arr = (
            denormalize_shape_slat(sample_arr)
            if sample_arr.ndim == 2 and sample_arr.shape[1] == 32
            else sample_arr
        )
        coords_3d_arr = (
            coords_arr[:, 1:]
            if coords_arr.ndim == 2 and coords_arr.shape[1] >= 1
            else coords_arr
        )
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        with temporary.open("wb") as f:
            np.savez(
                f,
                sample=sample_arr,
                feats=feats_arr,
                coords=coords_arr,
                coords_3d=coords_3d_arr,
            )
        temporary.replace(path)
        self.report["output"] = {
            "path": str(path),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        self.report["phase"] = "trajectory_final_validation"
        self._save()
        if (
            self.pending is not None
            or self._active_branch is not None
            or self.index != self.steps
        ):
            raise ValueError("incomplete trajectory")
        validate_output(sample_arr, coords_arr, self.coords)
        if self._prev_sample_next is None or not np.array_equal(
            sample_arr, self._prev_sample_next
        ):
            raise ValueError("final sample differs from last step sample_next")
        self.report["trajectory"]["final_sample_validated"] = True
        self.report["trajectory"]["final_sample_matches_last_step_exact"] = True
        self.report["output"].update(
            rows=int(coords_arr.shape[0]),
            sample_sha256=hashlib.sha256(
                np.ascontiguousarray(sample_arr).tobytes()
            ).hexdigest(),
            feats_sha256=hashlib.sha256(
                np.ascontiguousarray(feats_arr).tobytes()
            ).hexdigest(),
        )
        self._save()


def compute_cuda_step1_euler(
    x_t: Any,
    out_pos: Any,
    out_neg: Any,
    coords: np.ndarray,
) -> dict[str, np.ndarray]:
    """Execute upstream TRELLIS.2 ClassifierFreeGuidanceSamplerMixin + FlowEulerSampler step 1 on SparseTensor."""
    g = float(SAMPLER_PARAMS["guidance_strength"])
    r = float(SAMPLER_PARAMS["guidance_rescale"])
    sigma_min = float(SAMPLER_PARAMS["sigma_min"])
    t = 1.0
    t_prev = 33.0 / 34.0
    coeff = sigma_min + (1.0 - sigma_min) * t

    pred_cfg = g * out_pos + (1.0 - g) * out_neg
    x0_pos = (1.0 - sigma_min) * x_t - coeff * out_pos
    x0_cfg = (1.0 - sigma_min) * x_t - coeff * pred_cfg
    std_pos_t = x0_pos.std(dim=list(range(1, x0_pos.ndim)), keepdim=True)
    std_cfg_t = x0_cfg.std(dim=list(range(1, x0_cfg.ndim)), keepdim=True)
    ratio_t = std_pos_t / std_cfg_t
    x0_rescaled = x0_cfg * ratio_t
    x0_after_rescale = r * x0_rescaled + (1.0 - r) * x0_cfg
    pred_final = ((1.0 - sigma_min) * x_t - x0_after_rescale) / coeff
    sample_next = x_t - (t - t_prev) * pred_final

    def _feats_np(st: Any) -> np.ndarray:
        return np.ascontiguousarray(st.feats.detach().float().cpu().numpy(), dtype=np.float32)

    def _scalar_np(val: Any) -> np.ndarray:
        return np.asarray(val.detach().float().cpu().numpy(), dtype=np.float32).reshape(())

    ratio_np = _scalar_np(ratio_t)
    return {
        "coords": np.ascontiguousarray(coords, dtype=np.int32),
        "sample_in": _feats_np(x_t),
        "pred_pos": _feats_np(out_pos),
        "pred_neg": _feats_np(out_neg),
        "pred_cfg": _feats_np(pred_cfg),
        "x0_pos": _feats_np(x0_pos),
        "x0_cfg": _feats_np(x0_cfg),
        "std_pos": _scalar_np(std_pos_t),
        "std_cfg": _scalar_np(std_cfg_t),
        "ratio_raw": ratio_np,
        "std_ratio": ratio_np,
        "ratio_effective": ratio_np,
        "x0_rescaled": _feats_np(x0_rescaled),
        "x0_after_rescale": _feats_np(x0_after_rescale),
        "pred_final": _feats_np(pred_final),
        "sample_next": _feats_np(sample_next),
        "t": STEP1_T.reshape(()),
        "t_prev": STEP1_T_PREV.reshape(()),
    }


def compute_mlx_step1_euler(
    x_mx: Any,
    out_pos_mx: Any,
    out_neg_mx: Any,
    coords: np.ndarray,
) -> dict[str, np.ndarray]:
    """Execute native trellmlx.samplers step-1 CFG + rescale + Euler update on Metal."""
    import mlx.core as mx
    from trellmlx.samplers import _cfg_rescale_std, _pred_to_xstart, _xstart_to_pred

    guidance_strength = float(SAMPLER_PARAMS["guidance_strength"])
    guidance_rescale = float(SAMPLER_PARAMS["guidance_rescale"])
    sigma_min = float(SAMPLER_PARAMS["sigma_min"])
    t = 1.0
    t_prev = 33.0 / 34.0

    pred_cfg = guidance_strength * out_pos_mx + (1.0 - guidance_strength) * out_neg_mx
    x0_pos = _pred_to_xstart(x_mx, t, out_pos_mx, sigma_min)
    x0_cfg = _pred_to_xstart(x_mx, t, pred_cfg, sigma_min)
    std_pos = _cfg_rescale_std(x0_pos, sparse_tokens=True)
    std_cfg = _cfg_rescale_std(x0_cfg, sparse_tokens=True)
    safe_std_cfg = mx.where(std_cfg > 0, std_cfg, mx.ones_like(std_cfg))
    ratio_raw = std_pos / safe_std_cfg
    ratio_raw = mx.where(std_cfg > 0, ratio_raw, mx.ones_like(ratio_raw))
    std_ratio = ratio_raw
    x0_rescaled = x0_cfg * std_ratio
    x0_after_rescale = (
        guidance_rescale * x0_rescaled + (1.0 - guidance_rescale) * x0_cfg
    )
    pred_final = _xstart_to_pred(x_mx, t, x0_after_rescale, sigma_min)
    euler_dt = np.float32(t - t_prev).item()
    euler_delta = euler_dt * pred_final
    mx.eval(euler_delta)
    sample_next = x_mx - euler_delta
    mx.eval(
        pred_cfg,
        x0_pos,
        x0_cfg,
        std_pos,
        std_cfg,
        ratio_raw,
        std_ratio,
        x0_rescaled,
        x0_after_rescale,
        pred_final,
        sample_next,
    )

    def _mat_np(arr: Any) -> np.ndarray:
        return np.array(arr.astype(mx.float32), dtype=np.float32, copy=True)

    def _sca_np(arr: Any) -> np.ndarray:
        return np.array(arr.astype(mx.float32), dtype=np.float32, copy=True).reshape(())

    ratio_np = _sca_np(std_ratio)
    return {
        "coords": np.ascontiguousarray(coords, dtype=np.int32),
        "sample_in": _mat_np(x_mx),
        "pred_pos": _mat_np(out_pos_mx),
        "pred_neg": _mat_np(out_neg_mx),
        "pred_cfg": _mat_np(pred_cfg),
        "x0_pos": _mat_np(x0_pos),
        "x0_cfg": _mat_np(x0_cfg),
        "std_pos": _sca_np(std_pos),
        "std_cfg": _sca_np(std_cfg),
        "ratio_raw": _sca_np(ratio_raw),
        "std_ratio": ratio_np,
        "ratio_effective": ratio_np,
        "x0_rescaled": _mat_np(x0_rescaled),
        "x0_after_rescale": _mat_np(x0_after_rescale),
        "pred_final": _mat_np(pred_final),
        "sample_next": _mat_np(sample_next),
        "t": STEP1_T.reshape(()),
        "t_prev": STEP1_T_PREV.reshape(()),
    }


def validate_step1_bundle(
    bundle: dict[str, np.ndarray],
    expected_coords: np.ndarray,
    expected_sample_in: np.ndarray,
) -> None:
    required = {"coords", *STEP1_MATRIX_KEYS, *STEP1_SCALAR_KEYS}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"step-1 bundle missing keys: {missing}")
    coords = bundle["coords"]
    for key in STEP1_MATRIX_KEYS:
        validate_output(bundle[key], coords, expected_coords)
    if not np.array_equal(bundle["sample_in"], expected_sample_in):
        raise ValueError("step-1 bundle sample_in does not match input noise")
    for key in STEP1_SCALAR_KEYS:
        arr = np.asarray(bundle[key])
        if arr.dtype != np.float32 or arr.shape != () or not np.isfinite(arr).all():
            raise ValueError(f"invalid scalar {key}: shape={arr.shape} dtype={arr.dtype}")
    for key in ("std_pos", "std_cfg", "ratio_raw", "std_ratio", "ratio_effective"):
        if float(bundle[key]) <= 0.0:
            raise ValueError(f"non-positive guidance scalar {key}: {float(bundle[key])}")
    if float(bundle["t"]) != float(STEP1_T) or float(bundle["t_prev"]) != float(
        STEP1_T_PREV
    ):
        raise ValueError(
            f"unexpected step-1 schedule scalars: t={bundle['t']} t_prev={bundle['t_prev']}"
        )


def validate_route(
    route: dict[str, Any], backend: str, native_profile: str = "fp16-control"
) -> None:
    if backend == "cuda":
        if native_profile != "fp16-control":
            raise ValueError("native profile requires MLX backend")
        expected = dict(
            cuda_available=True,
            attention="xformers",
            sparse_container="none",
            trellis=TRELLIS_REVISION,
            xformers="0.0.35",
            xformers_built_with_cuda=True,
            xformers_fp16_block_diagonal_smoke=True,
            torso_dtype="float16",
            hr_resolution=64,
        )
        if not all(route.get(k) == v for k, v in expected.items()):
            raise ValueError(f"invalid CUDA route: {route}")
        if route.get("torch") not in ACCEPTED_TORCH_CUDA_VERSIONS:
            raise ValueError(f"unsupported CUDA torch runtime: {route.get('torch')!r}")
    elif backend == "mlx":
        if native_profile not in NATIVE_PROFILES:
            raise ValueError(f"unknown native profile: {native_profile}")
        expected_dtype = "float16" if native_profile == "fp16-control" else "bfloat16"
        expected_qk = (
            "mlx-sum"
            if native_profile in {"fp16-control", "bf16-mlx-sum"}
            else "source-cuda-warp32"
        )
        expected_modulation = (
            "float32"
            if native_profile in {"fp16-control", "bf16-f32-modulation"}
            else "bfloat16"
        )
        if route.get("torso_dtype") != expected_dtype:
            raise ValueError(
                f"MLX precision does not match {native_profile}: {route}"
            )
        if (
            "learned_modulation_dtype" in route
            and route.get("learned_modulation_dtype") != expected_modulation
        ):
            raise ValueError(
                f"MLX modulation precision does not match {native_profile}: {route}"
            )
        rev = route.get("native_revision")
        rev_req = route.get("native_revision_requested")
        rev_obs = route.get("native_revision_observed")
        if (
            not isinstance(rev, str)
            or not HEX40_RE.match(rev)
            or rev_req != rev
            or rev_obs != rev
        ):
            raise ValueError(f"invalid or unverified MLX native revision: {route}")
        if route.get("native_tree_clean") is not True:
            raise ValueError(f"MLX native source tree is not clean: {route}")
        expected_mlx = dict(
            mlx="0.32.4",
            device="gpu",
            attention="fast",
            query_chunk=1024,
            hr_attention_active=True,
            shape_flow_layernorm="mlx-two-pass",
            qk_norm=expected_qk,
            torso_dtype=expected_dtype,
        )
        if not all(route.get(k) == v for k, v in expected_mlx.items()):
            raise ValueError(f"invalid MLX route: {route}")
    else:
        raise ValueError(f"unknown backend: {backend}")


def load_inputs(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    expected_keys = {"noise", "cond", "neg_cond", "coords", "t", "t_prev"}
    if set(arrays) != expected_keys:
        raise ValueError(f"unexpected input keys: {sorted(arrays)}")
    coords = arrays["coords"]
    if (
        coords.dtype != np.int32
        or coords.ndim != 2
        or coords.shape[0] <= 0
        or coords.shape[1] != 4
    ):
        raise ValueError(f"invalid coords: shape={coords.shape} dtype={coords.dtype}")
    if not np.all(coords[:, 0] == 0):
        raise ValueError("coords batch column must be 0")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("duplicate coordinates in input support")
    rows = coords.shape[0]
    for k, shape in [
        ("noise", (rows, 32)),
        ("cond", (1, 4101, 1024)),
        ("neg_cond", (1, 4101, 1024)),
        ("t", (1,)),
        ("t_prev", (1,)),
    ]:
        arr = arrays[k]
        if arr.dtype != np.float32 or arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"invalid {k}: shape={arr.shape} dtype={arr.dtype}")
    if not np.all(arrays["neg_cond"] == 0.0):
        raise ValueError("neg_cond must be all zeros for shape-flow step 1")
    if float(arrays["t"][0]) != 1000.0:
        raise ValueError(f"unexpected timestep t: {arrays['t']}")
    if float(arrays["t_prev"][0]) != float(STEP1_T_PREV):
        raise ValueError(f"unexpected t_prev: {arrays['t_prev']}")
    return arrays


def trajectory_cuda(
    model: Any,
    x: Any,
    cond: Any,
    neg_cond: Any,
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    save: Any,
) -> dict[str, np.ndarray]:
    import torch
    from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

    trace = ShapeTrajectoryTrace(
        args.output_json.parent / "steps",
        arrays["coords"],
        arrays["noise"],
        args.steps,
        args.rescale_t,
        report,
        save,
    )
    report["trajectory"]["sampler_source"] = inspect.getfile(
        FlowEulerGuidanceIntervalSampler
    )
    report["trajectory"]["sampler_sha256"] = digest(
        report["trajectory"]["sampler_source"]
    )
    report["trajectory"]["cross_kv_cache"] = "original model has no sampler cache"
    forward_count = 0

    def observed_model(sample: Any, t: Any, conditioning: Any, **kw: Any) -> Any:
        nonlocal forward_count
        if conditioning is cond:
            branch = "pos"
        elif conditioning is neg_cond:
            branch = "neg"
        else:
            raise ValueError("unexpected conditioning tensor passed to observed_model")
        trace.begin(
            sample.feats.detach().float().cpu().numpy(),
            t.detach().float().cpu().numpy(),
            sample.coords.detach().cpu().numpy().astype(np.int32),
            branch=branch,
        )
        out = model(sample, t, conditioning, **kw)
        torch.cuda.synchronize()
        forward_count += 1
        trace.end(
            out.feats.detach().float().cpu().numpy(),
            out.coords.detach().cpu().numpy().astype(np.int32),
        )
        return out

    with torch.inference_mode():
        result = FlowEulerGuidanceIntervalSampler(
            sigma_min=float(SAMPLER_PARAMS["sigma_min"])
        ).sample(
            observed_model,
            x,
            cond,
            neg_cond,
            steps=args.steps,
            rescale_t=args.rescale_t,
            guidance_strength=float(SAMPLER_PARAMS["guidance_strength"]),
            guidance_rescale=float(SAMPLER_PARAMS["guidance_rescale"]),
            guidance_interval=tuple(SAMPLER_PARAMS["guidance_interval"]),
        )
        torch.cuda.synchronize()
    sample = (
        result.samples.feats.detach().float().cpu().numpy().astype(np.float32, copy=True)
    )
    coords = (
        result.samples.coords.detach().cpu().numpy().astype(np.int32, copy=True)
    )
    report["trajectory"]["observed_forwards"] = int(forward_count)
    if forward_count != trace.expected_forwards:
        raise ValueError(
            f"unexpected CUDA forward count: {forward_count} != {trace.expected_forwards}"
        )
    trace.finish(sample, coords, args.output_npz)
    return {
        "sample": sample,
        "feats": denormalize_shape_slat(sample),
        "coords": coords,
        "coords_3d": coords[:, 1:],
    }


def trajectory_mlx(
    model: Any,
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    save: Any,
) -> dict[str, np.ndarray]:
    import mlx.core as mx
    from trellmlx.samplers import flow_euler_sample

    trace = ShapeTrajectoryTrace(
        args.output_json.parent / "steps",
        arrays["coords"],
        arrays["noise"],
        args.steps,
        args.rescale_t,
        report,
        save,
    )
    report["trajectory"]["sampler_source"] = inspect.getfile(flow_euler_sample)
    report["trajectory"]["sampler_sha256"] = digest(
        report["trajectory"]["sampler_source"]
    )
    report["trajectory"]["cross_kv_cache"] = (
        "production sampler builds pos and neg caches once; every forward must consume the matching cache"
    )
    cond_mx = mx.array(arrays["cond"])
    neg_cond_mx = mx.array(arrays["neg_cond"])

    class ObservedModel:
        def __init__(self) -> None:
            self.pos_cache: Any = None
            self.neg_cache: Any = None
            self.builds = 0
            self.uses = 0

        def build_cross_kv_cache(self, cond: Any) -> Any:
            self.builds += 1
            cache = model.build_cross_kv_cache(cond)
            if len(cache) != len(model.blocks):
                raise ValueError("incomplete trajectory cache")
            if self.builds == 1 and cond is cond_mx:
                self.pos_cache = cache
                return self.pos_cache
            if self.builds == 2 and cond is neg_cond_mx:
                self.neg_cache = cache
                return self.neg_cache
            raise ValueError(f"unexpected trajectory cache build #{self.builds}")

        def __call__(self, sample: Any, t: Any, cond: Any, **kw: Any) -> Any:
            cache = kw.get("cross_kv_cache")
            if cond is cond_mx and cache is not None and cache is self.pos_cache:
                branch = "pos"
            elif cond is neg_cond_mx and cache is not None and cache is self.neg_cache:
                branch = "neg"
            else:
                raise ValueError("trajectory cache not consumed")
            self.uses += 1
            coords = np.column_stack(
                [
                    np.zeros(len(sample), dtype=np.int32),
                    np.array(kw["coords"], dtype=np.int32),
                ]
            )
            trace.begin(
                np.array(sample.astype(mx.float32), dtype=np.float32, copy=True),
                np.array(t.astype(mx.float32), dtype=np.float32, copy=True),
                coords,
                branch=branch,
            )
            out = model(sample, t, cond, **kw)
            mx.eval(out)
            trace.end(
                np.array(out.astype(mx.float32), dtype=np.float32, copy=True),
                coords,
            )
            return out

    observed = ObservedModel()
    out = flow_euler_sample(
        observed,
        mx.array(arrays["noise"]),
        cond_mx,
        neg_cond_mx,
        steps=args.steps,
        rescale_t=args.rescale_t,
        guidance_strength=float(SAMPLER_PARAMS["guidance_strength"]),
        guidance_rescale=float(SAMPLER_PARAMS["guidance_rescale"]),
        guidance_interval=tuple(SAMPLER_PARAMS["guidance_interval"]),
        sigma_min=float(SAMPLER_PARAMS["sigma_min"]),
        verbose=False,
        coords=mx.array(arrays["coords"][:, 1:]),
    )
    mx.eval(out)
    sample = np.array(out.astype(mx.float32), dtype=np.float32, copy=True)
    report["trajectory"]["cache_builds"] = int(observed.builds)
    report["trajectory"]["cache_uses"] = int(observed.uses)
    if observed.builds != 2 or observed.uses != trace.expected_forwards:
        raise ValueError(
            f"trajectory cache coverage differs: builds={observed.builds} uses={observed.uses} expected={trace.expected_forwards}"
        )
    coords = arrays["coords"].copy()
    trace.finish(sample, coords, args.output_npz)
    return {
        "sample": sample,
        "feats": denormalize_shape_slat(sample),
        "coords": coords,
        "coords_3d": coords[:, 1:],
    }


def run_cuda(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    save: Any = None,
) -> dict[str, np.ndarray]:
    def _save() -> None:
        if callable(save):
            save()

    def command(argv: list[str], cwd: Path | None = None) -> str:
        done = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
        report.setdefault("commands", []).append(
            dict(
                argv=argv,
                cwd=str(cwd),
                exit_code=done.returncode,
                stdout=done.stdout,
                stderr=done.stderr,
            )
        )
        _save()
        if done.returncode:
            raise RuntimeError(f"command failed: {argv}")
        return done.stdout.strip()

    report["phase"] = "cuda_setup"
    _save()
    import torch

    if not torch.cuda.is_available() or torch.__version__ not in ACCEPTED_TORCH_CUDA_VERSIONS:
        raise ValueError(f"expected measured CUDA Torch route, got {torch.__version__!r}")
    os.environ.update(
        SPARSE_CONV_BACKEND="none",
        SPARSE_ATTN_BACKEND="xformers",
        ATTN_BACKEND="xformers",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    command(xformers_pip_install_argv(torch.__version__, sys.executable))
    command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "easydict",
            "einops",
            "tqdm",
            "huggingface-hub",
            "safetensors",
        ]
    )
    import xformers._cpp_lib as xformers_cpp_lib
    import xformers.ops as xops

    xf_built_with_cuda = bool(getattr(xformers_cpp_lib, "_built_with_cuda", False))
    xf_build_meta = getattr(xformers_cpp_lib, "_build_metadata", None)
    xf_load_exc = getattr(xformers_cpp_lib, "_cpp_library_load_exception", None)
    smoke_ok = False
    if xf_built_with_cuda:
        with torch.inference_mode():
            q_smoke = torch.zeros((1, 16, 12, 128), device="cuda", dtype=torch.float16)
            mask_smoke = xops.fmha.BlockDiagonalMask.from_seqlens([16], [16])
            out_smoke = xops.memory_efficient_attention(
                q_smoke, q_smoke, q_smoke, attn_bias=mask_smoke
            )
            torch.cuda.synchronize()
            smoke_ok = tuple(out_smoke.shape) == (1, 16, 12, 128)
            del q_smoke, mask_smoke, out_smoke

    source = args.work_dir / "TRELLIS.2"
    if source.exists():
        raise ValueError("fresh CUDA source directory required")
    source.parent.mkdir(parents=True, exist_ok=True)
    command(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/microsoft/TRELLIS.2.git",
            str(source),
        ]
    )
    command(["git", "checkout", "--detach", TRELLIS_REVISION], source)
    sys.path.insert(0, str(source))
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from trellis2.models.structured_latent_flow import SLatFlowModel
    from trellis2.modules import sparse as sp
    from trellis2.modules.sparse import config

    files = {}
    for ext, expected in WEIGHT_HASHES.items():
        name = f"{STEM}{ext}"
        path = Path(
            hf_hub_download(
                MODEL_REPO,
                name,
                revision=MODEL_REVISION,
                cache_dir=str(args.work_dir / "hf"),
            )
        )
        got = digest(path)
        if got != expected:
            raise ValueError(f"model hash differs for {name}: {got} != {expected}")
        files[ext] = path

    cfg = json.loads(files[".json"].read_text())
    report["source_model_config"] = cfg
    kwargs = dict(cfg["args"], dtype="float16")
    route = dict(
        cuda_available=torch.cuda.is_available(),
        device=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),
        torch=torch.__version__,
        xformers=importlib.metadata.version("xformers"),
        xformers_built_with_cuda=xf_built_with_cuda,
        xformers_build_cuda=(
            getattr(xf_build_meta, "cuda_version", None)
            if xf_build_meta is not None
            else None
        ),
        xformers_cpp_lib_exception=(
            str(xf_load_exc) if xf_load_exc is not None else None
        ),
        xformers_fp16_block_diagonal_smoke=smoke_ok,
        attention=config.ATTN,
        sparse_container=config.CONV,
        torso_dtype="float16",
        hr_resolution=int(kwargs.get("resolution", -1)),
        trellis=command(["git", "rev-parse", "HEAD"], source),
        model_revision=MODEL_REVISION,
        source_path=inspect.getfile(SLatFlowModel),
        precision_change="original dtype argument bfloat16 replaced by float16; original convert_to retained",
        weights={k: digest(v) for k, v in files.items()},
    )
    validate_route(route, "cuda")
    if not Path(route["source_path"]).is_relative_to(source):
        raise ValueError("wrong source import")
    command(["git", "diff", "--exit-code", "HEAD"], source)
    report["effective_route"] = route
    report["phase"] = "model_load"
    _save()

    t0 = time.perf_counter()
    model = SLatFlowModel(**kwargs)
    model.load_state_dict(load_file(str(files[".safetensors"])), strict=True)
    model.eval().cuda()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    report["parameter_dtypes"] = {
        k: str(v.dtype) for k, v in model.named_parameters()
    }
    if (
        model.dtype != torch.float16
        or model.input_layer.weight.dtype != torch.float32
        or model.out_layer.weight.dtype != torch.float32
        or any(
            b.modulation.dtype != torch.float32
            or b.self_attn.to_qkv.weight.dtype != torch.float16
            for b in model.blocks
        )
    ):
        raise ValueError("source mixed precision contract differs")
    report["last_trustworthy_phase"] = "model_loaded"
    report["phase"] = "prediction"
    _save()

    coords = torch.from_numpy(arrays["coords"]).cuda()
    x = sp.SparseTensor(
        feats=torch.from_numpy(arrays["noise"]).cuda(),
        coords=coords,
    )
    t_tensor = torch.from_numpy(arrays["t"]).cuda()
    cond = torch.from_numpy(arrays["cond"]).cuda()
    neg_cond = torch.from_numpy(arrays["neg_cond"]).cuda()
    if getattr(args, "mode", "first-prediction") == "shape-trajectory":
        t1 = time.perf_counter()
        traj_bundle = trajectory_cuda(
            model, x, cond, neg_cond, args, arrays, report, _save
        )
        traj_s = time.perf_counter() - t1
        report["timing_seconds"] = {
            "model_load": load_s,
            "trajectory_sampling": traj_s,
        }
        report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        return traj_bundle

    with torch.inference_mode():
        t1 = time.perf_counter()
        out_pos = model(x, t_tensor, cond)
        out_neg = model(x, t_tensor, neg_cond)
        torch.cuda.synchronize()
        pred_s = time.perf_counter() - t1

        pos_coords = out_pos.coords.cpu().numpy().astype(np.int32)
        neg_coords = out_neg.coords.cpu().numpy().astype(np.int32)
        if not np.array_equal(pos_coords, neg_coords):
            raise ValueError("CUDA pos and neg branch output coordinates differ")
        backend_bundle = compute_cuda_step1_euler(x, out_pos, out_neg, pos_coords)

    cpu_bundle = compute_step1_euler(
        arrays["noise"],
        backend_bundle["pred_pos"],
        backend_bundle["pred_neg"],
        pos_coords,
    )
    report["step1_update_source"] = "cuda-sparse-tensor-sampler"
    report["standardized_cpu_step1_parity"] = compare_bundles(
        backend_bundle, cpu_bundle
    )
    report["timing_seconds"] = {
        "model_load": load_s,
        "prediction_both_branches": pred_s,
    }
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    return backend_bundle


def run_mlx(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    save: Any = None,
) -> dict[str, np.ndarray]:
    def _save() -> None:
        if callable(save):
            save()

    dtype = "float16" if args.native_profile == "fp16-control" else "bfloat16"
    qk = (
        "mlx-sum"
        if args.native_profile in {"fp16-control", "bf16-mlx-sum"}
        else "source-cuda-warp32"
    )
    modulation = (
        "float32"
        if args.native_profile in {"fp16-control", "bf16-f32-modulation"}
        else "bfloat16"
    )
    native_bf16 = dtype == "bfloat16"
    root = (args.source_root or Path(__file__).resolve().parents[1]).resolve()
    report["phase"] = "native_source_admission"
    native_source_info = {
        "source_root": str(root),
        "requested_revision": args.native_revision,
        "observed_revision": None,
        "tree_clean": None,
    }
    report["native_source"] = native_source_info
    _save()
    try:
        verified_source = verify_native_source(root, args.native_revision)
        native_source_info.update(verified_source)
    except BaseException:
        if root.is_dir():
            try:
                native_source_info["observed_revision"] = subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
                ).strip()
                diff_rc = subprocess.run(
                    ["git", "-C", str(root), "diff", "--exit-code", "HEAD"],
                    capture_output=True,
                ).returncode
                status_out = subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                    text=True,
                ).strip()
                native_source_info["tree_clean"] = diff_rc == 0 and status_out == ""
            except BaseException:
                pass
        _save()
        raise
    report["last_trustworthy_phase"] = "native_source_admission"
    _save()

    sys.path.insert(0, str(root))
    os.environ.update(
        TRELLIS2MLX_ATTENTION_BACKEND="fast",
        TRELLIS2MLX_FAST_ATTENTION_QUERY_CHUNK_SIZE="2048",
        TRELLIS2MLX_HR_SHAPE_FAST_ATTENTION_QUERY_CHUNK_SIZE="1024",
        TRELLIS2MLX_HR_SHAPE_ATTENTION_ACTIVE="1",
        TRELLIS2MLX_QK_NORM_BACKEND=qk,
    )
    import mlx.core as mx
    import mlx.utils
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import generate
    from trellmlx import sparse_flow_attention as sfa
    from trellmlx.greenroom_pause import pause_session
    from trellmlx.models.slat_flow import SLatFlowModel
    from trellmlx.models.sparse_structure_flow import _cast_block_linears
    from trellmlx.shape_flow_layernorm import get_shape_flow_layernorm_backend
    from trellmlx.weight_loader import _remap_key, load_weights

    sfa.configure_sparse_flow_attention_backend("inherit")
    try:
        ckpt_path = args.checkpoint
        json_path = None
        if ckpt_path is None:
            ckpt_path = Path(
                hf_hub_download(MODEL_REPO, f"{STEM}.safetensors", revision=MODEL_REVISION)
            )
            json_path = Path(
                hf_hub_download(MODEL_REPO, f"{STEM}.json", revision=MODEL_REVISION)
            )
        else:
            ckpt_path = Path(ckpt_path)
            candidate_json = ckpt_path.with_suffix(".json")
            json_path = (
                candidate_json
                if candidate_json.is_file()
                else Path(
                    hf_hub_download(MODEL_REPO, f"{STEM}.json", revision=MODEL_REVISION)
                )
            )
        if digest(ckpt_path) != WEIGHT_HASHES[".safetensors"]:
            raise ValueError("native checkpoint digest differs")
        if digest(json_path) != WEIGHT_HASHES[".json"]:
            raise ValueError("native config digest differs")

        actual = generate._shape_flow_attention_route_from_env()
        route = dict(
            mlx=importlib.metadata.version("mlx"),
            device="gpu" if mx.default_device() == mx.gpu else str(mx.default_device()),
            attention=actual["shape_flow_attention_backend_effective"],
            query_chunk=actual["shape_flow_fast_attention_query_chunk_size_effective"],
            hr_attention_active=(
                os.environ.get("TRELLIS2MLX_HR_SHAPE_ATTENTION_ACTIVE") == "1"
            ),
            shape_flow_layernorm=get_shape_flow_layernorm_backend(),
            qk_norm=generate.get_qk_norm_backend(),
            torso_dtype=dtype,
            learned_modulation_dtype=modulation,
            native_profile=args.native_profile,
            native_revision=native_source_info["observed_revision"],
            native_revision_requested=native_source_info["requested_revision"],
            native_revision_observed=native_source_info["observed_revision"],
            native_tree_clean=native_source_info["tree_clean"],
            source_path=inspect.getfile(SLatFlowModel),
            model_revision=MODEL_REVISION,
            weights={
                ".json": digest(json_path),
                ".safetensors": digest(ckpt_path),
            },
            precision_change={
                "production-bf16": "none: unmodified for_shape constructor and production weight loader",
                "fp16-control": "block linears float16, source-equivalent float32 learned block modulation; outer layers stay float32",
                "bf16-f32-modulation": "only learned block modulation held float32 before loading; production BF16 linears and QK retained",
                "bf16-mlx-sum": "none: production parameter dtypes; only QK normalization selects mlx-sum",
            }[args.native_profile],
        )
        validate_route(route, "mlx", native_profile=args.native_profile)
        if Path(route["source_path"]).resolve() != root / "trellmlx/models/slat_flow.py":
            raise ValueError("wrong native source import")
        report["effective_route"] = route
        report["phase"] = "model_load"
        _save()

        with pause_session(args.queue_dir, args.output_json.parent / "pause.json"):
            t0 = time.perf_counter()
            model = SLatFlowModel.for_shape()
            if not native_bf16:
                for block in model.blocks:
                    _cast_block_linears(block, mx.float16)
                    block.modulation = block.modulation.astype(mx.float32)
            elif modulation == "float32":
                for block in model.blocks:
                    block.modulation = block.modulation.astype(mx.float32)
            params = dict(mlx.utils.tree_flatten(model.parameters()))
            with safe_open(str(ckpt_path), framework="numpy") as sf:
                keys = {_remap_key(k) for k in sf.keys()}
            if set(params) != keys:
                raise ValueError("native/checkpoint parameter keys differ")
            if load_weights(model, str(ckpt_path), verbose=False):
                raise ValueError("unloaded native weights")
            mx.eval(model.parameters())
            load_s = time.perf_counter() - t0

            linear_dtype = mx.bfloat16 if native_bf16 else mx.float16
            modulation_dtype = mx.bfloat16 if modulation == "bfloat16" else mx.float32
            if (
                model.input_layer.weight.dtype != mx.float32
                or model.out_layer.weight.dtype != mx.float32
                or any(
                    b.modulation.dtype != modulation_dtype
                    or b.self_attn.to_qkv.weight.dtype != linear_dtype
                    for b in model.blocks
                )
            ):
                raise ValueError("native mixed precision contract differs")
            report["parameter_dtypes"] = {
                k: str(v.dtype) for k, v in mlx.utils.tree_flatten(model.parameters())
            }
            report["last_trustworthy_phase"] = "model_loaded"
            report["phase"] = "prediction"
            _save()

            if getattr(args, "mode", "first-prediction") == "shape-trajectory":
                t1 = time.perf_counter()
                traj_bundle = trajectory_mlx(model, args, arrays, report, _save)
                traj_s = time.perf_counter() - t1
                report["peak_allocated_bytes"] = int(mx.get_peak_memory())
                report["timing_seconds"] = {
                    "model_load": load_s,
                    "trajectory_sampling": traj_s,
                }
                return traj_bundle

            t1 = time.perf_counter()
            x_mx = mx.array(arrays["noise"])
            t_mx = mx.array(arrays["t"])
            coords3_mx = mx.array(arrays["coords"][:, 1:])
            cond_mx = mx.array(arrays["cond"])
            neg_cond_mx = mx.array(arrays["neg_cond"])
            out_pos = model(x_mx, t_mx, cond_mx, coords=coords3_mx)
            out_neg = model(x_mx, t_mx, neg_cond_mx, coords=coords3_mx)
            mx.eval(out_pos, out_neg)
            pred_s = time.perf_counter() - t1
            backend_bundle = compute_mlx_step1_euler(
                x_mx, out_pos, out_neg, arrays["coords"].copy()
            )
            report["peak_allocated_bytes"] = int(mx.get_peak_memory())

        cpu_bundle = compute_step1_euler(
            arrays["noise"],
            backend_bundle["pred_pos"],
            backend_bundle["pred_neg"],
            arrays["coords"].copy(),
        )
        report["step1_update_source"] = "mlx-trellmlx-sampler"
        report["standardized_cpu_step1_parity"] = compare_bundles(
            backend_bundle, cpu_bundle
        )
        report["timing_seconds"] = {
            "model_load": load_s,
            "prediction_both_branches": pred_s,
        }
        return backend_bundle
    finally:
        os.environ.pop("TRELLIS2MLX_HR_SHAPE_ATTENTION_ACTIVE", None)


def compare_bundles(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, Any]:
    if not np.array_equal(left["coords"], right["coords"]):
        raise ValueError("cannot compare bundles with mismatched coords")
    metrics: dict[str, Any] = {"matrices": {}, "scalars": {}}
    for key in STEP1_MATRIX_KEYS:
        a = np.asarray(left[key], dtype=np.float32)
        b = np.asarray(right[key], dtype=np.float32)
        diff = np.abs(a - b)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cosine = float(np.sum(a * b) / denom) if denom > 0.0 else 1.0
        metrics["matrices"][key] = {
            "shape": list(a.shape),
            "exact": bool(np.array_equal(a, b)),
            "nonzero_count": int(np.count_nonzero(diff)),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "p99_abs": float(np.quantile(diff, 0.99)),
            "p999_abs": float(np.quantile(diff, 0.999)),
            "cosine": cosine,
        }
    for key in STEP1_SCALAR_KEYS:
        av = float(left[key])
        bv = float(right[key])
        metrics["scalars"][key] = {
            "left": av,
            "right": bv,
            "abs_diff": abs(av - bv),
            "exact": bool(np.array_equal(left[key], right[key])),
        }
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["cuda", "mlx"], required=True)
    ap.add_argument(
        "--mode",
        choices=["first-prediction", "shape-trajectory"],
        default="first-prediction",
    )
    ap.add_argument("--steps", type=int)
    ap.add_argument("--rescale-t", type=float)
    ap.add_argument(
        "--native-profile", choices=NATIVE_PROFILES, default="fp16-control"
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--inputs-sha256", required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-npz", type=Path, required=True)
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/kaggle/temp/shape-first-prediction"),
    )
    ap.add_argument("--source-root", type=Path)
    ap.add_argument("--native-revision")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument(
        "--queue-dir",
        type=Path,
        default=Path.home() / ".local/state/gpu-greenroom",
    )
    args = ap.parse_args(argv)

    phase = "argument_parsing"
    last_trustworthy_phase: str | None = None
    admitted_paths: dict[str, Path] | None = None
    protected_inputs: list[Path] = [args.inputs]
    if args.checkpoint is not None:
        protected_inputs.append(args.checkpoint)
    if args.source_root is not None:
        protected_inputs.append(args.source_root)

    schema = (
        "trellis2mlx.shape_first_prediction_witness.v1"
        if args.mode == "first-prediction"
        else "trellis2mlx.shape_trajectory_witness.v1"
    )
    report: dict[str, Any] = {
        "schema": schema,
        "run_id": args.run_id,
        "backend_requested": args.backend,
        "mode_requested": args.mode,
        "native_profile_requested": args.native_profile,
        "status": "running",
        "claim_ceiling": claim_ceiling_for(
            args.backend, args.native_profile, mode=args.mode
        ),
        "inputs": {
            "path": str(args.inputs),
            "expected_sha256": args.inputs_sha256,
        },
    }

    def save() -> None:
        if admitted_paths is None:
            raise RuntimeError("cannot write to output_json before output_path_admission")
        write_json(admitted_paths["output_json"], report)

    try:
        phase = "profile_admission"
        report["phase"] = phase
        if args.backend != "mlx" and args.native_profile != "fp16-control":
            raise ValueError("native profile requires MLX backend")
        if args.mode == "shape-trajectory":
            if (
                args.steps is None
                or args.steps < 1
                or args.rescale_t is None
                or not np.isfinite(args.rescale_t)
                or args.rescale_t <= 0
            ):
                raise ValueError(
                    "shape-trajectory requires explicit positive --steps and --rescale-t"
                )
        elif args.steps is not None or args.rescale_t is not None:
            raise ValueError("--steps and --rescale-t require --mode shape-trajectory")
        last_trustworthy_phase = phase

        phase = "output_path_admission"
        report["phase"] = phase
        admitted_paths = admit_witness_output_paths(
            args.output_json,
            args.output_npz,
            protected_inputs,
            mode=args.mode,
        )
        last_trustworthy_phase = phase
        report["phase"] = phase
        report["last_trustworthy_phase"] = last_trustworthy_phase
        save()

        phase = "input_admission"
        report["phase"] = phase
        if not args.inputs.is_file() or args.inputs.stat().st_size <= 0:
            raise ValueError(f"missing or empty inputs file: {args.inputs}")
        got = digest(args.inputs)
        report["inputs"]["observed_sha256"] = got
        if got != args.inputs_sha256:
            raise ValueError(f"input digest mismatch: {got} != {args.inputs_sha256}")
        arrays = load_inputs(args.inputs)
        report["inputs"]["rows"] = int(arrays["coords"].shape[0])
        last_trustworthy_phase = phase
        report["last_trustworthy_phase"] = last_trustworthy_phase
        save()

        phase = "backend_execution"
        report["phase"] = phase
        runner = run_cuda if args.backend == "cuda" else run_mlx
        bundle = runner(args, arrays, report, save)
        last_trustworthy_phase = report.get("last_trustworthy_phase") or phase

        phase = "output_capture"
        report["phase"] = phase
        npz_path = admitted_paths["output_npz"]
        npz_tmp = admitted_paths["output_npz_tmp"]
        if args.mode == "shape-trajectory":
            if not npz_path.is_file():
                npz_path.parent.mkdir(parents=True, exist_ok=True)
                with npz_tmp.open("wb") as f:
                    np.savez(f, **bundle)
                npz_tmp.replace(npz_path)
                report["output"] = {
                    "path": str(args.output_npz),
                    "sha256": digest(npz_path),
                    "bytes": npz_path.stat().st_size,
                }
            validate_output(bundle["sample"], bundle["coords"], arrays["coords"])
            trace = report.get("trajectory", {})
            if (
                trace.get("completed_steps") != args.steps
                or trace.get("steps") != args.steps
                or trace.get("rescale_t") != args.rescale_t
                or trace.get("final_sample_validated") is not True
                or trace.get("final_sample_matches_last_step_exact") is not True
                or len(trace.get("records", [])) != args.steps
            ):
                raise ValueError(
                    "incomplete trajectory cannot be reported as success"
                )
            for idx, row in enumerate(trace["records"]):
                if (
                    row.get("index") != idx
                    or row.get("sampler_transition_exact") is not True
                    or digest(row["path"]) != row["sha256"]
                ):
                    raise ValueError(f"trajectory record differs at step {idx}")
            phase = "complete_trajectory"
            last_trustworthy_phase = phase
            report["phase"] = phase
            report["last_trustworthy_phase"] = last_trustworthy_phase
            report["status"] = "completed"
            save()
            return 0

        npz_path.parent.mkdir(parents=True, exist_ok=True)
        with npz_tmp.open("wb") as f:
            np.savez(f, **bundle)
        npz_tmp.replace(npz_path)
        report["output"] = {
            "path": str(args.output_npz),
            "sha256": digest(npz_path),
            "bytes": npz_path.stat().st_size,
        }
        validate_step1_bundle(bundle, arrays["coords"], arrays["noise"])
        report["output"].update(
            rows=int(bundle["coords"].shape[0]),
            pred_pos_sha256=hashlib.sha256(
                np.ascontiguousarray(bundle["pred_pos"]).tobytes()
            ).hexdigest(),
            pred_neg_sha256=hashlib.sha256(
                np.ascontiguousarray(bundle["pred_neg"]).tobytes()
            ).hexdigest(),
            pred_final_sha256=hashlib.sha256(
                np.ascontiguousarray(bundle["pred_final"]).tobytes()
            ).hexdigest(),
            sample_next_sha256=hashlib.sha256(
                np.ascontiguousarray(bundle["sample_next"]).tobytes()
            ).hexdigest(),
            std_pos=float(bundle["std_pos"]),
            std_cfg=float(bundle["std_cfg"]),
            std_ratio=float(bundle["std_ratio"]),
        )
        last_trustworthy_phase = phase
        report["last_trustworthy_phase"] = last_trustworthy_phase
        report["status"] = "completed"
        save()
        return 0
    except BaseException as exc:
        effective_failure_phase = str(report.get("phase") or phase)
        effective_last_trustworthy = report.get(
            "last_trustworthy_phase", last_trustworthy_phase
        )
        report.update(
            status="failed",
            failure_phase=effective_failure_phase,
            last_trustworthy_phase=effective_last_trustworthy,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        if admitted_paths is not None:
            try:
                save()
                return 1
            except OSError:
                pass
        pause_p = (args.output_json.resolve().parent / "pause.json").resolve()
        protected_set = {
            *(p.resolve() for p in protected_inputs),
            args.output_npz.resolve(),
            args.output_npz.with_name(
                f"{args.output_npz.stem}.tmp{args.output_npz.suffix}"
            ).resolve(),
            pause_p,
            report_tmp_path(pause_p).resolve(),
        }
        if args.mode == "shape-trajectory":
            protected_set.add((args.output_json.resolve().parent / "steps").resolve())
        safe_refusal = choose_safe_refusal_report_path(
            args.output_json, protected_set
        )
        report["requested_output_json"] = str(args.output_json)
        report["refusal_report_path"] = str(safe_refusal)
        write_json(safe_refusal, report)
        print(f"refusal_report_path={safe_refusal}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

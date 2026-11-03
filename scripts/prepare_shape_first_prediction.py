#!/usr/bin/env python3
"""Stage hash-verified inputs and Kaggle packet for the first 1024 HR shape-flow prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.shape_first_prediction import (  # noqa: E402
    FP16_CONTROL_CLAIM,
    digest,
    load_inputs,
)
from trellmlx.kaggle_cuda_witness import (  # noqa: E402
    KaggleCudaWitnessPacket,
    prepare_packet,
)

INPUTS = {
    "noise": {
        "path": "/Users/noahlyons/.local/state/trellis-metal-remesh-20261011/kiln-1024-hostedpixels-rowparallel-fullcapture-r1/samplers/hr_shape_noise.npz",
        "sha256": "20a236096051bcec5b438cc452e37106f5a6b1250b5b562bd9c0aec9ef8a7d16",
    },
    "conditioning": {
        "path": "/Users/noahlyons/.local/state/trellis-metal-remesh-20261011/kiln-dino1024-feasibility-1027-r1/conditioning1024.npz",
        "sha256": "8f7db9695441d18b9ea35913b8f3ebe5dfcd83f7e0b6aa25ee2aac37fd656449",
    },
    "first_step": {
        "path": "/Users/noahlyons/.local/state/trellis-metal-remesh-20261011/kiln-hr-conditioning-shape12-1027-r1/samplers/hr_shape_euler_01.npz",
        "sha256": "b659b4d9a36212053080fd49fab8b9ed3c740954aac3a619add1ff8b3ddfc0ff",
    },
    "weights": {
        "path": "/Users/noahlyons/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "sha256": "07cd0596f634c5adc1890023d16023afc5eed02fb84b22bb23aff5bf0030fbbd",
    },
}
DEFAULT_STAGE_ROOT = Path(
    "/Users/noahlyons/.local/state/trellis-metal-remesh-20261011/"
    "kiln-corrected-shape-cuda-decode-1102-r1/shape-first-prediction-1103-r3"
)
DEFAULT_RUN_ID = "c841529a-1103-4f7c-8a91-20a236098f7d"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def verify_retained_inputs(inputs_spec: dict = INPUTS) -> dict[str, np.ndarray]:
    for name, item in inputs_spec.items():
        got = digest(item["path"])
        if got != item["sha256"]:
            raise ValueError(f"{name} digest mismatch: {got} != {item['sha256']}")

    with np.load(inputs_spec["noise"]["path"], allow_pickle=False) as nz:
        noise = nz["sample"].astype(np.float32, copy=True)
        coords3 = nz["coords"].astype(np.int32, copy=True)
        params = json.loads(str(nz["parameters_json"].item()))
    expected_params = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "guidance_interval": [0.6, 1.0],
        "rescale_t": 3.0,
        "sigma_min": 1e-05,
    }
    if params != expected_params:
        raise ValueError(f"unexpected HR shape sampler parameters: {params}")
    if len(coords3) != 24578:
        raise ValueError(f"wrong retained kiln support row count: {len(coords3)}")

    with np.load(inputs_spec["conditioning"]["path"], allow_pickle=False) as cz:
        cond = cz["cond"].astype(np.float32, copy=True)
        neg_cond = cz["neg_cond"].astype(np.float32, copy=True)
    if cond.shape != (1, 4101, 1024) or neg_cond.shape != (1, 4101, 1024):
        raise ValueError(
            f"unexpected conditioning shapes: cond={cond.shape} neg_cond={neg_cond.shape}"
        )
    if not np.all(neg_cond == 0.0):
        raise ValueError("expected zero negative conditioning for HR shape flow")

    with np.load(inputs_spec["first_step"]["path"], allow_pickle=False) as e1:
        if not np.array_equal(e1["sample_in"], noise):
            raise ValueError(
                "retained hr_shape_euler_01 sample_in differs from hr_shape_noise"
            )
        t_val = float(e1["t"])
        t_prev_val = float(e1["t_prev"])
    if t_val != 1.0 or t_prev_val != float(np.float32(33.0 / 34.0)):
        raise ValueError(
            f"unexpected step-1 schedule in retained euler_01: t={t_val} t_prev={t_prev_val}"
        )

    coords = np.pad(coords3, ((0, 0), (1, 0))).astype(np.int32)
    t = np.array([1000.0], dtype=np.float32)
    t_prev = np.array([np.float32(33.0 / 34.0)], dtype=np.float32)
    return {
        "noise": noise,
        "cond": cond,
        "neg_cond": neg_cond,
        "coords": coords,
        "t": t,
        "t_prev": t_prev,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_STAGE_ROOT)
    ap.add_argument("--run-id", default=DEFAULT_RUN_ID)
    ap.add_argument(
        "--dataset-id",
        default="noahboo/t2mlx-shape-first-inputs-1103-r1",
    )
    ap.add_argument(
        "--kernel-id",
        default="noahboo/t2mlx-shape-first-cuda-1103-r1",
    )
    ap.add_argument(
        "--title",
        default="T2mlx Shape First Cuda 1103 R1",
    )
    args = ap.parse_args(argv)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema": "trellis2mlx.prepare_shape_first_prediction.v1",
        "status": "running",
        "phase": "input_admission",
        "run_id": args.run_id,
        "output_dir": str(root),
        "inputs": INPUTS,
        "claim_ceiling": FP16_CONTROL_CLAIM,
    }

    def save() -> None:
        write_json(root / "prepare-report.json", report)

    save()
    try:
        arrays = verify_retained_inputs(INPUTS)
        witness = Path(__file__).with_name("shape_first_prediction.py")
        report["witness_sha256"] = digest(witness)
        report["prepare_sha256"] = digest(__file__)
        report["phase"] = "packet_prepare"
        save()

        capsule = root / "capsule"
        capsule.mkdir(exist_ok=False)
        inputs_npz = capsule / "inputs.npz"
        np.savez(inputs_npz, **arrays)
        load_inputs(inputs_npz)
        shutil.copy2(witness, capsule / witness.name)

        report["input_arrays"] = {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
            }
            for k, v in arrays.items()
        }
        write_json(capsule / "provenance.json", report)

        inputs_sha = digest(inputs_npz)
        packet = KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=root / "packet",
            dataset_id=args.dataset_id,
            kernel_id=args.kernel_id,
            title=args.title,
            entrypoint=witness.name,
            inputs=(witness.name, "inputs.npz", "provenance.json"),
            run_id=args.run_id,
            enable_internet=True,
            entrypoint_args=(
                "--backend",
                "cuda",
                "--run-id",
                args.run_id,
                "--inputs",
                "inputs.npz",
                "--inputs-sha256",
                inputs_sha,
            ),
            output_json="cuda_prediction_report.json",
            output_npz="cuda_prediction.npz",
        )
        prepare_packet(packet)
        report.update(
            status="completed",
            phase="packet_prepared",
            packet_manifest_sha256=digest(packet.dataset_dir / "witness-manifest.json"),
            input_archive_sha256=inputs_sha,
        )
        save()
        return 0
    except BaseException as exc:
        report.update(
            status="failed",
            failure_phase=report["phase"],
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        save()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

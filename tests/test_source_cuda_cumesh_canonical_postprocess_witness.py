import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.source_cuda_cumesh_canonical_postprocess_witness import (
    INSTRUMENTATION_SCHEMA,
    _porcelain_changed_files,
    run_witness,
)
from scripts.source_cuda_cumesh_postprocess_witness import (
    CUMESH_COMMIT,
    EXPECTED_CUDA_CAPABILITY,
    EXPECTED_CUDA_DEVICE_NAME,
    TRELLIS_COMMIT,
    TRELLIS_POSTPROCESS_SHA256,
    WitnessError,
    sha256_file,
    write_binary_ply,
)


def _write_input(path: Path) -> None:
    write_binary_ply(
        path,
        np.zeros((4, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.int32),
    )


def _runtime(patch_sha256: str, *, schema: str) -> SimpleNamespace:
    return SimpleNamespace(
        effective_route={
            "trellis_commit": TRELLIS_COMMIT,
            "trellis_source_clean": True,
            "trellis_postprocess_sha256": TRELLIS_POSTPROCESS_SHA256,
            "cumesh_commit": CUMESH_COMMIT,
            "cumesh_source_clean_before_build": True,
            "cuda_device_name": EXPECTED_CUDA_DEVICE_NAME,
            "cuda_capability": list(EXPECTED_CUDA_CAPABILITY),
            "device_type": "cuda",
            "cumesh_instrumentation": {
                "schema": schema,
                "patch_sha256": patch_sha256,
            },
        }
    )


def test_canonical_witness_rejects_wrong_instrumentation_before_geometry(
    tmp_path,
):
    input_ply = tmp_path / "input.ply"
    patch = tmp_path / "canonical.patch"
    report_json = tmp_path / "report.json"
    _write_input(input_ply)
    patch.write_text("fixture patch")
    patch_sha256 = sha256_file(patch)

    with pytest.raises(WitnessError, match="wrong instrumentation schema"):
        run_witness(
            input_ply=input_ply,
            output_dir=tmp_path / "stages",
            report_json=report_json,
            expected_input_sha256=sha256_file(input_ply),
            target_faces=1,
            work_dir=tmp_path / "runtime",
            instrumentation_patch=patch,
            expected_patch_sha256=patch_sha256,
            runtime_factory=lambda **kwargs: _runtime(
                patch_sha256,
                schema="wrong.schema",
            ),
        )

    report = json.loads(report_json.read_text())
    assert report["failure_phase"] == "runtime_setup"
    assert report["primary_output_status"] == "not_started"
    assert report["requested_route"]["adjacency_order"] == (
        "ascending-face-id-per-vertex"
    )
    assert report["requested_route"]["expected_patch_sha256"] == patch_sha256


def test_canonical_entrypoint_imports_from_flat_kaggle_capsule(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    for source, target in (
        (
            repo_root
            / "scripts"
            / "source_cuda_cumesh_canonical_postprocess_witness.py",
            tmp_path / "source_cuda_cumesh_canonical_postprocess_witness.py",
        ),
        (
            repo_root / "scripts" / "source_cuda_cumesh_postprocess_witness.py",
            tmp_path / "source_cuda_cumesh_postprocess_witness.py",
        ),
        (
            repo_root / "trellmlx" / "canonical_cumesh.py",
            tmp_path / "canonical_cumesh.py",
        ),
    ):
        shutil.copy2(source, target)

    completed = subprocess.run(
        [
            sys.executable,
            str(
                tmp_path
                / "source_cuda_cumesh_canonical_postprocess_witness.py"
            ),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--instrumentation-patch" in completed.stdout
    assert "--record-simplify-step-digests" in completed.stdout
    assert INSTRUMENTATION_SCHEMA


def test_porcelain_parser_preserves_leading_status_column():
    status = (
        " M src/connectivity.cu\n"
        "M  src/cumesh.h\n"
        " M src/ext.cpp\n"
        " M src/simplify.cu\n"
    )

    assert _porcelain_changed_files(status) == [
        "src/connectivity.cu",
        "src/cumesh.h",
        "src/ext.cpp",
        "src/simplify.cu",
    ]

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _write_shape_slat_grid_fixture(tmp_path, *, points=None):
    import hashlib
    import json
    from pathlib import Path

    import numpy as np

    if points is None:
        points = [
            ("alpha-1_beta-1", 1.0, 1.0),
            ("alpha-0_beta-0", 0.0, 0.0),
            ("alpha-1_beta-0p5", 1.0, 0.5),
            ("alpha-0p5_beta-0", 0.5, 0.0),
        ]
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    arrays = {"coords": coords}
    point_rows = []
    for index, (coordinate_key, alpha, beta) in enumerate(points):
        output_key = f"point_{coordinate_key}_shape_slat"
        values = np.full((2, 32), index + 0.25, dtype=np.float32)
        arrays[output_key] = values
        point_rows.append(
            {
                "coordinate": {"alpha": alpha, "beta": beta},
                "coordinate_key": coordinate_key,
                "output_key": output_key,
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "shape": [2, 32],
                "vs_source_control": {"exact": coordinate_key == "alpha-1_beta-1"},
            }
        )

    grid = Path(tmp_path) / "cuda_result.npz"
    np.savez(grid, **arrays)
    grid_sha = hashlib.sha256(grid.read_bytes()).hexdigest()
    source_report = Path(tmp_path) / "cuda_result.json"
    source_report.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.source_cuda_shape_block29_basin_map.v1",
                "status": "done",
                "effective_route": {
                    "route": "official-source-cuda-full-eight-step-shape-flow-with-fixed-block29-endpoints",
                    "device_type": "cuda",
                    "cuda_device": "Tesla T4",
                    "attention_backend": "sdpa",
                    "conv_backend": "none",
                    "block_index": 29,
                    "step_index": 0,
                    "steps": 8,
                    "one_model_load": True,
                    "endpoint_semantics": "current + scale * (source - current)",
                },
                "primary_output": {
                    "path": grid.name,
                    "sha256": grid_sha,
                    "size_bytes": grid.stat().st_size,
                    "keys": sorted(arrays),
                },
                "source_control": {
                    "coordinate": {"alpha": 1.0, "beta": 1.0},
                    "exact": True,
                },
                "points": point_rows,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return grid, source_report, [row[0] for row in points]


def test_level0_trace_contract_loads_from_flat_kaggle_capsule(tmp_path):
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    runner_path = capsule / "source_cuda_postcond_full_decode_timing.py"
    contract_path = capsule / "decoder_level0_trace_contract.py"
    shutil.copy2(Path(source_runner.__file__), runner_path)
    shutil.copy2(
        Path(source_runner.__file__).with_name("decoder_level0_trace_contract.py"),
        contract_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import source_cuda_postcond_full_decode_timing as runner; "
                "contract = runner.load_decoder_level0_trace_contract(); "
                "assert contract.__name__ == 'decoder_level0_trace_contract'"
            ),
        ],
        cwd=capsule,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_level0_trace_contract_missing_from_flat_capsule_fails_locally(tmp_path):
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    shutil.copy2(
        Path(source_runner.__file__),
        capsule / "source_cuda_postcond_full_decode_timing.py",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import source_cuda_postcond_full_decode_timing as runner; "
                "\ntry:\n"
                "    runner.load_decoder_level0_trace_contract()\n"
                "except ModuleNotFoundError as exc:\n"
                "    assert exc.name == 'decoder_level0_trace_contract'\n"
                "    assert 'adjacent contract' in str(exc)\n"
                "else:\n"
                "    raise AssertionError('inherited package contract was accepted')\n"
            ),
        ],
        cwd=capsule,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_level0_trace_contract_package_import_keeps_package_identity():
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    contract = source_runner.load_decoder_level0_trace_contract()

    assert contract.__name__ == "scripts.decoder_level0_trace_contract"


def test_level1_trace_contract_loads_from_flat_kaggle_capsule(tmp_path):
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    shutil.copy2(
        Path(source_runner.__file__),
        capsule / "source_cuda_postcond_full_decode_timing.py",
    )
    shutil.copy2(
        Path(source_runner.__file__).with_name("decoder_level1_trace_contract.py"),
        capsule / "decoder_level1_trace_contract.py",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import source_cuda_postcond_full_decode_timing as runner; "
                "contract = runner.load_decoder_level1_trace_contract(); "
                "assert contract.__name__ == 'decoder_level1_trace_contract'"
            ),
        ],
        cwd=capsule,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_level1_trace_contract_missing_from_flat_capsule_fails_locally(tmp_path):
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    shutil.copy2(
        Path(source_runner.__file__),
        capsule / "source_cuda_postcond_full_decode_timing.py",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import source_cuda_postcond_full_decode_timing as runner; "
                "\ntry:\n"
                "    runner.load_decoder_level1_trace_contract()\n"
                "except ModuleNotFoundError as exc:\n"
                "    assert exc.name == 'decoder_level1_trace_contract'\n"
                "    assert 'adjacent contract' in str(exc)\n"
                "else:\n"
                "    raise AssertionError('inherited package contract was accepted')\n"
            ),
        ],
        cwd=capsule,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_parser_exposes_decoder_level1_trace_mode():
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    args = source_runner.build_parser().parse_args(
        [
            "--output-json",
            "report.json",
            "--decoder-level1-trace",
        ]
    )

    assert args.decoder_level1_trace is True


def test_full_decoder_hash_contract_loads_from_flat_kaggle_capsule(tmp_path):
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    runner_path = Path(source_runner.__file__)
    shutil.copy2(
        runner_path,
        capsule / "source_cuda_postcond_full_decode_timing.py",
    )
    shutil.copy2(
        runner_path.with_name("decoder_full_hash_ledger_contract.py"),
        capsule / "decoder_full_hash_ledger_contract.py",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import source_cuda_postcond_full_decode_timing as runner; "
                "contract = runner.load_decoder_full_hash_ledger_contract(); "
                "assert contract.__name__ == "
                "'decoder_full_hash_ledger_contract'"
            ),
        ],
        cwd=capsule,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_parser_exposes_full_decoder_hash_ledger_modifier():
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    args = source_runner.build_parser().parse_args(
        [
            "--output-json",
            "report.json",
            "--decoder-level1-trace",
            "--full-decoder-hash-ledger",
        ]
    )

    assert args.decoder_level1_trace is True
    assert args.full_decoder_hash_ledger is True


def test_full_decoder_hash_ledger_requires_level1_trace_mode(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    args = _shape_slat_decode_args(
        tmp_path,
        grid,
        source_report,
        point_names,
    )
    args.append("--full-decoder-hash-ledger")

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert "--decoder-level1-trace" in report["error"]


def test_full_decoder_hash_ledger_preflight_records_distinct_route(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    args = _shape_slat_decode_args(
        tmp_path,
        grid,
        source_report,
        point_names,
    )
    args.extend(
        ["--decoder-level1-trace", "--full-decoder-hash-ledger"]
    )

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["requested_route"]["full_decoder_hash_ledger"] is True
    assert report["requested_route"]["decoder_output_head_backend"] == (
        "torch-sparse-linear-fp32"
    )
    assert report["effective_route"]["route"] == (
        "official-source-cuda-shape-decoder-full-hash-ledger"
    )
    assert report["effective_route"]["full_decoder_hash_ledger"] is True
    assert report["effective_route"]["decoder_output_head_backend"] == (
        "torch-sparse-linear-fp32"
    )


def test_level2_block0_trace_contract_loads_from_flat_kaggle_capsule(tmp_path):
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    shutil.copy2(
        Path(source_runner.__file__),
        capsule / "source_cuda_postcond_full_decode_timing.py",
    )
    shutil.copy2(
        Path(source_runner.__file__).with_name(
            "decoder_level2_block0_trace_contract.py"
        ),
        capsule / "decoder_level2_block0_trace_contract.py",
    )
    shutil.copy2(
        Path(source_runner.__file__).with_name(
            "decoder_level1_trace_contract.py"
        ),
        capsule / "decoder_level1_trace_contract.py",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import source_cuda_postcond_full_decode_timing as runner; "
                "contract = runner.load_decoder_level2_block0_trace_contract(); "
                "assert contract.__name__ == "
                "'decoder_level2_block0_trace_contract'"
            ),
        ],
        cwd=capsule,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_parser_exposes_decoder_level2_block0_trace_mode():
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    args = source_runner.build_parser().parse_args(
        [
            "--output-json",
            "report.json",
            "--decoder-level2-block0-trace",
        ]
    )

    assert args.decoder_level2_block0_trace is True


def test_level2_subdiv_trace_contract_loads_from_flat_kaggle_capsule(
    tmp_path,
):
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    runner_path = Path(source_runner.__file__)
    shutil.copy2(
        runner_path,
        capsule / "source_cuda_postcond_full_decode_timing.py",
    )
    for contract_name in (
        "decoder_level2_subdiv_trace_contract.py",
        "decoder_level2_block0_trace_contract.py",
        "decoder_level1_trace_contract.py",
    ):
        shutil.copy2(
            runner_path.with_name(contract_name),
            capsule / contract_name,
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import source_cuda_postcond_full_decode_timing as runner; "
                "contract = runner.load_decoder_level2_subdiv_trace_contract(); "
                "assert contract.__name__ == "
                "'decoder_level2_subdiv_trace_contract'"
            ),
        ],
        cwd=capsule,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_parser_exposes_decoder_level2_subdiv_trace_mode():
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    args = source_runner.build_parser().parse_args(
        [
            "--output-json",
            "report.json",
            "--decoder-level2-subdiv-trace",
        ]
    )

    assert args.decoder_level2_subdiv_trace is True


def test_level2_norm2_trace_contract_loads_from_flat_kaggle_capsule(
    tmp_path,
):
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    runner_path = Path(source_runner.__file__)
    shutil.copy2(
        runner_path,
        capsule / "source_cuda_postcond_full_decode_timing.py",
    )
    shutil.copy2(
        runner_path.with_name("decoder_level2_norm2_trace_contract.py"),
        capsule / "decoder_level2_norm2_trace_contract.py",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import source_cuda_postcond_full_decode_timing as runner; "
                "contract = runner.load_decoder_level2_norm2_trace_contract(); "
                "assert contract.__name__ == "
                "'decoder_level2_norm2_trace_contract'"
            ),
        ],
        cwd=capsule,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_parser_exposes_decoder_level2_norm2_trace_mode():
    from scripts import source_cuda_postcond_full_decode_timing as source_runner

    args = source_runner.build_parser().parse_args(
        [
            "--output-json",
            "report.json",
            "--decoder-level2-norm2-trace",
        ]
    )

    assert args.decoder_level2_norm2_trace is True


def _shape_slat_decode_args(
    tmp_path,
    grid,
    source_report,
    point_names,
    *,
    bind_expected_suffix_digests=True,
):
    import hashlib

    args = [
        "--output-json",
        str(tmp_path / "decode-report.json"),
        "--shape-slat-grid",
        str(grid),
        "--shape-slat-grid-report",
        str(source_report),
        "--output-dir",
        str(tmp_path / "meshes"),
        "--no-download",
        *[
            item
            for point_name in point_names
            for item in ("--shape-slat-point", point_name)
        ],
    ]
    if bind_expected_suffix_digests and any(
        point_name.startswith("switch-") for point_name in point_names
    ):
        args.extend(
            [
                "--shape-slat-grid-sha256",
                hashlib.sha256(grid.read_bytes()).hexdigest(),
                "--shape-slat-grid-report-sha256",
                hashlib.sha256(source_report.read_bytes()).hexdigest(),
            ]
        )
    return args


def test_shape_slat_level0_trace_preflight_records_distinct_effective_route(
    tmp_path,
):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    args = _shape_slat_decode_args(
        tmp_path,
        grid,
        source_report,
        point_names,
    )
    args.append("--decoder-level0-trace")

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    trace_path = tmp_path / "meshes" / "alpha-1_beta-1.decoder-level0-trace.npz"
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["requested_route"]["decoder_level0_trace"] is True
    assert (
        report["effective_route"]["route"]
        == "official-source-cuda-shape-decoder-level0-trace"
    )
    assert report["effective_route"]["device_type"] == "not_loaded_no_download"
    assert report["decoder_trace_artifacts"] == [
        {
            "coordinate_key": "alpha-1_beta-1",
            "path": str(trace_path),
            "status": "not_written_no_download",
        }
    ]
    assert report["mesh_artifacts"] == []
    assert report["decoder_state_artifacts"] == []
    assert not trace_path.exists()


def test_shape_slat_level0_trace_and_raw_state_are_mutually_exclusive(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    args = _shape_slat_decode_args(
        tmp_path,
        grid,
        source_report,
        point_names,
    )
    args.extend(["--decoder-level0-trace", "--decoder-state-only"])

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert "mutually exclusive" in report["error"]


def test_shape_slat_level1_trace_preflight_records_distinct_effective_route(
    tmp_path,
):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    args = _shape_slat_decode_args(
        tmp_path,
        grid,
        source_report,
        point_names,
    )
    args.append("--decoder-level1-trace")

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    trace_path = tmp_path / "meshes" / "alpha-1_beta-1.decoder-level1-trace.npz"
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["requested_route"]["decoder_level1_trace"] is True
    assert report["requested_route"]["raw_meshes"] is False
    assert (
        report["effective_route"]["route"]
        == "official-source-cuda-shape-decoder-level1-trace"
    )
    assert report["effective_route"]["device_type"] == "not_loaded_no_download"
    assert report["decoder_trace_artifacts"] == [
        {
            "coordinate_key": "alpha-1_beta-1",
            "path": str(trace_path),
            "status": "not_written_no_download",
        }
    ]
    assert report["mesh_artifacts"] == []
    assert report["decoder_state_artifacts"] == []
    assert not trace_path.exists()


def test_shape_slat_level0_and_level1_trace_are_mutually_exclusive(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    args = _shape_slat_decode_args(
        tmp_path,
        grid,
        source_report,
        point_names,
    )
    args.extend(["--decoder-level0-trace", "--decoder-level1-trace"])

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert "mutually exclusive" in report["error"]


def test_shape_slat_level2_block0_trace_preflight_records_route(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    args = _shape_slat_decode_args(
        tmp_path,
        grid,
        source_report,
        point_names,
    )
    args.append("--decoder-level2-block0-trace")

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    trace_path = (
        tmp_path
        / "meshes"
        / "alpha-1_beta-1.decoder-level2-block0-trace.npz"
    )
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["requested_route"]["decoder_level2_block0_trace"] is True
    assert report["requested_route"]["raw_meshes"] is False
    assert report["effective_route"]["route"] == (
        "official-source-cuda-shape-decoder-level2-block0-trace"
    )
    assert report["effective_route"]["decoder_level2_block0_trace"] is True
    assert report["decoder_trace_artifacts"] == [
        {
            "coordinate_key": "alpha-1_beta-1",
            "path": str(trace_path),
            "status": "not_written_no_download",
        }
    ]
    assert not trace_path.exists()


def test_shape_slat_level1_and_level2_block0_are_mutually_exclusive(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    args = _shape_slat_decode_args(
        tmp_path,
        grid,
        source_report,
        point_names,
    )
    args.extend(
        ["--decoder-level1-trace", "--decoder-level2-block0-trace"]
    )

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert "mutually exclusive" in report["error"]


def test_shape_slat_level2_subdiv_trace_preflight_records_route(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    args = _shape_slat_decode_args(
        tmp_path,
        grid,
        source_report,
        point_names,
    )
    args.append("--decoder-level2-subdiv-trace")

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    trace_path = (
        tmp_path
        / "meshes"
        / "alpha-1_beta-1.decoder-level2-subdiv-trace.npz"
    )
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["requested_route"]["decoder_level2_subdiv_trace"] is True
    assert report["requested_route"]["raw_meshes"] is False
    assert report["effective_route"]["route"] == (
        "official-source-cuda-shape-decoder-level2-subdiv-trace"
    )
    assert report["effective_route"]["decoder_level2_subdiv_trace"] is True
    assert report["decoder_trace_artifacts"] == [
        {
            "coordinate_key": "alpha-1_beta-1",
            "path": str(trace_path),
            "status": "not_written_no_download",
        }
    ]
    assert not trace_path.exists()


def test_shape_slat_level2_norm2_trace_preflight_records_route(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    args = _shape_slat_decode_args(
        tmp_path,
        grid,
        source_report,
        point_names,
    )
    args.append("--decoder-level2-norm2-trace")

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    trace_path = (
        tmp_path
        / "meshes"
        / "alpha-1_beta-1.decoder-level2-norm2-trace.npz"
    )
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["requested_route"]["decoder_level2_norm2_trace"] is True
    assert report["requested_route"]["raw_meshes"] is False
    assert report["effective_route"]["route"] == (
        "official-source-cuda-shape-decoder-level2-norm2-trace"
    )
    assert report["effective_route"]["decoder_level2_norm2_trace"] is True
    assert report["effective_route"]["normalization_backend"] == (
        "official-source-module-layernorm"
    )
    assert report["decoder_trace_artifacts"] == [
        {
            "coordinate_key": "alpha-1_beta-1",
            "path": str(trace_path),
            "status": "not_written_no_download",
        }
    ]
    assert not trace_path.exists()


def test_shape_slat_level1_and_level2_subdiv_are_mutually_exclusive(
    tmp_path,
):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    args = _shape_slat_decode_args(
        tmp_path,
        grid,
        source_report,
        point_names,
    )
    args.extend(
        ["--decoder-level1-trace", "--decoder-level2-subdiv-trace"]
    )

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert "mutually exclusive" in report["error"]


def _write_shape_slat_suffix_fixture(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    import numpy as np

    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    arrays = {"coords": coords, "switch_steps": np.arange(9, dtype=np.int32)}
    points = []
    for step in range(9):
        output_key = f"switch_{step}_shape_slat"
        values = np.full((2, 32), step + 0.25, dtype=np.float32)
        arrays[output_key] = values
        points.append(
            {
                "switch_step": step,
                "source_step_indices": list(range(step, 8)),
                "source_step_count": 8 - step,
                "output_key": output_key,
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "shape": [2, 32],
            }
        )

    effective_route = {
        "route": (
            "official-source-cuda-shape-flow-suffix-ladder-"
            "from-exact-mlx-prefixes"
        ),
        "device_type": "cuda",
        "cuda_device": "Tesla T4",
        "attention_backend": "sdpa",
        "conv_backend": "none",
        "steps": 8,
        "switch_steps": list(range(9)),
        "one_model_load": True,
        "comparison_class": "exact-mlx-prefix-plus-source-cuda-suffix",
    }
    timing = {
        "source_steps_completed": 36,
        "source_steps_requested": 36,
        "switch_points_completed": 9,
        "switch_points_requested": 9,
    }
    inputs = {
        "conditioning_sha256": "conditioning-sha256",
        "accepted_source_baseline_sha256": "source-baseline-sha256",
    }
    pairwise = {
        f"{left}:{right}": {
            "mean_abs": float(abs(left - right)),
            "max_abs": float(abs(left - right)),
            "nonzero": 0 if left == right else 64,
        }
        for left in range(9)
        for right in range(9)
    }
    forbidden_inferences = [
        "not final mesh, texture, winding, or GLB evidence",
        "not proof of a visual basin until quotient-distinct endpoints are decoded",
    ]
    arrays["metadata_json"] = np.asarray(
        json.dumps(
            {
                "schema": (
                    "trellis2mlx.source_cuda_shape_flow_suffix_ladder.artifact.v1"
                ),
                "artifact_status": "computed_pending_serialization",
                "external_report_required": True,
                "effective_route": effective_route,
                "inputs": inputs,
                "points": points,
                "pairwise": pairwise,
                "timing": timing,
                "forbidden_inferences": forbidden_inferences,
            },
            sort_keys=True,
        )
    )
    grid = Path(tmp_path) / "suffix-result.npz"
    np.savez(grid, **arrays)
    report = Path(tmp_path) / "suffix-result.json"
    report.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.source_cuda_shape_flow_suffix_ladder.v1",
                "status": "done",
                "effective_route": effective_route,
                "inputs": inputs,
                "primary_output": {
                    "path": grid.name,
                    "sha256": hashlib.sha256(grid.read_bytes()).hexdigest(),
                    "size_bytes": grid.stat().st_size,
                    "keys": sorted(arrays),
                    "validation": {"point_arrays_bound": True, "switch_count": 9},
                },
                "points": points,
                "pairwise": pairwise,
                "timing": timing,
                "forbidden_inferences": forbidden_inferences,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return grid, report


def _write_source_shape_flow_steps_fixture(tmp_path):
    import hashlib
    import json

    import numpy as np

    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    sample_next = np.arange(8 * 2 * 32, dtype=np.float32).reshape(8, 2, 32)
    arrays = {
        "coords": coords,
        "sample_next": sample_next,
    }
    array_manifest = {
        name: {
            "dtype": str(values.dtype),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "shape": list(values.shape),
        }
        for name, values in arrays.items()
    }
    effective_route = {
        "route": "official-source-cuda-shape-flow-recurrence",
        "device_type": "cuda",
        "cuda_device": "Tesla T4",
        "attention_backend": "sdpa",
        "conv_backend": "none",
        "model_ref": (
            "microsoft/TRELLIS.2-4B/ckpts/"
            "slat_flow_img2shape_dit_1_3B_512_bf16"
        ),
        "rescale_t": 3.0,
        "candidate_names": ["source-native-control"],
        "steps": 8,
        "one_model_load": True,
        "comparison_class": "source-native-eight-step-recurrence",
    }
    metadata = {
        "schema": (
            "trellis2mlx.source_cuda_shape_flow_transition0_recoverability.v1."
            "source_recurrence.v2"
        ),
        "artifact_status": "computed_pending_external_report",
        "external_report_required": True,
        "effective_route": effective_route,
        "arrays": array_manifest,
        "source_candidate": {
            "name": "source-native-control",
            "source_step_count": 8,
            "source_step_indices": list(range(8)),
        },
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    primary = Path(tmp_path) / "source-steps.npz"
    np.savez(primary, **arrays)
    primary_sha256 = hashlib.sha256(primary.read_bytes()).hexdigest()
    report = Path(tmp_path) / "source-steps.json"
    report.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.source_cuda_shape_flow_steps.v1",
                "status": "done",
                "last_trustworthy_phase": "source_recurrence",
                "effective_route": effective_route,
                "candidates": [metadata["source_candidate"]],
                "primary_output": {
                    "path": primary.name,
                    "sha256": primary_sha256,
                    "size_bytes": primary.stat().st_size,
                    "validation": {
                        "all_arrays_bound": True,
                        "recurrence_exact": True,
                        "step_count": 8,
                        "token_count": 2,
                        "channel_count": 32,
                    },
                },
                "timing": {
                    "source_steps_completed": 8,
                    "source_steps_requested": 8,
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return primary, report


def _rewrite_source_shape_flow_route_self_consistently(
    primary,
    report,
    **route_updates,
):
    import hashlib
    import json

    import numpy as np

    payload = json.loads(report.read_text())
    payload["effective_route"].update(route_updates)
    with np.load(primary, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["effective_route"].update(route_updates)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(primary, **arrays)
    payload["primary_output"].update(
        {
            "sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
            "size_bytes": primary.stat().st_size,
        }
    )
    report.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _source_shape_flow_endpoint_decode_args(tmp_path, primary, report):
    import hashlib

    return [
        "--output-json",
        str(tmp_path / "decode-report.json"),
        "--source-shape-flow-steps",
        str(primary),
        "--source-shape-flow-steps-report",
        str(report),
        "--source-shape-flow-steps-sha256",
        hashlib.sha256(primary.read_bytes()).hexdigest(),
        "--source-shape-flow-steps-report-sha256",
        hashlib.sha256(report.read_bytes()).hexdigest(),
        "--output-dir",
        str(tmp_path / "meshes"),
        "--no-download",
    ]


def test_source_shape_flow_terminal_decode_preflight_binds_exact_endpoint(tmp_path):
    import hashlib
    import json

    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import main

    primary, source_report = _write_source_shape_flow_steps_fixture(tmp_path)
    rc = main(
        _source_shape_flow_endpoint_decode_args(tmp_path, primary, source_report)
    )

    report = json.loads((tmp_path / "decode-report.json").read_text())
    with np.load(primary, allow_pickle=False) as archive:
        terminal = np.asarray(archive["sample_next"][-1])
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["selected_point_names"] == ["source-terminal"]
    assert report["source_shape_slat_route"]["route"] == (
        "official-source-cuda-shape-flow-recurrence-terminal"
    )
    assert report["source_shape_slat_route"]["model_ref"] == (
        "microsoft/TRELLIS.2-4B/ckpts/"
        "slat_flow_img2shape_dit_1_3B_512_bf16"
    )
    assert report["source_shape_slat_route"]["rescale_t"] == 3.0
    assert report["source_shape_slat_route"]["candidate_names"] == [
        "source-native-control"
    ]
    assert report["selected_points"] == [
        {
            "coordinate_key": "source-terminal",
            "output_key": "sample_next[-1]",
            "sha256": hashlib.sha256(terminal.tobytes()).hexdigest(),
            "shape": [2, 32],
            "source_step_index": 7,
            "normalization_status": "deferred_to_bound_source_pipeline_config",
        }
    ]
    assert report["source_shape_slat_primary"]["sha256"] == hashlib.sha256(
        primary.read_bytes()
    ).hexdigest()
    assert report["effective_route"]["route"] == (
        "official-source-cuda-shape-slat-decoder"
    )


def test_source_shape_flow_terminal_decode_requires_bound_digests_before_cleanup(
    tmp_path,
):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    primary, source_report = _write_source_shape_flow_steps_fixture(tmp_path)
    args = _source_shape_flow_endpoint_decode_args(tmp_path, primary, source_report)
    del args[args.index("--source-shape-flow-steps-sha256") : args.index("--output-dir")]
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale = output_dir / "source-terminal.raw.ply"
    stale.write_bytes(b"stale")

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert "expected report and NPZ SHA256 values are required" in report["error"]
    assert stale.read_bytes() == b"stale"


def test_source_shape_flow_terminal_decode_rejects_substitute_before_cleanup(
    tmp_path,
):
    import json

    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import main

    primary, source_report = _write_source_shape_flow_steps_fixture(tmp_path)
    args = _source_shape_flow_endpoint_decode_args(tmp_path, primary, source_report)
    with np.load(primary, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["sample_next"] = arrays["sample_next"].copy()
    arrays["sample_next"][-1, 0, 0] += np.float32(1.0)
    np.savez(primary, **arrays)
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale = output_dir / "source-terminal.raw.ply"
    stale.write_bytes(b"stale")

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert "expected source shape-flow primary digest mismatch" in report["error"]
    assert stale.read_bytes() == b"stale"


@pytest.mark.parametrize("failure", ["metadata", "incomplete", "nonfinite"])
def test_source_shape_flow_terminal_decode_rejects_invalid_recurrence(
    tmp_path,
    failure,
):
    import hashlib
    import json

    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import main

    primary, source_report = _write_source_shape_flow_steps_fixture(tmp_path)
    with np.load(primary, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if failure == "metadata":
        arrays["metadata_json"] = np.asarray("{not-json")
    elif failure == "incomplete":
        arrays["sample_next"] = arrays["sample_next"][:-1]
    else:
        arrays["sample_next"] = arrays["sample_next"].copy()
        arrays["sample_next"][-1, 0, 0] = np.nan
    if failure != "metadata":
        metadata["arrays"]["sample_next"] = {
            "dtype": str(arrays["sample_next"].dtype),
            "sha256": hashlib.sha256(
                arrays["sample_next"].tobytes()
            ).hexdigest(),
            "shape": list(arrays["sample_next"].shape),
        }
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True)
        )
    np.savez(primary, **arrays)
    payload = json.loads(source_report.read_text())
    payload["primary_output"]["sha256"] = hashlib.sha256(
        primary.read_bytes()
    ).hexdigest()
    payload["primary_output"]["size_bytes"] = primary.stat().st_size
    source_report.write_text(json.dumps(payload, sort_keys=True) + "\n")
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale = output_dir / "source-terminal.raw.ply"
    stale.write_bytes(b"stale source decoder evidence")

    rc = main(
        _source_shape_flow_endpoint_decode_args(tmp_path, primary, source_report)
    )

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    if failure != "metadata":
        expected = (
            "exactly 8 steps" if failure == "incomplete" else "non-finite"
        )
        assert expected in report["error"]
    assert stale.read_bytes() == b"stale source decoder evidence"


@pytest.mark.parametrize(
    ("route_update", "expected_field"),
    [
        ({"model_ref": "microsoft/TRELLIS.2-4B/ckpts/not-the-model"}, "model_ref"),
        ({"rescale_t": 1.0}, "rescale_t"),
        ({"candidate_names": ["different-source-control"]}, "candidate_names"),
    ],
)
def test_source_shape_flow_terminal_decode_rejects_route_substitution_before_cleanup(
    tmp_path,
    route_update,
    expected_field,
):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    primary, source_report = _write_source_shape_flow_steps_fixture(tmp_path)
    _rewrite_source_shape_flow_route_self_consistently(
        primary,
        source_report,
        **route_update,
    )
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale = output_dir / "source-terminal.raw.ply"
    stale.write_bytes(b"stale source decoder evidence")

    rc = main(
        _source_shape_flow_endpoint_decode_args(tmp_path, primary, source_report)
    )

    decode_report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert decode_report["failure_phase"] == "input_validation"
    assert f"route mismatch for {expected_field}" in decode_report["error"]
    assert stale.read_bytes() == b"stale source decoder evidence"


def test_source_shape_flow_level0_trace_binds_denormalized_decoder_input(
    tmp_path,
    monkeypatch,
):
    import contextlib
    import hashlib
    import json
    import sys
    import types

    import numpy as np

    import scripts.source_cuda_postcond_full_decode_timing as runner
    from scripts import decoder_level0_trace_contract as trace_hash_contract

    primary, source_report = _write_source_shape_flow_steps_fixture(tmp_path)
    with np.load(primary, allow_pickle=False) as archive:
        coords = np.asarray(archive["coords"])
        normalized_endpoint = np.asarray(archive["sample_next"][-1])
    mean = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    std = np.linspace(0.5, 2.0, 32, dtype=np.float32)
    denormalized = normalized_endpoint * std[None] + mean[None]
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "args": {
                    "models": {
                        "shape_slat_decoder": "ckpts/shape-decoder",
                    },
                    "shape_slat_normalization": {
                        "mean": mean.tolist(),
                        "std": std.tolist(),
                    },
                }
            }
        )
        + "\n"
    )

    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values)
            self.device = "cuda"

        def __array__(self, dtype=None):
            return np.asarray(self.values, dtype=dtype)

        def to(self, *_args, **_kwargs):
            return self

        def __getitem__(self, item):
            return FakeTensor(self.values[item])

        def __mul__(self, other):
            return FakeTensor(self.values * np.asarray(other))

        def __add__(self, other):
            return FakeTensor(self.values + np.asarray(other))

    class FakeParameter:
        def numel(self):
            return 7

    class FakeDecoder:
        def __init__(self):
            self.training = True
            self.low_vram = False

        def set_resolution(self, resolution):
            self.resolution = resolution

        def eval(self):
            self.training = False
            return self

        def to(self, _device):
            return self

        def parameters(self):
            return [FakeParameter()]

    decoder = FakeDecoder()
    source_models = types.ModuleType("trellis2.models")
    source_models.from_pretrained = lambda _model_ref: decoder
    sparse_module = types.ModuleType("trellis2.modules.sparse")
    sparse_module.SparseTensor = lambda **kwargs: types.SimpleNamespace(**kwargs)
    sparse_module.config = types.SimpleNamespace(ATTN="sdpa", CONV="none")
    modules_package = types.ModuleType("trellis2.modules")
    modules_package.sparse = sparse_module
    trellis2_package = types.ModuleType("trellis2")
    trellis2_package.models = source_models
    trellis2_package.modules = modules_package

    torch_module = types.ModuleType("torch")
    torch_module.__version__ = "test-cuda"
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _index: "Tesla T4",
        synchronize=lambda: None,
    )
    torch_module.device = lambda value: value
    torch_module.set_grad_enabled = lambda _enabled: None
    torch_module.from_numpy = FakeTensor
    torch_module.tensor = lambda values: FakeTensor(
        np.asarray(values, dtype=np.float32)
    )
    torch_module.no_grad = contextlib.nullcontext
    hub_module = types.ModuleType("huggingface_hub")
    hub_module.hf_hub_download = lambda _repo, _path: str(config_path)

    for name, module in {
        "torch": torch_module,
        "huggingface_hub": hub_module,
        "trellis2": trellis2_package,
        "trellis2.models": source_models,
        "trellis2.modules": modules_package,
        "trellis2.modules.sparse": sparse_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner, "extract_source", lambda *_args: tmp_path)
    monkeypatch.setattr(
        runner,
        "install_mesh_override",
        lambda *_args: {"status": "installed"},
    )
    monkeypatch.setattr(
        runner,
        "capture_source_decoder_level0_trace",
        lambda effective_decoder, _shape_slat: {}
        if effective_decoder is decoder
        else (_ for _ in ()).throw(AssertionError("wrong decoder")),
    )

    def write_trace(path, _arrays, **_kwargs):
        path.write_bytes(b"authenticated mock trace")
        return {"reopened_exact": True}

    fake_contract = types.SimpleNamespace(
        write_decoder_level0_trace_npz=write_trace,
        decoder_trace_input_sha256=(
            trace_hash_contract.decoder_trace_input_sha256
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_decoder_level0_trace_contract",
        lambda: fake_contract,
    )

    args = _source_shape_flow_endpoint_decode_args(
        tmp_path,
        primary,
        source_report,
    )
    args.remove("--no-download")
    args.append("--decoder-level0-trace")

    rc = runner.main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    expected_hash = trace_hash_contract.decoder_trace_input_sha256(
        denormalized,
        coords,
    )
    normalized_hash = trace_hash_contract.decoder_trace_input_sha256(
        normalized_endpoint,
        coords,
    )
    assert rc == 0
    assert report["status"] == "done"
    assert report["source_endpoint_normalization"][
        "normalized_endpoint_sha256"
    ] == hashlib.sha256(normalized_endpoint.tobytes()).hexdigest()
    assert report["source_endpoint_normalization"][
        "decoder_input_sha256"
    ] == hashlib.sha256(denormalized.tobytes()).hexdigest()
    assert report["decoder_trace_artifacts"][0][
        "input_tensor_sha256"
    ] == expected_hash
    assert expected_hash != normalized_hash


@pytest.mark.parametrize(
    ("mode_args", "output_name"),
    [
        ([], "source-terminal.raw.ply"),
        (["--decoder-state-only"], "source-terminal.decoder-state.npz"),
        (
            ["--decoder-level0-trace"],
            "source-terminal.decoder-level0-trace.npz",
        ),
    ],
)
def test_source_shape_flow_terminal_output_collision_preserves_stale_evidence(
    tmp_path,
    mode_args,
    output_name,
):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    primary, source_report = _write_source_shape_flow_steps_fixture(tmp_path)
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale = output_dir / output_name
    stale.write_bytes(b"stale source decoder evidence")
    args = _source_shape_flow_endpoint_decode_args(
        tmp_path,
        primary,
        source_report,
    )
    args[args.index("--output-json") + 1] = str(stale)
    args.extend(mode_args)

    rc = main(args)

    fallback = stale.with_name(
        f"{stale.name}.selective-decode-failure.json"
    )
    report = json.loads(fallback.read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert report["requested_output_json"] == str(stale)
    assert report["effective_output_json"] == str(fallback)
    assert "collides with an expected" in report["error"]
    assert report["mesh_artifacts"] == []
    assert report["decoder_state_artifacts"] == []
    assert report["decoder_trace_artifacts"] == []
    assert stale.read_bytes() == b"stale source decoder evidence"


def _rewrite_shape_slat_suffix_fixture_self_consistently(grid, report):
    import hashlib
    import json

    import numpy as np

    payload = json.loads(report.read_text())
    with np.load(grid, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    substituted_inputs = dict(payload["inputs"])
    substituted_inputs["substitution_marker"] = "fabricated-but-self-consistent"
    payload["inputs"] = substituted_inputs
    metadata["inputs"] = substituted_inputs
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(grid, **arrays)
    payload["primary_output"]["sha256"] = hashlib.sha256(grid.read_bytes()).hexdigest()
    payload["primary_output"]["size_bytes"] = grid.stat().st_size
    report.write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_shape_slat_grid_decode_preflight_invalidates_stale_outputs(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale_raw = output_dir / "alpha-1_beta-1.raw.ply"
    stale_filled = output_dir / "alpha-1_beta-1.filled.ply"
    stale_raw.write_bytes(b"stale raw")
    stale_filled.write_bytes(b"stale filled")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, point_names))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 0
    assert report["schema"] == "trellis2mlx.source_cuda_shape_slat_grid_decode.v1"
    assert report["status"] == "preflight_stopped"
    assert report["failure_phase"] is None
    assert report["selected_point_names"] == point_names
    assert report["source_basin_route"]["route"].startswith("official-source-cuda-")
    assert report["effective_route"]["route"] == "official-source-cuda-shape-slat-decoder"
    assert report["expected_artifact_count"] == 8
    assert report["written_artifact_count"] == 0
    assert {row["status"] for row in report["mesh_artifacts"]} == {"not_written_no_download"}
    assert not stale_raw.exists()
    assert not stale_filled.exists()


def test_shape_slat_decoder_state_preflight_binds_raw_output_contract(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale = output_dir / "alpha-1_beta-1.decoder-state.npz"
    stale.write_bytes(b"stale")
    args = _shape_slat_decode_args(tmp_path, grid, source_report, point_names)
    args.append("--decoder-state-only")

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["requested_route"]["route"] == (
        "official-source-cuda-shape-slat-decoder-raw-state"
    )
    assert report["requested_route"]["decoder_state_only"] is True
    assert report["expected_artifact_count"] == 1
    assert report["mesh_artifacts"] == []
    assert report["decoder_state_artifacts"] == [
        {
            "coordinate_key": "alpha-1_beta-1",
            "path": str(output_dir / "alpha-1_beta-1.decoder-state.npz"),
            "status": "not_written_no_download",
        }
    ]
    assert report["effective_route"]["route"] == (
        "official-source-cuda-shape-slat-decoder-raw-state"
    )
    assert report["effective_route"]["mesh_conversion"] is False
    assert not stale.exists()


def test_shape_slat_suffix_decode_preflight_admits_exact_switch_identity(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report = _write_shape_slat_suffix_fixture(tmp_path)
    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, ["switch-1"]))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["selected_point_names"] == ["switch-1"]
    assert report["source_shape_slat_route"]["route"].startswith(
        "official-source-cuda-shape-flow-suffix-ladder"
    )
    assert report["selected_points"] == [
        {
            "switch_step": 1,
            "coordinate_key": "switch-1",
            "output_key": "switch_1_shape_slat",
            "sha256": report["selected_points"][0]["sha256"],
            "shape": [2, 32],
        }
    ]


def test_shape_slat_suffix_decode_requires_expected_digests_before_stale_cleanup(
    tmp_path,
):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report = _write_shape_slat_suffix_fixture(tmp_path)
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale = output_dir / "switch-1.raw.ply"
    stale.write_bytes(b"stale")

    rc = main(
        _shape_slat_decode_args(
            tmp_path,
            grid,
            source_report,
            ["switch-1"],
            bind_expected_suffix_digests=False,
        )
    )

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert "expected report and NPZ SHA256 values are required" in report["error"]
    assert stale.read_bytes() == b"stale"


def test_shape_slat_suffix_decode_rejects_self_consistent_substitute_before_cleanup(
    tmp_path,
):
    import hashlib
    import json

    from scripts.source_cuda_postcond_full_decode_timing import (
        build_parser,
        run_shape_slat_grid_decode,
    )

    grid, source_report = _write_shape_slat_suffix_fixture(tmp_path)
    args = build_parser().parse_args(
        _shape_slat_decode_args(tmp_path, grid, source_report, ["switch-1"])
    )
    args.shape_slat_grid_sha256 = hashlib.sha256(grid.read_bytes()).hexdigest()
    args.shape_slat_grid_report_sha256 = hashlib.sha256(
        source_report.read_bytes()
    ).hexdigest()
    _rewrite_shape_slat_suffix_fixture_self_consistently(grid, source_report)
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale = output_dir / "switch-1.raw.ply"
    stale.write_bytes(b"stale")

    rc = run_shape_slat_grid_decode(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert "expected source shape-SLat" in report["error"]
    assert stale.read_bytes() == b"stale"


def test_shape_slat_suffix_decode_rejects_partial_ladder(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report = _write_shape_slat_suffix_fixture(tmp_path)
    payload = json.loads(source_report.read_text())
    payload["timing"]["source_steps_completed"] = 35
    source_report.write_text(json.dumps(payload) + "\n")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, ["switch-1"]))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    assert "source_steps_completed" in report["error"]


def test_shape_slat_suffix_decode_rejects_substituted_artifact_route(tmp_path):
    import hashlib
    import json

    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report = _write_shape_slat_suffix_fixture(tmp_path)
    with np.load(grid, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["effective_route"]["device_type"] = "cpu"
    metadata["effective_route"]["route"] = "substituted-non-cuda-route"
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(grid, **arrays)
    payload = json.loads(source_report.read_text())
    payload["primary_output"]["sha256"] = hashlib.sha256(grid.read_bytes()).hexdigest()
    payload["primary_output"]["size_bytes"] = grid.stat().st_size
    source_report.write_text(json.dumps(payload) + "\n")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, ["switch-1"]))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    assert "artifact effective_route differs from external report" in report["error"]


def test_shape_slat_suffix_decode_rejects_substituted_artifact_provenance(
    tmp_path,
):
    import hashlib
    import json

    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report = _write_shape_slat_suffix_fixture(tmp_path)
    with np.load(grid, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["inputs"]["conditioning_sha256"] = "fabricated"
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(grid, **arrays)
    payload = json.loads(source_report.read_text())
    payload["primary_output"]["sha256"] = hashlib.sha256(grid.read_bytes()).hexdigest()
    payload["primary_output"]["size_bytes"] = grid.stat().st_size
    source_report.write_text(json.dumps(payload) + "\n")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, ["switch-1"]))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    assert "artifact inputs differs from external report" in report["error"]


def test_shape_slat_suffix_decode_rejects_missing_unselected_switch_array(tmp_path):
    import hashlib
    import json

    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report = _write_shape_slat_suffix_fixture(tmp_path)
    with np.load(grid, allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "switch_8_shape_slat"
        }
    np.savez(grid, **arrays)
    payload = json.loads(source_report.read_text())
    payload["primary_output"]["sha256"] = hashlib.sha256(grid.read_bytes()).hexdigest()
    payload["primary_output"]["size_bytes"] = grid.stat().st_size
    payload["primary_output"]["keys"] = sorted(arrays)
    source_report.write_text(json.dumps(payload) + "\n")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, ["switch-1"]))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    assert "missing canonical suffix array 'switch_8_shape_slat'" in report["error"]


def test_shape_slat_grid_decode_rejects_duplicate_selection_with_report(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    args = _shape_slat_decode_args(tmp_path, grid, source_report, [point_names[0], point_names[0]])

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert "duplicate --shape-slat-point" in report["error"]


def test_shape_slat_grid_decode_rejects_primary_digest_mismatch(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    payload = json.loads(source_report.read_text())
    payload["primary_output"]["sha256"] = "0" * 64
    source_report.write_text(json.dumps(payload) + "\n")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, point_names))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    assert "primary output digest mismatch" in report["error"]
    assert "primary output digest mismatch" in report["traceback"]
    assert "_load_selected_shape_slat_inputs" in report["traceback"]


def test_shape_slat_grid_decode_rejects_selected_array_digest_mismatch_and_stale_output(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    payload = json.loads(source_report.read_text())
    payload["points"][0]["sha256"] = "f" * 64
    source_report.write_text(json.dumps(payload) + "\n")
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale = output_dir / f"{point_names[0]}.raw.ply"
    stale.write_bytes(b"stale")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, point_names))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    assert "selected array digest mismatch" in report["error"]
    assert not stale.exists()


@pytest.mark.parametrize(
    ("eval_clears_training", "fill_holes_changes_geometry"),
    [(True, False), (True, True), (False, False)],
)
def test_shape_slat_grid_decode_records_direct_decoder_eval_mode(
    tmp_path,
    monkeypatch,
    eval_clears_training,
    fill_holes_changes_geometry,
):
    import contextlib
    import json
    import sys
    import types

    import numpy as np

    import scripts.source_cuda_postcond_full_decode_timing as runner

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "args": {
                    "models": {
                        "shape_slat_decoder": "ckpts/shape-decoder",
                    }
                }
            }
        )
        + "\n"
    )

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def to(self, **_kwargs):
            return self

    class FakeParameter:
        def numel(self):
            return 7

    class FakeMesh:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        def fill_holes(self):
            if fill_holes_changes_geometry:
                self.vertices = self.vertices.copy()
                self.vertices[0, 0] = 0.125

    class FakeDecoder:
        def __init__(self):
            self.training = True
            self.low_vram = False

        def set_resolution(self, resolution):
            self.resolution = resolution

        def eval(self):
            if eval_clears_training:
                self.training = False
            return self

        def to(self, _device):
            return self

        def parameters(self):
            return [FakeParameter()]

        def __call__(self, _shape_slat, *, return_subs):
            self.called = True
            assert return_subs is True
            if self.training:
                raise TypeError("'NoneType' object is not iterable")
            return [FakeMesh()], []

    decoder = FakeDecoder()
    source_models = types.ModuleType("trellis2.models")
    source_models.from_pretrained = lambda _model_ref: decoder
    sparse_module = types.ModuleType("trellis2.modules.sparse")
    sparse_module.SparseTensor = lambda **kwargs: kwargs
    sparse_module.config = types.SimpleNamespace(ATTN="sdpa", CONV="none")
    modules_package = types.ModuleType("trellis2.modules")
    modules_package.sparse = sparse_module
    trellis2_package = types.ModuleType("trellis2")
    trellis2_package.models = source_models
    trellis2_package.modules = modules_package

    torch_module = types.ModuleType("torch")
    torch_module.__version__ = "test-cuda"
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _index: "Tesla T4",
        synchronize=lambda: None,
    )
    torch_module.device = lambda value: value
    torch_module.set_grad_enabled = lambda _enabled: None
    torch_module.from_numpy = FakeTensor
    torch_module.no_grad = contextlib.nullcontext
    hub_module = types.ModuleType("huggingface_hub")
    hub_module.hf_hub_download = lambda _repo, _path: str(config_path)

    for name, module in {
        "torch": torch_module,
        "huggingface_hub": hub_module,
        "trellis2": trellis2_package,
        "trellis2.models": source_models,
        "trellis2.modules": modules_package,
        "trellis2.modules.sparse": sparse_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner, "extract_source", lambda *_args: tmp_path)
    monkeypatch.setattr(runner, "install_mesh_override", lambda *_args: {"status": "installed"})

    args = _shape_slat_decode_args(tmp_path, grid, source_report, point_names)
    args.remove("--no-download")
    rc = runner.main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    if eval_clears_training:
        assert rc == 0
        assert decoder.training is False
        assert decoder.called is True
        assert report["model_load"]["training_before_eval"] is True
        assert report["model_load"]["training"] is False
        assert report["effective_route"]["model_training"] is False
        assert "raw_and_filled_meshes" not in report["effective_route"]
        assert report["effective_route"]["post_fill_holes_snapshots"] is True
        assert report["status"] == "done"
        assert report["written_artifact_count"] == 2
        assert report["point_results"][0]["fill_holes_effective_change"] is fill_holes_changes_geometry
        artifacts = {row["variant"]: row for row in report["mesh_artifacts"]}
        assert artifacts["filled"]["fill_holes_effective_change"] is fill_holes_changes_geometry
        assert (artifacts["filled"]["sha256"] != artifacts["raw"]["sha256"]) is fill_holes_changes_geometry
    else:
        assert rc == 1
        assert decoder.training is True
        assert not hasattr(decoder, "called")
        assert report["failure_phase"] == "load_shape_decoder"
        assert report["model_load"]["training_before_eval"] is True
        assert report["model_load"]["training"] is True
        assert "remained in training mode" in report["error"]
        assert "remained in training mode" in report["traceback"]
        assert report["written_artifact_count"] == 0


def test_shape_slat_decoder_state_writes_reopened_validated_primary(
    tmp_path,
    monkeypatch,
):
    import contextlib
    import json
    import sys
    import types

    import numpy as np

    import scripts.source_cuda_postcond_full_decode_timing as runner

    grid, source_report, point_names = _write_shape_slat_grid_fixture(
        tmp_path,
        points=[("alpha-1_beta-1", 1.0, 1.0)],
    )
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "args": {
                    "models": {
                        "shape_slat_decoder": "ckpts/shape-decoder",
                    }
                }
            }
        )
        + "\n"
    )

    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __array__(self, dtype=None):
            return np.asarray(self.values, dtype=dtype)

        @property
        def shape(self):
            return self.values.shape

        def to(self, **_kwargs):
            return self

    class FakeSparseState:
        def __init__(self, feats, coords):
            self.feats = FakeTensor(feats)
            self.coords = FakeTensor(coords)

    class FakeParameter:
        def numel(self):
            return 7

    class FakeDecoder:
        def __init__(self):
            self.training = True
            self.low_vram = False

        def set_resolution(self, resolution):
            self.resolution = resolution

        def eval(self):
            self.training = False
            return self

        def to(self, _device):
            return self

        def parameters(self):
            return [FakeParameter()]

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("raw decoder-state route must not invoke mesh wrapper")

    decoder = FakeDecoder()
    source_models = types.ModuleType("trellis2.models")
    source_models.from_pretrained = lambda _model_ref: decoder
    sparse_module = types.ModuleType("trellis2.modules.sparse")
    sparse_module.SparseTensor = lambda **kwargs: kwargs
    sparse_module.config = types.SimpleNamespace(ATTN="sdpa", CONV="none")
    modules_package = types.ModuleType("trellis2.modules")
    modules_package.sparse = sparse_module
    trellis2_package = types.ModuleType("trellis2")
    trellis2_package.models = source_models
    trellis2_package.modules = modules_package

    torch_module = types.ModuleType("torch")
    torch_module.__version__ = "test-cuda"
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _index: "Tesla T4",
        synchronize=lambda: None,
    )
    torch_module.device = lambda value: value
    torch_module.set_grad_enabled = lambda _enabled: None
    torch_module.from_numpy = FakeTensor
    torch_module.no_grad = contextlib.nullcontext
    hub_module = types.ModuleType("huggingface_hub")
    hub_module.hf_hub_download = lambda _repo, _path: str(config_path)

    for name, module in {
        "torch": torch_module,
        "huggingface_hub": hub_module,
        "trellis2": trellis2_package,
        "trellis2.models": source_models,
        "trellis2.modules": modules_package,
        "trellis2.modules.sparse": sparse_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner, "extract_source", lambda *_args: tmp_path)
    monkeypatch.setattr(runner, "install_mesh_override", lambda *_args: {"status": "installed"})
    monkeypatch.setattr(
        runner,
        "decode_shape_slat_raw",
        lambda effective_decoder, _shape_slat: (
            FakeSparseState(
                np.array(
                    [
                        [0.1, 0.2, 0.3, 1.0, -1.0, 0.5, 0.25],
                        [0.4, 0.5, 0.6, -1.0, 1.0, -0.5, 0.75],
                    ],
                    dtype=np.float32,
                ),
                np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32),
            ),
            [
                FakeSparseState(
                    np.full((2, 8), level + 1, dtype=np.float16),
                    np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32),
                )
                for level in range(4)
            ],
        )
        if effective_decoder is decoder
        else (_ for _ in ()).throw(AssertionError("wrong decoder")),
    )

    args = _shape_slat_decode_args(tmp_path, grid, source_report, point_names)
    args.remove("--no-download")
    args.append("--decoder-state-only")
    rc = runner.main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    artifact = report["decoder_state_artifacts"][0]
    state_path = tmp_path / "meshes" / "alpha-1_beta-1.decoder-state.npz"
    assert rc == 0
    assert report["status"] == "done"
    assert report["written_artifact_count"] == 1
    assert report["effective_route"]["mesh_conversion"] is False
    assert artifact["status"] == "written"
    assert artifact["path"] == str(state_path)
    assert artifact["sha256"] == runner.sha256_file(state_path)
    assert artifact["validation"]["feats_shape"] == [2, 7]
    assert artifact["validation"]["feats_dtype"] == "float32"
    assert artifact["validation"]["coords_shape"] == [2, 4]
    assert artifact["validation"]["coords_dtype"] == "int32"
    assert artifact["validation"]["subdivision_shapes"] == [[2, 8]] * 4
    assert artifact["validation"]["subdivision_coordinate_shapes"] == [[2, 4]] * 4
    assert artifact["validation"]["subdivision_dtypes"] == [
        {"logits": "float16", "coords": "int32"}
    ] * 4
    with np.load(state_path, allow_pickle=False) as archive:
        assert archive.files == [
            "feats",
            "coords",
            "shape_subs_0",
            "shape_subs_0_coords",
            "shape_subs_1",
            "shape_subs_1_coords",
            "shape_subs_2",
            "shape_subs_2_coords",
            "shape_subs_3",
            "shape_subs_3_coords",
        ]
        assert archive["feats"].shape == (2, 7)
        assert archive["coords"].dtype == np.int32
        assert archive["shape_subs_0"].dtype == np.float16
        assert np.isfinite(archive["shape_subs_0"]).all()
        assert np.array_equal(
            archive["shape_subs_3_coords"],
            np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32),
        )

    monkeypatch.setattr(
        runner,
        "decode_shape_slat_raw",
        lambda effective_decoder, _shape_slat: (
            FakeSparseState(
                np.empty((0, 7), dtype=np.float32),
                np.empty((0, 4), dtype=np.int32),
            ),
            [],
        )
        if effective_decoder is decoder
        else (_ for _ in ()).throw(AssertionError("wrong decoder")),
    )

    rc = runner.main(args)

    failed_report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert failed_report["status"] == "failed"
    assert failed_report["failure_phase"] == "decode_selected_points"
    assert failed_report["written_artifact_count"] == 0
    assert failed_report["decoder_state_artifacts"][0]["status"] == "not_written"
    assert "decoder-state output must be nonempty" in failed_report["error"]
    assert not state_path.exists()

    monkeypatch.setattr(
        runner,
        "decode_shape_slat_raw",
        lambda effective_decoder, _shape_slat: (
            FakeSparseState(
                np.ones((2, 7), dtype=np.float32),
                np.array(
                    [[0.0, 1.5, 2.0, 3.0], [0.0, 4.0, 5.0, 6.0]],
                    dtype=np.float32,
                ),
            ),
            [
                FakeSparseState(
                    np.ones((2, 8), dtype=np.float16),
                    np.array(
                        [[0, 1, 2, 3], [0, 4, 5, 6]],
                        dtype=np.int32,
                    ),
                )
                for _ in range(4)
            ],
        )
        if effective_decoder is decoder
        else (_ for _ in ()).throw(AssertionError("wrong decoder")),
    )

    rc = runner.main(args)

    wrong_dtype_report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert wrong_dtype_report["status"] == "failed"
    assert wrong_dtype_report["failure_phase"] == "decode_selected_points"
    assert wrong_dtype_report["written_artifact_count"] == 0
    assert wrong_dtype_report["decoder_state_artifacts"][0]["status"] == "not_written"
    assert "decoder-state coords must have dtype int32" in wrong_dtype_report["error"]
    assert not state_path.exists()


def test_write_decoder_state_requires_four_coordinate_bound_subdivision_levels(
    tmp_path,
):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_decoder_state_npz

    class SparseState:
        def __init__(self, feats, coords):
            self.feats = feats
            self.coords = coords

    final_state = SparseState(
        np.ones((2, 7), dtype=np.float32),
        np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32),
    )
    subdivisions = [
        SparseState(
            np.ones((2, 8), dtype=np.float16),
            np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32),
        )
        for _ in range(3)
    ]

    with pytest.raises(ValueError, match="exactly 4 subdivision levels"):
        write_decoder_state_npz(
            tmp_path / "missing-level.decoder-state.npz",
            final_state,
            subdivisions,
        )

    subdivisions.append(
        SparseState(
            np.empty((0, 8), dtype=np.float16),
            np.empty((0, 4), dtype=np.int32),
        )
    )
    with pytest.raises(ValueError, match="subdivision level 3 must be nonempty"):
        write_decoder_state_npz(
            tmp_path / "empty-level.decoder-state.npz",
            final_state,
            subdivisions,
        )


@pytest.mark.parametrize(
    ("field", "wrong_dtype", "expected_error"),
    [
        ("feats", "float64", "decoder-state feats must have dtype float32"),
        ("coords", "float32", "decoder-state coords must have dtype int32"),
        (
            "subdivision_feats",
            "float32",
            "decoder subdivision logits must have dtype float16 at level 0",
        ),
        (
            "subdivision_coords",
            "float32",
            "decoder subdivision coords must have dtype int32 at level 0",
        ),
    ],
)
def test_write_decoder_state_rejects_raw_dtype_coercion(
    tmp_path,
    field,
    wrong_dtype,
    expected_error,
):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_decoder_state_npz

    class SparseState:
        def __init__(self, feats, coords):
            self.feats = feats
            self.coords = coords

    final_feats = np.ones((2, 7), dtype=np.float32)
    final_coords = np.array(
        [[0, 1, 2, 3], [0, 4, 5, 6]],
        dtype=np.int32,
    )
    subdivision_feats = np.ones((2, 8), dtype=np.float16)
    subdivision_coords = final_coords.copy()
    if field == "feats":
        final_feats = final_feats.astype(wrong_dtype)
    elif field == "coords":
        final_coords = final_coords.astype(wrong_dtype)
        final_coords[0, 1] = 1.5
    elif field == "subdivision_feats":
        subdivision_feats = subdivision_feats.astype(wrong_dtype)
    else:
        subdivision_coords = subdivision_coords.astype(wrong_dtype)
        subdivision_coords[0, 1] = 1.5

    output = tmp_path / f"{field}.decoder-state.npz"
    with pytest.raises(ValueError, match=expected_error):
        write_decoder_state_npz(
            output,
            SparseState(final_feats, final_coords),
            [
                SparseState(subdivision_feats, subdivision_coords)
                for _ in range(4)
            ],
        )
    assert not output.exists()


def test_write_decoder_state_reopen_failure_preserves_existing_primary(
    tmp_path,
    monkeypatch,
):
    import numpy as np

    import scripts.source_cuda_postcond_full_decode_timing as runner

    class SparseState:
        def __init__(self, feats, coords):
            self.feats = feats
            self.coords = coords

    output = tmp_path / "decoder-state.npz"
    original = b"existing-authoritative-primary"
    output.write_bytes(original)
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    state = SparseState(np.ones((2, 7), dtype=np.float32), coords)
    subdivisions = [
        SparseState(np.ones((2, 8), dtype=np.float16), coords)
        for _ in range(4)
    ]
    monkeypatch.setattr(
        runner.np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("synthetic reopen failure")
        ),
    )

    with pytest.raises(ValueError, match="synthetic reopen failure"):
        runner.write_decoder_state_npz(output, state, subdivisions)

    assert output.read_bytes() == original
    assert list(tmp_path.glob("*.tmp.npz")) == []


def test_shape_slat_grid_decode_report_collision_uses_durable_fallback(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    original = source_report.read_bytes()
    args = _shape_slat_decode_args(tmp_path, grid, source_report, point_names)
    args[1] = str(source_report)

    rc = main(args)

    fallback = source_report.with_name(f"{source_report.name}.selective-decode-failure.json")
    report = json.loads(fallback.read_text())
    assert rc == 1
    assert source_report.read_bytes() == original
    assert report["failure_phase"] == "request_validation"
    assert report["requested_output_json"] == str(source_report)
    assert report["effective_output_json"] == str(fallback)
    assert "collides with protected input" in report["error"]


def test_shape_slat_grid_decode_report_hardlink_collision_preserves_input(tmp_path):
    import json
    import os

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    original = source_report.read_bytes()
    aliased_report = tmp_path / "hardlinked-output.json"
    os.link(source_report, aliased_report)
    args = _shape_slat_decode_args(tmp_path, grid, source_report, point_names)
    args[1] = str(aliased_report)

    rc = main(args)

    fallback = aliased_report.with_name(
        f"{aliased_report.name}.selective-decode-failure.json"
    )
    report = json.loads(fallback.read_text())
    assert rc == 1
    assert source_report.read_bytes() == original
    assert aliased_report.read_bytes() == original
    assert report["failure_phase"] == "request_validation"
    assert report["requested_output_json"] == str(aliased_report)
    assert report["effective_output_json"] == str(fallback)
    assert "collides with protected input" in report["error"]


def test_shape_slat_grid_decode_expected_mesh_report_collision_uses_fallback(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    colliding_mesh = output_dir / f"{point_names[0]}.raw.ply"
    colliding_mesh.write_bytes(b"stale mesh")
    args = _shape_slat_decode_args(tmp_path, grid, source_report, point_names)
    args[1] = str(colliding_mesh)

    rc = main(args)

    fallback = colliding_mesh.with_name(
        f"{colliding_mesh.name}.selective-decode-failure.json"
    )
    report = json.loads(fallback.read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert report["requested_output_json"] == str(colliding_mesh)
    assert report["effective_output_json"] == str(fallback)
    assert "collides with an expected mesh output" in report["error"]
    assert not colliding_mesh.exists()
    assert not list(output_dir.glob("*.ply"))


def test_shape_slat_grid_decode_rejects_expected_mesh_collision_with_protected_input(
    tmp_path,
):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    colliding_grid = output_dir / f"{point_names[0]}.raw.ply"
    grid.replace(colliding_grid)
    original = colliding_grid.read_bytes()
    args = _shape_slat_decode_args(
        tmp_path,
        colliding_grid,
        source_report,
        point_names,
    )

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert "expected mesh output collides with protected input" in report["error"]
    assert colliding_grid.read_bytes() == original


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload: payload["points"][0].update(
                {"coordinate": {"alpha": 0.0, "beta": 0.0}}
            ),
            "coordinate mismatch for 'alpha-1_beta-1'",
        ),
        (
            lambda payload: payload["source_control"].update(
                {"coordinate": {"alpha": 0.0, "beta": 0.0}}
            ),
            "source control coordinate mismatch",
        ),
        (
            lambda payload: payload["primary_output"].update({"keys": ["coords"]}),
            "primary output key list omits selected array",
        ),
        (
            lambda payload: payload["effective_route"].pop("cuda_device"),
            "source basin route is missing cuda_device",
        ),
    ],
)
def test_shape_slat_grid_decode_rejects_semantic_route_or_label_contradiction(
    tmp_path,
    mutation,
    error,
):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    payload = json.loads(source_report.read_text())
    mutation(payload)
    source_report.write_text(json.dumps(payload) + "\n")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, point_names))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    assert error in report["error"]


def test_required_model_names_for_512_postcond_decode():
    from scripts.source_cuda_postcond_full_decode_timing import required_model_names

    assert required_model_names("512") == (
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "shape_slat_decoder",
        "shape_slat_flow_model_512",
        "tex_slat_decoder",
        "tex_slat_flow_model_512",
    )


def test_required_model_names_for_1024_cascade_postcond_decode():
    from scripts.source_cuda_postcond_full_decode_timing import required_model_names

    assert required_model_names("1024_cascade") == (
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "shape_slat_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "tex_slat_decoder",
        "tex_slat_flow_model_1024",
    )


def test_required_model_names_rejects_unknown_pipeline_type():
    from scripts.source_cuda_postcond_full_decode_timing import required_model_names

    with pytest.raises(ValueError, match="unsupported pipeline_type"):
        required_model_names("1536_cascade")


def test_resolve_model_ref_uses_pipeline_repo_for_relative_models():
    from scripts.source_cuda_postcond_full_decode_timing import resolve_model_ref

    assert (
        resolve_model_ref("microsoft/TRELLIS.2-4B", "ckpts/example")
        == "microsoft/TRELLIS.2-4B/ckpts/example"
    )
    assert (
        resolve_model_ref("microsoft/TRELLIS.2-4B", "microsoft/TRELLIS-image-large/ckpts/ss_dec")
        == "microsoft/TRELLIS-image-large/ckpts/ss_dec"
    )


def test_postcond_decode_runner_defaults_to_dependency_free_sparse_conv_backend():
    from scripts.source_cuda_postcond_full_decode_timing import build_parser

    args = build_parser().parse_args(["--output-json", "out.json", "--output-npz", "out.npz"])

    assert args.sparse_conv_backend == "none"
    assert args.sparse_attn_backend == "sdpa"


def test_apply_sparse_backend_env_sets_dense_attention_alias(monkeypatch):
    from scripts.source_cuda_postcond_full_decode_timing import apply_sparse_backend_env

    applied = apply_sparse_backend_env("none", "sdpa")

    assert applied == {
        "SPARSE_CONV_BACKEND": "none",
        "SPARSE_ATTN_BACKEND": "sdpa",
        "ATTN_BACKEND": "sdpa",
    }
    assert applied["ATTN_BACKEND"] == "sdpa"


def test_postcond_decode_runner_defaults_mesh_override_input():
    from pathlib import Path

    from scripts.source_cuda_postcond_full_decode_timing import build_parser

    args = build_parser().parse_args(["--output-json", "out.json", "--output-npz", "out.npz"])

    assert args.mesh_override == Path("o_voxel_override_convert.py")
    assert args.output_mesh_state is None
    assert args.output_shape_slat is None
    assert args.output_shape_flow_step is None
    assert args.sparse_flow_noise_sample is None
    assert args.sparse_flow_noise_sample_sha256 is None
    assert args.shape_flow_noise_sample is None


def test_install_mesh_override_copies_into_source_stubs(tmp_path):
    from scripts.source_cuda_postcond_full_decode_timing import install_mesh_override

    source_root = tmp_path / "source"
    override = tmp_path / "o_voxel_override_convert.py"
    override.write_text("SENTINEL = 1\n")

    result = install_mesh_override(source_root, override)

    installed = source_root / "stubs" / "o_voxel_override_convert.py"
    assert installed.read_text() == "SENTINEL = 1\n"
    assert result["status"] == "installed"
    assert result["path"] == str(installed)
    assert result["source"] == str(override)


def test_write_binary_mesh_ply_preserves_vertices_and_faces(tmp_path):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_binary_mesh_ply

    class Mesh:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)

    output = tmp_path / "mesh.ply"

    write_binary_mesh_ply(output, Mesh())

    payload = output.read_bytes()
    header, body = payload.split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    assert b"element vertex 3" in header
    assert b"element face 1" in header

    vertices = np.frombuffer(body[: 3 * 3 * 4], dtype="<f4").reshape(3, 3)
    faces = np.frombuffer(
        body[3 * 3 * 4 :],
        dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]),
    )
    np.testing.assert_allclose(vertices, Mesh.vertices)
    assert faces["count"].tolist() == [3]
    assert faces["indices"].tolist() == [[0, 1, 2]]


def test_write_mesh_state_npz_preserves_voxel_payload(tmp_path):
    import json
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_mesh_state_npz

    class Mesh:
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        faces = np.array([[0, 1, 1]], dtype=np.int64)
        attrs = np.array([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=np.float32)
        coords = np.array([[3, 4, 5]], dtype=np.int64)
        origin = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        voxel_size = 1 / 512
        voxel_shape = (1, 64, 64, 64)
        layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }

    output = tmp_path / "mesh_state.npz"

    write_mesh_state_npz(output, Mesh())

    with np.load(output) as data:
        np.testing.assert_allclose(data["vertices"], Mesh.vertices)
        np.testing.assert_array_equal(data["faces"], Mesh.faces.astype(np.int32))
        np.testing.assert_allclose(data["attrs"], Mesh.attrs)
        np.testing.assert_array_equal(data["coords"], Mesh.coords.astype(np.int32))
        np.testing.assert_allclose(data["origin"], Mesh.origin)
        assert float(data["voxel_size"]) == Mesh.voxel_size
        np.testing.assert_array_equal(data["voxel_shape"], np.array(Mesh.voxel_shape, dtype=np.int64))
        layout = json.loads(str(data["layout_json"]))

    assert layout == {
        "base_color": [0, 3, None],
        "metallic": [3, 4, None],
        "roughness": [4, 5, None],
        "alpha": [5, 6, None],
    }


def test_write_sparse_tensor_npz_preserves_coords_feats_and_metadata(tmp_path):
    import json
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_sparse_tensor_npz

    class SparseTensor:
        coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int64)
        feats = np.array([[1.5, -2.0], [3.25, 4.5]], dtype=np.float32)

    output = tmp_path / "shape_slat.npz"

    artifact = write_sparse_tensor_npz(
        output,
        SparseTensor(),
        stage="shape_slat",
        normalization="source-config",
    )

    assert artifact["path"] == str(output)
    assert artifact["format"] == "sparse_tensor_npz"
    assert artifact["artifact_scope"] == "source_cuda_shape_slat"
    assert artifact["coords_shape"] == [2, 4]
    assert artifact["feats_shape"] == [2, 2]
    assert artifact["sha256"]

    with np.load(output) as data:
        np.testing.assert_array_equal(data["coords"], SparseTensor.coords.astype(np.int32))
        np.testing.assert_allclose(data["feats"], SparseTensor.feats)
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata == {
        "artifact_scope": "source_cuda_shape_slat",
        "normalization": "source-config",
        "stage": "shape_slat",
    }


def test_write_source_shape_flow_step_npz_preserves_comparable_fields(tmp_path):
    import json
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_source_shape_flow_step_npz

    payload = {
        "noise": np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32),
        "sample_feats": np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32),
        "coords": np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32),
        "coords_3d": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "pred_pos": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "pred_neg": np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
        "pred_cfg": np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32),
        "x0_pos": np.array([[1.1, -0.8], [0.2, 0.1]], dtype=np.float32),
        "x0_cfg": np.array([[0.3, -0.4], [0.5, 0.6]], dtype=np.float32),
        "std_pos": np.array(0.75, dtype=np.float32),
        "std_cfg": np.array(1.5, dtype=np.float32),
        "ratio_raw": np.array(0.5, dtype=np.float32),
        "std_ratio": np.array(0.5, dtype=np.float32),
        "ratio_effective": np.array(0.5, dtype=np.float32),
        "x0_rescaled": np.array([[0.15, -0.2], [0.25, 0.3]], dtype=np.float32),
        "x0_after_rescale": np.array([[0.2, -0.3], [0.4, 0.45]], dtype=np.float32),
        "pred_final": np.array([[0.55, 0.65], [0.75, 0.85]], dtype=np.float32),
        "pred_v_feats": np.array([[0.55, 0.65], [0.75, 0.85]], dtype=np.float32),
        "sample_next": np.array([[0.97, -1.03], [0.46, 0.21]], dtype=np.float32),
        "t": np.array(1.0, dtype=np.float32),
        "t_prev": np.array(0.95, dtype=np.float32),
        "steps": np.array(8, dtype=np.int32),
        "guidance_strength": np.array(7.5, dtype=np.float32),
        "guidance_rescale": np.array(0.5, dtype=np.float32),
        "guidance_interval": np.array([0.8, 1.0], dtype=np.float32),
        "rescale_t": np.array(3.0, dtype=np.float32),
    }
    output = tmp_path / "shape_flow_step.npz"

    artifact = write_source_shape_flow_step_npz(output, payload, normalization="source-config")

    assert artifact["path"] == str(output)
    assert artifact["format"] == "shape_flow_step_npz"
    assert artifact["artifact_scope"] == "source_cuda_shape_flow_first_step"
    assert artifact["coords_shape"] == [2, 4]
    assert artifact["sample_next_shape"] == [2, 2]
    assert artifact["sha256"]

    with np.load(output) as data:
        for key, value in payload.items():
            np.testing.assert_array_equal(data[key], value)
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata == {
        "artifact_scope": "source_cuda_shape_flow_first_step",
        "normalization": "source-config",
        "stage": "shape_flow_step",
    }


def test_load_shape_flow_noise_sample_accepts_exact_coords(tmp_path):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import load_shape_flow_noise_sample

    sample = tmp_path / "mlx_shape_flow_step.npz"
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    noise = np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32)
    np.savez(sample, coords=coords, noise=noise)

    loaded, key = load_shape_flow_noise_sample(sample, coords)

    np.testing.assert_array_equal(loaded, noise)
    assert key == "noise"


def test_load_sparse_flow_noise_sample_accepts_exact_source_noise(tmp_path):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import (
        load_sparse_flow_noise_sample,
        sha256_file,
    )

    sample = tmp_path / "source_mps_sparse_flow_steps.npz"
    noise = np.arange(1 * 8 * 16 * 16 * 16, dtype=np.float32).reshape(1, 8, 16, 16, 16)
    np.savez(sample, noise=noise, sample_in=np.zeros((8, 1, 8, 16, 16, 16), dtype=np.float32))

    loaded, key = load_sparse_flow_noise_sample(
        sample,
        expected_shape=(1, 8, 16, 16, 16),
        expected_sha256=sha256_file(sample),
    )

    np.testing.assert_array_equal(loaded, noise)
    assert loaded.dtype == np.float32
    assert key == "noise"


def test_load_sparse_flow_noise_sample_rejects_same_shape_substitution(tmp_path):
    from scripts.source_cuda_postcond_full_decode_timing import (
        load_sparse_flow_noise_sample,
        sha256_file,
    )

    intended = tmp_path / "source_mps_sparse_flow_steps.npz"
    replacement = tmp_path / "replacement_sparse_flow_steps.npz"
    shape = (1, 8, 16, 16, 16)
    np.savez(intended, noise=np.zeros(shape, dtype=np.float32))
    np.savez(replacement, noise=np.ones(shape, dtype=np.float32))

    with pytest.raises(ValueError, match="digest mismatch"):
        load_sparse_flow_noise_sample(
            replacement,
            expected_shape=shape,
            expected_sha256=sha256_file(intended),
        )


@pytest.mark.parametrize(
    ("sample", "expected_sha256", "message"),
    [
        ("noise.npz", None, "required"),
        (None, "a" * 64, "provided together"),
        ("noise.npz", "A" * 64, "canonical lowercase"),
    ],
)
def test_validate_sparse_flow_noise_request_requires_paired_canonical_identity(
    sample,
    expected_sha256,
    message,
):
    from scripts.source_cuda_postcond_full_decode_timing import (
        validate_sparse_flow_noise_request,
    )

    with pytest.raises(ValueError, match=message):
        validate_sparse_flow_noise_request(
            Path(sample) if sample is not None else None,
            expected_sha256,
        )


def test_main_preserves_sparse_noise_request_when_expected_hash_is_missing(tmp_path):
    from scripts.source_cuda_postcond_full_decode_timing import main

    noise = tmp_path / "noise.npz"
    np.savez(noise, noise=np.zeros((1, 8, 16, 16, 16), dtype=np.float32))
    report = tmp_path / "report.json"

    result = main(
        [
            "--output-json",
            str(report),
            "--output-npz",
            str(tmp_path / "summary.npz"),
            "--sparse-flow-noise-sample",
            str(noise),
        ]
    )

    payload = json.loads(report.read_text())
    assert result == 1
    assert payload["status"] == "failed"
    assert payload["failure_phase"] == "validate_args"
    assert payload["sparse_flow_noise_sample"] == {
        "path": str(noise),
        "expected_sha256": None,
        "sha256": None,
        "noise_key": None,
        "noise_shape": None,
        "noise_dtype": None,
        "sampling_route": "official-source-sparse-flow-from-admitted-noise",
    }


@pytest.mark.parametrize(
    ("noise", "message"),
    [
        (np.zeros((1, 8, 8, 8, 8), dtype=np.float32), "shape"),
        (np.zeros((1, 8, 16, 16, 16), dtype=np.float64), "float32"),
        (
            np.full((1, 8, 16, 16, 16), np.nan, dtype=np.float32),
            "non-finite",
        ),
    ],
)
def test_load_sparse_flow_noise_sample_rejects_unfaithful_input(
    tmp_path,
    noise,
    message,
):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import (
        load_sparse_flow_noise_sample,
        sha256_file,
    )

    sample = tmp_path / "bad_sparse_noise.npz"
    np.savez(sample, noise=noise)

    with pytest.raises(ValueError, match=message):
        load_sparse_flow_noise_sample(
            sample,
            expected_shape=(1, 8, 16, 16, 16),
            expected_sha256=sha256_file(sample),
        )


def test_sample_sparse_structure_with_noise_uses_admitted_tensor_without_randn():
    from types import SimpleNamespace

    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import sample_sparse_structure_with_noise

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        @property
        def shape(self):
            return self.array.shape

        def to(self, *args, **kwargs):
            return self

        def cpu(self):
            return self

        def int(self):
            return FakeTensor(self.array.astype(np.int32))

        def __gt__(self, other):
            return FakeTensor(self.array > other)

        def __getitem__(self, key):
            return FakeTensor(self.array[key])

    class FakeTorch:
        @staticmethod
        def from_numpy(array):
            return FakeTensor(np.array(array, copy=True))

        @staticmethod
        def argwhere(tensor):
            return FakeTensor(np.argwhere(tensor.array))

    class Movable:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(("to", device))
            return self

        def cpu(self):
            self.moves.append(("cpu", None))
            return self

    flow_model = Movable()
    flow_model.resolution = 16
    flow_model.in_channels = 8
    decoder = Movable()
    decoded = np.zeros((1, 1, 32, 32, 32), dtype=np.float32)
    decoded[0, 0, 2, 3, 4] = 1.0
    decoder.__class__.__call__ = lambda self, latent: FakeTensor(decoded)
    seen = {}

    class Sampler:
        def sample(self, model, noise, **kwargs):
            seen["model"] = model
            seen["noise"] = noise.array.copy()
            seen["kwargs"] = kwargs
            return SimpleNamespace(samples=FakeTensor(np.zeros((1, 8, 16, 16, 16), dtype=np.float32)))

    pipeline = SimpleNamespace(
        models={
            "sparse_structure_flow_model": flow_model,
            "sparse_structure_decoder": decoder,
        },
        sparse_structure_sampler=Sampler(),
        sparse_structure_sampler_params={"guidance_strength": 7.5},
        low_vram=True,
        device="cuda",
    )
    noise = np.arange(1 * 8 * 16 * 16 * 16, dtype=np.float32).reshape(1, 8, 16, 16, 16)

    coords = sample_sparse_structure_with_noise(
        pipeline,
        {"cond": "positive", "neg_cond": "negative"},
        32,
        1,
        {"steps": 8},
        noise,
        torch_module=FakeTorch,
    )

    np.testing.assert_array_equal(seen["noise"], noise)
    assert seen["model"] is flow_model
    assert seen["kwargs"]["steps"] == 8
    assert seen["kwargs"]["guidance_strength"] == 7.5
    assert seen["kwargs"]["verbose"] is True
    np.testing.assert_array_equal(coords.array, np.array([[0, 2, 3, 4]], dtype=np.int32))
    assert flow_model.moves == [("to", "cuda"), ("cpu", None)]
    assert decoder.moves == [("to", "cuda"), ("cpu", None)]


def test_sample_sparse_structure_capture_observes_official_sampler_and_restores_hooks():
    from types import SimpleNamespace

    from scripts.source_cuda_postcond_full_decode_timing import (
        sample_sparse_structure_with_noise,
    )

    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array, dtype=np.float32)

        def __array__(self, dtype=None):
            return np.asarray(self.array, dtype=dtype)

        @property
        def shape(self):
            return self.array.shape

        @property
        def ndim(self):
            return self.array.ndim

        @property
        def device(self):
            return "cuda"

        def detach(self):
            return self

        def to(self, *args, **kwargs):
            return self

        def cpu(self):
            return self

        def int(self):
            return Tensor(self.array.astype(np.int32))

        def std(self, dim, keepdim):
            return Tensor(
                np.std(self.array, axis=tuple(dim), ddof=1, keepdims=keepdim)
            )

        def __getitem__(self, key):
            return Tensor(self.array[key])

        def __gt__(self, other):
            return Tensor(self.array > other)

        def __add__(self, other):
            return Tensor(self.array + np.asarray(other))

        def __radd__(self, other):
            return Tensor(np.asarray(other) + self.array)

        def __sub__(self, other):
            return Tensor(self.array - np.asarray(other))

        def __rsub__(self, other):
            return Tensor(np.asarray(other) - self.array)

        def __mul__(self, other):
            return Tensor(self.array * np.asarray(other))

        def __rmul__(self, other):
            return Tensor(np.asarray(other) * self.array)

        def __truediv__(self, other):
            return Tensor(self.array / np.asarray(other))

    class FakeTorch:
        @staticmethod
        def from_numpy(array):
            return Tensor(np.array(array, copy=True))

        @staticmethod
        def argwhere(tensor):
            return Tensor(np.argwhere(tensor.array))

    class FlowModel:
        resolution = 1
        in_channels = 2

        def __init__(self):
            self.moves = []

        def __call__(self, sample, t_tensor, cond):
            value = 0.25 if cond == "positive" else -0.125
            return Tensor(np.full(sample.shape, value, dtype=np.float32))

        def to(self, device):
            self.moves.append(("to", device))

        def cpu(self):
            self.moves.append(("cpu", None))

    class Sampler:
        sigma_min = 1e-5

        def _pred_to_xstart(self, sample, t, pred):
            return (1 - self.sigma_min) * sample - (
                self.sigma_min + (1 - self.sigma_min) * t
            ) * pred

        def _xstart_to_pred(self, sample, t, x0):
            return ((1 - self.sigma_min) * sample - x0) / (
                self.sigma_min + (1 - self.sigma_min) * t
            )

        def _inference_model(
            self,
            model,
            sample,
            t,
            cond,
            neg_cond,
            guidance_strength,
            guidance_rescale,
            guidance_interval,
        ):
            pred_pos = model(sample, t, cond)
            if guidance_interval[0] <= t <= guidance_interval[1]:
                pred_neg = model(sample, t, neg_cond)
                pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
                x0_pos = self._pred_to_xstart(sample, t, pred_pos)
                x0_cfg = self._pred_to_xstart(sample, t, pred)
                ratio = x0_pos.std(list(range(1, x0_pos.ndim)), True) / x0_cfg.std(
                    list(range(1, x0_cfg.ndim)), True
                )
                x0 = guidance_rescale * x0_cfg * ratio + (1 - guidance_rescale) * x0_cfg
                pred = self._xstart_to_pred(sample, t, x0)
                return pred
            return pred_pos

        def sample_once(self, model, sample, t, t_prev, cond, **kwargs):
            pred = self._inference_model(model, sample, t, cond=cond, **kwargs)
            return SimpleNamespace(
                pred_x_prev=sample - (t - t_prev) * pred,
                pred_x_0=self._pred_to_xstart(sample, t, pred),
            )

        def sample(
            self,
            model,
            noise,
            cond,
            steps,
            rescale_t,
            verbose=True,
            tqdm_desc="Sampling",
            **kwargs,
        ):
            sample = noise
            t_seq = np.linspace(1, 0, steps + 1)
            t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
            for index in range(steps):
                sample = self.sample_once(
                    model,
                    sample,
                    float(t_seq[index]),
                    float(t_seq[index + 1]),
                    cond,
                    **kwargs,
                ).pred_x_prev
            return SimpleNamespace(samples=sample)

    class Decoder:
        def __call__(self, latent):
            return Tensor(np.array([[[[[1.0]]]]], dtype=np.float32))

        def to(self, device):
            return self

        def cpu(self):
            return self

    flow_model = FlowModel()
    sampler = Sampler()
    pipeline = SimpleNamespace(
        models={
            "sparse_structure_flow_model": flow_model,
            "sparse_structure_decoder": Decoder(),
        },
        sparse_structure_sampler=sampler,
        sparse_structure_sampler_params={
            "steps": 2,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "guidance_interval": (0.6, 1.0),
            "rescale_t": 5.0,
        },
        low_vram=True,
        device="cuda",
    )
    capture = {}
    noise = np.array([[[[[1.0]]], [[[2.0]]]]], dtype=np.float32)

    coords = sample_sparse_structure_with_noise(
        pipeline,
        {"cond": "positive", "neg_cond": "negative"},
        1,
        1,
        {"steps": 2},
        noise,
        torch_module=FakeTorch,
        capture_steps=capture,
    )

    assert "_inference_model" not in sampler.__dict__
    assert "sample_once" not in sampler.__dict__
    assert capture["sample_in"].shape == (2, 1, 2, 1, 1, 1)
    assert capture["pred_pos"].shape == capture["sample_in"].shape
    assert capture["pred_neg"].shape == capture["sample_in"].shape
    assert capture["pred_final"].shape == capture["sample_in"].shape
    np.testing.assert_array_equal(capture["noise"], noise)
    np.testing.assert_array_equal(capture["sample_in"][1:], capture["sample_next"][:-1])
    np.testing.assert_array_equal(coords.array, np.array([[0, 0, 0, 0]], dtype=np.float32))


def _source_sparse_flow_steps_payload(*, steps=8, break_recurrence=False):
    shape = (steps, 1, 2, 1, 1, 1)
    sample_in = np.empty(shape, dtype=np.float32)
    sample_next = np.empty(shape, dtype=np.float32)
    pred_final = np.full(shape, np.float32(0.25), dtype=np.float32)
    t = np.linspace(1.0, 0.2, steps, dtype=np.float32)
    t_prev = t - np.float32(0.05)
    sample_in[0] = np.array([[[[[1.0]]], [[[2.0]]]]], dtype=np.float32)
    for index in range(steps):
        sample_next[index] = (
            sample_in[index]
            - np.float32(t[index] - t_prev[index]) * pred_final[index]
        )
        if index + 1 < steps:
            sample_in[index + 1] = sample_next[index]
    if break_recurrence and steps > 1:
        sample_in[1, 0, 0, 0, 0, 0] += np.float32(1.0)
    ones = np.ones((steps, 1, 1, 1, 1, 1), dtype=np.float32)
    return {
        "noise": sample_in[0].copy(),
        "sample_in": sample_in,
        "pred_pos": pred_final.copy(),
        "pred_neg": pred_final.copy(),
        "pred_cfg": pred_final.copy(),
        "x0_pos": pred_final.copy(),
        "x0_cfg": pred_final.copy(),
        "std_pos": ones.copy(),
        "std_cfg": ones.copy(),
        "ratio_raw": ones.copy(),
        "std_ratio": ones.copy(),
        "ratio_effective": ones.copy(),
        "x0_rescaled": pred_final.copy(),
        "x0_after_rescale": pred_final.copy(),
        "pred_final": pred_final,
        "sample_next": sample_next,
        "t": t,
        "t_prev": t_prev,
        "t_tensor": 1000 * t,
        "steps": np.asarray(steps, dtype=np.int32),
        "guidance_strength": np.asarray(7.5, dtype=np.float32),
        "guidance_rescale": np.asarray(0.7, dtype=np.float32),
        "guidance_interval": np.asarray([0.6, 1.0], dtype=np.float32),
        "rescale_t": np.asarray(5.0, dtype=np.float32),
        "sigma_min": np.asarray(1e-5, dtype=np.float32),
        "apply_guidance": np.ones(steps, dtype=np.bool_),
    }


def test_write_source_sparse_flow_steps_npz_binds_recurrence_and_support(tmp_path):
    from scripts.source_cuda_postcond_full_decode_timing import (
        write_source_sparse_flow_steps_npz,
    )

    output = tmp_path / "source_cuda_sparse_flow_steps.npz"
    payload = _source_sparse_flow_steps_payload()
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)

    artifact = write_source_sparse_flow_steps_npz(output, payload, coords)

    assert artifact["artifact_scope"] == "source_cuda_sparse_flow_steps"
    assert artifact["format"] == "sparse_flow_steps_npz"
    assert artifact["step_count"] == 8
    assert artifact["support_count"] == 2
    assert artifact["official_sampler_fields"] == [
        "sample_in",
        "pred_pos",
        "pred_neg",
        "pred_final",
        "sample_next",
    ]
    assert artifact["reconstructed_diagnostic_fields"] == [
        "pred_cfg",
        "x0_pos",
        "x0_cfg",
        "std_pos",
        "std_cfg",
        "ratio_raw",
        "std_ratio",
        "ratio_effective",
        "x0_rescaled",
        "x0_after_rescale",
    ]
    with np.load(output, allow_pickle=False) as archive:
        assert set(payload) <= set(archive.files)
        np.testing.assert_array_equal(archive["decoded_coords"], coords)
        np.testing.assert_array_equal(archive["decoded_coords_3d"], coords[:, 1:])
        np.testing.assert_array_equal(archive["sample_in"][1:], archive["sample_next"][:-1])
        np.testing.assert_array_equal(archive["sample_in"][0], archive["noise"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("step_count", "exactly 8 steps"),
        ("recurrence", "recurrence"),
        ("nonfinite", "non-finite"),
    ],
)
def test_write_source_sparse_flow_steps_npz_rejects_false_closure(
    tmp_path,
    mutation,
    message,
):
    from scripts.source_cuda_postcond_full_decode_timing import (
        write_source_sparse_flow_steps_npz,
    )

    payload = _source_sparse_flow_steps_payload(
        steps=7 if mutation == "step_count" else 8,
        break_recurrence=mutation == "recurrence",
    )
    if mutation == "nonfinite":
        payload["pred_final"] = payload["pred_final"].copy()
        payload["pred_final"][0, 0, 0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match=message):
        write_source_sparse_flow_steps_npz(
            tmp_path / "invalid.npz",
            payload,
            np.array([[0, 1, 2, 3]], dtype=np.int32),
        )

    assert not (tmp_path / "invalid.npz").exists()


def test_sparse_structure_stop_route_requires_exact_noise_before_source_extract(tmp_path):
    from scripts.source_cuda_postcond_full_decode_timing import main

    report = tmp_path / "report.json"
    result = main(
        [
            "--output-json",
            str(report),
            "--output-sparse-flow-steps",
            str(tmp_path / "steps.npz"),
            "--stop-after-sparse-structure",
        ]
    )

    payload = json.loads(report.read_text())
    assert result == 1
    assert payload["failure_phase"] == "validate_args"
    assert "exact sparse-flow noise" in payload["error"]
    assert not (tmp_path / "steps.npz").exists()


def test_load_shape_flow_noise_sample_rejects_coord_order_mismatch(tmp_path):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import load_shape_flow_noise_sample

    sample = tmp_path / "mlx_shape_flow_step.npz"
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    swapped = coords[::-1]
    np.savez(sample, coords=swapped, noise=np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="coords do not exactly match"):
        load_shape_flow_noise_sample(sample, coords)


def test_load_shape_flow_noise_sample_rejects_feature_row_mismatch(tmp_path):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import load_shape_flow_noise_sample

    sample = tmp_path / "mlx_shape_flow_step.npz"
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    np.savez(sample, coords=coords, sample_feats=np.zeros((1, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="noise row mismatch"):
        load_shape_flow_noise_sample(sample, coords)

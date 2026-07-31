import shutil
import subprocess
import sys
from pathlib import Path

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

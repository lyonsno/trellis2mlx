import json
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _valid_sweep():
    grid_outputs = np.array(
        [1.0, 1.1892071, 1.4142135, 1.6817929],
        dtype=np.float32,
    )
    fixture_expected = np.array(
        [1.0, 0.36787945, 0.13533528],
        dtype=np.float32,
    )
    probes = np.array(
        [0.0, 2.0**-24, 0.25, 0.5, np.nextafter(1.0, 0.0)],
        dtype=np.float32,
    )
    probe_outputs = np.exp2(probes).astype(np.float32)
    return {
        "grid_output_bits": grid_outputs.view(np.uint32),
        "probe_input_bits": probes.view(np.uint32),
        "probe_output_bits": probe_outputs.view(np.uint32),
        "probe_repeat_bits": probe_outputs.view(np.uint32).copy(),
        "fixture_expected_bits": fixture_expected.view(np.uint32),
        "fixture_expf_bits": fixture_expected.view(np.uint32).copy(),
        "fixture_manual_bits": fixture_expected.view(np.uint32).copy(),
        "runtime": {
            "torch": "2.10.0+cu128",
            "cuda": "12.8",
            "device": "Tesla T4",
            "device_ordinal": 0,
            "nvcc": {
                "path": "/usr/local/cuda/bin/nvcc",
                "release": "12.8",
                "version_output": "release 12.8, V12.8.93",
            },
        },
        "implementation": {
            "instruction": "ex2.approx.ftz.f32",
            "grid_fraction_bits": 2,
            "grid_points": 4,
            "probe_generator": "float32-bit-stratified-v1",
        },
        "expected_grid_points": 4,
    }


def test_sweep_rejects_partial_grid():
    from scripts.cuda_ex2_primary_sweep import analyze_ex2_sweep_unit

    payload = _valid_sweep()
    payload["grid_output_bits"] = payload["grid_output_bits"][:-1]

    with pytest.raises(ValueError, match="grid is partial"):
        analyze_ex2_sweep_unit(**payload)


def test_sweep_rejects_substituted_runtime_route():
    from scripts.cuda_ex2_primary_sweep import analyze_ex2_sweep_unit

    payload = _valid_sweep()
    payload["runtime"] = dict(payload["runtime"], device="NVIDIA A100-SXM4-40GB")

    with pytest.raises(ValueError, match="expected CUDA device Tesla T4"):
        analyze_ex2_sweep_unit(**payload)


def test_sweep_rejects_substituted_instruction_route():
    from scripts.cuda_ex2_primary_sweep import analyze_ex2_sweep_unit

    payload = _valid_sweep()
    payload["implementation"] = dict(
        payload["implementation"],
        instruction="exp2f",
    )

    with pytest.raises(ValueError, match="inline PTX instruction"):
        analyze_ex2_sweep_unit(**payload)


@pytest.mark.parametrize("field", ["fixture_expf_bits", "fixture_manual_bits"])
def test_sweep_rejects_unbound_trellis_fixture(field):
    from scripts.cuda_ex2_primary_sweep import analyze_ex2_sweep_unit

    payload = _valid_sweep()
    payload[field] = payload[field].copy()
    payload[field][1] += np.uint32(1)

    with pytest.raises(ValueError, match="Trellis fixture"):
        analyze_ex2_sweep_unit(**payload)


def test_sweep_rejects_nondeterministic_probe_replay():
    from scripts.cuda_ex2_primary_sweep import analyze_ex2_sweep_unit

    payload = _valid_sweep()
    payload["probe_repeat_bits"] = payload["probe_repeat_bits"].copy()
    payload["probe_repeat_bits"][-1] += np.uint32(1)

    with pytest.raises(ValueError, match="probe replay"):
        analyze_ex2_sweep_unit(**payload)


def test_unit_sweep_cannot_claim_complete_primary_surface():
    from scripts.cuda_ex2_primary_sweep import analyze_ex2_sweep_unit

    analysis = analyze_ex2_sweep_unit(**_valid_sweep())

    assert all(analysis["self_authentication"].values())
    assert "complete_grid" not in analysis["self_authentication"]
    assert analysis["grid"]["points"] == 4
    assert analysis["grid"]["monotonic"] is True
    assert analysis["grid"]["coverage"] == "unit-reduced"


def test_written_primary_rejects_missing_array(tmp_path):
    from scripts.cuda_ex2_primary_sweep import validate_written_primary

    path = tmp_path / "sweep.npz"
    arrays = {
        "grid_output_bits": np.arange(4, dtype=np.uint32),
        "probe_input_bits": np.arange(3, dtype=np.uint32),
    }
    route_identity_json = json.dumps({"schema": "test"})
    np.savez(
        path,
        grid_output_bits=arrays["grid_output_bits"],
        route_identity_json=np.array(route_identity_json),
    )

    with pytest.raises(ValueError, match="primary keys mismatch"):
        validate_written_primary(
            path,
            expected_arrays=arrays,
            expected_route_identity_json=route_identity_json,
        )


def test_failure_report_survives_missing_fixture(tmp_path):
    from scripts import cuda_ex2_primary_sweep as sweep

    report_path = tmp_path / "report.json"
    output_path = tmp_path / "sweep.npz"

    exit_code = sweep.main(
        [
            "--fixture-npz",
            str(tmp_path / "missing.npz"),
            "--expected-fixture-sha256",
            "a" * 64,
            "--output-json",
            str(report_path),
            "--output-npz",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert report_path.exists()
    assert not output_path.exists()
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "fixture_load"
    assert report["last_trustworthy_phase"] == "request_validated"
    assert report["primary_output_status"] == "missing"


def _write_fixture(
    path: Path,
    *,
    rows: int = 4,
    width: int = 7697,
    route_width: int | None = None,
) -> str:
    scores = np.zeros((rows, width), dtype=np.float32)
    row_maxes = np.zeros((rows,), dtype=np.float32)
    exponents = np.ones_like(scores)
    route_identity = {
        "schema": "trellis2mlx.source_cuda_softmax_fixture.v1",
        "selected_rows": [182, 1059, 1261, 3821][:rows],
        "selection": [
            "max_abs",
            "max_nonzero",
            "min_nonzero",
            "last_control",
        ][:rows],
        "source_oracle_script_sha256": "1" * 64,
        "source_oracle_sha256": "2" * 64,
        "source_stage_sha256": "3" * 64,
        "width": width if route_width is None else route_width,
    }
    np.savez(
        path,
        scores_fp32=scores,
        exponents_fp32=exponents,
        row_maxes_fp32=row_maxes,
        route_identity_json=np.array(
            json.dumps(route_identity, sort_keys=True)
        ),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("colliding_output", ["json", "npz"])
def test_output_paths_cannot_alias_protected_fixture(
    tmp_path, colliding_output
):
    from scripts import cuda_ex2_primary_sweep as sweep

    fixture = tmp_path / "fixture.npz"
    fixture_bytes = b"protected-ex2-fixture"
    fixture.write_bytes(fixture_bytes)
    digest = hashlib.sha256(fixture_bytes).hexdigest()
    requested_json = tmp_path / "report.json"
    requested_npz = tmp_path / "sweep.npz"
    if colliding_output == "json":
        requested_json = fixture
    else:
        requested_npz = fixture

    exit_code = sweep.main(
        [
            "--fixture-npz",
            str(fixture),
            "--expected-fixture-sha256",
            digest,
            "--output-json",
            str(requested_json),
            "--output-npz",
            str(requested_npz),
        ]
    )

    assert exit_code == 1
    assert fixture.read_bytes() == fixture_bytes
    fallback = tmp_path / "fixture.npz.ex2-sweep-failure.json"
    report_path = fallback if colliding_output == "json" else requested_json
    report = json.loads(report_path.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "request_received"
    expected_status = (
        "protected_input" if colliding_output == "npz" else "missing"
    )
    assert report["primary_output_status"] == expected_status


@pytest.mark.parametrize("colliding_output", ["json", "npz"])
def test_hard_linked_output_paths_cannot_alias_protected_fixture(
    tmp_path, colliding_output
):
    from scripts import cuda_ex2_primary_sweep as sweep

    fixture = tmp_path / "fixture.npz"
    fixture_bytes = b"protected-hard-link-ex2-fixture"
    fixture.write_bytes(fixture_bytes)
    digest = hashlib.sha256(fixture_bytes).hexdigest()
    alias = tmp_path / f"alias.{colliding_output}"
    os.link(fixture, alias)
    requested_json = tmp_path / "report.json"
    requested_npz = tmp_path / "sweep.npz"
    if colliding_output == "json":
        requested_json = alias
    else:
        requested_npz = alias

    exit_code = sweep.main(
        [
            "--fixture-npz",
            str(fixture),
            "--expected-fixture-sha256",
            digest,
            "--output-json",
            str(requested_json),
            "--output-npz",
            str(requested_npz),
        ]
    )

    assert exit_code == 1
    assert fixture.read_bytes() == fixture_bytes
    fallback = tmp_path / "fixture.npz.ex2-sweep-failure.json"
    report_path = fallback if colliding_output == "json" else requested_json
    report = json.loads(report_path.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "request_received"
    expected_status = (
        "protected_input" if colliding_output == "npz" else "missing"
    )
    assert report["primary_output_status"] == expected_status


@pytest.mark.parametrize("fraction_bits", [23, 25])
def test_primary_rejects_noncanonical_grid_before_fixture_load(
    tmp_path, fraction_bits
):
    from scripts import cuda_ex2_primary_sweep as sweep

    report_path = tmp_path / "report.json"
    exit_code = sweep.main(
        [
            "--fixture-npz",
            str(tmp_path / "missing.npz"),
            "--expected-fixture-sha256",
            "a" * 64,
            "--output-json",
            str(report_path),
            "--output-npz",
            str(tmp_path / "sweep.npz"),
            "--grid-fraction-bits",
            str(fraction_bits),
        ]
    )

    assert exit_code == 1
    report = json.loads(report_path.read_text())
    assert report["failure_phase"] == "request_validation"
    assert "exactly 24" in report["error"]


def test_cuda_scale_reconstruction_uses_defined_unsigned_shift():
    from scripts.cuda_ex2_primary_sweep import CUDA_SOURCE

    assert "__float_as_uint(exponent_magic) << 23" in CUDA_SOURCE
    assert "__uint_as_float(" in CUDA_SOURCE
    assert "__float_as_int(exponent_magic) << 23" not in CUDA_SOURCE


def _write_fake_nvcc(path: Path, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        "echo 'Cuda compilation tools, "
        f"release {version}, V{version}.93'\n"
    )
    path.chmod(0o755)


def test_nvcc_identity_uses_pytorch_cuda_home_not_path(
    tmp_path, monkeypatch
):
    from scripts.cuda_ex2_primary_sweep import validated_nvcc_identity

    path_nvcc = tmp_path / "path-bin" / "nvcc"
    home_nvcc = tmp_path / "cuda-home" / "bin" / "nvcc"
    _write_fake_nvcc(path_nvcc, "99.0")
    _write_fake_nvcc(home_nvcc, "12.8")
    monkeypatch.setenv(
        "PATH",
        str(path_nvcc.parent) + os.pathsep + os.environ.get("PATH", ""),
    )

    identity = validated_nvcc_identity(tmp_path / "cuda-home")

    assert identity["path"] == str(home_nvcc.resolve())
    assert identity["release"] == "12.8"


def test_nvcc_identity_rejects_non_cuda_12_8(tmp_path):
    from scripts.cuda_ex2_primary_sweep import validated_nvcc_identity

    nvcc = tmp_path / "cuda-home" / "bin" / "nvcc"
    _write_fake_nvcc(nvcc, "12.9")

    with pytest.raises(RuntimeError, match="expected NVCC release 12.8"):
        validated_nvcc_identity(tmp_path / "cuda-home")


def test_analyzer_rejects_unbound_nvcc_route():
    from scripts.cuda_ex2_primary_sweep import analyze_ex2_sweep_unit

    payload = _valid_sweep()
    payload["runtime"] = dict(
        payload["runtime"],
        device_ordinal=0,
        nvcc={
            "path": "/usr/local/cuda/bin/nvcc",
            "release": "99.0",
            "version_output": "release 99.0",
        },
    )

    with pytest.raises(ValueError, match="NVCC release 12.8"):
        analyze_ex2_sweep_unit(**payload)


@pytest.mark.parametrize(
    ("rows", "width", "route_width"),
    [(1, 1, 1), (4, 7697, 7696)],
)
def test_fixture_substitution_fails_before_cuda_import(
    tmp_path, rows, width, route_width
):
    from scripts import cuda_ex2_primary_sweep as sweep

    fixture = tmp_path / "fixture.npz"
    digest = _write_fixture(
        fixture,
        rows=rows,
        width=width,
        route_width=route_width,
    )
    report_path = tmp_path / "report.json"

    exit_code = sweep.main(
        [
            "--fixture-npz",
            str(fixture),
            "--expected-fixture-sha256",
            digest,
            "--output-json",
            str(report_path),
            "--output-npz",
            str(tmp_path / "sweep.npz"),
            "--probe-points",
            "16",
        ]
    )

    assert exit_code == 1
    report = json.loads(report_path.read_text())
    assert report["failure_phase"] == "fixture_load"
    assert "fixture" in report["error"]


def test_current_cuda_device_must_match_authenticated_ordinal_zero(
    tmp_path, monkeypatch
):
    from scripts import cuda_ex2_primary_sweep as sweep

    fixture = tmp_path / "fixture.npz"
    digest = _write_fixture(fixture)
    report_path = tmp_path / "report.json"
    extension_called = False

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 1

        @staticmethod
        def get_device_name(ordinal):
            return "Tesla T4" if ordinal == 0 else "Different GPU"

    fake_torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=FakeCuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def forbidden_build():
        nonlocal extension_called
        extension_called = True
        raise AssertionError("extension build must not run")

    monkeypatch.setattr(sweep, "_build_extension", forbidden_build)

    exit_code = sweep.main(
        [
            "--fixture-npz",
            str(fixture),
            "--expected-fixture-sha256",
            digest,
            "--output-json",
            str(report_path),
            "--output-npz",
            str(tmp_path / "sweep.npz"),
            "--probe-points",
            "16",
        ]
    )

    assert exit_code == 1
    assert extension_called is False
    report = json.loads(report_path.read_text())
    assert report["failure_phase"] == "runtime_validation"
    assert "current CUDA device must be ordinal 0" in report["error"]

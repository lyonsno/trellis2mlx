import json
import subprocess

import pytest


def test_prepare_packet_writes_private_dataset_kernel_and_manifest(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    (capsule / "witness.npz").write_bytes(b"npz bytes")

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/trellis2mlx-block7-rmsnorm-inputs",
            kernel_id="operator/trellis2mlx-block7-rmsnorm-cuda-witness",
            title="Trellis2MLX Block7 RMSNorm CUDA Witness",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py", "witness.npz"),
            accelerator="NvidiaTeslaT4",
        )
    )

    dataset_metadata = json.loads((packet.dataset_dir / "dataset-metadata.json").read_text())
    kernel_metadata = json.loads((packet.kernel_dir / "kernel-metadata.json").read_text())
    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()

    assert dataset_metadata["id"] == "operator/trellis2mlx-block7-rmsnorm-inputs"
    assert dataset_metadata["licenses"] == [{"name": "unknown"}]
    assert sorted(resource["path"] for resource in dataset_metadata["resources"]) == [
        "cuda_probe.py",
        "witness-manifest.json",
        "witness.npz",
    ]
    assert kernel_metadata == {
        "id": "operator/trellis2mlx-block7-rmsnorm-cuda-witness",
        "title": "Trellis2MLX Block7 RMSNorm CUDA Witness",
        "code_file": "run_kaggle_cuda_witness.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["operator/trellis2mlx-block7-rmsnorm-inputs"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    assert manifest["schema"] == "trellis2mlx.kaggle_cuda_witness.inputs.v1"
    assert manifest["dataset_id"] == "operator/trellis2mlx-block7-rmsnorm-inputs"
    assert manifest["entrypoint"] == "cuda_probe.py"
    assert manifest["outputs"] == ["cuda_result.json", "cuda_result.npz"]
    assert manifest["files"]["cuda_probe.py"]["sha256"]
    assert manifest["files"]["witness.npz"]["size_bytes"] == len(b"npz bytes")
    assert "kaggle_cuda_witness_receipt.json" in runner
    assert "torch.cuda.is_available()" in runner
    assert "find_manifest()" in runner
    assert "rglob(\"witness-manifest.json\")" in runner
    assert "mounted_input_files" in runner
    assert '"missing_input": str(source)' in runner
    assert "mounted_input_snapshot()" in runner


def test_prepare_packet_rejects_missing_input_before_metadata(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, WitnessPacketError, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()

    with pytest.raises(WitnessPacketError, match="missing input"):
        prepare_packet(
            KaggleCudaWitnessPacket(
                capsule_dir=capsule,
                output_dir=tmp_path / "packet",
                dataset_id="operator/missing-inputs",
                kernel_id="operator/missing-inputs",
                title="Missing Inputs",
                entrypoint="cuda_probe.py",
                inputs=("cuda_probe.py",),
            )
        )

    assert not (tmp_path / "packet").exists()


def test_prepare_packet_rejects_kernel_slug_title_mismatch(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, WitnessPacketError, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")

    with pytest.raises(WitnessPacketError, match="kernel_id slug"):
        prepare_packet(
            KaggleCudaWitnessPacket(
                capsule_dir=capsule,
                output_dir=tmp_path / "packet",
                dataset_id="operator/capsule-inputs",
                kernel_id="operator/capsule-cuda",
                title="Capsule CUDA Witness",
                entrypoint="cuda_probe.py",
                inputs=("cuda_probe.py",),
            )
        )


def test_build_commands_preserve_dataset_kernel_and_accelerator(tmp_path):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        build_dataset_command,
        build_kernel_output_command,
        build_kernel_push_command,
        build_kernel_status_command,
        prepare_packet,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/capsule-inputs",
            kernel_id="operator/capsule-cuda",
            title="Capsule CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
            accelerator="NvidiaTeslaP100",
        )
    )

    assert build_dataset_command(packet) == [
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(packet.dataset_dir),
        "-q",
        "-t",
        "-r",
        "skip",
    ]
    assert build_dataset_command(packet, version=True) == [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(packet.dataset_dir),
        "-m",
        "update CUDA witness inputs",
        "-q",
        "-t",
        "-r",
        "skip",
    ]
    assert build_kernel_push_command(packet, timeout_seconds=600) == [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(packet.kernel_dir),
        "--accelerator",
        "NvidiaTeslaP100",
        "--timeout",
        "600",
    ]
    assert build_kernel_status_command(packet) == ["kaggle", "kernels", "status", "operator/capsule-cuda"]
    assert build_kernel_output_command(packet, tmp_path / "outputs") == [
        "kaggle",
        "kernels",
        "output",
        "operator/capsule-cuda",
        "-p",
        str(tmp_path / "outputs"),
        "-o",
        "--file-pattern",
        ".*(cuda_result|kaggle_cuda_witness_receipt).*",
    ]


def test_run_command_writes_report_with_failure_phase(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, prepare_packet, run_command

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/capsule-inputs",
            kernel_id="operator/capsule-cuda",
            title="Capsule CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    report_path = tmp_path / "command-report.json"

    def failing_runner(cmd, capture_output, text, check):
        assert check is False
        return subprocess.CompletedProcess(cmd, 9, stdout="out", stderr="bad auth")

    report = run_command(
        ["kaggle", "kernels", "status", packet.kernel_id],
        phase="kernel_status",
        report_path=report_path,
        runner=failing_runner,
    )

    assert report["status"] == "failed"
    assert report["failure_phase"] == "kernel_status"
    assert report["exit_code"] == 9
    assert report["stderr"] == "bad auth"
    assert json.loads(report_path.read_text()) == report


def test_run_command_treats_kaggle_textual_error_as_failed(tmp_path):
    from trellmlx.kaggle_cuda_witness import run_command

    report_path = tmp_path / "kaggle-text-error.json"

    def textual_error_runner(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="Dataset creation error: Invalid Owner Id\n", stderr="")

    report = run_command(
        ["kaggle", "datasets", "create"],
        phase="dataset_create",
        report_path=report_path,
        runner=textual_error_runner,
    )

    assert report["status"] == "failed"
    assert report["failure_phase"] == "dataset_create"
    assert report["exit_code"] == 0
    assert report["stdout"] == "Dataset creation error: Invalid Owner Id\n"


def test_run_command_treats_invalid_dataset_source_as_failed(tmp_path):
    from trellmlx.kaggle_cuda_witness import run_command

    report_path = tmp_path / "kaggle-invalid-source.json"

    def invalid_source_runner(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                "The following are not valid dataset sources and could not be added to the kernel: "
                "['operator/missing-dataset']\n"
                "Kernel version 1 successfully pushed.\n"
            ),
            stderr="",
        )

    report = run_command(
        ["kaggle", "kernels", "push"],
        phase="kernel_push",
        report_path=report_path,
        runner=invalid_source_runner,
    )

    assert report["status"] == "failed"
    assert report["failure_phase"] == "kernel_push"
    assert report["exit_code"] == 0


def test_run_command_writes_report_when_executable_is_missing(tmp_path):
    from trellmlx.kaggle_cuda_witness import run_command

    report_path = tmp_path / "missing-cli.json"

    def missing_runner(cmd, capture_output, text, check):
        raise FileNotFoundError("kaggle")

    report = run_command(
        ["kaggle", "kernels", "status", "operator/capsule-cuda"],
        phase="kernel_status",
        report_path=report_path,
        runner=missing_runner,
    )

    assert report["status"] == "failed"
    assert report["failure_phase"] == "kernel_status_launch"
    assert report["exit_code"] == 127
    assert "FileNotFoundError" in report["stderr"]
    assert json.loads(report_path.read_text()) == report


def test_load_prepared_packet_round_trips_drive_commands(tmp_path):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        build_kernel_push_command,
        load_prepared_packet,
        prepare_packet,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    prepared = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/capsule-inputs",
            kernel_id="operator/capsule-cuda",
            title="Capsule CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
            accelerator="NvidiaTeslaP100",
        )
    )

    loaded = load_prepared_packet(tmp_path / "packet")

    assert loaded == KaggleCudaWitnessPacket(
        capsule_dir=prepared.dataset_dir,
        output_dir=prepared.output_dir,
        dataset_id=prepared.dataset_id,
        kernel_id=prepared.kernel_id,
        title=prepared.title,
        entrypoint=prepared.entrypoint,
        inputs=prepared.inputs,
        accelerator=prepared.accelerator,
    )
    assert build_kernel_push_command(loaded, timeout_seconds=30)[-2:] == ["--timeout", "30"]

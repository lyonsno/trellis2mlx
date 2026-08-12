import json
from pathlib import Path
import re
import subprocess
import sys
import types
import zipfile

import numpy as np
import pytest


def _write_valid_ply(path):
    path.write_bytes(
        b"ply\n"
        b"format ascii 1.0\n"
        b"element vertex 3\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"element face 1\n"
        b"property list uchar int vertex_indices\n"
        b"end_header\n"
        b"0 0 0\n"
        b"1 0 0\n"
        b"0 1 0\n"
        b"3 0 1 2\n"
    )


def _write_success_receipt(
    packet,
    output_dir,
    *,
    status="done",
    failure_phase=None,
    cuda_available=True,
    cuda_device="Tesla T4",
):
    from trellmlx.kaggle_cuda_witness import sha256_file

    manifest_path = packet.dataset_dir / "witness-manifest.json"
    if not manifest_path.is_file():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": "trellis2mlx.kaggle_cuda_witness.inputs.v1",
                    "dataset_id": packet.dataset_id,
                    "kernel_id": packet.kernel_id,
                },
                sort_keys=True,
            )
            + "\n"
        )
    outputs = {}
    for name in packet.outputs:
        path = output_dir / name
        outputs[name] = {
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    receipt = {
        "schema": "trellis2mlx.kaggle_cuda_witness.receipt.v1",
        "status": status,
        "failure_phase": failure_phase,
        "requested_dataset_id": packet.dataset_id,
        "requested_kernel_id": packet.kernel_id,
        "requested_accelerator": packet.accelerator,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "input_manifest": {
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
        "source_identity": {
            "dataset_sources": [packet.dataset_id],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
        "outputs": outputs,
    }
    (output_dir / "kaggle_cuda_witness_receipt.json").write_text(
        json.dumps(receipt) + "\n"
    )
    return receipt


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
    assert 'phase="cuda_route"' in runner
    assert "CUDA route is unavailable" in runner
    assert "find_manifest()" in runner
    assert "rglob(\"witness-manifest.json\")" in runner
    assert "mounted_input_files" in runner
    assert '"missing_input": str(source)' in runner
    assert "mounted_input_snapshot()" in runner


def test_prepare_packet_can_enable_kernel_internet_explicitly(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, load_prepared_packet, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/internet-inputs",
            kernel_id="operator/internet-cuda",
            title="Internet CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
            enable_internet=True,
        )
    )

    kernel_metadata = json.loads((packet.kernel_dir / "kernel-metadata.json").read_text())
    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    loaded = load_prepared_packet(packet.output_dir, expected_capsule_dir=capsule, failure_report_dir=tmp_path / "reports")

    assert kernel_metadata["enable_internet"] == "true"
    assert manifest["enable_internet"] is True
    assert loaded.enable_internet is True


def test_prepare_packet_runner_config_compiles_with_default_optional_outputs(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/default-outputs-inputs",
            kernel_id="operator/default-outputs-cuda",
            title="Default Outputs CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )

    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    config_line = next(line for line in runner.splitlines() if line.startswith("CONFIG = "))

    compile(runner, "run_kaggle_cuda_witness.py", "exec")
    assert "json.loads" in config_line
    assert not config_line.startswith("CONFIG = {")


def test_prepared_runner_hashes_every_output_into_its_receipt(tmp_path, monkeypatch):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/output-hash-inputs",
            kernel_id="operator/output-hash-cuda",
            title="Output Hash CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    namespace = {"__name__": "runner_test"}
    exec((packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text(), namespace)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuda_result.json").write_text('{"status":"done"}\n')
    (tmp_path / "cuda_result.npz").write_bytes(b"npz result")

    snapshot = namespace["output_snapshot"]()

    assert snapshot["cuda_result.json"] == {
        "exists": True,
        "sha256": namespace["sha256_file"](tmp_path / "cuda_result.json"),
        "size_bytes": (tmp_path / "cuda_result.json").stat().st_size,
    }
    assert snapshot["cuda_result.npz"] == {
        "exists": True,
        "sha256": namespace["sha256_file"](tmp_path / "cuda_result.npz"),
        "size_bytes": (tmp_path / "cuda_result.npz").stat().st_size,
    }


def test_prepared_runner_records_uncapped_mount_and_exact_source_identity(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/exact-mount-inputs",
            kernel_id="operator/exact-mount-cuda",
            title="Exact Mount CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()

    assert "files[:200]" not in runner
    assert '"source_identity"' in runner
    assert '"mounted_input_snapshot": mounted_input_snapshot()' in runner


def test_prepared_runner_rejects_substituted_mounted_manifest_identity(
    tmp_path,
    monkeypatch,
):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--output-json', required=True)\n"
        "p.add_argument('--output-npz', required=True)\n"
        "a=p.parse_args()\n"
        "Path(a.output_json).write_text('{\"status\":\"done\"}\\n')\n"
        "Path(a.output_npz).write_bytes(b'npz')\n"
    )
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/requested-inputs",
            kernel_id="operator/requested-cuda",
            title="Requested CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    manifest_path = packet.dataset_dir / "witness-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_id"] = "attacker/substituted-inputs"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    runner = runner.replace(
        'Path("/kaggle/input")',
        f"Path({str(packet.dataset_dir)!r})",
    )
    namespace = {"__name__": "runner_test"}
    exec(runner, namespace)

    rc = namespace["main"]()

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert rc != 0
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] in {
        "input_manifest_digest",
        "input_manifest_identity",
    }
    assert not (work / "cuda_result.json").exists()


def test_prepared_runner_rejects_missing_effective_cuda_before_probe(
    tmp_path,
    monkeypatch,
):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text(
        "from pathlib import Path\n"
        "Path('probe-executed').write_text('yes')\n"
    )
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/no-cuda-inputs",
            kernel_id="operator/no-cuda-witness",
            title="No CUDA Witness",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    fake_torch = types.SimpleNamespace(
        __version__="test",
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            get_device_name=lambda _index: "",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    runner = runner.replace(
        'Path("/kaggle/input")',
        f"Path({str(packet.dataset_dir)!r})",
    )
    namespace = {"__name__": "runner_test"}
    exec(runner, namespace)

    rc = namespace["main"]()

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert rc != 0
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "cuda_route"
    assert receipt["cuda_available"] is False
    assert receipt["cuda_device"] is None
    assert receipt["input_manifest"]["sha256"]
    assert not (work / "probe-executed").exists()


def test_prepare_packet_preserves_entrypoint_args_in_runner(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, load_prepared_packet, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/args-inputs",
            kernel_id="operator/args-cuda",
            title="Args CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
            entrypoint_args=("--branch", "neg", "--block-indices", "0"),
        )
    )

    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    loaded = load_prepared_packet(packet.output_dir, expected_capsule_dir=capsule, failure_report_dir=tmp_path / "reports")

    assert manifest["entrypoint_args"] == ["--branch", "neg", "--block-indices", "0"]
    assert loaded.entrypoint_args == ("--branch", "neg", "--block-indices", "0")
    assert '\\"entrypoint_args\\": [\\"--branch\\", \\"neg\\", \\"--block-indices\\", \\"0\\"]' in runner
    assert "command += CONFIG.get(\"entrypoint_args\", [])" in runner


def test_prepare_packet_can_declare_mesh_ply_output(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, load_prepared_packet, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/mesh-artifact-inputs",
            kernel_id="operator/mesh-artifact-cuda",
            title="Mesh Artifact CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
            output_ply="cuda_result_mesh.ply",
        )
    )

    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    loaded = load_prepared_packet(packet.output_dir, expected_capsule_dir=capsule, failure_report_dir=tmp_path / "reports")

    assert manifest["outputs"] == ["cuda_result.json", "cuda_result.npz", "cuda_result_mesh.ply"]
    assert loaded.output_ply == "cuda_result_mesh.ply"
    assert '\\"output_ply\\": \\"cuda_result_mesh.ply\\"' in runner
    assert 'command += ["--output-ply", CONFIG["output_ply"]]' in runner


def test_prepare_packet_can_declare_json_and_nested_expected_outputs_without_npz(
    tmp_path,
):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        build_kernel_output_command,
        load_prepared_packet,
        prepare_packet,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "decode.py").write_text("print('decode')\n")
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/selective-decode-inputs",
            kernel_id="operator/selective-decode-cuda",
            title="Selective Decode CUDA",
            entrypoint="decode.py",
            inputs=("decode.py",),
            output_json="selective_decode_report.json",
            output_npz=None,
            expected_outputs=(
                "meshes/switch-1.raw.ply",
                "meshes/switch-1.filled.ply",
            ),
            entrypoint_args=(
                "--shape-slat-grid",
                "cuda_result.npz",
                "--shape-slat-grid-sha256",
                "3" * 64,
                "--shape-slat-grid-report",
                "cuda_result.json",
                "--shape-slat-grid-report-sha256",
                "4" * 64,
            ),
        )
    )

    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    loaded = load_prepared_packet(packet.output_dir, expected_capsule_dir=capsule, failure_report_dir=tmp_path / "reports")
    output_command = build_kernel_output_command(packet, tmp_path / "downloaded")

    assert manifest["outputs"] == [
        "selective_decode_report.json",
        "meshes/switch-1.raw.ply",
        "meshes/switch-1.filled.ply",
    ]
    assert manifest["output_roles"]["npz"] is None
    assert manifest["output_roles"]["expected"] == [
        "meshes/switch-1.raw.ply",
        "meshes/switch-1.filled.ply",
    ]
    assert loaded.output_npz is None
    assert loaded.expected_outputs == (
        "meshes/switch-1.raw.ply",
        "meshes/switch-1.filled.ply",
    )
    assert loaded.entrypoint_args == tuple(manifest["entrypoint_args"])
    assert manifest["entrypoint_args"][-3:] == [
        "cuda_result.json",
        "--shape-slat-grid-report-sha256",
        "4" * 64,
    ]
    assert '\\"output_json\\": \\"selective_decode_report.json\\"' in runner
    assert '\\"output_npz\\": null' in runner
    assert '\\"--shape-slat-grid-sha256\\"' in runner
    assert f'\\\"{"3" * 64}\\\"' in runner
    assert '\\"--shape-slat-grid-report-sha256\\"' in runner
    assert f'\\\"{"4" * 64}\\\"' in runner
    assert 'if CONFIG["output_npz"]:' in runner
    assert '"--output-npz", CONFIG["output_npz"]' in runner
    file_pattern = output_command[output_command.index("--file-pattern") + 1]
    assert "selective_decode_report\\.json" in file_pattern
    assert "meshes/switch\\-1\\.raw\\.ply" in file_pattern
    assert "meshes/switch\\-1\\.filled\\.ply" in file_pattern
    output_pattern = re.compile(file_pattern)
    expected_names = (*packet.outputs, "kaggle_cuda_witness_receipt.json")
    for name in expected_names:
        assert output_pattern.search(name)
        assert not output_pattern.search(f"../../{name}")
        assert not output_pattern.search(f"prefix-{name}")
        assert not output_pattern.search(f"{name}.stale")
        assert not output_pattern.search(f"./{name}")


def test_validate_downloaded_outputs_rejects_missing_declared_nested_mesh(tmp_path):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        validate_downloaded_outputs,
    )

    output_dir = tmp_path / "outputs"
    (output_dir / "meshes").mkdir(parents=True)
    packet = KaggleCudaWitnessPacket(
        capsule_dir=tmp_path,
        output_dir=tmp_path / "packet",
        dataset_id="operator/selective-output-inputs",
        kernel_id="operator/selective-output-cuda",
        title="Selective Output CUDA",
        entrypoint="decode.py",
        inputs=("decode.py",),
        output_json="selective_decode_report.json",
        output_npz=None,
        expected_outputs=(
            "meshes/switch-1.raw.ply",
            "meshes/switch-1.filled.ply",
        ),
    )
    (output_dir / "selective_decode_report.json").write_text('{"status": "done"}\n')
    _write_valid_ply(output_dir / "meshes" / "switch-1.raw.ply")
    _write_success_receipt(packet, output_dir)

    with pytest.raises(WitnessPacketError, match="switch-1.filled.ply"):
        validate_downloaded_outputs(packet, output_dir)

    (output_dir / "meshes" / "switch-1.filled.ply").write_bytes(b"not a ply")
    _write_success_receipt(packet, output_dir)
    with pytest.raises(WitnessPacketError, match="invalid PLY"):
        validate_downloaded_outputs(packet, output_dir)


def test_prepare_packet_can_declare_full_mesh_state_output(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, load_prepared_packet, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/mesh-state-inputs",
            kernel_id="operator/mesh-state-cuda",
            title="Mesh State CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
            output_ply="cuda_result_mesh.ply",
            output_mesh_state="cuda_result_mesh_state.npz",
        )
    )

    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    loaded = load_prepared_packet(packet.output_dir, expected_capsule_dir=capsule, failure_report_dir=tmp_path / "reports")

    assert manifest["outputs"] == [
        "cuda_result.json",
        "cuda_result.npz",
        "cuda_result_mesh.ply",
        "cuda_result_mesh_state.npz",
    ]
    assert loaded.output_ply == "cuda_result_mesh.ply"
    assert loaded.output_mesh_state == "cuda_result_mesh_state.npz"
    assert '\\"output_mesh_state\\": \\"cuda_result_mesh_state.npz\\"' in runner
    assert 'command += ["--output-mesh-state", CONFIG["output_mesh_state"]]' in runner


def test_prepare_packet_can_declare_shape_slat_output(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, load_prepared_packet, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/shape-slat-inputs",
            kernel_id="operator/shape-slat-cuda",
            title="Shape SLat CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
            output_shape_slat="cuda_result_shape_slat.npz",
        )
    )

    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    loaded = load_prepared_packet(packet.output_dir, expected_capsule_dir=capsule, failure_report_dir=tmp_path / "reports")

    assert manifest["outputs"] == [
        "cuda_result.json",
        "cuda_result.npz",
        "cuda_result_shape_slat.npz",
    ]
    assert manifest["output_roles"]["shape_slat"] == "cuda_result_shape_slat.npz"
    assert loaded.output_shape_slat == "cuda_result_shape_slat.npz"
    assert '\\"output_shape_slat\\": \\"cuda_result_shape_slat.npz\\"' in runner
    assert 'command += ["--output-shape-slat", CONFIG["output_shape_slat"]]' in runner


def test_prepare_packet_can_declare_shape_flow_step_output(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, load_prepared_packet, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/shape-step-inputs",
            kernel_id="operator/shape-flow-step-cuda",
            title="Shape Flow Step CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
            output_shape_flow_step="cuda_result_shape_flow_step.npz",
        )
    )

    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    loaded = load_prepared_packet(packet.output_dir, expected_capsule_dir=capsule, failure_report_dir=tmp_path / "reports")

    assert manifest["outputs"] == [
        "cuda_result.json",
        "cuda_result.npz",
        "cuda_result_shape_flow_step.npz",
    ]
    assert manifest["output_roles"]["shape_flow_step"] == "cuda_result_shape_flow_step.npz"
    assert loaded.output_shape_flow_step == "cuda_result_shape_flow_step.npz"
    assert '\\"output_shape_flow_step\\": \\"cuda_result_shape_flow_step.npz\\"' in runner
    assert 'command += ["--output-shape-flow-step", CONFIG["output_shape_flow_step"]]' in runner


def test_prepare_packet_can_declare_shape_flow_noise_sample(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, load_prepared_packet, prepare_packet

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    (capsule / "mlx_shape_flow_step.npz").write_bytes(b"npz bytes")

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/shape-step-inputs",
            kernel_id="operator/shape-step-cuda",
            title="Shape Step CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py", "mlx_shape_flow_step.npz"),
            shape_flow_noise_sample="mlx_shape_flow_step.npz",
            output_shape_flow_step="cuda_result_shape_flow_step.npz",
        )
    )

    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    loaded = load_prepared_packet(packet.output_dir, expected_capsule_dir=capsule, failure_report_dir=tmp_path / "reports")

    assert manifest["input_roles"]["shape_flow_noise_sample"] == "mlx_shape_flow_step.npz"
    assert loaded.shape_flow_noise_sample == "mlx_shape_flow_step.npz"
    assert '\\"shape_flow_noise_sample\\": \\"mlx_shape_flow_step.npz\\"' in runner
    assert 'command += ["--shape-flow-noise-sample", CONFIG["shape_flow_noise_sample"]]' in runner


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


@pytest.mark.parametrize("output_shape", ("capsule", "capsule_parent"))
def test_prepare_packet_rejects_destructive_output_overlap_before_rmtree(
    tmp_path,
    monkeypatch,
    output_shape,
):
    from trellmlx import kaggle_cuda_witness as witness

    capsule = tmp_path / "custody" / "capsule"
    capsule.mkdir(parents=True)
    source = capsule / "cuda_probe.py"
    source.write_text("print('protected')\n")
    original = source.read_bytes()
    output_dir = capsule if output_shape == "capsule" else capsule.parent
    rmtree_called = False

    def forbidden_rmtree(path):
        nonlocal rmtree_called
        rmtree_called = True
        raise AssertionError(f"unsafe rmtree reached: {path}")

    monkeypatch.setattr(witness.shutil, "rmtree", forbidden_rmtree)
    with pytest.raises(witness.WitnessPacketError, match="overlap|protected|output"):
        witness.prepare_packet(
            witness.KaggleCudaWitnessPacket(
                capsule_dir=capsule,
                output_dir=output_dir,
                dataset_id="operator/path-custody-inputs",
                kernel_id="operator/path-custody",
                title="Path Custody",
                entrypoint="cuda_probe.py",
                inputs=("cuda_probe.py",),
            )
        )

    assert rmtree_called is False
    assert source.read_bytes() == original
    failure_report = output_dir.with_name(output_dir.name + ".failure.json")
    assert json.loads(failure_report.read_text())["failure_phase"] == "packet_path_validation"


def test_kernel_output_rejects_destructive_overlap_before_rmtree(tmp_path, monkeypatch):
    from trellmlx import kaggle_cuda_witness as witness

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('protected')\n")
    packet = witness.prepare_packet(
        witness.KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/kernel-output-custody-inputs",
            kernel_id="operator/kernel-output-custody",
            title="Kernel Output Custody",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    protected = packet.dataset_dir / "cuda_probe.py"
    original = protected.read_bytes()
    rmtree_called = False

    def forbidden_rmtree(path):
        nonlocal rmtree_called
        rmtree_called = True
        raise AssertionError(f"unsafe rmtree reached: {path}")

    monkeypatch.setattr(witness.shutil, "rmtree", forbidden_rmtree)
    with pytest.raises(witness.WitnessPacketError, match="overlap|protected|output"):
        witness.prepare_kernel_output_paths(
            packet,
            output_dir=packet.output_dir,
            report_dir=packet.output_dir / "reports",
        )

    assert rmtree_called is False
    assert protected.read_bytes() == original
    failure_report = packet.output_dir / "reports" / "kernel_output_path_validation.json"
    assert json.loads(failure_report.read_text())["failure_phase"] == "kernel_output_path_validation"


@pytest.mark.parametrize(
    ("reload_packet", "unsafe_role", "unsafe_shape"),
    (
        (False, "output", "capsule"),
        (False, "output", "capsule_parent"),
        (False, "output", "capsule_child"),
        (False, "report", "capsule"),
        (False, "report", "capsule_parent"),
        (False, "report", "capsule_child"),
        (True, "output", "capsule"),
        (True, "output", "capsule_parent"),
        (True, "output", "capsule_child"),
        (True, "report", "capsule"),
        (True, "report", "capsule_parent"),
        (True, "report", "capsule_child"),
    ),
)
def test_kernel_output_paths_preserve_original_capsule_direct_and_round_trip(
    tmp_path,
    monkeypatch,
    reload_packet,
    unsafe_role,
    unsafe_shape,
):
    from trellmlx import kaggle_cuda_witness as witness

    capsule = tmp_path / "source" / "capsule"
    capsule.mkdir(parents=True)
    protected = capsule / "cuda_probe.py"
    protected.write_text("print('protected')\n")
    packet = witness.prepare_packet(
        witness.KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/original-capsule-inputs",
            kernel_id="operator/original-capsule",
            title="Original Capsule",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    if reload_packet:
        packet = witness.load_prepared_packet(
            packet.output_dir,
            expected_capsule_dir=capsule,
            failure_report_dir=tmp_path / "reports",
        )
    assert packet.capsule_dir == capsule
    original = protected.read_bytes()
    unsafe = {
        "capsule": capsule,
        "capsule_parent": capsule.parent,
        "capsule_child": capsule / "download-child",
    }[unsafe_shape]
    output_dir = unsafe if unsafe_role == "output" else tmp_path / "download"
    report_dir = unsafe if unsafe_role == "report" else tmp_path / "reports"
    rmtree_called = False

    def forbidden_rmtree(path):
        nonlocal rmtree_called
        rmtree_called = True
        raise AssertionError(f"unsafe rmtree reached: {path}")

    monkeypatch.setattr(witness.shutil, "rmtree", forbidden_rmtree)
    with pytest.raises(witness.WitnessPacketError, match="capsule|protected|overlap"):
        witness.prepare_kernel_output_paths(
            packet,
            output_dir=output_dir,
            report_dir=report_dir,
        )

    assert rmtree_called is False
    assert protected.read_bytes() == original
    failure_reports = list(tmp_path.rglob("kernel_output_path_validation.json"))
    assert failure_reports
    assert not any(path.is_relative_to(unsafe) for path in failure_reports)


@pytest.mark.parametrize("mutation", ("substitute", "delete"))
def test_load_prepared_packet_requires_caller_bound_original_capsule_identity(
    tmp_path,
    monkeypatch,
    mutation,
):
    from trellmlx import kaggle_cuda_witness as witness

    capsule = tmp_path / "source" / "capsule"
    capsule.mkdir(parents=True)
    protected = capsule / "cuda_probe.py"
    protected.write_text("print('protected original')\n")
    packet = witness.prepare_packet(
        witness.KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/caller-bound-capsule-inputs",
            kernel_id="operator/caller-bound-capsule",
            title="Caller-bound Capsule",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    state_path = packet.output_dir / witness.PACKET_STATE_NAME
    state = json.loads(state_path.read_text())
    substitute = tmp_path / "substituted-capsule"
    substitute.mkdir()
    if mutation == "substitute":
        state["capsule_dir"] = str(substitute)
    else:
        del state["capsule_dir"]
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    original = protected.read_bytes()
    rmtree_called = False

    def forbidden_rmtree(path):
        nonlocal rmtree_called
        rmtree_called = True
        raise AssertionError(f"unsafe rmtree reached: {path}")

    monkeypatch.setattr(witness.shutil, "rmtree", forbidden_rmtree)
    with pytest.raises(witness.WitnessPacketError, match="capsule.*identity|capsule_dir"):
        witness.load_prepared_packet(
            packet.output_dir,
            expected_capsule_dir=capsule,
            failure_report_dir=tmp_path / "trusted-reports",
        )

    assert rmtree_called is False
    assert protected.read_bytes() == original
    failure_path = tmp_path / "trusted-reports" / "packet_capsule_identity_validation.json"
    failure = json.loads(failure_path.read_text())
    assert failure["status"] == "failed"
    assert failure["failure_phase"] == "packet_capsule_identity_validation"
    assert failure["expected_capsule_dir"] == str(capsule)
    assert not failure_path.is_relative_to(capsule)
    assert not failure_path.is_relative_to(substitute)


@pytest.mark.parametrize("claimed_shape", ("report_temporary", "broad_ancestor", "anchor"))
def test_load_prepared_packet_publishes_capsule_identity_failure_outside_all_mutations(
    tmp_path,
    monkeypatch,
    claimed_shape,
):
    from trellmlx import kaggle_cuda_witness as witness

    capsule = tmp_path / "source" / "capsule"
    capsule.mkdir(parents=True)
    protected = capsule / "cuda_probe.py"
    protected.write_text("print('protected original')\n")
    report_dir = tmp_path.parent / f"trusted-capsule-reports-{tmp_path.name}"
    packet = witness.prepare_packet(
        witness.KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/diagnostic-capsule-inputs",
            kernel_id="operator/diagnostic-capsule",
            title="Diagnostic Capsule",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    failure_path = report_dir / "packet_capsule_identity_validation.json"
    failure_temporary = failure_path.with_name(failure_path.name + ".tmp")
    claimed = {
        "report_temporary": failure_temporary,
        "broad_ancestor": tmp_path,
        "anchor": Path(tmp_path.anchor),
    }[claimed_shape]
    claimed_bytes = None
    if claimed_shape == "report_temporary":
        report_dir.mkdir(parents=True)
        claimed.write_text("claimed bytes must survive\n")
        claimed_bytes = claimed.read_bytes()
    state_path = packet.output_dir / witness.PACKET_STATE_NAME
    state = json.loads(state_path.read_text())
    state["capsule_dir"] = str(claimed)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    original = protected.read_bytes()
    rmtree_called = False

    def forbidden_rmtree(path):
        nonlocal rmtree_called
        rmtree_called = True
        raise AssertionError(f"unsafe rmtree reached: {path}")

    monkeypatch.setattr(witness.shutil, "rmtree", forbidden_rmtree)
    with pytest.raises(witness.WitnessPacketError, match="capsule.*identity|capsule_dir"):
        witness.load_prepared_packet(
            packet.output_dir,
            expected_capsule_dir=capsule,
            failure_report_dir=report_dir,
        )

    assert rmtree_called is False
    assert protected.read_bytes() == original
    if claimed_bytes is not None:
        assert claimed.read_bytes() == claimed_bytes
    if claimed_shape == "report_temporary":
        failure_path = packet.output_dir / "reports" / "packet_capsule_identity_validation.json"
        failure_temporary = failure_path.with_name(failure_path.name + ".tmp")
    failure = json.loads(failure_path.read_text())
    assert failure["failure_phase"] == "packet_capsule_identity_validation"
    assert failure["effective_report_path"] == str(failure_path)
    assert failure["effective_report_temporary"] == str(failure_temporary)
    if claimed_shape != "anchor":
        assert not witness._paths_overlap(failure_path, claimed)
        assert not witness._paths_overlap(failure_temporary, claimed)
    if claimed_shape in {"report_temporary", "broad_ancestor", "anchor"}:
        assert failure["claimed_capsule_valid"] is False


def test_prepare_packet_rejects_normalized_output_aliases(tmp_path):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        prepare_packet,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")

    with pytest.raises(WitnessPacketError, match="canonical|unique"):
        prepare_packet(
            KaggleCudaWitnessPacket(
                capsule_dir=capsule,
                output_dir=tmp_path / "packet",
                dataset_id="operator/alias-inputs",
                kernel_id="operator/alias-cuda",
                title="Alias CUDA",
                entrypoint="cuda_probe.py",
                inputs=("cuda_probe.py",),
                output_json="result.json",
                output_npz=None,
                expected_outputs=("./result.json",),
            )
        )


@pytest.mark.parametrize("unsafe_output", ("../outside.json", "/tmp/outside.json"))
def test_load_prepared_packet_rejects_unsafe_manifest_output(
    tmp_path,
    unsafe_output,
):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        load_prepared_packet,
        prepare_packet,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/unsafe-load-inputs",
            kernel_id="operator/unsafe-load-cuda",
            title="Unsafe Load CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    manifest_path = packet.dataset_dir / "witness-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"] = [unsafe_output]
    manifest["output_roles"] = {
        "json": unsafe_output,
        "npz": None,
        "ply": None,
        "mesh_state": None,
        "shape_slat": None,
        "shape_flow_step": None,
        "expected": [],
    }
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(
        WitnessPacketError,
        match="manifest_sha256|relative|canonical",
    ):
        load_prepared_packet(packet.output_dir, expected_capsule_dir=capsule, failure_report_dir=tmp_path / "reports")


def test_prepare_packet_expands_directory_inputs(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, prepare_packet

    capsule = tmp_path / "capsule"
    source_tree = capsule / "source_tree"
    (source_tree / "pkg").mkdir(parents=True)
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    (source_tree / "__init__.py").write_text("")
    (source_tree / "pkg" / "module.py").write_text("VALUE = 3\n")

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/source-tree-inputs",
            kernel_id="operator/source-tree-cuda",
            title="Source Tree CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py", "source_tree"),
        )
    )

    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    dataset_metadata = json.loads((packet.dataset_dir / "dataset-metadata.json").read_text())

    assert (packet.dataset_dir / "source_tree" / "pkg" / "module.py").read_text() == "VALUE = 3\n"
    assert sorted(manifest["files"]) == [
        "cuda_probe.py",
        "source_tree/__init__.py",
        "source_tree/pkg/module.py",
    ]
    assert sorted(resource["path"] for resource in dataset_metadata["resources"]) == [
        "cuda_probe.py",
        "source_tree/__init__.py",
        "source_tree/pkg/module.py",
        "witness-manifest.json",
    ]


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
        (
            "\\A(?:cuda_result\\.json|cuda_result\\.npz|"
            "kaggle_cuda_witness_receipt\\.json)\\Z"
        ),
        "--page-size",
        "100",
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


def test_wait_for_published_dataset_manifest_rejects_stale_ready_version(
    tmp_path,
):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        prepare_packet,
        wait_for_published_dataset_manifest,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/publication-barrier-inputs",
            kernel_id="operator/publication-barrier-cuda",
            title="Publication Barrier CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    expected_manifest = (
        packet.dataset_dir / "witness-manifest.json"
    ).read_bytes()
    downloads = 0

    def runner(cmd, capture_output, text, check):
        nonlocal downloads
        if cmd[:3] == ["kaggle", "datasets", "status"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"status": "ready", "current_version_number": 2}',
                stderr="",
            )
        assert cmd[:3] == ["kaggle", "datasets", "download"]
        downloads += 1
        output_dir = tmp_path / cmd[cmd.index("-p") + 1]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "witness-manifest.json").write_bytes(
            b"stale manifest\n" if downloads == 1 else expected_manifest
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    report = wait_for_published_dataset_manifest(
        packet,
        report_path=tmp_path / "publication.json",
        runner=runner,
        sleeper=lambda _seconds: None,
        scratch_root=tmp_path,
    )

    assert downloads == 2
    assert report["status"] == "done"
    assert report["remote_manifest_sha256"] == report[
        "expected_manifest_sha256"
    ]
    assert report["stale_observations"] == 1
    assert report["dataset_status"] == {
        "status": "ready",
        "current_version_number": 2,
    }


@pytest.mark.parametrize(
    ("mode", "failure_phase"),
    [
        ("malformed_status", "dataset_status"),
        ("stale_manifest", "stale_manifest"),
        ("missing_manifest", "manifest_download"),
    ],
)
def test_wait_for_published_dataset_manifest_writes_bounded_failure_report(
    tmp_path,
    mode,
    failure_phase,
):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        prepare_packet,
        wait_for_published_dataset_manifest,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/publication-failure-inputs",
            kernel_id="operator/publication-failure-cuda",
            title="Publication Failure CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    report_path = tmp_path / "publication.json"

    def runner(cmd, capture_output, text, check):
        if cmd[:3] == ["kaggle", "datasets", "status"]:
            stdout = (
                "not json"
                if mode == "malformed_status"
                else '{"status": "ready"}'
            )
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=stdout,
                stderr="",
            )
        output_dir = Path(cmd[cmd.index("-p") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        if mode == "stale_manifest":
            (output_dir / "witness-manifest.json").write_text("stale\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    report = wait_for_published_dataset_manifest(
        packet,
        report_path=report_path,
        runner=runner,
        sleeper=lambda _seconds: None,
        scratch_root=tmp_path,
        max_attempts=2,
    )

    assert report["status"] == "failed"
    assert report["failure_phase"] == failure_phase
    assert report["attempts"] == 2
    assert report["last_trustworthy_phase"] == "publication_observed"
    assert json.loads(report_path.read_text()) == report


def test_wait_for_published_dataset_manifest_writes_running_observation(
    tmp_path,
):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        prepare_packet,
        wait_for_published_dataset_manifest,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/publication-running-inputs",
            kernel_id="operator/publication-running-cuda",
            title="Publication Running CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py",),
        )
    )
    report_path = tmp_path / "publication.json"

    def runner(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"status": "creating"}',
            stderr="",
        )

    def observe_and_stop(_seconds):
        running = json.loads(report_path.read_text())
        assert running["status"] == "running"
        assert running["current_phase"] == "dataset_status"
        assert running["attempts"] == 1
        assert running["max_attempts"] is None
        raise StopIteration

    with pytest.raises(StopIteration):
        wait_for_published_dataset_manifest(
            packet,
            report_path=report_path,
            runner=runner,
            sleeper=observe_and_stop,
        )


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


def test_load_prepared_packet_round_trips_drive_commands(tmp_path, capsys):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        build_kernel_push_command,
        load_prepared_packet,
        main,
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

    loaded = load_prepared_packet(
        tmp_path / "packet",
        expected_capsule_dir=capsule,
        failure_report_dir=tmp_path / "reports",
    )

    assert loaded == KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=prepared.output_dir,
        dataset_id=prepared.dataset_id,
        kernel_id=prepared.kernel_id,
        title=prepared.title,
        entrypoint=prepared.entrypoint,
        inputs=prepared.inputs,
        accelerator=prepared.accelerator,
    )
    assert build_kernel_push_command(loaded, timeout_seconds=30)[-2:] == ["--timeout", "30"]
    assert main(
        [
            "print-commands",
            "--packet-dir",
            str(prepared.output_dir),
            "--capsule-dir",
            str(capsule),
            "--report-dir",
            str(tmp_path / "reports"),
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset_id"] == prepared.dataset_id

    with pytest.raises(SystemExit) as exc_info:
        main(["print-commands", "--packet-dir", str(prepared.output_dir)])
    assert exc_info.value.code == 2


def test_validate_downloaded_outputs_rejects_blank_or_partial_evidence(tmp_path):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        validate_downloaded_outputs,
    )

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    packet = KaggleCudaWitnessPacket(
        capsule_dir=tmp_path,
        output_dir=tmp_path / "packet",
        dataset_id="operator/output-validation-inputs",
        kernel_id="operator/output-validation-cuda",
        title="Output Validation CUDA",
        entrypoint="probe.py",
        inputs=("probe.py",),
    )
    (output_dir / "cuda_result.json").write_text('{"status": "done"}\n')
    (output_dir / "cuda_result.npz").write_bytes(b"")
    _write_success_receipt(packet, output_dir)

    with pytest.raises(WitnessPacketError, match="blank downloaded output"):
        validate_downloaded_outputs(packet, output_dir)

    (output_dir / "cuda_result.npz").write_bytes(b"partial zip")
    _write_success_receipt(packet, output_dir)
    with pytest.raises(WitnessPacketError, match="invalid NPZ"):
        validate_downloaded_outputs(packet, output_dir)

    with zipfile.ZipFile(output_dir / "cuda_result.npz", "w") as archive:
        archive.writestr("not-an-array.txt", "this is not an npz")
    _write_success_receipt(packet, output_dir)
    with pytest.raises(WitnessPacketError, match="invalid NPZ"):
        validate_downloaded_outputs(packet, output_dir)

    np.savez(output_dir / "cuda_result.npz", witness=np.asarray([1], dtype=np.int32))
    _write_success_receipt(packet, output_dir)
    records = validate_downloaded_outputs(packet, output_dir)

    assert records["cuda_result.json"]["size_bytes"] > 0
    assert records["cuda_result.json"]["sha256"]
    assert records["cuda_result.npz"]["size_bytes"] > 0
    assert records["cuda_result.npz"]["sha256"]


def test_validate_downloaded_outputs_rejects_failed_or_mismatched_receipt(
    tmp_path,
):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        validate_downloaded_outputs,
    )

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    packet = KaggleCudaWitnessPacket(
        capsule_dir=tmp_path,
        output_dir=tmp_path / "packet",
        dataset_id="operator/receipt-inputs",
        kernel_id="operator/receipt-cuda",
        title="Receipt CUDA",
        entrypoint="probe.py",
        inputs=("probe.py",),
    )
    (output_dir / "cuda_result.json").write_text('{"status": "done"}\n')
    np.savez(output_dir / "cuda_result.npz", witness=np.asarray([1]))
    _write_success_receipt(
        packet,
        output_dir,
        status="failed",
        failure_phase="execution",
    )

    with pytest.raises(WitnessPacketError, match="receipt status"):
        validate_downloaded_outputs(packet, output_dir)

    receipt = _write_success_receipt(packet, output_dir)
    receipt["outputs"]["cuda_result.npz"]["sha256"] = "0" * 64
    (output_dir / "kaggle_cuda_witness_receipt.json").write_text(
        json.dumps(receipt) + "\n"
    )
    with pytest.raises(WitnessPacketError, match="receipt.*digest|digest.*receipt"):
        validate_downloaded_outputs(packet, output_dir)

    receipt = _write_success_receipt(packet, output_dir)
    receipt["input_manifest"]["sha256"] = "0" * 64
    (output_dir / "kaggle_cuda_witness_receipt.json").write_text(
        json.dumps(receipt) + "\n"
    )
    with pytest.raises(WitnessPacketError, match="manifest.*digest|digest.*manifest"):
        validate_downloaded_outputs(packet, output_dir)

    _write_success_receipt(
        packet,
        output_dir,
        cuda_available=False,
        cuda_device=None,
    )
    with pytest.raises(WitnessPacketError, match="CUDA"):
        validate_downloaded_outputs(packet, output_dir)


def test_validate_downloaded_outputs_rejects_header_only_ply(tmp_path):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        validate_downloaded_outputs,
    )

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    packet = KaggleCudaWitnessPacket(
        capsule_dir=tmp_path,
        output_dir=tmp_path / "packet",
        dataset_id="operator/ply-validation-inputs",
        kernel_id="operator/ply-validation-cuda",
        title="PLY Validation CUDA",
        entrypoint="probe.py",
        inputs=("probe.py",),
        output_npz=None,
        expected_outputs=("mesh.ply",),
    )
    (output_dir / "cuda_result.json").write_text('{"status": "done"}\n')
    (output_dir / "mesh.ply").write_bytes(b"ply\nend_header\n")
    _write_success_receipt(packet, output_dir)

    with pytest.raises(WitnessPacketError, match="invalid PLY"):
        validate_downloaded_outputs(packet, output_dir)


def test_wait_for_downloaded_outputs_allows_materializing_npz(tmp_path):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, wait_for_downloaded_outputs

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    packet = KaggleCudaWitnessPacket(
        capsule_dir=tmp_path,
        output_dir=tmp_path / "packet",
        dataset_id="operator/output-stabilization-inputs",
        kernel_id="operator/output-stabilization-cuda",
        title="Output Stabilization CUDA",
        entrypoint="probe.py",
        inputs=("probe.py",),
    )
    (output_dir / "cuda_result.json").write_text('{"status": "done"}\n')
    (output_dir / "cuda_result.npz").write_bytes(b"")
    _write_success_receipt(packet, output_dir)
    sleeps = []

    def materialize_npz(seconds):
        sleeps.append(seconds)
        np.savez(output_dir / "cuda_result.npz", witness=np.asarray([1], dtype=np.int32))
        _write_success_receipt(packet, output_dir)

    records = wait_for_downloaded_outputs(
        packet,
        output_dir,
        max_wait_seconds=1.0,
        poll_seconds=0.01,
        sleeper=materialize_npz,
    )

    assert sleeps == [0.01]
    assert records["cuda_result.npz"]["size_bytes"] > 0

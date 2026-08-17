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
        "run_id": packet.run_id,
        "expected_image_sha256": packet.expected_image_sha256,
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
    loaded = load_prepared_packet(packet.output_dir)

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
    loaded = load_prepared_packet(packet.output_dir)

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
    loaded = load_prepared_packet(packet.output_dir)

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
    loaded = load_prepared_packet(packet.output_dir)
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
    loaded = load_prepared_packet(packet.output_dir)

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
    loaded = load_prepared_packet(packet.output_dir)

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
    loaded = load_prepared_packet(packet.output_dir)

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
    loaded = load_prepared_packet(packet.output_dir)

    assert manifest["input_roles"]["shape_flow_noise_sample"] == "mlx_shape_flow_step.npz"
    assert loaded.shape_flow_noise_sample == "mlx_shape_flow_step.npz"
    assert '\\"shape_flow_noise_sample\\": \\"mlx_shape_flow_step.npz\\"' in runner
    assert 'command += ["--shape-flow-noise-sample", CONFIG["shape_flow_noise_sample"]]' in runner


def test_prepare_packet_binds_sparse_flow_noise_role_and_expected_digest(tmp_path):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        _prepared_summary,
        load_prepared_packet,
        prepare_packet,
        sha256_file,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    sparse_noise = capsule / "source_mps_sparse_flow_steps.npz"
    np.savez(sparse_noise, noise=np.zeros((1, 8, 16, 16, 16), dtype=np.float32))
    expected_sha256 = sha256_file(sparse_noise)

    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/sparse-noise-inputs",
            kernel_id="operator/sparse-noise-cuda",
            title="Sparse Noise CUDA",
            entrypoint="cuda_probe.py",
            inputs=("cuda_probe.py", sparse_noise.name),
            sparse_flow_noise_sample=sparse_noise.name,
            sparse_flow_noise_sample_sha256=expected_sha256,
        )
    )

    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    loaded = load_prepared_packet(packet.output_dir)

    assert manifest["input_roles"]["sparse_flow_noise_sample"] == sparse_noise.name
    assert manifest["input_roles"]["sparse_flow_noise_sample_sha256"] == expected_sha256
    assert loaded.sparse_flow_noise_sample == sparse_noise.name
    assert loaded.sparse_flow_noise_sample_sha256 == expected_sha256
    assert _prepared_summary(packet)["input_roles"] == manifest["input_roles"]
    assert '\\"sparse_flow_noise_sample\\": \\"source_mps_sparse_flow_steps.npz\\"' in runner
    assert f'\\"sparse_flow_noise_sample_sha256\\": \\"{expected_sha256}\\"' in runner
    assert 'command += ["--sparse-flow-noise-sample", CONFIG["sparse_flow_noise_sample"]]' in runner
    assert (
        'command += ["--sparse-flow-noise-sample-sha256", '
        'CONFIG["sparse_flow_noise_sample_sha256"]]'
    ) in runner


@pytest.mark.parametrize(
    ("include_input", "expected_sha256", "message"),
    [
        (False, "a" * 64, "staged inputs"),
        (True, None, "expected SHA256"),
        (True, "A" * 64, "canonical lowercase"),
    ],
)
def test_prepare_packet_rejects_unbound_sparse_flow_noise_role(
    tmp_path,
    include_input,
    expected_sha256,
    message,
):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        prepare_packet,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    (capsule / "noise.npz").write_bytes(b"noise")
    inputs = ("cuda_probe.py", "noise.npz") if include_input else ("cuda_probe.py",)

    with pytest.raises(WitnessPacketError, match=message):
        prepare_packet(
            KaggleCudaWitnessPacket(
                capsule_dir=capsule,
                output_dir=tmp_path / "packet",
                dataset_id="operator/unbound-noise-inputs",
                kernel_id="operator/unbound-noise-cuda",
                title="Unbound Noise CUDA",
                entrypoint="cuda_probe.py",
                inputs=inputs,
                sparse_flow_noise_sample="noise.npz",
                sparse_flow_noise_sample_sha256=expected_sha256,
            )
        )

    assert not (tmp_path / "packet").exists()


@pytest.mark.parametrize(
    ("run_id", "image_sha256", "entrypoint_args", "message"),
    [
        (
            None,
            "a" * 64,
            ("--run-id", "11111111-1111-4111-8111-111111111111", "--expected-image-sha256", "a" * 64),
            "requires packet field run_id",
        ),
        (
            "11111111-1111-4111-8111-111111111111",
            "a" * 64,
            ("--expected-image-sha256", "a" * 64),
            "run_id must appear exactly once",
        ),
        (
            "11111111-1111-4111-8111-111111111111",
            "a" * 64,
            (
                "--run-id",
                "11111111-1111-4111-8111-111111111111",
                "--run-id",
                "11111111-1111-4111-8111-111111111111",
                "--expected-image-sha256",
                "a" * 64,
            ),
            "run_id must appear exactly once",
        ),
        (
            "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
            "a" * 64,
            (
                "--run-id",
                "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
                "--expected-image-sha256",
                "a" * 64,
            ),
            "canonical lowercase UUID",
        ),
        (
            "11111111-1111-4111-8111-111111111111",
            None,
            (
                "--run-id",
                "11111111-1111-4111-8111-111111111111",
                "--expected-image-sha256",
                "a" * 64,
            ),
            "requires packet field expected_image_sha256",
        ),
        (
            "11111111-1111-4111-8111-111111111111",
            "A" * 64,
            (
                "--run-id",
                "11111111-1111-4111-8111-111111111111",
                "--expected-image-sha256",
                "A" * 64,
            ),
            "canonical lowercase hex",
        ),
        (
            None,
            None,
            ("--run-id=11111111-1111-4111-8111-111111111111",),
            "assignment form",
        ),
        (
            None,
            None,
            ("--expected-image-sha256=" + "a" * 64,),
            "assignment form",
        ),
    ],
)
def test_prepare_packet_rejects_unbound_attempt_identity(
    tmp_path,
    run_id,
    image_sha256,
    entrypoint_args,
    message,
):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        prepare_packet,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "cuda_probe.py").write_text("print('probe')\n")

    with pytest.raises(WitnessPacketError, match=message):
        prepare_packet(
            KaggleCudaWitnessPacket(
                capsule_dir=capsule,
                output_dir=tmp_path / "packet",
                dataset_id="operator/attempt-inputs",
                kernel_id="operator/attempt-cuda",
                title="Attempt CUDA",
                entrypoint="cuda_probe.py",
                inputs=("cuda_probe.py",),
                run_id=run_id,
                expected_image_sha256=image_sha256,
                entrypoint_args=entrypoint_args,
            )
        )

    assert not (tmp_path / "packet").exists()


def test_validate_downloaded_outputs_rejects_missing_sparse_noise_report_identity(tmp_path):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        prepare_packet,
        sha256_file,
        validate_downloaded_outputs,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "probe.py").write_text("print('probe')\n")
    noise = capsule / "noise.npz"
    np.savez(noise, noise=np.zeros((1, 8, 16, 16, 16), dtype=np.float32))
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/noise-output-inputs",
            kernel_id="operator/noise-output-cuda",
            title="Noise Output CUDA",
            entrypoint="probe.py",
            inputs=("probe.py", "noise.npz"),
            sparse_flow_noise_sample="noise.npz",
            sparse_flow_noise_sample_sha256=sha256_file(noise),
        )
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "cuda_result.json").write_text('{"status": "done"}\n')
    np.savez(output_dir / "cuda_result.npz", witness=np.asarray([1]))
    _write_success_receipt(packet, output_dir)

    with pytest.raises(WitnessPacketError, match="sparse-flow noise identity"):
        validate_downloaded_outputs(packet, output_dir)

    expected_sha256 = packet.sparse_flow_noise_sample_sha256
    (output_dir / "cuda_result.json").write_text(
        json.dumps(
            {
                "status": "done",
                "sparse_flow_noise_sample": {
                    "path": "noise.npz",
                    "expected_sha256": expected_sha256,
                    "sha256": expected_sha256,
                    "noise_key": "noise",
                    "noise_shape": [1, 8, 16, 16, 16],
                    "noise_dtype": "float32",
                    "sampling_route": "official-source-sparse-flow-from-admitted-noise",
                },
            }
        )
        + "\n"
    )
    _write_success_receipt(packet, output_dir)

    records = validate_downloaded_outputs(packet, output_dir)
    assert records["cuda_result.json"]["sha256"] == sha256_file(
        output_dir / "cuda_result.json"
    )

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

    with pytest.raises(WitnessPacketError, match="relative|canonical"):
        load_prepared_packet(packet.output_dir)


def test_prepare_packet_rejects_directory_inputs_omitted_by_kaggle_skip_mode(tmp_path):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        prepare_packet,
    )

    capsule = tmp_path / "capsule"
    source_tree = capsule / "source_tree"
    (source_tree / "pkg").mkdir(parents=True)
    (capsule / "cuda_probe.py").write_text("print('probe')\n")
    (source_tree / "__init__.py").write_text("")
    (source_tree / "pkg" / "module.py").write_text("VALUE = 3\n")

    with pytest.raises(WitnessPacketError, match="directory input.*dir-mode skip"):
        prepare_packet(
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

    assert not (tmp_path / "packet").exists()

    with pytest.raises(WitnessPacketError, match="nested input.*dir-mode skip"):
        prepare_packet(
            KaggleCudaWitnessPacket(
                capsule_dir=capsule,
                output_dir=tmp_path / "packet",
                dataset_id="operator/source-tree-inputs",
                kernel_id="operator/source-tree-cuda",
                title="Source Tree CUDA",
                entrypoint="cuda_probe.py",
                inputs=("cuda_probe.py", "source_tree/pkg/module.py"),
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

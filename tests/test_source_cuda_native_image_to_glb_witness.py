import ast
import errno
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import re
import shutil
import subprocess
import sys
import types
import zipfile

import numpy as np
import pytest


EXPECTED_STAGES = (
    "preprocessed_image",
    "conditioning_512",
    "sparse_flow",
    "sparse_support",
    "shape_flow",
    "shape_slat",
    "texture_flow",
    "decoder_raw_mesh",
    "texture_voxels",
    "pipeline_filled_mesh",
    "postprocess_stage11_pre_orientation",
    "postprocess_stage12_post_orientation",
    "consumer_glb",
)

FINAL_CONSUMER_STAGES = (
    "decoder_raw_mesh",
    "postprocess_stage11_pre_orientation",
    "postprocess_stage12_post_orientation",
    "consumer_glb",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_args(tmp_path: Path, image: Path) -> list[str]:
    return [
        "--image",
        str(image),
        "--expected-image-sha256",
        _sha256(image),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--work-dir",
        str(tmp_path / "runtime"),
    ]


def _stage_packet_rembg(capsule: Path, witness, monkeypatch):
    payloads = {
        "model.safetensors": b"packet-rembg-model",
        "config.json": b"packet-rembg-config",
        "birefnet.py": b"packet-rembg-implementation",
        "BiRefNet_config.py": b"packet-rembg-configuration",
    }
    coordinates = {
        "model.safetensors": "rembg-model.safetensors",
        "config.json": "rembg-config.json",
        "birefnet.py": "rembg-birefnet.py",
        "BiRefNet_config.py": "rembg-BiRefNet-config.py",
    }
    expected = {}
    for role, payload in payloads.items():
        path = capsule / coordinates[role]
        path.write_bytes(payload)
        expected[role] = _sha256(path)
    monkeypatch.setattr(witness, "REMBG_FILES", expected)
    arguments = tuple(
        item
        for role, attribute in witness.REMBG_FILE_ARGUMENTS.items()
        for item in (f"--{attribute.replace('_', '-')}", coordinates[role])
    )
    return coordinates, arguments, expected


def _stage_authority_helper(capsule: Path) -> str:
    import trellmlx.witness_authority as authority

    coordinate = "witness_authority.py"
    shutil.copy2(authority.__file__, capsule / coordinate)
    return coordinate


def _native_packet_contract(tmp_path: Path, monkeypatch):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    authority_helper = _stage_authority_helper(capsule)
    image = capsule / "9_img.png"
    image.write_bytes(b"native-packet-input")
    coordinates, rembg_arguments, expected_rembg = _stage_packet_rembg(
        capsule, witness, monkeypatch
    )
    run_id = "31fce6b7-853b-4a0f-b99d-518be23ebabc"
    image_sha256 = _sha256(image)
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    packet = KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=tmp_path / "packet",
        dataset_id="operator/native-image-anchor-inputs",
        kernel_id="operator/native-image-anchor-cuda",
        title="Native Image Anchor CUDA",
        entrypoint=entrypoint.name,
        inputs=(entrypoint.name, authority_helper, image.name, *coordinates.values()),
        output_json="report.json",
        output_npz=None,
        expected_outputs=expected_outputs,
        run_id=run_id,
        expected_image_sha256=image_sha256,
        entrypoint_args=(
            "--image",
            image.name,
            "--expected-image-sha256",
            image_sha256,
            "--run-id",
            run_id,
            "--output-dir",
            "outputs",
            "--work-dir",
            "runtime",
            *rembg_arguments,
        ),
    )
    return witness, packet, coordinates, expected_rembg


def _write_download_receipt(packet, bundle_dir: Path) -> None:
    from trellmlx.kaggle_cuda_witness import sha256_file

    receipt_outputs = {
        name: {
            "exists": True,
            "sha256": sha256_file(bundle_dir / name),
            "size_bytes": (bundle_dir / name).stat().st_size,
        }
        for name in packet.outputs
    }
    manifest = packet.dataset_dir / "witness-manifest.json"
    receipt = {
        "schema": "trellis2mlx.kaggle_cuda_witness.receipt.v1",
        "status": "done",
        "failure_phase": None,
        "requested_dataset_id": packet.dataset_id,
        "requested_kernel_id": packet.kernel_id,
        "requested_accelerator": packet.accelerator,
        "source_identity": {
            "dataset_sources": [packet.dataset_id],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
        "run_id": packet.run_id,
        "expected_image_sha256": packet.expected_image_sha256,
        "cuda_available": True,
        "cuda_device": "Tesla T4",
        "input_manifest": {
            "sha256": sha256_file(manifest),
            "size_bytes": manifest.stat().st_size,
        },
        "outputs": receipt_outputs,
    }
    (bundle_dir / "kaggle_cuda_witness_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n"
    )


def test_native_packet_preparation_binds_final_consumer_profile(tmp_path, monkeypatch):
    witness, packet, _coordinates, _expected = _native_packet_contract(
        tmp_path,
        monkeypatch,
    )
    packet = replace(
        packet,
        expected_outputs=tuple(
            witness.EXPECTED_ARTIFACT_FILENAMES[stage]
            for stage in FINAL_CONSUMER_STAGES
        ),
        entrypoint_args=(
            *packet.entrypoint_args,
            "--capture-profile",
            "final-consumer",
        ),
    )

    admitted = witness.prepare_native_image_to_glb_packet(packet)

    manifest = json.loads(
        (admitted.dataset_dir / "witness-manifest.json").read_text()
    )
    assert manifest["output_roles"]["expected"] == list(packet.expected_outputs)
    assert manifest["entrypoint_args"][-2:] == [
        "--capture-profile",
        "final-consumer",
    ]


def _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.native_image_to_glb_attempt import (
        AttemptAsset,
        CAPTURE_PROFILE_OUTPUTS,
        NativeImageToGLBAttemptSpec,
        build_attempt_packet,
    )

    sources = tmp_path / "attempt-sources"
    sources.mkdir()

    entrypoint = sources / "synthetic-native-entrypoint.py"
    entrypoint.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--output-json', required=True)\n"
        "p.add_argument('--output-dir', required=True)\n"
        "p.add_argument('--capture-profile', required=True)\n"
        "args, _ = p.parse_known_args()\n"
        "Path('entrypoint-invoked.json').write_text(json.dumps({'capture_profile': args.capture_profile}))\n"
        "output = Path(args.output_dir)\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "for name in ('07-decoder_raw_mesh.npz', '10-postprocess_stage11_pre_orientation.npz', '11-postprocess_stage12_post_orientation.npz'):\n"
        "    (output / name).write_bytes(b'synthetic-npz')\n"
        "(output / '12-consumer_glb.glb').write_bytes(b'synthetic-glb')\n"
        "Path(args.output_json).write_text(json.dumps({'status': 'completed'}))\n"
    )
    authority_source = Path(witness.witness_authority_module.__file__).resolve()
    image = sources / "9_img.png"
    image.write_bytes(b"final-consumer-image")

    def asset(source: Path, coordinate: str) -> AttemptAsset:
        return AttemptAsset(
            source=source,
            coordinate=coordinate,
            sha256=_sha256(source),
            size_bytes=source.stat().st_size,
        )

    dinov3_files = {}
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        source = sources / f"dinov3-{name}"
        source.write_bytes(f"dinov3-{name}".encode())
        dinov3_files[name] = asset(source, name)
    monkeypatch.setattr(
        witness,
        "DINOV3_FILES",
        {name: attempt_asset.sha256 for name, attempt_asset in dinov3_files.items()},
    )
    rembg_files = {}
    for name in ("model.safetensors", "config.json", "birefnet.py", "BiRefNet_config.py"):
        source = sources / f"rembg-{name}"
        source.write_bytes(f"rembg-{name}".encode())
        rembg_files[name] = asset(source, f"rembg-{name}")
    monkeypatch.setattr(
        witness,
        "REMBG_FILES",
        {name: attempt_asset.sha256 for name, attempt_asset in rembg_files.items()},
    )

    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/final-consumer-inputs",
        kernel_id="operator/final-consumer-cuda",
        title="Final Consumer CUDA",
        capsule_dir=tmp_path / "attempt-capsule",
        output_dir=tmp_path / "attempt-packet",
        entrypoint=asset(entrypoint, "synthetic-native-entrypoint.py"),
        authority_helper=asset(authority_source, "witness_authority.py"),
        image=asset(image, "9_img.png"),
        dinov3_files=dinov3_files,
        rembg_files=rembg_files,
        expected_outputs=CAPTURE_PROFILE_OUTPUTS["final-consumer"],
        capture_profile="final-consumer",
    )
    return witness.prepare_native_image_to_glb_packet(build_attempt_packet(spec))


def _execute_prepared_runner(
    packet,
    tmp_path,
    monkeypatch,
    *,
    corrupt_contract=False,
    cuda_available=True,
    forbid_input_copy=False,
    forbid_output_copy=False,
    sha256_failures=(),
    staging_failure=None,
    snapshot_failure=False,
    stat_failures=(),
):
    fake_torch = types.SimpleNamespace(
        __version__="2.10.0+cu128",
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda_available,
            get_device_name=lambda _index: "Tesla T4",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    work = tmp_path / "runner-work"
    work.mkdir()
    monkeypatch.chdir(work)
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text().replace(
        'Path("/kaggle/input")',
        f"Path({str(packet.dataset_dir)!r})",
    )
    namespace = {"__name__": "runner_test"}
    exec(runner, namespace)
    if corrupt_contract:
        namespace["CONFIG"]["attempt_contract"]["schema"] = (
            "trellis2mlx.native_image_to_glb_attempt.v2"
        )
        namespace["CONFIG"]["attempt_contract"].pop("capture_profile", None)
    if forbid_input_copy:
        original_copy2 = namespace["shutil"].copy2
        mounted_root = packet.dataset_dir.resolve()

        def reject_mounted_input_copy(source, destination, *args, **kwargs):
            resolved_source = Path(source).resolve()
            if resolved_source == mounted_root or mounted_root in resolved_source.parents:
                raise OSError(
                    errno.ENOSPC,
                    "synthetic mounted-input copy budget exhausted",
                )
            return original_copy2(source, destination, *args, **kwargs)

        monkeypatch.setattr(namespace["shutil"], "copy2", reject_mounted_input_copy)
    if forbid_output_copy:
        original_copy2 = namespace["shutil"].copy2
        execution_output_root = (work / "outputs").resolve()

        def reject_execution_output_copy(source, destination, *args, **kwargs):
            resolved_source = Path(source).resolve()
            if (
                resolved_source == execution_output_root
                or execution_output_root in resolved_source.parents
            ):
                raise OSError(errno.ENOSPC, "synthetic output-copy budget exhausted")
            return original_copy2(source, destination, *args, **kwargs)

        monkeypatch.setattr(namespace["shutil"], "copy2", reject_execution_output_copy)
    if staging_failure is not None:
        operation, relative_name = staging_failure
        destination = (work / relative_name).absolute()
        source = (packet.dataset_dir / relative_name).resolve()
        if operation == "symlink":
            original_symlink_to = namespace["Path"].symlink_to

            def reject_selected_symlink(path, target, *args, **kwargs):
                if path.absolute() == destination:
                    raise OSError(errno.EIO, "synthetic symlink failure")
                return original_symlink_to(path, target, *args, **kwargs)

            monkeypatch.setattr(namespace["Path"], "symlink_to", reject_selected_symlink)
        elif operation == "destination_resolve":
            original_resolve = namespace["Path"].resolve

            def reject_selected_resolve(path, *args, **kwargs):
                if path.absolute() == destination and path.is_symlink():
                    raise OSError(errno.EIO, "synthetic destination resolve failure")
                return original_resolve(path, *args, **kwargs)

            monkeypatch.setattr(namespace["Path"], "resolve", reject_selected_resolve)
        elif operation == "source_stat":
            original_stat = namespace["Path"].stat

            def reject_selected_source_stat(path, *args, **kwargs):
                if path.absolute() == source and destination.is_symlink():
                    raise OSError(errno.EIO, "synthetic source stat failure")
                return original_stat(path, *args, **kwargs)

            monkeypatch.setattr(namespace["Path"], "stat", reject_selected_source_stat)
        else:
            raise AssertionError(f"unsupported staging failure: {operation}")
    if snapshot_failure:
        original_rglob = namespace["Path"].rglob
        mounted_root = packet.dataset_dir.resolve()

        def reject_mounted_snapshot(path, pattern):
            if path.resolve() == mounted_root and (work / packet.inputs[0]).exists():
                raise OSError(errno.EIO, "synthetic mounted snapshot failure")
            return original_rglob(path, pattern)

        monkeypatch.setattr(namespace["Path"], "rglob", reject_mounted_snapshot)
    if sha256_failures:
        original_sha256_file = namespace["sha256_file"]
        rejected_paths = {(work / path).resolve() for path in sha256_failures}

        def reject_selected_digest(path):
            if Path(path).resolve() in rejected_paths:
                raise OSError(errno.EIO, "synthetic output digest failure")
            return original_sha256_file(path)

        namespace["sha256_file"] = reject_selected_digest
    if stat_failures:
        original_stat = namespace["Path"].stat
        original_run = namespace["subprocess"].run
        rejected_paths = {
            str((work / path).absolute()) for path in stat_failures
        }

        def reject_selected_stat(path, *args, **kwargs):
            if str(path.absolute()) in rejected_paths:
                raise OSError(errno.EIO, "synthetic persistent output stat failure")
            return original_stat(path, *args, **kwargs)

        def run_then_reject_selected_stat(*args, **kwargs):
            completed = original_run(*args, **kwargs)
            monkeypatch.setattr(namespace["Path"], "stat", reject_selected_stat)
            return completed

        monkeypatch.setattr(
            namespace["subprocess"],
            "run",
            run_then_reject_selected_stat,
        )
    return namespace["main"](), work


def test_generated_runner_executes_final_consumer_v3_attempt(tmp_path, monkeypatch):
    packet = _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch)

    rc, work = _execute_prepared_runner(packet, tmp_path, monkeypatch)

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    invocation = json.loads((work / "entrypoint-invoked.json").read_text())
    assert rc == 0
    assert invocation == {"capture_profile": "final-consumer"}
    assert receipt["status"] == "done"
    assert receipt["failure_phase"] is None
    assert set(receipt["outputs"]) == set(packet.outputs)
    assert all(record["exists"] for record in receipt["outputs"].values())
    assert not (work / "00-preprocessed_image.png").exists()
    assert not (work / "08-texture_voxels.npz").exists()


def test_generated_runner_stages_immutable_mount_without_copying_inputs(
    tmp_path,
    monkeypatch,
):
    packet = _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch)

    rc, work = _execute_prepared_runner(
        packet,
        tmp_path,
        monkeypatch,
        forbid_input_copy=True,
    )

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert rc == 0
    assert receipt["status"] == "done"
    assert receipt["input_staging"]["mode"] == "immutable-mount-symlink"
    assert receipt["input_staging"]["copied_bytes"] == 0
    assert receipt["input_staging"]["linked_files"] == len(packet.inputs)
    assert all(
        record["staging_method"] == "immutable-mount-symlink"
        for record in receipt["inputs"].values()
    )
    assert all((work / name).is_symlink() for name in packet.inputs)


def test_attempt_runner_receipts_staging_mode_before_staging_initialization(
    tmp_path,
    monkeypatch,
):
    packet = _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch)

    rc, work = _execute_prepared_runner(
        packet,
        tmp_path,
        monkeypatch,
        cuda_available=False,
    )

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert rc == 6
    assert receipt["failure_phase"] == "cuda_route"
    assert receipt["input_staging_mode"] == "immutable-mount-symlink"
    assert "input_staging" not in receipt


@pytest.mark.parametrize(
    "operation",
    ("symlink", "destination_resolve", "source_stat"),
)
def test_generated_runner_receipts_complete_active_input_staging_failure(
    tmp_path,
    monkeypatch,
    operation,
):
    packet = _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch)
    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    staged_names = [
        name for name in manifest["files"] if name != "witness-manifest.json"
    ]
    failed_name = staged_names[1]

    rc, work = _execute_prepared_runner(
        packet,
        tmp_path,
        monkeypatch,
        staging_failure=(operation, failed_name),
    )

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    active = receipt["inputs"][failed_name]
    assert rc == 9
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "input_staging"
    assert receipt["input_staging"]["active_coordinate"] == failed_name
    assert receipt["input_staging"]["linked_files"] == 1
    assert active["status"] == "failed"
    assert active["failure_phase"]
    assert active["source_path"]
    assert active["destination_path"] == failed_name
    assert active["staging_method"] == "immutable-mount-symlink"
    assert active["error"].startswith("OSError:")


def test_generated_runner_preserves_staging_error_when_mount_snapshot_fails(
    tmp_path,
    monkeypatch,
):
    packet = _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch)
    failed_name = packet.inputs[1]

    rc, work = _execute_prepared_runner(
        packet,
        tmp_path,
        monkeypatch,
        staging_failure=("symlink", failed_name),
        snapshot_failure=True,
    )

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert rc == 9
    assert receipt["inputs"][failed_name]["error"].startswith("OSError:")
    assert receipt["mounted_input_snapshot"]["inspection_error"].startswith(
        "OSError:"
    )


def test_generated_runner_publishes_attempt_outputs_without_copying_bundle(
    tmp_path,
    monkeypatch,
):
    packet = _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch)

    rc, work = _execute_prepared_runner(
        packet,
        tmp_path,
        monkeypatch,
        forbid_output_copy=True,
    )

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert rc == 0
    assert receipt["status"] == "done"
    assert receipt["output_publication"]["mode"] == "same-filesystem-replace"
    assert all((work / name).is_file() for name in packet.outputs)
    assert all(not (work / "outputs" / name).exists() for name in packet.outputs)


def test_generated_runner_receipts_source_digest_failure_before_any_move(
    tmp_path,
    monkeypatch,
):
    packet = _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch)
    failed_output = packet.expected_outputs[0]

    rc, work = _execute_prepared_runner(
        packet,
        tmp_path,
        monkeypatch,
        sha256_failures=(f"outputs/{failed_output}",),
    )

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    publication = receipt["output_publication"]
    assert rc == 8
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "output_publication"
    assert set(publication["outputs"]) == set(packet.outputs)
    assert publication["outputs"][failed_output]["failure_phase"] == "source_digest"
    assert all(not (work / name).exists() for name in packet.outputs)
    assert receipt["execution_outputs"][failed_output]["inspection_error"].startswith(
        "OSError:"
    )


def test_generated_runner_receipts_destination_digest_failure_after_report_move(
    tmp_path,
    monkeypatch,
):
    packet = _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch)

    rc, work = _execute_prepared_runner(
        packet,
        tmp_path,
        monkeypatch,
        sha256_failures=("report.json",),
    )

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    publication = receipt["output_publication"]
    report_publication = publication["outputs"]["report.json"]
    assert rc == 8
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "output_publication"
    assert report_publication["moved"] is True
    assert report_publication["destination_verified"] is False
    assert report_publication["failure_phase"] == "destination_digest"
    assert receipt["outputs"]["report.json"]["inspection_error"].startswith(
        "OSError:"
    )
    assert receipt["child_report_fallback"]["exists"] is True
    assert (work / "kaggle_cuda_witness_child_report.json").is_file()


def test_generated_runner_receipts_persistent_source_stat_failure(
    tmp_path,
    monkeypatch,
):
    packet = _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch)
    failed_output = packet.expected_outputs[0]

    rc, work = _execute_prepared_runner(
        packet,
        tmp_path,
        monkeypatch,
        stat_failures=(f"outputs/{failed_output}",),
    )

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    publication = receipt["output_publication"]
    failed_record = publication["outputs"][failed_output]
    assert rc == 8
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "output_publication"
    assert set(publication["outputs"]) == set(packet.outputs)
    assert failed_record["failure_phase"] == "source_inspection"
    assert failed_record["source_inspection_error"].startswith("OSError:")
    assert failed_record["moved"] is False
    assert all(not (work / name).exists() for name in packet.outputs)


def test_generated_runner_receipts_persistent_destination_stat_after_report_move(
    tmp_path,
    monkeypatch,
):
    packet = _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch)

    rc, work = _execute_prepared_runner(
        packet,
        tmp_path,
        monkeypatch,
        stat_failures=("report.json",),
    )

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    report_publication = receipt["output_publication"]["outputs"]["report.json"]
    assert rc == 8
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "output_publication"
    assert report_publication["moved"] is True
    assert report_publication["destination_verified"] is False
    assert report_publication["failure_phase"] == "destination_inspection"
    assert report_publication["destination_inspection_error"].startswith("OSError:")
    assert receipt["child_report_fallback"]["exists"] is True
    assert (work / "kaggle_cuda_witness_child_report.json").is_file()


def test_generated_runner_rejects_final_consumer_contract_mismatch_before_entrypoint(
    tmp_path,
    monkeypatch,
):
    packet = _prepared_final_consumer_attempt_packet(tmp_path, monkeypatch)

    rc, work = _execute_prepared_runner(
        packet,
        tmp_path,
        monkeypatch,
        corrupt_contract=True,
    )

    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert rc == 7
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "attempt_manifest"
    assert not (work / "entrypoint-invoked.json").exists()


def test_native_packet_rejects_final_consumer_outputs_without_profile(
    tmp_path,
    monkeypatch,
):
    from trellmlx.kaggle_cuda_witness import WitnessPacketError

    witness, packet, _coordinates, _expected = _native_packet_contract(
        tmp_path,
        monkeypatch,
    )
    packet = replace(
        packet,
        expected_outputs=tuple(
            witness.EXPECTED_ARTIFACT_FILENAMES[stage]
            for stage in FINAL_CONSUMER_STAGES
        ),
    )

    with pytest.raises(WitnessPacketError, match="outputs do not match capture profile"):
        witness.prepare_native_image_to_glb_packet(packet)


def test_native_packet_rejects_ambiguous_capture_profile_via_packet_error(
    tmp_path,
    monkeypatch,
):
    from trellmlx.kaggle_cuda_witness import WitnessPacketError

    witness, packet, _coordinates, _expected = _native_packet_contract(
        tmp_path,
        monkeypatch,
    )
    packet = replace(
        packet,
        entrypoint_args=(*packet.entrypoint_args, "--capture-profile"),
    )

    with pytest.raises(WitnessPacketError, match="capture profile is ambiguous"):
        witness.prepare_native_image_to_glb_packet(packet)


def _write_clean_git_checkout(root: Path) -> None:
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=root,
        check=True,
    )
    (root / "setup.py").write_text("# pinned source\n")
    subprocess.run(["git", "add", "setup.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=root, check=True)


def _write_minimal_wheel(
    path: Path,
    *,
    distribution: str,
    version: str,
    package_files: dict[str, str],
    requires: tuple[str, ...] = (),
) -> None:
    normalized = distribution.replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {distribution}",
        f"Version: {version}",
        *(f"Requires-Dist: {requirement}" for requirement in requires),
        "",
        "",
    ]
    files = {
        **package_files,
        f"{dist_info}/METADATA": "\n".join(metadata),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: trellis2mlx-conformance-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record_path = f"{dist_info}/RECORD"
    files[record_path] = "".join(f"{name},,\n" for name in (*files, record_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _create_test_venv(root: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(root)],
        check=True,
    )
    return root / "bin" / "python"


def _pip_install(python: Path, wheel: Path, *, force: bool = False) -> None:
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
    ]
    if force:
        command.append("--force-reinstall")
    command.extend(["--no-deps", str(wheel)])
    subprocess.run(command, check=True, capture_output=True, text=True)


def _venv_import_record(python: Path, repo_root: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m, json, xformers, xformers.ops as ops; "
        "print(json.dumps({'marker': xformers.MARKER, 'ops_marker': ops.MARKER, "
        "'dependency_version': m.version('xformers-conformance-dependency')}))"
    )
    completed = subprocess.run(
        [str(python), "-I", "-c", code],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_expected_post_build_products_are_recorded_removed_and_leave_clean_source(
    tmp_path,
):
    from scripts.source_cuda_native_image_to_glb_witness import (
        _remove_expected_build_products,
    )

    root = tmp_path / "nvdiffrast"
    _write_clean_git_checkout(root)
    (root / "build" / "temp.linux-test").mkdir(parents=True)
    (root / "build" / "temp.linux-test" / "extension.o").write_bytes(b"object")
    (root / "nvdiffrast.egg-info").mkdir()
    (root / "nvdiffrast.egg-info" / "PKG-INFO").write_text("Name: nvdiffrast\n")

    removed = _remove_expected_build_products(
        root,
        allowed_roots=("build", "nvdiffrast.egg-info"),
    )

    assert removed == ["build", "nvdiffrast.egg-info"]
    assert not (root / "build").exists()
    assert not (root / "nvdiffrast.egg-info").exists()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""


def test_pinned_xformers_install_forces_same_version_after_digest_verification(tmp_path):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    payload = b"exact-xformers-wheel"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    commands = []

    def download(url, destination):
        assert url == "https://example.invalid/xformers.whl"
        Path(destination).write_bytes(payload)

    def run(command, cwd=None):
        stdout = "pip 25.1 from /test/site-packages/pip (python 3.12)\n" if command[-1] == "--version" else ""
        receipt = {
            "command": command,
            "cwd": cwd,
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
        }
        commands.append(receipt)
        return receipt

    report = {"setup_commands": []}
    record = witness._install_pinned_xformers(
        python="/usr/bin/python3",
        work_dir=tmp_path,
        report=report,
        wheel_url="https://example.invalid/xformers.whl",
        expected_sha256=expected_sha256,
        downloader=download,
        runner=run,
    )

    wheel_path = (
        tmp_path
        / "pinned-wheels"
        / "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
    )
    assert record == {
        "version": "0.0.35",
        "url": "https://example.invalid/xformers.whl",
        "path": str(wheel_path),
        "sha256": expected_sha256,
        "size_bytes": len(payload),
        "install_mode": "forced-local-wheel-no-deps",
        "pip_version": "pip 25.1 from /test/site-packages/pip (python 3.12)",
    }
    assert commands == [
        {
            "command": [
                "/usr/bin/python3",
                "-m",
                "pip",
                "--disable-pip-version-check",
                "--version",
            ],
            "cwd": None,
            "exit_code": 0,
            "stdout": "pip 25.1 from /test/site-packages/pip (python 3.12)\n",
            "stderr": "",
        },
        {
            "command": [
                "/usr/bin/python3",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--force-reinstall",
                "--no-deps",
                str(wheel_path),
            ],
            "cwd": None,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }
    ]
    assert report["setup_commands"] == commands


def test_real_pip_replaces_same_version_and_joins_pep610_provenance(
    tmp_path, monkeypatch
):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    wheel_name = "xformers-0.0.35-py3-none-any.whl"
    dependency_wheel = tmp_path / "wheels" / "dependency" / (
        "xformers_conformance_dependency-1.0-py3-none-any.whl"
    )
    wheel_a = tmp_path / "wheels" / "a" / wheel_name
    wheel_b = tmp_path / "wheels" / "b" / wheel_name
    _write_minimal_wheel(
        dependency_wheel,
        distribution="xformers-conformance-dependency",
        version="1.0",
        package_files={"xformers_conformance_dependency/__init__.py": ""},
    )
    _write_minimal_wheel(
        wheel_a,
        distribution="xformers",
        version="0.0.35",
        package_files={
            "xformers/__init__.py": "__version__ = '0.0.35'\nMARKER = 'wheel-a'\n",
            "xformers/ops/__init__.py": "MARKER = 'ops-a'\n",
        },
        requires=("xformers-conformance-dependency==1.0",),
    )
    _write_minimal_wheel(
        wheel_b,
        distribution="xformers",
        version="0.0.35",
        package_files={
            "xformers/__init__.py": "__version__ = '0.0.35'\nMARKER = 'wheel-b'\n",
            "xformers/ops/__init__.py": "MARKER = 'ops-b'\n",
        },
        requires=("xformers-conformance-dependency==2.0",),
    )

    baseline_python = _create_test_venv(tmp_path / "baseline-venv")
    _pip_install(baseline_python, dependency_wheel)
    _pip_install(baseline_python, wheel_a)
    _pip_install(baseline_python, wheel_b)
    assert _venv_import_record(baseline_python, Path.cwd()) == {
        "marker": "wheel-a",
        "ops_marker": "ops-a",
        "dependency_version": "1.0",
    }

    forced_python = _create_test_venv(tmp_path / "forced-venv")
    _pip_install(forced_python, dependency_wheel)
    _pip_install(forced_python, wheel_a)
    monkeypatch.setattr(witness, "XFORMERS_WHEEL_FILENAME", wheel_name)
    wheel_b_sha256 = _sha256(wheel_b)
    report = {"setup_commands": []}
    record = witness._install_pinned_xformers(
        python=str(forced_python),
        work_dir=tmp_path / "runtime",
        report=report,
        wheel_url=wheel_b.resolve().as_uri(),
        expected_sha256=wheel_b_sha256,
        downloader=lambda _url, destination: shutil.copyfile(wheel_b, destination),
    )

    assert record["pip_version"].startswith("pip ")
    assert report["setup_commands"][0]["command"][-1] == "--version"
    assert "--force-reinstall" in report["setup_commands"][1]["command"]
    assert "--no-deps" in report["setup_commands"][1]["command"]
    assert _venv_import_record(forced_python, Path.cwd()) == {
        "marker": "wheel-b",
        "ops_marker": "ops-b",
        "dependency_version": "1.0",
    }

    repo_root = Path(__file__).resolve().parents[1]
    provenance_code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "from scripts import source_cuda_native_image_to_glb_witness as w; "
        "import xformers, xformers.ops as ops; "
        f"r=w.read_xformers_install_provenance(xformers, ops, wheel_path={record['path']!r}, "
        f"expected_sha256={wheel_b_sha256!r}); "
        "print(json.dumps(r, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(forced_python), "-I", "-c", provenance_code],
        check=True,
        capture_output=True,
        text=True,
    )
    provenance = json.loads(completed.stdout)
    assert provenance["wheel_path"] == record["path"]
    assert provenance["wheel_sha256"] == wheel_b_sha256
    assert provenance["distribution_name"] == "xformers"
    assert provenance["distribution_version"] == "0.0.35"
    assert provenance["distribution_files"] == {
        "xformers": "xformers/__init__.py",
        "xformers.ops": "xformers/ops/__init__.py",
    }
    assert provenance["direct_url"]["url"] == Path(record["path"]).resolve().as_uri()
    assert witness._direct_url_archive_sha256(provenance["direct_url"]) == wheel_b_sha256


def test_pinned_xformers_install_rejects_digest_before_pip(tmp_path):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    called = []

    def download(_url, destination):
        Path(destination).write_bytes(b"substituted-wheel")

    with pytest.raises(RuntimeError, match="xformers wheel digest mismatch"):
        witness._install_pinned_xformers(
            python="/usr/bin/python3",
            work_dir=tmp_path,
            report={"setup_commands": []},
            wheel_url="https://example.invalid/xformers.whl",
            expected_sha256="0" * 64,
            downloader=download,
            runner=lambda *_args, **_kwargs: called.append(True),
        )

    assert called == []


def test_native_image_xformers_provenance_owns_imports_from_exact_wheel(tmp_path):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    wheel_path = tmp_path / witness.XFORMERS_WHEEL_FILENAME
    wheel_path.write_bytes(b"exact-wheel")
    wheel_sha256 = _sha256(wheel_path)
    site_root = tmp_path / "site-packages"
    package = site_root / "xformers"
    ops_package = package / "ops"
    ops_package.mkdir(parents=True)
    xformers_path = package / "__init__.py"
    ops_path = ops_package / "__init__.py"
    xformers_path.write_text("")
    ops_path.write_text("")

    class Distribution:
        metadata = {"Name": "xformers"}
        version = "0.0.35"
        files = [Path("xformers/__init__.py"), Path("xformers/ops/__init__.py")]

        def locate_file(self, relative):
            return site_root / relative

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": wheel_path.resolve().as_uri(),
                    "archive_info": {"hashes": {"sha256": wheel_sha256}},
                }
            )

    record = witness.read_xformers_install_provenance(
        types.SimpleNamespace(
            __name__="xformers", __version__="0.0.35", __file__=str(xformers_path)
        ),
        types.SimpleNamespace(__name__="xformers.ops", __file__=str(ops_path)),
        wheel_path=wheel_path,
        expected_sha256=wheel_sha256,
        distribution_loader=lambda _name: Distribution(),
    )

    assert record["wheel_sha256"] == wheel_sha256
    assert record["module_paths"] == {
        "xformers": str(xformers_path.resolve()),
        "xformers.ops": str(ops_path.resolve()),
    }


@pytest.mark.parametrize("mutation", ["unknown_untracked", "tracked_modified"])
def test_post_build_cleanup_rejects_unattributable_source_changes_without_deleting_them(
    tmp_path, mutation
):
    from scripts.source_cuda_native_image_to_glb_witness import (
        _remove_expected_build_products,
    )

    root = tmp_path / "nvdiffrast"
    _write_clean_git_checkout(root)
    (root / "build").mkdir()
    (root / "build" / "extension.o").write_bytes(b"object")
    if mutation == "unknown_untracked":
        evidence = root / "unexpected.txt"
        evidence.write_text("not a build product\n")
    else:
        evidence = root / "setup.py"
        evidence.write_text("# source mutation\n")

    with pytest.raises(RuntimeError, match="unattributable post-build source changes"):
        _remove_expected_build_products(
            root,
            allowed_roots=("build", "nvdiffrast.egg-info"),
        )

    assert evidence.exists()
    assert (root / "build" / "extension.o").exists()


def test_parser_defaults_bind_the_authorized_native_route():
    from scripts.source_cuda_native_image_to_glb_witness import (
        CUMESH_COMMIT,
        CUMESH_REPOSITORY,
        DINOV3_REVISION,
        EXPECTED_TORCH_VERSION,
        MODEL_REVISION,
        NVDIFFRAST_COMMIT,
        REMBG_REVISION,
        REMBG_FILES,
        SPARSE_DECODER_REVISION,
        TRELLIS_COMMIT,
        TRELLIS_REPOSITORY,
        build_parser,
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "--image",
            "image.png",
            "--expected-image-sha256",
            "a" * 64,
            "--output-dir",
            "outputs",
            "--work-dir",
            "runtime",
        ]
    )

    assert TRELLIS_REPOSITORY == "https://github.com/microsoft/TRELLIS.2.git"
    assert TRELLIS_COMMIT == "5565d240c4a494caaf9ece7a554542b76ffa36d3"
    assert CUMESH_REPOSITORY == "https://github.com/JeffreyXiang/CuMesh.git"
    assert CUMESH_COMMIT == "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87"
    assert EXPECTED_TORCH_VERSION == "2.10.0+cu128"
    assert MODEL_REVISION == "af44b45f2e35a493886929c6d786e563ec68364d"
    assert SPARSE_DECODER_REVISION == "25e0d31ffbebe4b5a97464dd851910efc3002d96"
    assert DINOV3_REVISION == "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    assert REMBG_REVISION == "5df4c9c76d8170882c34f6986e848ee07fd0ba43"
    assert REMBG_FILES == {
        "model.safetensors": "566ed80c3d95f87ada6864d4cbe2290a1c5eb1c7bb0b123e984f60f76b02c3a7",
        "config.json": "c97ea21569daf66b205491a4635147dd3bc42c7c168b89d7d75b53f67ef548ae",
        "birefnet.py": "e499d75224b8819e985e68fb78b7a8e8c99316840474e74e16b5529f03ca2860",
        "BiRefNet_config.py": "e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52",
    }
    assert NVDIFFRAST_COMMIT == "253ac4fcea7de5f396371124af597e6cc957bfae"
    assert args.pipeline_type == "512"
    assert args.seed == 42
    assert args.steps == 8
    assert args.target_faces == 350000
    assert args.texture_size == 1024
    assert args.attention_backend == "xformers"
    assert args.sparse_conv_backend == "flex_gemm"


@pytest.mark.parametrize(
    "provided_flags",
    [
        ("--rembg-model-file",),
        ("--rembg-model-file", "--rembg-config-file"),
        (
            "--rembg-model-file",
            "--rembg-config-file",
            "--rembg-birefnet-file",
        ),
    ],
)
def test_request_rejects_partial_local_rembg_group(tmp_path, provided_flags):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    image = tmp_path / "image.png"
    image.write_bytes(b"authorized-image")
    arguments = _base_args(tmp_path, image) + ["--no-download"]
    for flag in provided_flags:
        arguments.extend([flag, str(tmp_path / flag.removeprefix("--"))])
    args = witness.build_parser().parse_args(arguments)
    args.run_id = "11111111-1111-4111-8111-111111111111"

    with pytest.raises(ValueError, match="complete local RMBG file group"):
        witness._validate_request(args)


def test_request_rejects_substituted_local_rembg_file(tmp_path, monkeypatch):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    image = tmp_path / "image.png"
    image.write_bytes(b"authorized-image")
    rembg_payloads = {
        "model.safetensors": b"model",
        "config.json": b"config",
        "birefnet.py": b"implementation",
        "BiRefNet_config.py": b"configuration",
    }
    rembg_paths = {}
    for name, payload in rembg_payloads.items():
        path = tmp_path / name
        path.write_bytes(payload)
        rembg_paths[name] = path
    expected = {name: _sha256(path) for name, path in rembg_paths.items()}
    expected["birefnet.py"] = "0" * 64
    monkeypatch.setattr(witness, "REMBG_FILES", expected)
    arguments = _base_args(tmp_path, image) + ["--no-download"]
    for name, attribute in witness.REMBG_FILE_ARGUMENTS.items():
        arguments.extend(
            [f"--{attribute.replace('_', '-')}", str(rembg_paths[name])]
        )
    args = witness.build_parser().parse_args(arguments)
    args.run_id = "11111111-1111-4111-8111-111111111111"

    with pytest.raises(ValueError, match="RMBG birefnet.py SHA256 mismatch"):
        witness._validate_request(args)


def test_failed_rembg_admission_removes_partial_run_custody(tmp_path):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    image = tmp_path / "image.png"
    image.write_bytes(b"image-authority")
    dino = tmp_path / "dino"
    dino.mkdir()
    (dino / "config.json").write_bytes(b"dino")
    rembg = tmp_path / "rembg"
    rembg.mkdir()
    for name in witness.REMBG_FILES:
        (rembg / name).write_bytes(name.encode())
    args = types.SimpleNamespace(
        image=image,
        expected_image_sha256=_sha256(image),
        dinov3_model_path=dino,
        **{
            attribute: rembg / name
            for name, attribute in witness.REMBG_FILE_ARGUMENTS.items()
        },
        work_dir=tmp_path / "runtime",
        run_id="11111111-1111-4111-8111-111111111111",
    )

    with pytest.raises(RuntimeError, match="authority changed while admitting"):
        witness.admit_run_inputs(
            args,
            {},
            expected_dinov3_files={"config.json": _sha256(dino / "config.json")},
            expected_rembg_files={
                name: (_sha256(rembg / name) if name != "birefnet.py" else "0" * 64)
                for name in witness.REMBG_FILES
            },
        )

    assert not (
        tmp_path
        / "runtime"
        / "admitted-inputs"
        / "11111111-1111-4111-8111-111111111111"
    ).exists()


def test_wrong_image_digest_fails_before_touching_existing_output(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import main

    image = tmp_path / "image.png"
    image.write_bytes(b"not-the-authorized-image")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    stale = output_dir / "consumer.glb"
    stale.write_bytes(b"preserve-until-request-validates")

    args = _base_args(tmp_path, image)
    args[args.index("--expected-image-sha256") + 1] = "0" * 64
    rc = main(args + ["--no-download"])

    report = json.loads((output_dir / "report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "arguments_parsed"
    assert report["primary_output_status"] == "not_attempted"
    assert "image SHA256" in report["error"]
    assert stale.read_bytes() == b"preserve-until-request-validates"


def test_no_download_preflight_records_exact_route_without_primary_artifacts(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import main

    image = tmp_path / "image.png"
    image.write_bytes(b"authorized-image-fixture")
    args = _base_args(tmp_path, image)

    rc = main(args + ["--no-download"])

    output_dir = tmp_path / "outputs"
    report = json.loads((output_dir / "report.json").read_text())
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["failure_phase"] is None
    assert report["last_trustworthy_phase"] == "request_validated"
    assert report["primary_output_status"] == "not_written_no_download"
    assert report["effective_route"]["device_type"] == "not_loaded_no_download"
    assert report["effective_route"]["trellis_source_clean"] == "not_checked_no_download"
    assert report["requested_route"]["native_conditioning"] is True
    assert report["requested_route"]["native_rng"] is True
    assert report["requested_route"]["pipeline_run_called_once"] is True
    assert report["requested_route"]["sampler_steps"] == {
        "shape": 8,
        "sparse": 8,
        "texture": 8,
    }
    assert report["requested_route"]["postprocess"] == {
        "decimation_target": 350000,
        "remesh": False,
        "texture_size": 1024,
    }
    assert report["expected_capture_order"] == list(EXPECTED_STAGES)
    assert list(output_dir.iterdir()) == [output_dir / "report.json"]


def test_no_download_final_consumer_profile_records_bounded_capture_contract(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import main

    image = tmp_path / "image.png"
    image.write_bytes(b"authorized-image-fixture")

    rc = main(
        _base_args(tmp_path, image)
        + ["--capture-profile", "final-consumer", "--no-download"]
    )

    report = json.loads((tmp_path / "outputs" / "report.json").read_text())
    assert rc == 0
    assert report["capture_profile"] == "final-consumer"
    assert report["expected_capture_order"] == list(FINAL_CONSUMER_STAGES)
    assert report["capture_order"] == []


def test_final_consumer_recorder_preserves_global_stage_coordinates(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import ArtifactRecorder

    recorder = ArtifactRecorder(
        tmp_path / "outputs",
        expected_capture_order=FINAL_CONSUMER_STAGES,
    )
    path = recorder.save_npz(
        "decoder_raw_mesh",
        {
            "vertices": np.zeros((3, 3), dtype=np.float32),
            "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
        },
    )

    assert path.name == "07-decoder_raw_mesh.npz"
    assert recorder.capture_order == ["decoder_raw_mesh"]


def test_recorder_publishes_each_boundary_to_progress_report(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import ArtifactRecorder

    observations = []

    def publish(recorder):
        observations.append(
            {
                "capture_order": list(recorder.capture_order),
                "artifacts": dict(recorder.artifacts),
            }
        )

    recorder = ArtifactRecorder(tmp_path / "outputs", on_capture=publish)
    recorder.save_image(
        "preprocessed_image",
        type(
            "FakeImage",
            (),
            {
                "mode": "RGBA",
                "size": (2, 2),
                "save": lambda self, path, format: Path(path).write_bytes(b"png"),
            },
        )(),
    )

    assert observations == [
        {
            "capture_order": ["preprocessed_image"],
            "artifacts": {
                "preprocessed_image": recorder.artifacts["preprocessed_image"]
            },
        }
    ]


def test_orientation_observer_does_not_mutate_native_extension_class():
    from scripts.source_cuda_native_image_to_glb_witness import (
        _restore_orientation_observer,
        install_orientation_observer,
    )

    class ImmutableNativeType(type):
        def __setattr__(cls, name, value):
            raise TypeError("immutable extension type")

    class NativeCuMesh(metaclass=ImmutableNativeType):
        def read(self):
            return [1, 2, 3], [[0, 1, 2]]

        def unify_face_orientations(self):
            return "native-return"

    class Recorder:
        def __init__(self):
            self.stages = []

        def save_npz(self, stage, arrays):
            self.stages.append((stage, arrays))

    module = types.SimpleNamespace(CuMesh=NativeCuMesh)
    recorder = Recorder()

    observer = install_orientation_observer(module, recorder)
    observed = module.CuMesh()
    result = observed.unify_face_orientations()
    _restore_orientation_observer(observer)

    assert result == "native-return"
    assert module.CuMesh is NativeCuMesh
    assert [stage for stage, _ in recorder.stages] == [
        "postprocess_stage11_pre_orientation",
        "postprocess_stage12_post_orientation",
    ]
    assert observer["state"] == {
        "call_count": 1,
        "native_method_return_preserved": True,
        "pre_readback_written": True,
        "post_readback_written": True,
    }


def test_actual_kaggle_runner_reaches_witness_request_validation(tmp_path, monkeypatch):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket
    from scripts import source_cuda_native_image_to_glb_witness as witness

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    authority_helper = _stage_authority_helper(capsule)
    image = capsule / "9_img.png"
    image.write_bytes(b"synthetic-transport-contract-image")
    rembg_coordinates, rembg_arguments, expected_rembg = _stage_packet_rembg(
        capsule, witness, monkeypatch
    )
    entrypoint.write_text(
        entrypoint.read_text().replace(
            "REMBG_FILE_ARGUMENTS = {",
            f"REMBG_FILES = {expected_rembg!r}\nREMBG_FILE_ARGUMENTS = {{",
            1,
        )
    )
    run_id = "22222222-2222-4222-8222-222222222222"
    image_sha256 = _sha256(image)
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    packet = witness.prepare_native_image_to_glb_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/native-image-anchor-inputs",
            kernel_id="operator/native-image-anchor-cuda",
            title="Native Image Anchor CUDA",
            entrypoint=entrypoint.name,
            inputs=(entrypoint.name, authority_helper, image.name, *rembg_coordinates.values()),
            run_id=run_id,
            expected_image_sha256=image_sha256,
            output_json="report.json",
            output_npz=None,
            expected_outputs=expected_outputs,
            entrypoint_args=(
                "--image",
                image.name,
                "--expected-image-sha256",
                image_sha256,
                "--run-id",
                run_id,
                "--output-dir",
                ".",
                "--work-dir",
                str(tmp_path / "runtime"),
                *rembg_arguments,
                "--no-download",
            ),
        )
    )
    fake_torch = types.SimpleNamespace(
        __version__="2.10.0+cu128",
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Tesla T4",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text().replace(
        'Path("/kaggle/input")',
        f"Path({str(packet.dataset_dir)!r})",
    )
    namespace = {"__name__": "runner_test"}
    exec(runner, namespace)

    rc = namespace["main"]()

    report = json.loads((work / "report.json").read_text())
    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert rc != 0
    assert report["status"] == "preflight_stopped"
    assert report["last_trustworthy_phase"] == "request_validated"
    assert receipt["failure_phase"] == "output"
    assert receipt["run_id"] == run_id
    assert receipt["expected_image_sha256"] == image_sha256
    assert receipt["effective_command"][:3] == [
        sys.executable,
        entrypoint.name,
        "--output-json",
    ]
    assert set(packet.expected_outputs).issuperset(expected_outputs)
    assert receipt["effective_command"].count("--run-id") == 1
    assert receipt["effective_command"][receipt["effective_command"].index("--run-id") + 1] == run_id


def test_native_packet_rejects_missing_attempt_identities_before_output_mutation(tmp_path):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        prepare_packet,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    image = capsule / "9_img.png"
    image.write_bytes(b"native-packet-input")
    output_dir = tmp_path / "packet"
    output_dir.mkdir()
    marker = output_dir / "must-survive-identity-rejection.txt"
    marker.write_text("preserved")
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    packet = KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=output_dir,
        dataset_id="operator/native-image-anchor-inputs",
        kernel_id="operator/native-image-anchor-cuda",
        title="Native Image Anchor CUDA",
        entrypoint=entrypoint.name,
        inputs=(entrypoint.name, image.name),
        output_json="report.json",
        output_npz=None,
        expected_outputs=expected_outputs,
        entrypoint_args=(
            "--image",
            image.name,
            "--output-dir",
            ".",
            "--work-dir",
            str(tmp_path / "runtime"),
        ),
    )
    # On the reviewed parent there is no native constructor, so this exercises
    # the actual generic route and records its mutation as the fail-first witness.
    prepare_native = getattr(
        witness,
        "prepare_native_image_to_glb_packet",
        prepare_packet,
    )

    with pytest.raises(WitnessPacketError, match="run identity.*image identity"):
        prepare_native(packet)

    assert marker.read_text() == "preserved"


def test_native_packet_rejects_missing_authority_helper_before_output_mutation(
    tmp_path,
    monkeypatch,
):
    from trellmlx.kaggle_cuda_witness import WitnessPacketError

    witness, packet, _coordinates, _expected = _native_packet_contract(
        tmp_path, monkeypatch
    )
    packet = replace(
        packet,
        inputs=tuple(
            coordinate
            for coordinate in packet.inputs
            if coordinate != "witness_authority.py"
        ),
    )
    packet.output_dir.mkdir()
    marker = packet.output_dir / "must-survive-helper-rejection.txt"
    marker.write_text("preserved")

    with pytest.raises(WitnessPacketError, match="authority helper"):
        witness.prepare_native_image_to_glb_packet(packet)

    assert marker.read_text() == "preserved"


@pytest.mark.parametrize(
    ("declared_output_dir", "declared_work_dir", "assignment_form"),
    (
        (".", "/kaggle/working/native-pixal9-runtime", False),
        ("//kaggle/working", "/kaggle/working/native-pixal9-runtime", False),
        ("/kaggle/working", "//kaggle/working/native-pixal9-runtime", False),
        ("outputs", "outputs/runtime", True),
        ("same", "same", False),
        ("runtime/output", "runtime", False),
    ),
)
def test_native_packet_rejects_remote_output_work_overlap_before_output_mutation(
    tmp_path,
    declared_output_dir,
    declared_work_dir,
    assignment_form,
    monkeypatch,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    authority_helper = _stage_authority_helper(capsule)
    image = capsule / "9_img.png"
    image.write_bytes(b"native-packet-input")
    rembg_coordinates, rembg_arguments, _expected_rembg = _stage_packet_rembg(
        capsule, witness, monkeypatch
    )
    image_sha256 = _sha256(image)
    output_dir = tmp_path / "packet"
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    path_arguments = (
        (
            f"--output-dir={declared_output_dir}",
            f"--work-dir={declared_work_dir}",
        )
        if assignment_form
        else (
            "--output-dir",
            declared_output_dir,
            "--work-dir",
            declared_work_dir,
        )
    )
    packet = KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=output_dir,
        dataset_id="operator/native-image-anchor-inputs",
        kernel_id="operator/native-image-anchor-cuda",
        title="Native Image Anchor CUDA",
        entrypoint=entrypoint.name,
        inputs=(entrypoint.name, authority_helper, image.name, *rembg_coordinates.values()),
        output_json="report.json",
        output_npz=None,
        expected_outputs=expected_outputs,
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        expected_image_sha256=image_sha256,
        entrypoint_args=(
            "--image",
            image.name,
            "--expected-image-sha256",
            image_sha256,
            "--run-id",
            "31fce6b7-853b-4a0f-b99d-518be23ebabc",
            *path_arguments,
            *rembg_arguments,
        ),
    )

    with pytest.raises(WitnessPacketError, match="Kaggle output and work directories overlap"):
        witness.prepare_native_image_to_glb_packet(packet)

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("declared_output_dir", "declared_work_dir"),
    (("outputs", "runtime"), ("outputs", "outputs-archive"), ("-1", "runtime")),
)
def test_native_packet_accepts_disjoint_assignment_form_paths(
    tmp_path,
    declared_output_dir,
    declared_work_dir,
    monkeypatch,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    authority_helper = _stage_authority_helper(capsule)
    image = capsule / "9_img.png"
    image.write_bytes(b"native-packet-input")
    rembg_coordinates, rembg_arguments, _expected_rembg = _stage_packet_rembg(
        capsule, witness, monkeypatch
    )
    image_sha256 = _sha256(image)
    output_dir = tmp_path / "packet"
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    packet = KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=output_dir,
        dataset_id="operator/native-image-anchor-inputs",
        kernel_id="operator/native-image-anchor-cuda",
        title="Native Image Anchor CUDA",
        entrypoint=entrypoint.name,
        inputs=(entrypoint.name, authority_helper, image.name, *rembg_coordinates.values()),
        output_json="report.json",
        output_npz=None,
        expected_outputs=expected_outputs,
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        expected_image_sha256=image_sha256,
        entrypoint_args=(
            "--image",
            image.name,
            "--expected-image-sha256",
            image_sha256,
            "--run-id",
            "31fce6b7-853b-4a0f-b99d-518be23ebabc",
            f"--output-dir={declared_output_dir}",
            f"--work-dir={declared_work_dir}",
            *rembg_arguments,
        ),
    )

    assert witness.prepare_native_image_to_glb_packet(packet) is packet
    assert output_dir.is_dir()


@pytest.mark.parametrize(
    "path_arguments",
    (
        ("--output-dir", "--work-dir", "runtime"),
        ("--output-dir", "-x", "--work-dir", "runtime"),
        ("--", "--output-dir", "outputs", "--work-dir", "runtime"),
        (
            "--output-dir",
            "outputs",
            "--output-dir=other",
            "--work-dir",
            "runtime",
        ),
    ),
)
def test_native_packet_rejects_malformed_path_arguments_before_output_mutation(
    tmp_path,
    path_arguments,
    monkeypatch,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    authority_helper = _stage_authority_helper(capsule)
    image = capsule / "9_img.png"
    image.write_bytes(b"native-packet-input")
    rembg_coordinates, rembg_arguments, _expected_rembg = _stage_packet_rembg(
        capsule, witness, monkeypatch
    )
    image_sha256 = _sha256(image)
    output_dir = tmp_path / "packet"
    output_dir.mkdir()
    marker = output_dir / "must-survive-argument-rejection.txt"
    marker.write_text("preserved")
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    packet = KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=output_dir,
        dataset_id="operator/native-image-anchor-inputs",
        kernel_id="operator/native-image-anchor-cuda",
        title="Native Image Anchor CUDA",
        entrypoint=entrypoint.name,
        inputs=(entrypoint.name, authority_helper, image.name, *rembg_coordinates.values()),
        output_json="report.json",
        output_npz=None,
        expected_outputs=expected_outputs,
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        expected_image_sha256=image_sha256,
        entrypoint_args=(
            "--image",
            image.name,
            "--expected-image-sha256",
            image_sha256,
            "--run-id",
            "31fce6b7-853b-4a0f-b99d-518be23ebabc",
            *path_arguments,
            *rembg_arguments,
        ),
    )

    with pytest.raises(WitnessPacketError):
        witness.prepare_native_image_to_glb_packet(packet)

    assert marker.read_text() == "preserved"


@pytest.mark.parametrize(
    "mutation",
    (
        "all_omitted",
        "partial_1",
        "partial_2",
        "partial_3",
        "duplicate_flag",
        "duplicate_coordinate",
        "coordinate_absent_from_inputs",
        "nested_coordinate",
        "escaping_coordinate",
        "missing_source",
        "wrong_digest:model.safetensors",
        "wrong_digest:config.json",
        "wrong_digest:birefnet.py",
        "wrong_digest:BiRefNet_config.py",
    ),
)
def test_native_packet_rejects_invalid_rembg_custody_before_output_mutation(
    tmp_path,
    monkeypatch,
    mutation,
):
    from trellmlx.kaggle_cuda_witness import WitnessPacketError

    witness, packet, coordinates, _expected = _native_packet_contract(
        tmp_path, monkeypatch
    )
    arguments = list(packet.entrypoint_args)
    inputs = list(packet.inputs)
    rembg_pairs = [
        (f"--{attribute.replace('_', '-')}", coordinates[role])
        for role, attribute in witness.REMBG_FILE_ARGUMENTS.items()
    ]

    if mutation == "all_omitted":
        for flag, coordinate in rembg_pairs:
            arguments[arguments.index(flag) : arguments.index(flag) + 2] = []
    elif mutation.startswith("partial_"):
        keep = int(mutation.rsplit("_", 1)[1])
        for flag, coordinate in rembg_pairs[keep:]:
            arguments[arguments.index(flag) : arguments.index(flag) + 2] = []
    elif mutation == "duplicate_flag":
        arguments.extend(rembg_pairs[0])
    elif mutation == "duplicate_coordinate":
        second_flag, _second_coordinate = rembg_pairs[1]
        arguments[arguments.index(second_flag) + 1] = rembg_pairs[0][1]
    elif mutation == "coordinate_absent_from_inputs":
        inputs.remove(rembg_pairs[0][1])
    elif mutation == "nested_coordinate":
        flag, coordinate = rembg_pairs[0]
        nested = f"nested/{coordinate}"
        nested_path = packet.capsule_dir / nested
        nested_path.parent.mkdir()
        nested_path.write_bytes((packet.capsule_dir / coordinate).read_bytes())
        arguments[arguments.index(flag) + 1] = nested
        inputs[inputs.index(coordinate)] = nested
    elif mutation == "escaping_coordinate":
        flag, coordinate = rembg_pairs[0]
        escaping = f"../{coordinate}"
        (packet.capsule_dir.parent / coordinate).write_bytes(
            (packet.capsule_dir / coordinate).read_bytes()
        )
        arguments[arguments.index(flag) + 1] = escaping
        inputs[inputs.index(coordinate)] = escaping
    elif mutation == "missing_source":
        (packet.capsule_dir / rembg_pairs[0][1]).unlink()
    elif mutation.startswith("wrong_digest:"):
        role = mutation.split(":", 1)[1]
        (packet.capsule_dir / coordinates[role]).write_bytes(b"substituted")
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    packet = replace(
        packet,
        entrypoint_args=tuple(arguments),
        inputs=tuple(inputs),
    )
    packet.output_dir.mkdir()
    marker = packet.output_dir / "must-survive-rembg-rejection.txt"
    marker.write_text("preserved")

    with pytest.raises(WitnessPacketError, match="RMBG|rembg"):
        witness.prepare_native_image_to_glb_packet(packet)

    assert marker.read_text() == "preserved"


def test_native_packet_manifest_binds_exact_rembg_inputs_and_runner_coordinates(
    tmp_path,
    monkeypatch,
):
    witness, packet, coordinates, expected = _native_packet_contract(
        tmp_path, monkeypatch
    )

    admitted = witness.prepare_native_image_to_glb_packet(packet)

    manifest = json.loads((admitted.dataset_dir / "witness-manifest.json").read_text())
    runner = (admitted.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    config_match = re.search(r"^CONFIG = json\.loads\((.+)\)$", runner, re.MULTILINE)
    assert config_match is not None
    runner_config = json.loads(ast.literal_eval(config_match.group(1)))
    for role, attribute in witness.REMBG_FILE_ARGUMENTS.items():
        coordinate = coordinates[role]
        assert manifest["files"][coordinate]["sha256"] == expected[role]
        assert manifest["entrypoint_args"].count(
            f"--{attribute.replace('_', '-')}"
        ) == 1
        assert manifest["entrypoint_args"].count(coordinate) == 1
        assert runner_config["entrypoint_args"].count(coordinate) == 1


def test_native_packet_rejects_rembg_substitution_between_precheck_and_copy(
    tmp_path,
    monkeypatch,
):
    from trellmlx import kaggle_cuda_witness
    from trellmlx.kaggle_cuda_witness import WitnessPacketError

    witness, packet, coordinates, _expected = _native_packet_contract(
        tmp_path, monkeypatch
    )
    real_prepare = kaggle_cuda_witness.prepare_packet

    def substitute_then_prepare(candidate):
        (candidate.capsule_dir / coordinates["config.json"]).write_bytes(
            b"post-precheck-substitution"
        )
        return real_prepare(candidate)

    monkeypatch.setattr(kaggle_cuda_witness, "prepare_packet", substitute_then_prepare)
    packet.output_dir.mkdir()
    marker = packet.output_dir / "must-survive-copy-race.txt"
    marker.write_text("preserved")

    with pytest.raises(WitnessPacketError, match="RMBG.*SHA256|RMBG.*digest"):
        witness.prepare_native_image_to_glb_packet(packet)

    assert marker.read_text() == "preserved"


@pytest.mark.parametrize(
    "mutated_role",
    (
        "image",
        "dinov3:model.safetensors",
        "dinov3:config.json",
        "dinov3:preprocessor_config.json",
        "rembg:model.safetensors",
        "rembg:config.json",
        "rembg:birefnet.py",
        "rembg:BiRefNet_config.py",
    ),
)
def test_admitted_inputs_reject_requested_path_substitution_before_use(
    tmp_path,
    mutated_role,
):
    from scripts.source_cuda_native_image_to_glb_witness import (
        admit_run_inputs,
        verify_admitted_inputs_before_use,
    )

    image = tmp_path / "image.png"
    image.write_bytes(b"image-authority")
    dino = tmp_path / "dino"
    dino.mkdir()
    dino_files = {
        "model.safetensors": b"model-authority",
        "config.json": b"config-authority",
        "preprocessor_config.json": b"preprocessor-authority",
    }
    for name, payload in dino_files.items():
        (dino / name).write_bytes(payload)
    rembg = tmp_path / "rembg"
    rembg.mkdir()
    rembg_files = {
        "model.safetensors": b"rembg-model-authority",
        "config.json": b"rembg-config-authority",
        "birefnet.py": b"rembg-code-authority",
        "BiRefNet_config.py": b"rembg-config-code-authority",
    }
    rembg_args = {
        "model.safetensors": "rembg_model_file",
        "config.json": "rembg_config_file",
        "birefnet.py": "rembg_birefnet_file",
        "BiRefNet_config.py": "rembg_birefnet_config_file",
    }
    for name, payload in rembg_files.items():
        (rembg / name).write_bytes(payload)
    args = types.SimpleNamespace(
        image=image,
        expected_image_sha256=_sha256(image),
        dinov3_model_path=dino,
        **{attribute: rembg / name for name, attribute in rembg_args.items()},
        work_dir=tmp_path / "runtime",
        run_id="11111111-1111-4111-8111-111111111111",
    )
    report = {}
    admitted = admit_run_inputs(
        args,
        report,
        expected_dinov3_files={name: _sha256(dino / name) for name in dino_files},
        expected_rembg_files={name: _sha256(rembg / name) for name in rembg_files},
    )
    if mutated_role == "image":
        requested = image
    else:
        family, name = mutated_role.split(":", 1)
        requested = (dino if family == "dinov3" else rembg) / name
    requested.write_bytes(b"substituted-after-admission")

    with pytest.raises(RuntimeError, match=mutated_role.split(":")[-1]):
        verify_admitted_inputs_before_use(
            admitted,
            report,
            expected_dinov3_files={name: _sha256(admitted.dinov3_model_path / name) for name in dino_files},
            expected_rembg_files={name: _sha256(admitted.rembg_model_path / name) for name in rembg_files},
        )

    assert Path(admitted.image).read_bytes() == b"image-authority"
    for name, payload in dino_files.items():
        assert (Path(admitted.dinov3_model_path) / name).read_bytes() == payload
    for name, payload in rembg_files.items():
        assert (Path(admitted.rembg_model_path) / name).read_bytes() == payload


def test_prepare_model_view_uses_admitted_rembg_without_remote_fetch(tmp_path, monkeypatch):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    trellis = tmp_path / "trellis"
    trellis.mkdir()
    pipeline = {
        "args": {
            "models": {
                "shape": "ckpts/shape",
                "decoder": f"{witness.SPARSE_DECODER_REPOSITORY}/decoder",
            },
            "image_cond_model": {"args": {"model_name": "remote-dino"}},
            "rembg_model": {"args": {"model_name": "remote-rembg"}},
        }
    }
    pipeline_path = trellis / "pipeline.json"
    pipeline_path.write_text(json.dumps(pipeline, sort_keys=True))
    decoder = tmp_path / "decoder"
    decoder.mkdir()
    dino = tmp_path / "admitted-dino"
    dino.mkdir()
    rembg = tmp_path / "admitted-rembg"
    rembg.mkdir()
    calls = []

    def snapshot_download(*, repo_id, revision, cache_dir, allow_patterns):
        calls.append((repo_id, revision, Path(cache_dir), tuple(allow_patterns)))
        if repo_id == witness.MODEL_REPOSITORY:
            return trellis
        if repo_id == witness.SPARSE_DECODER_REPOSITORY:
            return decoder
        raise AssertionError(f"unexpected remote model fetch: {repo_id}")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setattr(witness, "MODEL_PIPELINE_SHA256", _sha256(pipeline_path))
    monkeypatch.setattr(witness, "verify_admitted_inputs_before_use", lambda *_args: None)
    monkeypatch.setattr(
        witness.shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(free=10**12),
    )
    args = types.SimpleNamespace(
        work_dir=tmp_path / "runtime",
        dinov3_model_path=dino,
        rembg_model_path=rembg,
    )
    report = {}

    model_view = witness.prepare_model_view(args, report)

    rewritten = json.loads((model_view / "pipeline.json").read_text())
    assert rewritten["args"]["rembg_model"]["args"]["model_name"] == str(rembg.resolve())
    assert report["model_assets"]["rembg"]["snapshot_path"] == str(rembg.resolve())
    assert calls == [
        (
            witness.MODEL_REPOSITORY,
            witness.MODEL_REVISION,
            tmp_path / "runtime" / "huggingface",
            witness._selective_model_patterns("trellis"),
        ),
        (
            witness.SPARSE_DECODER_REPOSITORY,
            witness.SPARSE_DECODER_REVISION,
            tmp_path / "runtime" / "huggingface",
            witness._selective_model_patterns("sparse_decoder"),
        ),
    ]


def test_prepare_model_view_uses_verified_mounted_blobs_without_remote_fetch(
    tmp_path,
    monkeypatch,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    blob_root = tmp_path / "mounted-kernel-output" / "runtime" / "huggingface"
    pipeline = {
        "args": {
            "models": {
                "shape": "ckpts/shape",
                "decoder": f"{witness.SPARSE_DECODER_REPOSITORY}/decoder",
            },
            "image_cond_model": {"args": {"model_name": "remote-dino"}},
            "rembg_model": {"args": {"model_name": "remote-rembg"}},
        }
    }
    payloads = {
        "trellis": {
            "pipeline.json": json.dumps(pipeline, sort_keys=True).encode(),
            "ckpts/shape.json": b"shape-config",
            "ckpts/shape.safetensors": b"shape-weights",
        },
        "sparse_decoder": {
            "decoder.json": b"decoder-config",
            "decoder.safetensors": b"decoder-weights",
        },
    }
    cache_dirs = {
        "trellis": "models--microsoft--TRELLIS.2-4B",
        "sparse_decoder": "models--microsoft--TRELLIS-image-large",
    }
    manifest = {}
    for family, files in payloads.items():
        records = {}
        blobs = blob_root / cache_dirs[family] / "blobs"
        blobs.mkdir(parents=True)
        for index, (coordinate, payload) in enumerate(files.items()):
            blob = f"{family}-blob-{index}"
            path = blobs / blob
            path.write_bytes(payload)
            records[coordinate] = {
                "blob": blob,
                "sha256": _sha256(path),
                "size_bytes": len(payload),
            }
        manifest[family] = {
            "cache_dir": cache_dirs[family],
            "files": records,
        }

    def unexpected_snapshot_download(**_kwargs):
        raise AssertionError("mounted model admission must not fetch remotely")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=unexpected_snapshot_download),
    )
    monkeypatch.setattr(witness, "MODEL_BLOB_MANIFEST", manifest)
    monkeypatch.setattr(
        witness,
        "MODEL_PIPELINE_SHA256",
        manifest["trellis"]["files"]["pipeline.json"]["sha256"],
    )
    monkeypatch.setattr(witness, "verify_admitted_inputs_before_use", lambda *_args: None)
    dino = tmp_path / "admitted-dino"
    rembg = tmp_path / "admitted-rembg"
    dino.mkdir()
    rembg.mkdir()
    args = types.SimpleNamespace(
        work_dir=tmp_path / "runtime",
        dinov3_model_path=dino,
        rembg_model_path=rembg,
        model_blob_root=blob_root,
        model_source_kernel="operator/pinned-model-output",
    )
    report = {}

    model_view = witness.prepare_model_view(args, report)

    rewritten = json.loads((model_view / "pipeline.json").read_text())
    shape = Path(rewritten["args"]["models"]["shape"])
    decoder = Path(rewritten["args"]["models"]["decoder"])
    assert shape.with_suffix(".json").is_symlink()
    assert shape.with_suffix(".safetensors").is_symlink()
    assert decoder.with_suffix(".json").is_symlink()
    assert decoder.with_suffix(".safetensors").is_symlink()
    assert report["model_assets"]["source_mode"] == "mounted-kernel-output"
    assert report["model_assets"]["source_kernel"] == "operator/pinned-model-output"
    assert report["model_assets"]["mounted_blob_bytes"] == sum(
        len(payload)
        for files in payloads.values()
        for payload in files.values()
    )


def test_prepare_model_view_rejects_corrupt_mounted_blob_without_remote_fallback(
    tmp_path,
    monkeypatch,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    blob_root = tmp_path / "mounted" / "runtime" / "huggingface"
    blobs = blob_root / "models--microsoft--TRELLIS.2-4B" / "blobs"
    blobs.mkdir(parents=True)
    pipeline_blob = blobs / "pipeline-blob"
    pipeline_blob.write_text('{"args": {}}')
    expected = hashlib.sha256(b"different-authority").hexdigest()
    monkeypatch.setattr(
        witness,
        "MODEL_BLOB_MANIFEST",
        {
            "trellis": {
                "cache_dir": "models--microsoft--TRELLIS.2-4B",
                "files": {
                    "pipeline.json": {
                        "blob": "pipeline-blob",
                        "sha256": expected,
                        "size_bytes": pipeline_blob.stat().st_size,
                    }
                },
            },
            "sparse_decoder": {
                "cache_dir": "models--microsoft--TRELLIS-image-large",
                "files": {},
            },
        },
    )
    monkeypatch.setattr(witness, "verify_admitted_inputs_before_use", lambda *_args: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            snapshot_download=lambda **_kwargs: pytest.fail("remote fallback occurred")
        ),
    )
    dino = tmp_path / "dino"
    rembg = tmp_path / "rembg"
    dino.mkdir()
    rembg.mkdir()
    args = types.SimpleNamespace(
        work_dir=tmp_path / "runtime",
        dinov3_model_path=dino,
        rembg_model_path=rembg,
        model_blob_root=blob_root,
        model_source_kernel="operator/pinned-model-output",
    )

    with pytest.raises(RuntimeError, match="mounted model blob.*pipeline.json"):
        witness.prepare_model_view(args, {})


@pytest.mark.parametrize(
    "mutation",
    (
        None,
        "wrong_kernel",
        "missing_assets_mode",
        "missing_assets_root",
        "missing_assets_kernel",
        "missing_storage_mode",
        "missing_storage_root",
        "missing_storage_kernel",
        "wrong_blob",
        "writable_payload",
        "low_reserve",
    ),
)
def test_completed_report_requires_full_mounted_model_authority(tmp_path, mutation):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    source_kernel = "operator/pinned-model-output"
    source_root = "/kaggle/input/pinned-model-output/runtime/huggingface"
    file_records = {
        family: {
            coordinate: {
                **expected,
                "source_path": (
                    f"{source_root}/{family_manifest['cache_dir']}/blobs/"
                    f"{expected['blob']}"
                ),
                "linked_path": f"/kaggle/working/runtime/{family}/{coordinate}",
            }
            for coordinate, expected in family_manifest["files"].items()
        }
        for family, family_manifest in witness.MODEL_BLOB_MANIFEST.items()
    }
    report["requested_route"].update(
        model_blob_root=source_root,
        model_source_kernel=source_kernel,
    )
    report["model_assets"].update(
        source_mode="mounted-kernel-output",
        source_kernel=source_kernel,
        source_root=source_root,
        mounted_blob_bytes=witness.MODEL_REQUIRED_BYTES,
        writable_model_bytes=0,
        files=file_records,
    )
    report["model_storage_admission"] = {
        "source_mode": "mounted-kernel-output",
        "source_kernel": source_kernel,
        "source_root": source_root,
        "mounted_blob_bytes": witness.MODEL_REQUIRED_BYTES,
        "writable_model_bytes": 0,
        "output_reserve_bytes": witness.MODEL_OUTPUT_RESERVE_BYTES,
        "free_bytes": witness.MODEL_OUTPUT_RESERVE_BYTES + 1,
    }
    if mutation == "wrong_kernel":
        report["model_assets"]["source_kernel"] = "operator/substituted-output"
    elif mutation == "missing_assets_mode":
        report["model_assets"].pop("source_mode")
    elif mutation == "missing_assets_root":
        report["model_assets"].pop("source_root")
    elif mutation == "missing_assets_kernel":
        report["model_assets"].pop("source_kernel")
    elif mutation == "missing_storage_mode":
        report["model_storage_admission"].pop("source_mode")
    elif mutation == "missing_storage_root":
        report["model_storage_admission"].pop("source_root")
    elif mutation == "missing_storage_kernel":
        report["model_storage_admission"].pop("source_kernel")
    elif mutation == "wrong_blob":
        report["model_assets"]["files"]["trellis"]["pipeline.json"][
            "sha256"
        ] = "0" * 64
    elif mutation == "writable_payload":
        report["model_assets"]["writable_model_bytes"] = 1
    elif mutation == "low_reserve":
        report["model_storage_admission"]["free_bytes"] = (
            witness.MODEL_OUTPUT_RESERVE_BYTES - 1
        )
    report_path.write_text(json.dumps(report))

    if mutation is None:
        assert witness.validate_completed_report(report_path)["status"] == "completed"
    else:
        with pytest.raises(ValueError, match="mounted model"):
            witness.validate_completed_report(report_path)


@pytest.mark.parametrize(
    "mutation",
    (
        None,
        "missing",
        "wrong_kernel",
        "wrong_report_kernel",
        "missing_mount_root",
        "relative_mount_root",
        "nested_mount_root",
        "wrong_root",
        "missing_marker_coordinate",
        "wrong_marker_coordinate",
        "ambiguous",
        "wrong_marker",
    ),
)
def test_mounted_model_download_receipt_reconciles_effective_source(
    tmp_path,
    mutation,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    kernel_source = "operator/pinned-model-output"
    mount_root = "/kaggle/input/pinned-model-output"
    blob_root = f"{mount_root}/runtime/huggingface"
    marker_coordinate = (
        "runtime/huggingface/models--microsoft--TRELLIS.2-4B/"
        "blobs/f5ec14c7f71b3d7f2cb0221c5f568a6871dc5e90"
    )
    packet = types.SimpleNamespace(kernel_sources=(kernel_source,))
    report = {
        "requested_route": {
            "model_source_kernel": kernel_source,
            "model_blob_root": blob_root,
        },
        "model_assets": {
            "source_mode": "mounted-kernel-output",
            "source_kernel": kernel_source,
            "source_root": blob_root,
        },
        "model_storage_admission": {
            "source_mode": "mounted-kernel-output",
            "source_kernel": kernel_source,
            "source_root": blob_root,
        },
    }
    mount = {
        "requested_kernel_source": kernel_source,
        "effective_mount_root": mount_root,
        "effective_blob_root": blob_root,
        "marker": marker_coordinate,
        "candidate_count": 1,
        "marker_sha256": witness.MODEL_PIPELINE_SHA256,
    }
    receipt = {"model_source_mount": mount}
    if mutation == "missing":
        receipt.pop("model_source_mount")
    elif mutation == "wrong_kernel":
        mount["requested_kernel_source"] = "operator/substituted-output"
    elif mutation == "wrong_report_kernel":
        substituted = "operator/substituted-output"
        report["requested_route"]["model_source_kernel"] = substituted
        report["model_assets"] = {"source_kernel": substituted}
        report["model_storage_admission"] = {"source_kernel": substituted}
    elif mutation == "missing_mount_root":
        mount.pop("effective_mount_root")
    elif mutation == "relative_mount_root":
        mount["effective_mount_root"] = "kaggle/input/pinned-model-output"
    elif mutation == "nested_mount_root":
        mount["effective_mount_root"] = "/kaggle/input/parent/pinned-model-output"
    elif mutation == "wrong_root":
        mount["effective_blob_root"] = "/kaggle/input/substituted/runtime/huggingface"
    elif mutation == "missing_marker_coordinate":
        mount.pop("marker")
    elif mutation == "wrong_marker_coordinate":
        mount["marker"] = "runtime/huggingface/substituted/pipeline.json"
    elif mutation == "ambiguous":
        mount["candidate_count"] = 2
    elif mutation == "wrong_marker":
        mount["marker_sha256"] = "0" * 64
    (tmp_path / "kaggle_cuda_witness_receipt.json").write_text(json.dumps(receipt))

    if mutation is None:
        witness._validate_mounted_model_receipt(packet, tmp_path, report)
    else:
        with pytest.raises(ValueError, match="mounted model"):
            witness._validate_mounted_model_receipt(packet, tmp_path, report)


def test_mounted_report_rejects_empty_packet_kernel_sources(tmp_path):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    packet = types.SimpleNamespace(kernel_sources=())
    report = {
        "requested_route": {
            "model_source_kernel": "operator/substituted-output",
            "model_blob_root": "/kaggle/input/substituted/runtime/huggingface",
        }
    }
    (tmp_path / "kaggle_cuda_witness_receipt.json").write_text("{}")

    with pytest.raises(ValueError, match="mounted model"):
        witness._validate_mounted_model_receipt(packet, tmp_path, report)


@pytest.mark.parametrize("receipt_mount", ("absent", "null"))
def test_generic_report_without_mounted_route_admits_empty_packet_kernel_sources(
    tmp_path,
    receipt_mount,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    packet = types.SimpleNamespace(kernel_sources=())
    report = {
        "requested_route": {
            "model_source_kernel": None,
            "model_blob_root": None,
        }
    }
    receipt = {}
    if receipt_mount == "null":
        receipt["model_source_mount"] = None
    (tmp_path / "kaggle_cuda_witness_receipt.json").write_text(json.dumps(receipt))

    witness._validate_mounted_model_receipt(packet, tmp_path, report)


@pytest.mark.parametrize("packet_source_present", (False, True))
@pytest.mark.parametrize("report_root_present", (False, True))
@pytest.mark.parametrize("report_kernel_present", (False, True))
@pytest.mark.parametrize("receipt_mount_state", ("absent", "null", "mounted"))
def test_model_source_authority_presence_matrix(
    tmp_path,
    packet_source_present,
    report_root_present,
    report_kernel_present,
    receipt_mount_state,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    kernel_source = "operator/pinned-model-output"
    mount_root = "/kaggle/input/pinned-model-output"
    blob_root = f"{mount_root}/runtime/huggingface"
    packet = types.SimpleNamespace(
        kernel_sources=(kernel_source,) if packet_source_present else ()
    )
    report = {
        "requested_route": {
            "model_source_kernel": kernel_source if report_kernel_present else None,
            "model_blob_root": blob_root if report_root_present else None,
        }
    }
    if report_root_present and report_kernel_present:
        source_record = {
            "source_mode": "mounted-kernel-output",
            "source_kernel": kernel_source,
            "source_root": blob_root,
        }
    else:
        source_record = {"source_mode": "selective-snapshot-download"}
    report["model_assets"] = dict(source_record)
    report["model_storage_admission"] = dict(source_record)
    receipt = {}
    if receipt_mount_state == "null":
        receipt["model_source_mount"] = None
    elif receipt_mount_state == "mounted":
        receipt["model_source_mount"] = {
            "requested_kernel_source": kernel_source,
            "effective_mount_root": mount_root,
            "effective_blob_root": blob_root,
            "marker": witness.MODEL_SOURCE_MARKER,
            "candidate_count": 1,
            "marker_sha256": witness.MODEL_PIPELINE_SHA256,
        }
    (tmp_path / "kaggle_cuda_witness_receipt.json").write_text(json.dumps(receipt))

    generic = (
        not packet_source_present
        and not report_root_present
        and not report_kernel_present
        and receipt_mount_state in {"absent", "null"}
    )
    mounted = (
        packet_source_present
        and report_root_present
        and report_kernel_present
        and receipt_mount_state == "mounted"
    )
    if generic or mounted:
        witness._validate_mounted_model_receipt(packet, tmp_path, report)
    else:
        with pytest.raises(ValueError, match="mounted model"):
            witness._validate_mounted_model_receipt(packet, tmp_path, report)


@pytest.mark.parametrize(
    ("carrier", "field", "value"),
    (
        ("model_assets", "source_mode", "mounted-kernel-output"),
        ("model_assets", "source_root", "/kaggle/input/pinned/runtime/huggingface"),
        ("model_assets", "source_kernel", "operator/pinned-model-output"),
        ("model_storage_admission", "source_mode", "mounted-kernel-output"),
        (
            "model_storage_admission",
            "source_root",
            "/kaggle/input/pinned/runtime/huggingface",
        ),
        (
            "model_storage_admission",
            "source_kernel",
            "operator/pinned-model-output",
        ),
    ),
)
def test_generic_report_rejects_mounted_effective_source_authority(
    tmp_path,
    carrier,
    field,
    value,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    report["requested_route"].update(
        model_blob_root=None,
        model_source_kernel=None,
    )
    report["model_assets"]["source_mode"] = "selective-snapshot-download"
    report["model_storage_admission"] = {
        "source_mode": "selective-snapshot-download"
    }
    report[carrier][field] = value
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="mounted model"):
        witness.validate_completed_report(report_path)

    receipt = {"model_source_mount": None}
    packet = types.SimpleNamespace(kernel_sources=())
    with pytest.raises(ValueError, match="mounted model"):
        witness._normalize_model_source_authority(packet, report, receipt)


def test_model_manifest_has_measured_required_size_and_remote_capacity_fails_first(
    tmp_path,
    monkeypatch,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    assert witness.MODEL_REQUIRED_BYTES == 14_967_470_615
    assert sum(
        len(family["files"])
        for family in witness.MODEL_BLOB_MANIFEST.values()
    ) == 17
    monkeypatch.setattr(witness, "verify_admitted_inputs_before_use", lambda *_args: None)
    monkeypatch.setattr(
        witness.shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(
            free=witness.MODEL_REQUIRED_BYTES
            + witness.MODEL_OUTPUT_RESERVE_BYTES
            - 1
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            snapshot_download=lambda **_kwargs: pytest.fail(
                "capacity failure must precede remote acquisition"
            )
        ),
    )
    dino = tmp_path / "dino"
    rembg = tmp_path / "rembg"
    dino.mkdir()
    rembg.mkdir()
    args = types.SimpleNamespace(
        work_dir=tmp_path / "runtime",
        dinov3_model_path=dino,
        rembg_model_path=rembg,
    )
    report = {}

    with pytest.raises(RuntimeError, match="insufficient writable storage"):
        witness.prepare_model_view(args, report)

    assert report["model_storage_admission"] == {
        "source_mode": "selective-snapshot-download",
        "free_bytes": witness.MODEL_REQUIRED_BYTES
        + witness.MODEL_OUTPUT_RESERVE_BYTES
        - 1,
        "model_required_bytes": witness.MODEL_REQUIRED_BYTES,
        "output_reserve_bytes": witness.MODEL_OUTPUT_RESERVE_BYTES,
        "required_bytes": witness.MODEL_REQUIRED_BYTES
        + witness.MODEL_OUTPUT_RESERVE_BYTES,
    }


def _write_completed_fixture(
    tmp_path: Path,
    *,
    run_id: str = "11111111-1111-4111-8111-111111111111",
    image_sha256: str = "a" * 64,
) -> Path:
    from scripts.source_cuda_native_image_to_glb_witness import (
        CUMESH_COMMIT,
        CUMESH_REPOSITORY,
        DINOV3_FILES,
        DINOV3_REPOSITORY,
        DINOV3_REVISION,
        FLEX_GEMM_COMMIT,
        FLEX_GEMM_REPOSITORY,
        MODEL_PIPELINE_SHA256,
        MODEL_REPOSITORY,
        MODEL_REVISION,
        NVDIFFRAST_COMMIT,
        NVDIFFRAST_REPOSITORY,
        REMBG_REPOSITORY,
        REMBG_REVISION,
        REMBG_FILES,
        SPARSE_DECODER_REPOSITORY,
        SPARSE_DECODER_REVISION,
        TRELLIS_COMMIT,
        TRELLIS_REPOSITORY,
    )
    from PIL import Image
    import trimesh

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    coords = np.asarray(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=np.int32,
    )
    voxel_coords = coords[:, 1:].copy()
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int32,
    )
    arrays_by_stage = {
        "conditioning_512": {
            "cond": np.ones((1, 2, 3), dtype=np.float32),
            "neg_cond": np.zeros((1, 2, 3), dtype=np.float32),
        },
        "sparse_flow": {
            "noise": np.ones((1, 2, 2, 2, 2), dtype=np.float32),
            "sample_next": np.ones((8, 1, 2, 2, 2, 2), dtype=np.float32),
            "pred_x0": np.zeros((8, 1, 2, 2, 2, 2), dtype=np.float32),
        },
        "sparse_support": {"coords": coords},
        "shape_flow": {
            "noise_feats": features,
            "noise_coords": coords,
            "coords": coords,
            "sample_next": np.repeat(features[None], 8, axis=0),
            "pred_x0": np.repeat((features + 1)[None], 8, axis=0),
        },
        "shape_slat": {"shape_slat_feats": features, "shape_slat_coords": coords},
        "texture_flow": {
            "noise_feats": features,
            "noise_coords": coords,
            "coords": coords,
            "sample_next": np.repeat(features[None], 8, axis=0),
            "pred_x0": np.repeat((features + 1)[None], 8, axis=0),
        },
        "decoder_raw_mesh": {"vertices": vertices, "faces": faces},
        "texture_voxels": {
            "texture_voxels_feats": features,
            "texture_voxels_coords": coords,
        },
        "pipeline_filled_mesh": {
            "vertices": vertices,
            "faces": faces,
            "texture_coords": voxel_coords,
            "texture_attrs": np.ones((4, 6), dtype=np.float32),
        },
        "postprocess_stage11_pre_orientation": {"vertices": vertices, "faces": faces},
        "postprocess_stage12_post_orientation": {"vertices": vertices, "faces": faces},
    }
    artifacts = {}
    for index, stage in enumerate(EXPECTED_STAGES):
        suffix = ".png" if stage == "preprocessed_image" else ".glb" if stage == "consumer_glb" else ".npz"
        path = output_dir / f"{index:02d}-{stage}{suffix}"
        metadata = {}
        if stage == "preprocessed_image":
            image = Image.new("RGB", (2, 2), (80, 120, 160))
            image.save(path, format="PNG")
            metadata = {"mode": "RGB", "size": [2, 2]}
        elif stage == "consumer_glb":
            uv = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.new("RGB", (2, 2), (200, 100, 50))
            )
            visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                visual=visual,
                process=False,
            )
            path.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
        else:
            arrays = arrays_by_stage[stage]
            np.savez(path, **arrays)
            metadata = {
                "arrays": {
                    name: {
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                    }
                    for name, value in arrays.items()
                }
            }
        artifacts[stage] = {
            "path": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "run_id": run_id,
            **metadata,
        }

    report = {
        "schema": "trellis2mlx.source_cuda_native_image_to_glb.v1",
        "run_id": run_id,
        "status": "completed",
        "failure_phase": None,
        "last_trustworthy_phase": "consumer_glb_validated",
        "primary_output_status": "validated",
        "capture_profile": "full",
        "expected_capture_order": list(EXPECTED_STAGES),
        "xformers_wheel": {
            "version": "0.0.35",
            "url": (
                "https://files.pythonhosted.org/packages/a4/85/"
                "6d71f9b16f2ac647877e66ed4af723b3fbd477806ab8b8a89d39a362b85f/"
                "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
            ),
            "path": (
                "/run/pinned-wheels/"
                "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
            ),
            "sha256": "ccc73c7db9890224ab05f5fb60e2034f9e6c8672a10be0cf00e95cbbae3eda7c",
            "size_bytes": 3264751,
            "install_mode": "forced-local-wheel-no-deps",
            "pip_version": "pip 25.1 from /run/site-packages/pip (python 3.12)",
        },
        "effective_route": {
            "device_type": "cuda",
            "cuda_device_name": "Tesla T4",
            "torch_version": "2.10.0+cu128",
            "xformers_build_identity": {
                "version": "0.0.35",
                "torch": "2.10.0+cu128",
                "cuda": 1208,
                "torch_cuda_arch_list": "7.5 8.0+PTX 8.0 9.0a",
                "package_from": "wheel-v0.0.35",
            },
            "xformers_import_provenance": {
                "mode": "pep610-local-wheel",
                "wheel_path": (
                    "/run/pinned-wheels/"
                    "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
                ),
                "wheel_sha256": (
                    "ccc73c7db9890224ab05f5fb60e2034f9e6c8672a10be0cf00e95cbbae3eda7c"
                ),
                "distribution_name": "xformers",
                "distribution_version": "0.0.35",
                "distribution_root": "/run/site-packages",
                "module_paths": {
                    "xformers": "/run/site-packages/xformers/__init__.py",
                    "xformers.ops": "/run/site-packages/xformers/ops/__init__.py",
                },
                "distribution_files": {
                    "xformers": "xformers/__init__.py",
                    "xformers.ops": "xformers/ops/__init__.py",
                },
                "direct_url": {
                    "url": (
                        "file:///run/pinned-wheels/"
                        "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
                    ),
                    "archive_info": {
                        "hashes": {
                            "sha256": (
                                "ccc73c7db9890224ab05f5fb60e2034f9e6c8672a10be0cf00e95cbbae3eda7c"
                            )
                        }
                    },
                },
            },
            "attention_backend": "xformers",
            "sparse_attention_backend": "xformers",
            "sparse_conv_backend": "flex_gemm",
            "trellis_commit": TRELLIS_COMMIT,
            "trellis_source_clean": True,
            "cumesh_commit": CUMESH_COMMIT,
            "cumesh_source_clean_before_build": True,
            "nvdiffrast_commit": "253ac4fcea7de5f396371124af597e6cc957bfae",
            "model_revision": "af44b45f2e35a493886929c6d786e563ec68364d",
            "sparse_decoder_revision": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
            "dinov3_revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
            "rembg_revision": "5df4c9c76d8170882c34f6986e848ee07fd0ba43",
            "pipeline_type": "512",
            "seed": 42,
            "sampler_steps": {"sparse": 8, "shape": 8, "texture": 8},
            "pipeline_run_call_count": 1,
            "native_conditioning": True,
            "native_rng": True,
            "observation_only_instrumentation": True,
            "run_id": run_id,
        },
        "requested_route": {
            "run_id": run_id,
            "image_sha256": image_sha256,
            "rembg_repository": REMBG_REPOSITORY,
            "rembg_revision": REMBG_REVISION,
            "rembg_files": {
                name: f"/request/rembg/{name}" for name in REMBG_FILES
            },
        },
        "requested_inputs": {
            "rembg": {
                "repository": REMBG_REPOSITORY,
                "revision": REMBG_REVISION,
                "files": {
                    name: {
                        "path": f"/request/rembg/{name}",
                        "sha256": digest,
                        "size_bytes": 100,
                    }
                    for name, digest in REMBG_FILES.items()
                },
            },
        },
        "effective_inputs": {
            "run_id": run_id,
            "image": {"path": "/run/image.png", "sha256": image_sha256, "size_bytes": 100},
            "dinov3": {
                "path": "/run/dinov3",
                "files": {
                    name: {"path": f"/run/dinov3/{name}", "sha256": digest, "size_bytes": 100}
                    for name, digest in DINOV3_FILES.items()
                },
            },
            "rembg": {
                "path": f"/run/admitted-inputs/{run_id}/rembg",
                "repository": REMBG_REPOSITORY,
                "revision": REMBG_REVISION,
                "files": {
                    name: {
                        "path": f"/run/admitted-inputs/{run_id}/rembg/{name}",
                        "sha256": digest,
                        "size_bytes": 100,
                    }
                    for name, digest in REMBG_FILES.items()
                },
            },
        },
        "model_assets": {
            "trellis": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "pipeline_json_sha256": MODEL_PIPELINE_SHA256,
            },
            "sparse_decoder": {
                "repository": SPARSE_DECODER_REPOSITORY,
                "revision": SPARSE_DECODER_REVISION,
            },
            "dinov3": {
                "repository": DINOV3_REPOSITORY,
                "revision": DINOV3_REVISION,
                "files": dict(DINOV3_FILES),
            },
            "rembg": {
                "repository": REMBG_REPOSITORY,
                "revision": REMBG_REVISION,
                "files": dict(REMBG_FILES),
                "source": "admitted-local-files",
                "snapshot_path": f"/run/admitted-inputs/{run_id}/rembg",
            },
            "path_rewrite_only": True,
        },
        "source_identities_after_build": {
            "trellis": {"repository": TRELLIS_REPOSITORY, "commit": TRELLIS_COMMIT, "clean": True},
            "cumesh": {"repository": CUMESH_REPOSITORY, "commit": CUMESH_COMMIT, "clean": True},
            "flex_gemm": {"repository": FLEX_GEMM_REPOSITORY, "commit": FLEX_GEMM_COMMIT, "clean": True},
            "nvdiffrast": {"repository": NVDIFFRAST_REPOSITORY, "commit": NVDIFFRAST_COMMIT, "clean": True},
        },
        "capture_order": list(EXPECTED_STAGES),
        "orientation_observer": {
            "call_count": 1,
            "native_method_return_preserved": True,
            "pre_readback_written": True,
            "post_readback_written": True,
        },
        "artifacts": artifacts,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    return report_path


def test_completed_report_admission_reopens_every_boundary(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import (
        validate_completed_report,
    )

    report_path = _write_completed_fixture(tmp_path)
    admitted = validate_completed_report(report_path)

    assert admitted["status"] == "completed"
    assert admitted["primary_output_status"] == "validated"
    assert admitted["capture_order"] == list(EXPECTED_STAGES)


def test_completed_report_admits_final_consumer_profile_without_dense_intermediates(
    tmp_path,
):
    from scripts.source_cuda_native_image_to_glb_witness import (
        validate_completed_report,
    )

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    for stage in set(EXPECTED_STAGES) - set(FINAL_CONSUMER_STAGES):
        (report_path.parent / report["artifacts"][stage]["path"]).unlink()
        report["artifacts"].pop(stage)
    report["capture_profile"] = "final-consumer"
    report["expected_capture_order"] = list(FINAL_CONSUMER_STAGES)
    report["capture_order"] = list(FINAL_CONSUMER_STAGES)
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    admitted = validate_completed_report(report_path)

    assert admitted["capture_profile"] == "final-consumer"
    assert set(admitted["artifacts"]) == set(FINAL_CONSUMER_STAGES)


def test_completed_report_rejects_plaintext_npz_and_glb_bytes(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    record = report["artifacts"]["shape_flow"]
    path = report_path.parent / record["path"]
    path.write_bytes(b"plain text pretending to be npz")
    record["sha256"] = _sha256(path)
    record["size_bytes"] = path.stat().st_size
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="NPZ|GLB|PNG|artifact structure"):
        validate_completed_report(report_path)


def test_completed_report_rejects_malformed_glb_even_with_matching_receipt(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    record = report["artifacts"]["consumer_glb"]
    path = report_path.parent / record["path"]
    path.write_bytes(b"plain text pretending to be a glb")
    record["sha256"] = _sha256(path)
    record["size_bytes"] = path.stat().st_size
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="GLB"):
        validate_completed_report(report_path)


def test_completed_report_rejects_duplicate_stage_path_even_with_matching_receipt(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    duplicate = report["artifacts"]["shape_flow"]
    report["artifacts"]["texture_flow"] = dict(duplicate)
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="distinct|duplicate"):
        validate_completed_report(report_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_array", "NPZ keys"),
        ("retyped_array", "NPZ dtype"),
        ("empty_array", "NPZ array is empty"),
    ],
)
def test_completed_report_rejects_invalid_npz_array_contract(tmp_path, mutation, match):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    record = report["artifacts"]["conditioning_512"]
    path = report_path.parent / record["path"]
    with np.load(path, allow_pickle=False) as reopened:
        arrays = {name: np.ascontiguousarray(reopened[name]) for name in reopened.files}
    if mutation == "missing_array":
        arrays.pop("neg_cond")
    elif mutation == "retyped_array":
        arrays["cond"] = arrays["cond"].astype(np.float64)
    elif mutation == "empty_array":
        arrays["cond"] = np.empty((0, 2, 3), dtype=np.float32)
    np.savez(path, **arrays)
    record["sha256"] = _sha256(path)
    record["size_bytes"] = path.stat().st_size
    record["arrays"] = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
        for name, value in arrays.items()
    }
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=match):
        validate_completed_report(report_path)


def test_completed_report_rejects_stale_bundle_from_another_run(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)

    with pytest.raises(ValueError, match="run identity mismatch"):
        validate_completed_report(
            report_path,
            expected_run_id="22222222-2222-4222-8222-222222222222",
        )


def test_current_packet_consumer_rejects_complete_prior_attempt_bundle(
    tmp_path,
    monkeypatch,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        sha256_file,
    )

    old_run_id = "11111111-1111-4111-8111-111111111111"
    current_run_id = "22222222-2222-4222-8222-222222222222"
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    authority_helper = _stage_authority_helper(capsule)
    image = capsule / "9_img.png"
    image.write_bytes(b"packet-owned-image-authority")
    rembg_coordinates, rembg_arguments, _expected_rembg = _stage_packet_rembg(
        capsule, witness, monkeypatch
    )
    image_sha256 = _sha256(image)
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )

    def packet_for(run_id, name):
        return witness.prepare_native_image_to_glb_packet(
            KaggleCudaWitnessPacket(
                capsule_dir=capsule,
                output_dir=tmp_path / name,
                dataset_id="operator/native-image-anchor-inputs",
                kernel_id="operator/native-image-anchor-cuda",
                title="Native Image Anchor CUDA",
                entrypoint=entrypoint.name,
                inputs=(entrypoint.name, authority_helper, image.name, *rembg_coordinates.values()),
                run_id=run_id,
                expected_image_sha256=image_sha256,
                output_json="report.json",
                output_npz=None,
                expected_outputs=expected_outputs,
                entrypoint_args=(
                    "--image",
                    image.name,
                    "--expected-image-sha256",
                    image_sha256,
                    "--run-id",
                    run_id,
                    "--output-dir",
                    ".",
                    "--work-dir",
                    str(tmp_path / f"runtime-{name}"),
                    *rembg_arguments,
                ),
            )
        )

    old_packet = packet_for(old_run_id, "old-packet")
    current_packet = packet_for(current_run_id, "current-packet")
    report_path = _write_completed_fixture(
        tmp_path / "old-bundle",
        run_id=old_run_id,
        image_sha256=image_sha256,
    )
    output_dir = report_path.parent
    def write_receipt(packet, bundle_dir):
        receipt_outputs = {
            name: {
                "exists": True,
                "sha256": sha256_file(bundle_dir / name),
                "size_bytes": (bundle_dir / name).stat().st_size,
            }
            for name in packet.outputs
        }
        manifest = packet.dataset_dir / "witness-manifest.json"
        receipt = {
            "schema": "trellis2mlx.kaggle_cuda_witness.receipt.v1",
            "status": "done",
            "failure_phase": None,
            "requested_dataset_id": packet.dataset_id,
            "requested_kernel_id": packet.kernel_id,
            "requested_accelerator": packet.accelerator,
            "source_identity": {
                "dataset_sources": [packet.dataset_id],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            "run_id": packet.run_id,
            "expected_image_sha256": packet.expected_image_sha256,
            "cuda_available": True,
            "cuda_device": "Tesla T4",
            "input_manifest": {
                "sha256": sha256_file(manifest),
                "size_bytes": manifest.stat().st_size,
            },
            "outputs": receipt_outputs,
        }
        (bundle_dir / "kaggle_cuda_witness_receipt.json").write_text(
            json.dumps(receipt, sort_keys=True) + "\n"
        )

    write_receipt(old_packet, output_dir)

    with pytest.raises(WitnessPacketError, match="run identity|run_id"):
        witness.validate_downloaded_native_image_to_glb_outputs(
            current_packet,
            output_dir,
        )

    current_report_path = _write_completed_fixture(
        tmp_path / "current-bundle",
        run_id=current_run_id,
        image_sha256=image_sha256,
    )
    write_receipt(current_packet, current_report_path.parent)
    admitted = witness.validate_downloaded_native_image_to_glb_outputs(
        current_packet,
        current_report_path.parent,
    )
    assert admitted["report"]["status"] == "completed"
    assert admitted["report"]["run_id"] == current_run_id


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing_requested_inputs", "requested inputs|requested RMBG"),
        ("canonical_alias", "requested RMBG"),
        ("size_disagreement", "requested RMBG|effective RMBG"),
        ("missing_model_view_path", "model assets RMBG"),
    ),
)
def test_downloaded_consumer_rejects_incoherent_rembg_authority(
    tmp_path,
    monkeypatch,
    mutation,
    match,
):
    witness, packet, _coordinates, _expected = _native_packet_contract(
        tmp_path, monkeypatch
    )
    packet = witness.prepare_native_image_to_glb_packet(packet)
    report_path = _write_completed_fixture(
        tmp_path / "downloaded-bundle",
        run_id=packet.run_id,
        image_sha256=packet.expected_image_sha256,
    )
    report = json.loads(report_path.read_text())
    if mutation == "missing_requested_inputs":
        report.pop("requested_inputs")
    elif mutation == "canonical_alias":
        alias = "/request/rembg/./model.safetensors"
        report["requested_route"]["rembg_files"]["config.json"] = alias
        report["requested_inputs"]["rembg"]["files"]["config.json"]["path"] = alias
    elif mutation == "size_disagreement":
        report["requested_inputs"]["rembg"]["files"]["config.json"][
            "size_bytes"
        ] += 1
    elif mutation == "missing_model_view_path":
        report["model_assets"]["rembg"].pop("snapshot_path")
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    _write_download_receipt(packet, report_path.parent)

    with pytest.raises(ValueError, match=match):
        witness.validate_downloaded_native_image_to_glb_outputs(
            packet,
            report_path.parent,
        )


@pytest.mark.parametrize("consumer", ("direct", "downloaded"))
@pytest.mark.parametrize(
    "raw_alias",
    (
        "/run/admitted-inputs/{run_id}/./rembg",
        "/run/admitted-inputs/{run_id}/sibling/../rembg",
    ),
)
def test_rembg_effective_root_rejects_raw_aliases(
    tmp_path,
    monkeypatch,
    consumer,
    raw_alias,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    if consumer == "downloaded":
        witness, packet, _coordinates, _expected = _native_packet_contract(
            tmp_path, monkeypatch
        )
        packet = witness.prepare_native_image_to_glb_packet(packet)
        report_path = _write_completed_fixture(
            tmp_path / "downloaded-alias-bundle",
            run_id=packet.run_id,
            image_sha256=packet.expected_image_sha256,
        )
    else:
        packet = None
        report_path = _write_completed_fixture(tmp_path / "direct-alias-bundle")
    report = json.loads(report_path.read_text())
    report["effective_inputs"]["rembg"]["path"] = raw_alias.format(
        run_id=report["run_id"]
    )
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="effective RMBG|canonical|declared"):
        if consumer == "downloaded":
            _write_download_receipt(packet, report_path.parent)
            witness.validate_downloaded_native_image_to_glb_outputs(
                packet,
                report_path.parent,
            )
        else:
            witness.validate_completed_report(report_path)


@pytest.mark.parametrize(
    "alias_kind",
    ("dot", "parent", "repeated_separator", "double_leading_slash"),
)
def test_producer_rejects_raw_rembg_alias_before_input_admission(
    tmp_path,
    monkeypatch,
    alias_kind,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    rembg = tmp_path / "rembg"
    rembg.mkdir()
    files = {}
    for name in witness.REMBG_FILES:
        path = rembg / name
        path.write_bytes(name.encode())
        files[name] = path
    monkeypatch.setattr(
        witness,
        "REMBG_FILES",
        {name: _sha256(path) for name, path in files.items()},
    )
    target = str(files["config.json"])
    aliases = {
        "dot": f"{rembg}/./config.json",
        "parent": f"{rembg}/sibling/../config.json",
        "repeated_separator": f"{rembg}//config.json",
        "double_leading_slash": f"/{target}",
    }
    files["config.json"] = aliases[alias_kind]
    arguments = [
        *_base_args(tmp_path, image),
        "--run-id",
        "31fce6b7-853b-4a0f-b99d-518be23ebabc",
        *(
            item
            for name, attribute in witness.REMBG_FILE_ARGUMENTS.items()
            for item in (f"--{attribute.replace('_', '-')}", str(files[name]))
        ),
        "--no-download",
    ]

    assert witness.main(arguments) == 1
    report = json.loads((tmp_path / "outputs" / "report.json").read_text())
    assert report["failure_phase"] == "request_validation"
    assert "canonical" in report["error"] or "declared" in report["error"]
    assert not (tmp_path / "runtime" / "admitted-inputs").exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_model_assets", "model assets"),
        ("missing_requested_inputs", "requested inputs|requested RMBG"),
        ("missing_requested_rembg", "requested RMBG"),
        ("wrong_requested_repository", "requested RMBG"),
        ("wrong_requested_revision", "requested RMBG"),
        ("missing_requested_route_coordinate", "requested RMBG"),
        ("null_requested_route_coordinate", "requested RMBG"),
        ("duplicate_requested_route_coordinate", "requested RMBG"),
        ("renamed_requested_route_coordinate", "requested RMBG"),
        ("requested_path_mismatch", "requested RMBG"),
        ("dot_alias_requested_coordinate", "requested RMBG"),
        ("repeated_separator_requested_coordinate", "requested RMBG"),
        ("parent_alias_requested_coordinate", "requested RMBG"),
        ("wrong_requested_digest", "requested RMBG"),
        ("nonpositive_requested_size", "requested RMBG"),
        ("requested_effective_size_disagreement", "requested RMBG|effective RMBG"),
        ("requested_effective_digest_disagreement", "requested RMBG|effective RMBG"),
        ("missing_effective_rembg", "effective RMBG"),
        ("wrong_effective_rembg_digest", "effective RMBG"),
        ("missing_model_asset_rembg_files", "model assets RMBG"),
        ("missing_model_asset_rembg_snapshot", "model assets RMBG"),
        ("wrong_model_asset_rembg_snapshot", "model assets RMBG"),
        ("wrong_effective_image", "effective image"),
        ("missing_run_id", "run identity"),
        ("missing_xformers_wheel", "xformers wheel"),
        ("missing_xformers_pip_identity", "effective pip identity"),
        ("missing_xformers_provenance", "xformers import provenance"),
        ("foreign_xformers_module", "xformers import provenance|module"),
        ("foreign_xformers_wheel_path", "xformers import provenance|wheel path"),
        ("foreign_xformers_wheel_digest", "xformers import provenance|wheel digest"),
    ],
)
def test_completed_report_rejects_missing_or_wrong_authority(tmp_path, mutation, match):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    if mutation == "missing_model_assets":
        report.pop("model_assets", None)
    elif mutation == "missing_requested_inputs":
        report.pop("requested_inputs", None)
    elif mutation == "missing_requested_rembg":
        report["requested_inputs"].pop("rembg")
    elif mutation == "wrong_requested_repository":
        report["requested_route"]["rembg_repository"] = "substituted/RMBG"
    elif mutation == "wrong_requested_revision":
        report["requested_inputs"]["rembg"]["revision"] = "0" * 40
    elif mutation == "missing_requested_route_coordinate":
        report["requested_route"]["rembg_files"].pop("config.json")
    elif mutation == "null_requested_route_coordinate":
        report["requested_route"]["rembg_files"]["config.json"] = None
    elif mutation == "duplicate_requested_route_coordinate":
        report["requested_route"]["rembg_files"]["config.json"] = report[
            "requested_route"
        ]["rembg_files"]["model.safetensors"]
    elif mutation == "renamed_requested_route_coordinate":
        value = report["requested_route"]["rembg_files"].pop("config.json")
        report["requested_route"]["rembg_files"]["configuration.json"] = value
    elif mutation == "requested_path_mismatch":
        report["requested_inputs"]["rembg"]["files"]["config.json"][
            "path"
        ] = "/request/rembg/other-config.json"
    elif mutation == "dot_alias_requested_coordinate":
        alias = "/request/rembg/./model.safetensors"
        report["requested_route"]["rembg_files"]["config.json"] = alias
        report["requested_inputs"]["rembg"]["files"]["config.json"]["path"] = alias
    elif mutation == "repeated_separator_requested_coordinate":
        alias = "/request/rembg//model.safetensors"
        report["requested_route"]["rembg_files"]["config.json"] = alias
        report["requested_inputs"]["rembg"]["files"]["config.json"]["path"] = alias
    elif mutation == "parent_alias_requested_coordinate":
        alias = "/request/rembg/subdir/../model.safetensors"
        report["requested_route"]["rembg_files"]["config.json"] = alias
        report["requested_inputs"]["rembg"]["files"]["config.json"]["path"] = alias
    elif mutation == "wrong_requested_digest":
        report["requested_inputs"]["rembg"]["files"]["config.json"][
            "sha256"
        ] = "0" * 64
    elif mutation == "nonpositive_requested_size":
        report["requested_inputs"]["rembg"]["files"]["config.json"][
            "size_bytes"
        ] = 0
    elif mutation == "requested_effective_size_disagreement":
        report["requested_inputs"]["rembg"]["files"]["config.json"][
            "size_bytes"
        ] += 1
    elif mutation == "requested_effective_digest_disagreement":
        report["requested_inputs"]["rembg"]["files"]["config.json"][
            "sha256"
        ] = "0" * 64
    elif mutation == "missing_effective_rembg":
        report["effective_inputs"].pop("rembg")
    elif mutation == "wrong_effective_rembg_digest":
        report["effective_inputs"]["rembg"]["files"]["model.safetensors"][
            "sha256"
        ] = "0" * 64
    elif mutation == "missing_model_asset_rembg_files":
        report["model_assets"]["rembg"].pop("files")
    elif mutation == "missing_model_asset_rembg_snapshot":
        report["model_assets"]["rembg"].pop("snapshot_path")
    elif mutation == "wrong_model_asset_rembg_snapshot":
        report["model_assets"]["rembg"]["snapshot_path"] = "/run/foreign/rembg"
    elif mutation == "wrong_effective_image":
        report.setdefault("effective_inputs", {})["image"] = {
            "sha256": "0" * 64,
            "size_bytes": 1,
        }
    elif mutation == "missing_run_id":
        report.pop("run_id", None)
    elif mutation == "missing_xformers_wheel":
        report.pop("xformers_wheel", None)
    elif mutation == "missing_xformers_pip_identity":
        report["xformers_wheel"].pop("pip_version", None)
    elif mutation == "missing_xformers_provenance":
        report["effective_route"].pop("xformers_import_provenance", None)
    elif mutation == "foreign_xformers_module":
        report["effective_route"]["xformers_import_provenance"]["module_paths"][
            "xformers"
        ] = "/tmp/foreign/xformers/__init__.py"
    elif mutation == "foreign_xformers_wheel_path":
        report["effective_route"]["xformers_import_provenance"]["direct_url"]["url"] = (
            "file:///tmp/foreign/xformers-0.0.35.whl"
        )
    elif mutation == "foreign_xformers_wheel_digest":
        report["effective_route"]["xformers_import_provenance"]["direct_url"][
            "archive_info"
        ]["hashes"]["sha256"] = "0" * 64
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=match):
        validate_completed_report(report_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_stage", "capture order"),
        ("wrong_digest", "digest mismatch"),
        ("wrong_device", "device_type"),
        ("dirty_source", "trellis_source_clean"),
        ("orientation_not_observed", "orientation observer"),
    ],
)
def test_completed_report_admission_rejects_false_closure(tmp_path, mutation, match):
    from scripts.source_cuda_native_image_to_glb_witness import (
        validate_completed_report,
    )

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    if mutation == "missing_stage":
        report["capture_order"].remove("shape_flow")
    elif mutation == "wrong_digest":
        report["artifacts"]["shape_flow"]["sha256"] = "0" * 64
    elif mutation == "wrong_device":
        report["effective_route"]["device_type"] = "cpu"
    elif mutation == "dirty_source":
        report["effective_route"]["trellis_source_clean"] = False
    elif mutation == "orientation_not_observed":
        report["orientation_observer"]["call_count"] = 0
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=match):
        validate_completed_report(report_path)

import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from scripts.prepare_native_image_to_glb_attempt import prepare_attempt_from_path
from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket


def _write_valid_spec(tmp_path: Path, *, spec_path: Path | None = None) -> Path:
    sources = tmp_path / "sources"
    sources.mkdir(parents=True, exist_ok=True)

    def asset(name: str, coordinate: str) -> dict:
        path = sources / name
        payload = name.encode()
        path.write_bytes(payload)
        return {
            "source": str(path),
            "coordinate": coordinate,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    payload = {
        "schema": "trellis2mlx.native_image_to_glb_attempt_spec.v2",
        "run_id": "31fce6b7-853b-4a0f-b99d-518be23ebabc",
        "dataset_id": "operator/topology-inputs",
        "kernel_id": "operator/topology-cuda",
        "title": "Topology CUDA",
        "capsule_dir": str(tmp_path / "capsule"),
        "output_dir": str(tmp_path / "packet"),
        "entrypoint": asset("entrypoint.py", "entrypoint.py"),
        "authority_helper": asset("authority.py", "witness_authority.py"),
        "image": asset("image.png", "image.png"),
        "dinov3_files": {
            name: asset(f"dinov3-{name}", name)
            for name in (
                "model.safetensors",
                "config.json",
                "preprocessor_config.json",
            )
        },
        "rembg_files": {
            name: asset(name, f"rembg-{name}")
            for name in (
                "model.safetensors",
                "config.json",
                "birefnet.py",
                "BiRefNet_config.py",
            )
        },
        "expected_outputs": ["12-consumer_glb.glb"],
        "output_coordinate": "outputs",
        "work_coordinate": "runtime",
        "accelerator": "NvidiaTeslaT4",
        "enable_internet": True,
    }
    path = spec_path or (tmp_path / "attempt.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_preparation_receipt_rejects_spec_mutation_after_single_read(tmp_path):
    spec_path = tmp_path / "attempt.json"
    original = json.dumps({"schema": "original"}).encode()
    spec_path.write_bytes(original)
    report_path = tmp_path / "report.json"

    def mutate():
        spec_path.write_text(json.dumps({"schema": "substituted"}))

    report = prepare_attempt_from_path(
        spec_path,
        report_path,
        after_read=mutate,
    )

    assert report["status"] == "failed"
    assert report["failure_phase"] == "spec_identity_changed"
    assert report["spec"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert report["final_spec"]["sha256"] != report["spec"]["sha256"]


def test_v3_final_consumer_spec_binds_profile_to_runner_and_outputs(tmp_path):
    from trellmlx.native_image_to_glb_attempt import (
        build_attempt_packet,
        load_attempt_spec,
    )

    spec_path = _write_valid_spec(tmp_path)
    payload = json.loads(spec_path.read_text())
    payload["schema"] = "trellis2mlx.native_image_to_glb_attempt_spec.v3"
    payload["capture_profile"] = "final-consumer"
    payload["expected_outputs"] = [
        "07-decoder_raw_mesh.npz",
        "10-postprocess_stage11_pre_orientation.npz",
        "11-postprocess_stage12_post_orientation.npz",
        "12-consumer_glb.glb",
    ]
    spec_path.write_text(json.dumps(payload))

    packet = build_attempt_packet(load_attempt_spec(spec_path))

    assert packet.expected_outputs == tuple(payload["expected_outputs"])
    profile_index = packet.entrypoint_args.index("--capture-profile")
    assert packet.entrypoint_args[profile_index + 1] == "final-consumer"
    attempt = json.loads(
        (packet.capsule_dir / "native-image-to-glb-attempt.json").read_text()
    )
    assert attempt["schema"] == "trellis2mlx.native_image_to_glb_attempt.v3"
    assert attempt["capture_profile"] == "final-consumer"


def test_v3_final_consumer_spec_rejects_mismatched_output_set(tmp_path):
    from trellmlx.native_image_to_glb_attempt import (
        AttemptSpecError,
        build_attempt_packet,
        load_attempt_spec,
    )

    spec_path = _write_valid_spec(tmp_path)
    payload = json.loads(spec_path.read_text())
    payload["schema"] = "trellis2mlx.native_image_to_glb_attempt_spec.v3"
    payload["capture_profile"] = "final-consumer"
    spec_path.write_text(json.dumps(payload))

    with pytest.raises(AttemptSpecError, match="outputs do not match capture profile"):
        build_attempt_packet(load_attempt_spec(spec_path))


def test_v3_spec_rejects_explicit_full_profile_before_packet_build(tmp_path):
    from trellmlx.native_image_to_glb_attempt import AttemptSpecError, load_attempt_spec

    spec_path = _write_valid_spec(tmp_path)
    payload = json.loads(spec_path.read_text())
    payload["schema"] = "trellis2mlx.native_image_to_glb_attempt_spec.v3"
    payload["capture_profile"] = "full"
    payload["expected_outputs"] = [
        "07-decoder_raw_mesh.npz",
        "10-postprocess_stage11_pre_orientation.npz",
        "11-postprocess_stage12_post_orientation.npz",
        "12-consumer_glb.glb",
    ]
    spec_path.write_text(json.dumps(payload))

    with pytest.raises(AttemptSpecError, match="explicit full capture profile"):
        load_attempt_spec(spec_path)


@pytest.mark.parametrize("alias", ("report_in_output", "spec_in_output", "same_file"))
def test_preparer_rejects_spec_and_report_aliases_before_managed_mutation(
    tmp_path,
    alias,
):
    output = tmp_path / "packet"
    spec_path = _write_valid_spec(
        tmp_path,
        spec_path=(output / "attempt.json" if alias == "spec_in_output" else None),
    )
    original_spec = spec_path.read_bytes()
    report_path = (
        output / "report.json"
        if alias == "report_in_output"
        else spec_path
        if alias == "same_file"
        else tmp_path / "report.json"
    )
    marker = output / "preserve-me.txt"
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text("preserved")

    report = prepare_attempt_from_path(spec_path, report_path)

    assert report["status"] == "failed"
    assert "topology" in report["error"] or "alias" in report["error"]
    assert marker.read_text() == "preserved"
    assert spec_path.read_bytes() == original_spec


@pytest.mark.parametrize(
    "asset_role",
    (
        "entrypoint",
        "authority_helper",
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
def test_preparer_rejects_report_alias_with_every_protected_asset(
    tmp_path,
    asset_role,
):
    spec_path = _write_valid_spec(tmp_path)
    payload = json.loads(spec_path.read_text())
    if asset_role.startswith("rembg:"):
        asset = payload["rembg_files"][asset_role.split(":", 1)[1]]
    elif asset_role.startswith("dinov3:"):
        asset = payload["dinov3_files"][asset_role.split(":", 1)[1]]
    else:
        asset = payload[asset_role]
    report_path = Path(asset["source"])
    original = report_path.read_bytes()
    output = Path(payload["output_dir"])
    output.mkdir()
    marker = output / "preserve-me.txt"
    marker.write_text("preserved")

    report = prepare_attempt_from_path(spec_path, report_path)

    assert report["status"] == "failed"
    assert "asset" in report["error"] or "topology" in report["error"]
    assert report_path.read_bytes() == original
    assert marker.read_text() == "preserved"


@pytest.mark.parametrize("mutation_phase", ("packet_build", "packet_prepare"))
def test_preparer_rejects_spec_drift_during_packet_construction(
    tmp_path,
    monkeypatch,
    mutation_phase,
):
    from scripts import prepare_native_image_to_glb_attempt as preparer

    spec_path = _write_valid_spec(tmp_path)
    payload = json.loads(spec_path.read_text())
    output = Path(payload["output_dir"])
    capsule = Path(payload["capsule_dir"])
    output.mkdir()
    capsule.mkdir()
    output_marker = output / "prior-output.txt"
    capsule_marker = capsule / "prior-capsule.txt"
    output_marker.write_text("prior output")
    capsule_marker.write_text("prior capsule")

    def mutate_spec():
        mutated = json.loads(spec_path.read_text())
        mutated["title"] = "Mutated During Construction"
        spec_path.write_text(json.dumps(mutated))

    def fake_build(spec):
        if mutation_phase == "packet_build":
            mutate_spec()
        spec.capsule_dir.mkdir(parents=True, exist_ok=True)
        (spec.capsule_dir / "native-image-to-glb-attempt.json").write_text("{}")
        return types.SimpleNamespace(
            run_id=spec.run_id,
            dataset_id=spec.dataset_id,
            kernel_id=spec.kernel_id,
            capsule_dir=spec.capsule_dir,
            output_dir=spec.output_dir,
            dataset_dir=spec.output_dir / "dataset",
        )

    def fake_prepare(packet):
        if mutation_phase == "packet_prepare":
            mutate_spec()
        packet.dataset_dir.mkdir(parents=True, exist_ok=True)
        (packet.dataset_dir / "witness-manifest.json").write_text("{}")
        return packet

    monkeypatch.setattr(preparer, "build_attempt_packet", fake_build)
    monkeypatch.setattr(preparer, "prepare_native_image_to_glb_packet", fake_prepare)

    report = prepare_attempt_from_path(spec_path, tmp_path / "report.json")

    assert report["status"] == "failed"
    assert report["failure_phase"] == "spec_identity_changed"
    assert report["spec"]["sha256"] != report["final_spec"]["sha256"]
    assert output_marker.read_text() == "prior output"
    assert capsule_marker.read_text() == "prior capsule"


def test_preparer_rolls_back_spec_drift_after_pair_installation(
    tmp_path,
    monkeypatch,
):
    from scripts import prepare_native_image_to_glb_attempt as preparer

    spec_path = _write_valid_spec(tmp_path)
    payload = json.loads(spec_path.read_text())
    output = Path(payload["output_dir"])
    capsule = Path(payload["capsule_dir"])
    output.mkdir()
    capsule.mkdir()
    output_marker = output / "prior-output.txt"
    capsule_marker = capsule / "prior-capsule.txt"
    output_marker.write_text("prior output")
    capsule_marker.write_text("prior capsule")

    def fake_build(spec):
        spec.capsule_dir.mkdir(parents=True, exist_ok=True)
        (spec.capsule_dir / "native-image-to-glb-attempt.json").write_text("{}")
        return KaggleCudaWitnessPacket(
            run_id=spec.run_id,
            dataset_id=spec.dataset_id,
            kernel_id=spec.kernel_id,
            title=spec.title,
            capsule_dir=spec.capsule_dir,
            output_dir=spec.output_dir,
            entrypoint="entrypoint.py",
            inputs=(),
        )

    def fake_prepare(packet):
        packet.dataset_dir.mkdir(parents=True, exist_ok=True)
        (packet.dataset_dir / "witness-manifest.json").write_text("{}")
        return packet

    real_publish = preparer._publish_pair

    def publish_then_mutate(*args, **kwargs):
        publication = real_publish(*args, **kwargs)
        mutated = json.loads(spec_path.read_text())
        mutated["title"] = "Mutated After Pair Installation"
        spec_path.write_text(json.dumps(mutated))
        return publication

    monkeypatch.setattr(preparer, "build_attempt_packet", fake_build)
    monkeypatch.setattr(preparer, "prepare_native_image_to_glb_packet", fake_prepare)
    monkeypatch.setattr(preparer, "_publish_pair", publish_then_mutate)

    report = prepare_attempt_from_path(spec_path, tmp_path / "report.json")

    assert report["status"] == "failed"
    assert report["failure_phase"] == "spec_identity_changed"
    assert report["spec"]["sha256"] != report["final_spec"]["sha256"]
    assert output_marker.read_text() == "prior output"
    assert capsule_marker.read_text() == "prior capsule"


def _prepare_synthetic_attempt_packet(tmp_path, entrypoint_source):
    from trellmlx.kaggle_cuda_witness import prepare_packet
    from trellmlx.native_image_to_glb_attempt import (
        build_attempt_packet,
        load_attempt_spec,
    )

    spec_path = _write_valid_spec(tmp_path)
    payload = json.loads(spec_path.read_text())
    entrypoint = Path(payload["entrypoint"]["source"])
    entrypoint.write_text(entrypoint_source)
    entrypoint_bytes = entrypoint.read_bytes()
    payload["entrypoint"]["sha256"] = hashlib.sha256(entrypoint_bytes).hexdigest()
    payload["entrypoint"]["size_bytes"] = len(entrypoint_bytes)
    spec_path.write_text(json.dumps(payload))
    return prepare_packet(build_attempt_packet(load_attempt_spec(spec_path)))


def _run_synthetic_attempt_packet(packet, work, monkeypatch):
    fake_torch = types.SimpleNamespace(
        __version__="2.10.0+cu128",
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Tesla T4",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    work.mkdir(exist_ok=True)
    monkeypatch.chdir(work)
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text().replace(
        'Path("/kaggle/input")',
        f"Path({str(packet.dataset_dir)!r})",
    )
    namespace = {"__name__": "runner_test"}
    exec(runner, namespace)
    rc = namespace["main"]()
    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    return rc, receipt


def test_structured_runner_bridges_attempt_outputs_to_kaggle_publication(
    tmp_path,
    monkeypatch,
):
    packet = _prepare_synthetic_attempt_packet(
        tmp_path,
        "import argparse\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--output-json', required=True)\n"
        "p.add_argument('--output-dir', required=True)\n"
        "a, _ = p.parse_known_args()\n"
        "output_dir = Path(a.output_dir)\n"
        "output_dir.mkdir(parents=True, exist_ok=True)\n"
        "Path(a.output_json).write_text('{\"status\": \"completed\"}\\n')\n"
        "(output_dir / '12-consumer_glb.glb').write_bytes(b'glb')\n",
    )
    work = tmp_path / "work"
    work.mkdir()
    (work / "report.json").write_text('{"status": "stale"}\n')
    (work / "12-consumer_glb.glb").write_bytes(b"stale")

    rc, receipt = _run_synthetic_attempt_packet(packet, work, monkeypatch)

    command = receipt["effective_command"]
    assert rc == 0
    assert command[command.index("--output-json") + 1] == "outputs/report.json"
    assert (work / "outputs" / "report.json").is_file()
    assert (work / "outputs" / "12-consumer_glb.glb").is_file()
    assert json.loads((work / "report.json").read_text())["status"] == "completed"
    assert (work / "12-consumer_glb.glb").read_bytes() == b"glb"
    assert receipt["status"] == "done"
    assert set(receipt["outputs"]) == set(packet.outputs)


def test_structured_runner_publishes_failure_report_from_attempt_coordinate(
    tmp_path,
    monkeypatch,
):
    packet = _prepare_synthetic_attempt_packet(
        tmp_path,
        "import argparse\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--output-json', required=True)\n"
        "p.add_argument('--output-dir', required=True)\n"
        "a, _ = p.parse_known_args()\n"
        "Path(a.output_dir).mkdir(parents=True, exist_ok=True)\n"
        "Path(a.output_json).write_text('{\"status\": \"failed\"}\\n')\n"
        "raise SystemExit(9)\n",
    )
    work = tmp_path / "work"

    rc, receipt = _run_synthetic_attempt_packet(packet, work, monkeypatch)

    command = receipt["effective_command"]
    assert rc == 9
    assert command[command.index("--output-json") + 1] == "outputs/report.json"
    assert json.loads((work / "report.json").read_text())["status"] == "failed"
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "execution"
    assert receipt["outputs"]["report.json"]["exists"] is True
    assert receipt["outputs"]["12-consumer_glb.glb"]["exists"] is False


def test_structured_runner_preserves_nested_report_when_publication_fails(
    tmp_path,
    monkeypatch,
):
    packet = _prepare_synthetic_attempt_packet(
        tmp_path,
        "import argparse\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--output-json', required=True)\n"
        "p.add_argument('--output-dir', required=True)\n"
        "a, _ = p.parse_known_args()\n"
        "Path(a.output_dir).mkdir(parents=True, exist_ok=True)\n"
        "Path(a.output_json).write_text('{\"status\": \"failed\"}\\n')\n"
        "Path('report.json').mkdir()\n"
        "raise SystemExit(9)\n",
    )
    work = tmp_path / "work"

    rc, receipt = _run_synthetic_attempt_packet(packet, work, monkeypatch)

    fallback = work / "kaggle_cuda_witness_child_report.json"
    assert rc == 8
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "output_publication"
    assert receipt["execution_outputs"]["report.json"]["exists"] is True
    assert receipt["child_report_fallback"]["exists"] is True
    assert json.loads(fallback.read_text())["status"] == "failed"


@pytest.mark.parametrize("spec_kind", ("malformed", "wrong_schema"))
def test_preparer_preserves_invalid_spec_when_report_aliases_it(
    tmp_path,
    spec_kind,
):
    spec_path = _write_valid_spec(tmp_path)
    if spec_kind == "malformed":
        original = b'{"schema":'
    else:
        payload = json.loads(spec_path.read_text())
        payload["schema"] = "trellis2mlx.native_image_to_glb_attempt_spec.v999"
        original = json.dumps(payload).encode()
    spec_path.write_bytes(original)

    report = prepare_attempt_from_path(spec_path, spec_path)

    assert report["status"] == "failed"
    assert report["failure_phase"] == "spec_load"
    assert report["requested_report_path"] == str(spec_path.resolve())
    assert report["effective_report_path"] != str(spec_path.resolve())
    assert spec_path.read_bytes() == original
    effective_report = Path(report["effective_report_path"])
    assert effective_report.is_file()
    assert json.loads(effective_report.read_text()) == report

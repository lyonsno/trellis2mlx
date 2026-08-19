import hashlib
import json
from pathlib import Path
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
        "schema": "trellis2mlx.native_image_to_glb_attempt_spec.v1",
        "run_id": "31fce6b7-853b-4a0f-b99d-518be23ebabc",
        "dataset_id": "operator/topology-inputs",
        "kernel_id": "operator/topology-cuda",
        "title": "Topology CUDA",
        "capsule_dir": str(tmp_path / "capsule"),
        "output_dir": str(tmp_path / "packet"),
        "entrypoint": asset("entrypoint.py", "entrypoint.py"),
        "authority_helper": asset("authority.py", "witness_authority.py"),
        "image": asset("image.png", "image.png"),
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

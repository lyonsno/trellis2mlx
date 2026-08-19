import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from trellmlx.native_image_to_glb_attempt import (
    AttemptAsset,
    AttemptSpecError,
    NativeImageToGLBAttemptSpec,
    build_attempt_packet,
    load_attempt_spec,
)


def _asset(path: Path, coordinate: str, payload: bytes) -> AttemptAsset:
    path.write_bytes(payload)
    return AttemptAsset(
        source=path,
        coordinate=coordinate,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _dinov3_assets(sources: Path) -> dict[str, AttemptAsset]:
    return {
        name: _asset(sources / f"dinov3-{name}", name, name.encode())
        for name in (
            "model.safetensors",
            "config.json",
            "preprocessor_config.json",
        )
    }


def test_structured_attempt_requires_hash_bound_dinov3_model_path(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    rembg = {
        name: _asset(sources / f"rembg-{name}", f"rembg-{name}", name.encode())
        for name in (
            "model.safetensors",
            "config.json",
            "birefnet.py",
            "BiRefNet_config.py",
        )
    }
    dinov3 = _dinov3_assets(sources)
    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/native-image-r6-inputs",
        kernel_id="operator/native-image-r6-cuda",
        title="Native Image R6 CUDA",
        capsule_dir=tmp_path / "capsule",
        output_dir=tmp_path / "packet",
        entrypoint=_asset(sources / "entrypoint.py", "entrypoint.py", b"entry"),
        authority_helper=_asset(
            sources / "authority.py", "witness_authority.py", b"authority"
        ),
        image=_asset(sources / "image.png", "image.png", b"image"),
        dinov3_files=dinov3,
        rembg_files=rembg,
        expected_outputs=("12-consumer_glb.glb",),
    )

    packet = build_attempt_packet(spec)

    position = packet.entrypoint_args.index("--dinov3-model-path")
    assert packet.entrypoint_args[position + 1] == "."
    assert all(asset.coordinate in packet.inputs for asset in dinov3.values())
    manifest = json.loads(
        (spec.capsule_dir / "native-image-to-glb-attempt.json").read_text()
    )
    assert {
        role: manifest["assets"][f"dinov3:{role}"]["coordinate"]
        for role in dinov3
    } == {role: role for role in dinov3}


@pytest.mark.parametrize(
    "mutation", ("missing_role", "renamed_coordinate", "aliased_sources")
)
def test_structured_attempt_rejects_unusable_dinov3_custody(tmp_path, mutation):
    sources = tmp_path / "sources"
    sources.mkdir()
    dinov3 = _dinov3_assets(sources)
    if mutation == "missing_role":
        dinov3.pop("preprocessor_config.json")
    elif mutation == "renamed_coordinate":
        dinov3["config.json"] = _asset(
            sources / "renamed-dinov3-config.json",
            "dinov3-config.json",
            b"config.json",
        )
    else:
        shared = dinov3["model.safetensors"]
        dinov3["config.json"] = AttemptAsset(
            source=shared.source,
            coordinate="config.json",
            sha256=shared.sha256,
            size_bytes=shared.size_bytes,
        )
    rembg = {
        name: _asset(sources / f"rembg-{name}", f"rembg-{name}", name.encode())
        for name in (
            "model.safetensors",
            "config.json",
            "birefnet.py",
            "BiRefNet_config.py",
        )
    }
    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/native-image-r6-inputs",
        kernel_id="operator/native-image-r6-cuda",
        title="Native Image R6 CUDA",
        capsule_dir=tmp_path / "capsule",
        output_dir=tmp_path / "packet",
        entrypoint=_asset(sources / "entrypoint.py", "entrypoint.py", b"entry"),
        authority_helper=_asset(
            sources / "authority.py", "witness_authority.py", b"authority"
        ),
        image=_asset(sources / "image.png", "image.png", b"image"),
        dinov3_files=dinov3,
        rembg_files=rembg,
        expected_outputs=("12-consumer_glb.glb",),
    )

    with pytest.raises(AttemptSpecError, match="DINOv3"):
        build_attempt_packet(spec)


def test_structured_attempt_builds_packet_without_lifecycle_text_rewriting(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    rembg = {
        name: _asset(sources / name, coordinate, name.encode())
        for name, coordinate in {
            "model.safetensors": "rembg-model.safetensors",
            "config.json": "rembg-config.json",
            "birefnet.py": "rembg-birefnet.py",
            "BiRefNet_config.py": "rembg-BiRefNet-config.py",
        }.items()
    }
    dinov3 = _dinov3_assets(sources)
    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/native-image-r5-inputs",
        kernel_id="operator/native-image-r5-cuda",
        title="Native Image R5 CUDA",
        capsule_dir=tmp_path / "capsule",
        output_dir=tmp_path / "packet",
        entrypoint=_asset(
            sources / "source_cuda_native_image_to_glb_witness.py",
            "source_cuda_native_image_to_glb_witness.py",
            b"print('entrypoint')\n",
        ),
        authority_helper=_asset(
            sources / "witness_authority.py",
            "witness_authority.py",
            b"# shared authority helper\n",
        ),
        image=_asset(sources / "9_img.png", "9_img.png", b"image"),
        dinov3_files=dinov3,
        rembg_files=rembg,
        expected_outputs=("00-preprocessed_image.png", "12-consumer_glb.glb"),
    )

    packet = build_attempt_packet(spec)

    assert packet.run_id == spec.run_id
    assert packet.dataset_id == spec.dataset_id
    assert packet.kernel_id == spec.kernel_id
    assert packet.inputs == (
        spec.entrypoint.coordinate,
        spec.authority_helper.coordinate,
        spec.image.coordinate,
        *(asset.coordinate for asset in dinov3.values()),
        *(asset.coordinate for asset in rembg.values()),
        "native-image-to-glb-attempt.json",
    )
    assert packet.entrypoint_args[:8] == (
        "--image",
        spec.image.coordinate,
        "--expected-image-sha256",
        spec.image.sha256,
        "--run-id",
        spec.run_id,
        "--output-dir",
        spec.output_coordinate,
    )
    for asset in (
        spec.entrypoint,
        spec.authority_helper,
        spec.image,
        *dinov3.values(),
        *rembg.values(),
    ):
        staged = spec.capsule_dir / asset.coordinate
        assert staged.read_bytes() == asset.source.read_bytes()


def test_structured_attempt_preserves_prior_capsule_when_source_identity_drifts(
    tmp_path,
):
    sources = tmp_path / "sources"
    sources.mkdir()
    rembg = {
        name: _asset(sources / name, coordinate, name.encode())
        for name, coordinate in {
            "model.safetensors": "rembg-model.safetensors",
            "config.json": "rembg-config.json",
            "birefnet.py": "rembg-birefnet.py",
            "BiRefNet_config.py": "rembg-BiRefNet-config.py",
        }.items()
    }
    dinov3 = _dinov3_assets(sources)
    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/native-image-r5-inputs",
        kernel_id="operator/native-image-r5-cuda",
        title="Native Image R5 CUDA",
        capsule_dir=tmp_path / "capsule",
        output_dir=tmp_path / "packet",
        entrypoint=_asset(sources / "entrypoint.py", "entrypoint.py", b"entrypoint"),
        authority_helper=_asset(
            sources / "witness_authority.py",
            "witness_authority.py",
            b"authority",
        ),
        image=_asset(sources / "image.png", "image.png", b"image"),
        dinov3_files=dinov3,
        rembg_files=rembg,
        expected_outputs=("12-consumer_glb.glb",),
    )
    build_attempt_packet(spec)
    marker = spec.capsule_dir / "preserve-me.txt"
    marker.write_text("old capsule")
    spec.image.source.write_bytes(b"mutated after declaration")

    with pytest.raises(AttemptSpecError, match="digest|size"):
        build_attempt_packet(spec)

    assert marker.read_text() == "old capsule"


def test_attempt_spec_loads_complete_structured_json(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    assets = {
        "entrypoint": _asset(sources / "entrypoint.py", "entrypoint.py", b"entry"),
        "authority_helper": _asset(
            sources / "authority.py", "witness_authority.py", b"authority"
        ),
        "image": _asset(sources / "image.png", "image.png", b"image"),
    }
    rembg = {
        name: _asset(sources / name, f"rembg-{name}", name.encode())
        for name in (
            "model.safetensors",
            "config.json",
            "birefnet.py",
            "BiRefNet_config.py",
        )
    }
    dinov3 = _dinov3_assets(sources)

    def record(asset):
        return {
            "source": str(asset.source),
            "coordinate": asset.coordinate,
            "sha256": asset.sha256,
            "size_bytes": asset.size_bytes,
        }

    path = tmp_path / "attempt.json"
    path.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.native_image_to_glb_attempt_spec.v2",
                "run_id": "31fce6b7-853b-4a0f-b99d-518be23ebabc",
                "dataset_id": "operator/native-image-r5-inputs",
                "kernel_id": "operator/native-image-r5-cuda",
                "title": "Native Image R5 CUDA",
                "capsule_dir": str(tmp_path / "capsule"),
                "output_dir": str(tmp_path / "packet"),
                "entrypoint": record(assets["entrypoint"]),
                "authority_helper": record(assets["authority_helper"]),
                "image": record(assets["image"]),
                "dinov3_files": {
                    name: record(asset) for name, asset in dinov3.items()
                },
                "rembg_files": {name: record(asset) for name, asset in rembg.items()},
                "expected_outputs": ["12-consumer_glb.glb"],
                "output_coordinate": "outputs",
                "work_coordinate": "runtime",
                "accelerator": "NvidiaTeslaT4",
                "enable_internet": True,
            }
        )
    )

    spec = load_attempt_spec(path)

    assert spec.run_id == "31fce6b7-853b-4a0f-b99d-518be23ebabc"
    assert spec.entrypoint == assets["entrypoint"]
    assert spec.dinov3_files == dinov3
    assert spec.rembg_files == rembg


def test_attempt_spec_rejects_unknown_authority_fields(tmp_path):
    path = tmp_path / "attempt.json"
    path.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.native_image_to_glb_attempt_spec.v2",
                "unexpected": "silently ignored fields are not authority",
            }
        )
    )

    with pytest.raises(AttemptSpecError, match="field|schema"):
        load_attempt_spec(path)


def test_structured_attempt_composes_with_native_packet_preparation(
    tmp_path,
    monkeypatch,
):
    import trellmlx.witness_authority as authority
    from scripts import source_cuda_native_image_to_glb_witness as witness

    sources = tmp_path / "sources"
    sources.mkdir()
    rembg = {
        name: _asset(sources / name, coordinate, name.encode())
        for name, coordinate in {
            "model.safetensors": "rembg-model.safetensors",
            "config.json": "rembg-config.json",
            "birefnet.py": "rembg-birefnet.py",
            "BiRefNet_config.py": "rembg-BiRefNet-config.py",
        }.items()
    }
    dinov3 = _dinov3_assets(sources)
    monkeypatch.setattr(
        witness,
        "DINOV3_FILES",
        {name: asset.sha256 for name, asset in dinov3.items()},
    )
    monkeypatch.setattr(
        witness,
        "REMBG_FILES",
        {name: asset.sha256 for name, asset in rembg.items()},
    )
    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/native-image-r5-inputs",
        kernel_id="operator/native-image-r5-cuda",
        title="Native Image R5 CUDA",
        capsule_dir=tmp_path / "capsule",
        output_dir=tmp_path / "packet",
        entrypoint=_asset(
            sources / "source_cuda_native_image_to_glb_witness.py",
            "source_cuda_native_image_to_glb_witness.py",
            Path(witness.__file__).read_bytes(),
        ),
        authority_helper=_asset(
            sources / "witness_authority.py",
            "witness_authority.py",
            Path(authority.__file__).read_bytes(),
        ),
        image=_asset(sources / "9_img.png", "9_img.png", b"image"),
        dinov3_files=dinov3,
        rembg_files=rembg,
        expected_outputs=("12-consumer_glb.glb",),
    )

    packet = witness.prepare_native_image_to_glb_packet(build_attempt_packet(spec))

    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    assert "native-image-to-glb-attempt.json" in manifest["files"]
    assert manifest["files"]["witness_authority.py"]["sha256"] == spec.authority_helper.sha256
    assert manifest["run_id"] == spec.run_id


def test_native_preparer_rejects_semantically_substituted_attempt_manifest(
    tmp_path,
    monkeypatch,
):
    import trellmlx.witness_authority as authority
    from scripts import source_cuda_native_image_to_glb_witness as witness

    sources = tmp_path / "sources"
    sources.mkdir()
    rembg = {
        name: _asset(sources / name, f"rembg-{name}", name.encode())
        for name in witness.REMBG_FILES
    }
    dinov3 = _dinov3_assets(sources)
    monkeypatch.setattr(
        witness,
        "DINOV3_FILES",
        {name: asset.sha256 for name, asset in dinov3.items()},
    )
    monkeypatch.setattr(
        witness,
        "REMBG_FILES",
        {name: asset.sha256 for name, asset in rembg.items()},
    )
    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/native-image-r5-inputs",
        kernel_id="operator/native-image-r5-cuda",
        title="Native Image R5 CUDA",
        capsule_dir=tmp_path / "capsule",
        output_dir=tmp_path / "packet",
        entrypoint=_asset(
            sources / "entrypoint.py",
            "source_cuda_native_image_to_glb_witness.py",
            Path(witness.__file__).read_bytes(),
        ),
        authority_helper=_asset(
            sources / "authority.py",
            "witness_authority.py",
            Path(authority.__file__).read_bytes(),
        ),
        image=_asset(sources / "image.png", "image.png", b"image"),
        dinov3_files=dinov3,
        rembg_files=rembg,
        expected_outputs=("12-consumer_glb.glb",),
    )
    packet = build_attempt_packet(spec)
    manifest_path = spec.capsule_dir / "native-image-to-glb-attempt.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["run_id"] = "22222222-2222-4222-8222-222222222222"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="attempt manifest|run identity"):
        witness.prepare_native_image_to_glb_packet(packet)


def test_generated_cuda_runner_consumes_attempt_manifest_before_execution(
    tmp_path,
    monkeypatch,
):
    from trellmlx.kaggle_cuda_witness import prepare_packet

    sources = tmp_path / "sources"
    sources.mkdir()
    rembg = {
        name: _asset(sources / name, f"rembg-{name}", name.encode())
        for name in (
            "model.safetensors",
            "config.json",
            "birefnet.py",
            "BiRefNet_config.py",
        )
    }
    dinov3 = _dinov3_assets(sources)
    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/attempt-runner-inputs",
        kernel_id="operator/attempt-runner-cuda",
        title="Attempt Runner CUDA",
        capsule_dir=tmp_path / "capsule",
        output_dir=tmp_path / "packet",
        entrypoint=_asset(sources / "entrypoint.py", "entrypoint.py", b"print('must not run')\n"),
        authority_helper=_asset(
            sources / "authority.py", "witness_authority.py", b"authority"
        ),
        image=_asset(sources / "image.png", "image.png", b"image"),
        dinov3_files=dinov3,
        rembg_files=rembg,
        expected_outputs=("12-consumer_glb.glb",),
    )
    packet = prepare_packet(build_attempt_packet(spec))
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
    namespace["CONFIG"]["attempt_manifest"]["run_id"] = (
        "22222222-2222-4222-8222-222222222222"
    )
    monkeypatch.setattr(
        namespace["subprocess"],
        "run",
        lambda *args, **kwargs: pytest.fail("entrypoint executed before attempt admission"),
    )

    assert namespace["main"]() == 7
    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert receipt["failure_phase"] == "attempt_manifest"
    assert "semantic payload mismatch" in receipt["message"]


def test_generated_cuda_runner_admits_untampered_v2_attempt_before_execution(
    tmp_path,
    monkeypatch,
):
    from trellmlx.kaggle_cuda_witness import prepare_packet

    sources = tmp_path / "sources"
    sources.mkdir()
    rembg = {
        name: _asset(sources / name, f"rembg-{name}", name.encode())
        for name in (
            "model.safetensors",
            "config.json",
            "birefnet.py",
            "BiRefNet_config.py",
        )
    }
    dinov3 = _dinov3_assets(sources)
    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/attempt-runner-v2-inputs",
        kernel_id="operator/attempt-runner-v2-cuda",
        title="Attempt Runner V2 CUDA",
        capsule_dir=tmp_path / "capsule",
        output_dir=tmp_path / "packet",
        entrypoint=_asset(
            sources / "entrypoint.py", "entrypoint.py", b"print('must run')\n"
        ),
        authority_helper=_asset(
            sources / "authority.py", "witness_authority.py", b"authority"
        ),
        image=_asset(sources / "image.png", "image.png", b"image"),
        dinov3_files=dinov3,
        rembg_files=rembg,
        expected_outputs=("12-consumer_glb.glb",),
    )
    packet = prepare_packet(build_attempt_packet(spec))
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
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return types.SimpleNamespace(stdout="", stderr="expected stop", returncode=23)

    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)

    assert namespace["main"]() == 23
    assert len(calls) == 1
    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert receipt["failure_phase"] == "execution"


def test_native_preparer_rejects_substituted_dinov3_before_packet_copy(
    tmp_path,
    monkeypatch,
):
    import trellmlx.witness_authority as authority
    from scripts import source_cuda_native_image_to_glb_witness as witness

    sources = tmp_path / "sources"
    sources.mkdir()
    rembg = {
        name: _asset(sources / name, f"rembg-{name}", name.encode())
        for name in witness.REMBG_FILES
    }
    monkeypatch.setattr(
        witness,
        "REMBG_FILES",
        {name: asset.sha256 for name, asset in rembg.items()},
    )
    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/substituted-dinov3-inputs",
        kernel_id="operator/substituted-dinov3-cuda",
        title="Substituted DINOv3 CUDA",
        capsule_dir=tmp_path / "capsule",
        output_dir=tmp_path / "packet",
        entrypoint=_asset(
            sources / "entrypoint.py",
            "source_cuda_native_image_to_glb_witness.py",
            Path(witness.__file__).read_bytes(),
        ),
        authority_helper=_asset(
            sources / "authority.py",
            "witness_authority.py",
            Path(authority.__file__).read_bytes(),
        ),
        image=_asset(sources / "image.png", "image.png", b"image"),
        dinov3_files=_dinov3_assets(sources),
        rembg_files=rembg,
        expected_outputs=("12-consumer_glb.glb",),
    )

    with pytest.raises(ValueError, match="DINOv3.*SHA256 mismatch"):
        witness.prepare_native_image_to_glb_packet(build_attempt_packet(spec))


def test_native_preparer_rejects_attempt_manifest_drift_during_packet_copy(
    tmp_path,
    monkeypatch,
):
    import trellmlx.kaggle_cuda_witness as kaggle_witness
    import trellmlx.witness_authority as authority
    from scripts import source_cuda_native_image_to_glb_witness as witness

    sources = tmp_path / "sources"
    sources.mkdir()
    rembg = {
        name: _asset(sources / name, f"rembg-{name}", name.encode())
        for name in witness.REMBG_FILES
    }
    dinov3 = _dinov3_assets(sources)
    monkeypatch.setattr(
        witness,
        "DINOV3_FILES",
        {name: asset.sha256 for name, asset in dinov3.items()},
    )
    monkeypatch.setattr(
        witness,
        "REMBG_FILES",
        {name: asset.sha256 for name, asset in rembg.items()},
    )
    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/attempt-drift-inputs",
        kernel_id="operator/attempt-drift-cuda",
        title="Attempt Drift CUDA",
        capsule_dir=tmp_path / "capsule",
        output_dir=tmp_path / "packet",
        entrypoint=_asset(
            sources / "entrypoint.py",
            "source_cuda_native_image_to_glb_witness.py",
            Path(witness.__file__).read_bytes(),
        ),
        authority_helper=_asset(
            sources / "authority.py",
            "witness_authority.py",
            Path(authority.__file__).read_bytes(),
        ),
        image=_asset(sources / "image.png", "image.png", b"image"),
        dinov3_files=dinov3,
        rembg_files=rembg,
        expected_outputs=("12-consumer_glb.glb",),
    )
    packet = build_attempt_packet(spec)
    spec.output_dir.mkdir()
    marker = spec.output_dir / "preserve-me.txt"
    marker.write_text("prior output")
    original_prepare = kaggle_witness.prepare_packet

    def mutate_then_prepare(candidate):
        manifest_path = candidate.capsule_dir / "native-image-to-glb-attempt.json"
        payload = json.loads(manifest_path.read_text())
        payload["output_coordinate"] = "substituted-output"
        manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        return original_prepare(candidate)

    monkeypatch.setattr(kaggle_witness, "prepare_packet", mutate_then_prepare)

    with pytest.raises(ValueError, match="attempt manifest|authority changed"):
        witness.prepare_native_image_to_glb_packet(packet)

    assert marker.read_text() == "prior output"


@pytest.mark.parametrize(
    "topology",
    ("equal", "output_under_capsule", "capsule_under_output", "source_under_output"),
)
def test_attempt_builder_rejects_destructive_local_topology(tmp_path, topology):
    sources = tmp_path / "sources"
    sources.mkdir()
    capsule = tmp_path / "capsule"
    output = tmp_path / "packet"
    if topology == "equal":
        output = capsule
    elif topology == "output_under_capsule":
        output = capsule / "packet"
    elif topology == "capsule_under_output":
        capsule = output / "capsule"
    elif topology == "source_under_output":
        sources = output / "sources"
        sources.mkdir(parents=True)
    rembg = {
        name: _asset(sources / name, f"rembg-{name}", name.encode())
        for name in (
            "model.safetensors",
            "config.json",
            "birefnet.py",
            "BiRefNet_config.py",
        )
    }
    dinov3 = _dinov3_assets(sources)
    spec = NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/native-image-r5-inputs",
        kernel_id="operator/native-image-r5-cuda",
        title="Native Image R5 CUDA",
        capsule_dir=capsule,
        output_dir=output,
        entrypoint=_asset(sources / "entrypoint.py", "entrypoint.py", b"entry"),
        authority_helper=_asset(
            sources / "authority.py", "witness_authority.py", b"authority"
        ),
        image=_asset(sources / "image.png", "image.png", b"image"),
        dinov3_files=dinov3,
        rembg_files=rembg,
        expected_outputs=("12-consumer_glb.glb",),
    )
    marker_root = capsule if capsule.exists() else output
    marker_root.mkdir(parents=True, exist_ok=True)
    marker = marker_root / "preserve-me.txt"
    marker.write_text("preserved")

    with pytest.raises(AttemptSpecError, match="overlap|managed|topology"):
        build_attempt_packet(spec)

    assert marker.read_text() == "preserved"

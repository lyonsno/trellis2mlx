from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from trellmlx.native_image_to_glb_attempt import (
    AttemptAsset,
    AttemptSpecError,
    CAPTURE_PROFILE_OUTPUTS,
    NativeImageToGLBAttemptSpec,
    build_attempt_packet,
    capture_contract_from_entrypoint_args,
    load_attempt_manifest,
    load_attempt_spec,
    validate_attempt_manifest,
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


def _review_attempt_spec(tmp_path: Path) -> NativeImageToGLBAttemptSpec:
    sources = tmp_path / "review-sources"
    sources.mkdir()
    return NativeImageToGLBAttemptSpec(
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        dataset_id="operator/native-image-review-inputs",
        kernel_id="operator/native-image-review-cuda",
        title="Native Image Review CUDA",
        capsule_dir=tmp_path / "review-capsule",
        output_dir=tmp_path / "review-packet",
        entrypoint=_asset(sources / "entrypoint.py", "entrypoint.py", b"entry"),
        authority_helper=_asset(
            sources / "authority.py", "witness_authority.py", b"authority"
        ),
        image=_asset(sources / "image.png", "image.png", b"image"),
        dinov3_files=_dinov3_assets(sources),
        rembg_files={
            name: _asset(
                sources / f"rembg-{name}",
                f"rembg-{name}",
                name.encode(),
            )
            for name in (
                "model.safetensors",
                "config.json",
                "birefnet.py",
                "BiRefNet_config.py",
            )
        },
        expected_outputs=("12-consumer_glb.glb",),
    )


def test_joined_explicit_full_capture_profile_does_not_downgrade_to_v2():
    with pytest.raises(AttemptSpecError, match="explicit full capture profile"):
        capture_contract_from_entrypoint_args(
            ("--capture-profile=full",),
            ("legacy.glb",),
            context="attempt packet",
        )


def test_joined_final_consumer_capture_profile_is_explicit_and_bound():
    contract = capture_contract_from_entrypoint_args(
        ("--capture-profile=final-consumer",),
        CAPTURE_PROFILE_OUTPUTS["final-consumer"],
        context="attempt packet",
    )

    assert contract.capture_profile == "final-consumer"
    assert contract.profile_is_explicit is True
    assert contract.profile_binds_outputs is True


@pytest.mark.parametrize(
    "arguments",
    (
        (
            "--capture-profile=final-consumer",
            "--capture-profile",
            "final-consumer",
        ),
        (
            "--capture-profile=final-consumer",
            "--capture-profile=final-consumer",
        ),
    ),
)
def test_mixed_or_duplicate_capture_profile_declarations_are_ambiguous(arguments):
    with pytest.raises(AttemptSpecError, match="capture profile is ambiguous"):
        capture_contract_from_entrypoint_args(
            arguments,
            CAPTURE_PROFILE_OUTPUTS["final-consumer"],
            context="attempt packet",
        )


def test_structured_attempt_rejects_escaping_output_coordinate_before_mutation(
    tmp_path,
):
    spec = replace(_review_attempt_spec(tmp_path), output_coordinate="../escape")
    spec.capsule_dir.mkdir()
    marker = spec.capsule_dir / "preserve-me.txt"
    marker.write_text("preserved")

    with pytest.raises(AttemptSpecError, match="output coordinate"):
        build_attempt_packet(spec)

    assert marker.read_text() == "preserved"


def test_structured_attempt_rejects_output_publication_alias_graph_before_mutation(
    tmp_path,
):
    spec = replace(
        _review_attempt_spec(tmp_path),
        expected_outputs=("outputs/a.bin", "a.bin"),
    )
    spec.capsule_dir.mkdir()
    marker = spec.capsule_dir / "preserve-me.txt"
    marker.write_text("preserved")

    with pytest.raises(AttemptSpecError, match="expected output"):
        build_attempt_packet(spec)

    assert marker.read_text() == "preserved"


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


def test_structured_attempt_binds_read_only_model_kernel_source(tmp_path):
    spec = replace(
        _review_attempt_spec(tmp_path),
        capture_profile="final-consumer",
        expected_outputs=CAPTURE_PROFILE_OUTPUTS["final-consumer"],
        model_kernel_source="operator/pinned-model-output",
    )

    packet = build_attempt_packet(spec)
    manifest = json.loads(
        (spec.capsule_dir / "native-image-to-glb-attempt.json").read_text()
    )

    assert manifest["schema"] == "trellis2mlx.native_image_to_glb_attempt.v4"
    assert manifest["model_kernel_source"] == "operator/pinned-model-output"
    assert packet.kernel_sources == ("operator/pinned-model-output",)
    assert "--model-blob-root" not in packet.entrypoint_args
    assert "--model-source-kernel" not in packet.entrypoint_args


def test_v5_full_capture_binds_feature_request_settings(tmp_path):
    from trellmlx.kaggle_cuda_witness import prepare_packet

    spec = replace(
        _review_attempt_spec(tmp_path),
        expected_outputs=CAPTURE_PROFILE_OUTPUTS["full"],
        model_kernel_source="operator/pinned-model-output",
        pipeline_type="512",
        seed=81414,
        steps=8,
        target_faces=100000,
        texture_size=512,
        request_settings_bound=True,
    )

    packet = build_attempt_packet(spec)
    manifest_path = spec.capsule_dir / "native-image-to-glb-attempt.json"
    manifest = load_attempt_manifest(manifest_path)
    validate_attempt_manifest(packet, manifest)
    prepared = prepare_packet(packet)
    runner = (prepared.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    namespace = {"__name__": "runner_test"}
    exec(runner, namespace)

    assert manifest["schema"] == "trellis2mlx.native_image_to_glb_attempt.v5"
    assert manifest["capture_profile"] == "full"
    assert {
        field: manifest[field]
        for field in (
            "pipeline_type",
            "seed",
            "steps",
            "target_faces",
            "texture_size",
        )
    } == {
        "pipeline_type": "512",
        "seed": 81414,
        "steps": 8,
        "target_faces": 100000,
        "texture_size": 512,
    }
    assert "--capture-profile" not in packet.entrypoint_args
    assert namespace["CONFIG"]["attempt_contract"] == manifest


def test_v5_manifest_rejects_request_setting_substitution(tmp_path):
    spec = replace(
        _review_attempt_spec(tmp_path),
        expected_outputs=CAPTURE_PROFILE_OUTPUTS["full"],
        model_kernel_source="operator/pinned-model-output",
        seed=81414,
        target_faces=100000,
        texture_size=512,
        request_settings_bound=True,
    )
    packet = build_attempt_packet(spec)
    manifest = load_attempt_manifest(
        spec.capsule_dir / "native-image-to-glb-attempt.json"
    )
    manifest["seed"] = 81415

    with pytest.raises(AttemptSpecError, match="seed request setting mismatch"):
        validate_attempt_manifest(packet, manifest)


def test_v5_runner_rejects_partially_bound_request_settings(tmp_path):
    from trellmlx.kaggle_cuda_witness import prepare_packet

    spec = replace(
        _review_attempt_spec(tmp_path),
        expected_outputs=CAPTURE_PROFILE_OUTPUTS["full"],
        model_kernel_source="operator/pinned-model-output",
        seed=81414,
        target_faces=100000,
        texture_size=512,
        request_settings_bound=True,
    )
    packet = build_attempt_packet(spec)
    texture_index = packet.entrypoint_args.index("--texture-size")
    packet = replace(
        packet,
        entrypoint_args=(
            packet.entrypoint_args[:texture_index]
            + packet.entrypoint_args[texture_index + 2 :]
        ),
    )

    with pytest.raises(ValueError, match="partially bound"):
        prepare_packet(packet)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("seed", -1, "seed"),
        ("steps", 7, "steps"),
        ("target_faces", 0, "target_faces"),
        ("texture_size", 513, "texture_size"),
    ),
)
def test_v5_request_settings_reject_invalid_values(tmp_path, field, value, message):
    spec = replace(
        _review_attempt_spec(tmp_path),
        expected_outputs=CAPTURE_PROFILE_OUTPUTS["full"],
        model_kernel_source="operator/pinned-model-output",
        request_settings_bound=True,
        **{field: value},
    )

    with pytest.raises(AttemptSpecError, match=message):
        build_attempt_packet(spec)


def test_non_default_request_settings_require_v5_binding(tmp_path):
    spec = replace(_review_attempt_spec(tmp_path), seed=81414)

    with pytest.raises(AttemptSpecError, match="v5-bound"):
        build_attempt_packet(spec)


def test_model_kernel_source_runner_rejects_ambiguous_mount_and_admits_one(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import MODEL_SOURCE_MARKER
    from trellmlx.kaggle_cuda_witness import prepare_packet

    spec = replace(
        _review_attempt_spec(tmp_path),
        capture_profile="final-consumer",
        expected_outputs=CAPTURE_PROFILE_OUTPUTS["final-consumer"],
        model_kernel_source="operator/pinned-model-output",
    )
    packet = prepare_packet(build_attempt_packet(spec))
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    namespace = {"__name__": "runner_test"}
    exec(runner, namespace)
    marker = Path(namespace["CONFIG"]["model_source_marker"])
    assert marker.as_posix() == MODEL_SOURCE_MARKER
    mount_a = tmp_path / "kaggle-input" / "mount-a"
    mount_b = tmp_path / "kaggle-input" / "mount-b"
    for mount in (mount_a, mount_b):
        source = mount / marker
        source.parent.mkdir(parents=True)
        source.write_text("pinned pipeline")
    namespace["KAGGLE_INPUT_ROOT"] = tmp_path / "kaggle-input"

    with pytest.raises(RuntimeError, match="missing or ambiguous"):
        namespace["find_model_blob_root"]()

    (mount_b / marker).unlink()
    mounted = namespace["find_model_blob_root"]()
    assert mounted["requested_kernel_source"] == "operator/pinned-model-output"
    assert mounted["effective_mount_root"] == str(mount_a)
    assert mounted["effective_blob_root"] == str(
        mount_a / "runtime" / "huggingface"
    )


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


def test_attempt_spec_loads_v5_feature_request_contract(tmp_path):
    original = replace(
        _review_attempt_spec(tmp_path),
        expected_outputs=CAPTURE_PROFILE_OUTPUTS["full"],
        model_kernel_source="operator/pinned-model-output",
        pipeline_type="512",
        seed=81414,
        steps=8,
        target_faces=100000,
        texture_size=512,
        request_settings_bound=True,
    )

    def record(asset):
        return {
            "source": str(asset.source),
            "coordinate": asset.coordinate,
            "sha256": asset.sha256,
            "size_bytes": asset.size_bytes,
        }

    path = tmp_path / "feature-attempt.json"
    path.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.native_image_to_glb_attempt_spec.v5",
                "run_id": original.run_id,
                "dataset_id": original.dataset_id,
                "kernel_id": original.kernel_id,
                "title": original.title,
                "capsule_dir": str(original.capsule_dir),
                "output_dir": str(original.output_dir),
                "entrypoint": record(original.entrypoint),
                "authority_helper": record(original.authority_helper),
                "image": record(original.image),
                "dinov3_files": {
                    name: record(asset)
                    for name, asset in original.dinov3_files.items()
                },
                "rembg_files": {
                    name: record(asset)
                    for name, asset in original.rembg_files.items()
                },
                "expected_outputs": list(original.expected_outputs),
                "capture_profile": original.capture_profile,
                "model_kernel_source": original.model_kernel_source,
                "pipeline_type": original.pipeline_type,
                "seed": original.seed,
                "steps": original.steps,
                "target_faces": original.target_faces,
                "texture_size": original.texture_size,
                "output_coordinate": original.output_coordinate,
                "work_coordinate": original.work_coordinate,
                "accelerator": original.accelerator,
                "enable_internet": original.enable_internet,
            }
        )
    )

    loaded = load_attempt_spec(path)

    assert loaded.request_settings_bound is True
    assert loaded.capture_profile == "full"
    assert loaded.model_kernel_source == "operator/pinned-model-output"
    assert (
        loaded.pipeline_type,
        loaded.seed,
        loaded.steps,
        loaded.target_faces,
        loaded.texture_size,
    ) == ("512", 81414, 8, 100000, 512)
    packet = build_attempt_packet(loaded)
    validate_attempt_manifest(
        packet,
        load_attempt_manifest(loaded.capsule_dir / "native-image-to-glb-attempt.json"),
    )


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

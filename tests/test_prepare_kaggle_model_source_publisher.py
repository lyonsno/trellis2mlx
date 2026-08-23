from dataclasses import replace
from pathlib import Path

import pytest

from scripts.prepare_kaggle_model_source_publisher import (
    PUBLISHER_KERNEL_PREFIX,
    build_packet,
    canonical_repositories,
    validate_r11_publisher_authority,
)
from scripts.source_cuda_native_image_to_glb_witness import (
    MODEL_BLOB_MANIFEST,
    MODEL_PIPELINE_SHA256,
    MODEL_REPOSITORY,
    MODEL_REQUIRED_BYTES,
    MODEL_REVISION,
    MODEL_SOURCE_MARKER,
    SPARSE_DECODER_REPOSITORY,
    SPARSE_DECODER_REVISION,
)
from trellmlx.kaggle_model_source_publisher import ModelSourcePublisherError


def test_canonical_publisher_packet_matches_native_model_authority(tmp_path: Path):
    packet = build_packet(output_dir=tmp_path / "packet", attempt_id="a" * 32)

    assert packet.kernel_id == f"{PUBLISHER_KERNEL_PREFIX}-{packet.attempt_id[:8]}"
    assert packet.title == packet.kernel_id.split("/", 1)[1]
    assert packet.marker == MODEL_SOURCE_MARKER
    assert packet.marker_sha256 == MODEL_PIPELINE_SHA256
    assert packet.blob_count == 17
    assert packet.published_blob_bytes == MODEL_REQUIRED_BYTES == 14_967_470_615
    assert canonical_repositories() == packet.repositories
    assert [
        (repository.repo_id, repository.revision)
        for repository in packet.repositories
    ] == sorted(
        [
            (MODEL_REPOSITORY, MODEL_REVISION),
            (SPARSE_DECODER_REPOSITORY, SPARSE_DECODER_REVISION),
        ]
    )
    assert {
        repository.name: {
            file.coordinate: {
                "blob": file.blob,
                "sha256": file.sha256,
                "size_bytes": file.size_bytes,
            }
            for file in repository.files
        }
        for repository in packet.repositories
    } == {
        family: manifest["files"]
        for family, manifest in MODEL_BLOB_MANIFEST.items()
    }
    validate_r11_publisher_authority(packet)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kernel_id", "operator/wrong-source"),
        ("marker_sha256", "0" * 64),
        ("repositories", ()),
    ],
)
def test_r11_publisher_authority_rejects_route_or_payload_drift(
    tmp_path: Path,
    field: str,
    value: object,
):
    packet = build_packet(output_dir=tmp_path / "packet", attempt_id="b" * 32)
    drifted = replace(packet, **{field: value})

    with pytest.raises(ModelSourcePublisherError, match="R11 publisher authority mismatch"):
        validate_r11_publisher_authority(drifted)

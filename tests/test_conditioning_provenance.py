import hashlib
import json
from pathlib import Path

import numpy as np


def test_artifact_and_array_identities_are_content_addressed(tmp_path):
    import generate

    artifact = tmp_path / "conditioning.npz"
    artifact.write_bytes(b"conditioning-artifact")
    array = np.array([[1.0, 2.0]], dtype=np.float32)

    assert generate._artifact_identity(artifact) == {
        "path": str(artifact.resolve()),
        "sha256": hashlib.sha256(b"conditioning-artifact").hexdigest(),
        "size_bytes": len(b"conditioning-artifact"),
    }
    assert generate._array_identity(array) == {
        "shape": [1, 2],
        "dtype": "float32",
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def test_upstream_conditioning_route_loads_checkpoint_sidecar(tmp_path):
    import generate
    from trellmlx.checkpoint import save_checkpoint

    upstream = {"schema": "trellis2mlx-conditioning-v1", "route": "image"}
    save_checkpoint(
        str(tmp_path),
        "conditioning",
        cond=np.zeros((1, 2)),
        neg_cond=np.zeros((1, 2)),
        conditioning_route_json=json.dumps(upstream),
    )
    sample_path = tmp_path / "conditioning.npz"

    with np.load(sample_path, allow_pickle=False) as archive:
        assert generate._upstream_conditioning_route(sample_path, archive) == upstream


def test_generate_records_preprocess_and_conditioning_route_contract():
    source = Path(__file__).resolve().parents[1].joinpath("generate.py").read_text()

    assert "preprocess_image_with_provenance" in source
    assert "rembg_session = new_session(DEFAULT_BACKGROUND_MODEL)" in source
    assert 'conditioning_route["cond"] = _array_identity(cond_np)' in source
    assert 'conditioning_route["neg_cond"] = _array_identity(neg_cond_np)' in source
    assert "conditioning_route_json=json.dumps(" in source

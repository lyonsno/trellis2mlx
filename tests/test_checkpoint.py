"""Tests for checkpoint save/load/resume functionality.

Verifies:
- Round-trip: save → load produces identical arrays
- List-of-arrays (subdivision masks) survive serialization
- Metadata (scalars, strings) round-trips through JSON
- has_checkpoint detects both .npz and .json-only stages
- list_checkpoints finds all saved stages
- Edge cases: empty arrays, missing stages, overwrite
"""

import numpy as np
import pytest

from trellmlx.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    has_checkpoint,
    list_checkpoints,
)


@pytest.fixture
def ckpt_dir(tmp_path):
    return str(tmp_path / "checkpoints")


class TestSaveLoadRoundTrip:
    def test_arrays_survive_roundtrip(self, ckpt_dir):
        vertices = np.random.randn(1000, 3).astype(np.float32)
        faces = np.random.randint(0, 1000, (2000, 3)).astype(np.int64)

        save_checkpoint(ckpt_dir, "mesh_raw", vertices=vertices, faces=faces)
        loaded = load_checkpoint(ckpt_dir, "mesh_raw")

        np.testing.assert_array_equal(loaded["vertices"], vertices)
        np.testing.assert_array_equal(loaded["faces"], faces)

    def test_scalars_survive_roundtrip(self, ckpt_dir):
        save_checkpoint(ckpt_dir, "mesh_raw",
                        vertices=np.zeros((1, 3), dtype=np.float32),
                        mesh_grid_size=1024)
        loaded = load_checkpoint(ckpt_dir, "mesh_raw")

        assert loaded["mesh_grid_size"] == 1024

    def test_string_metadata_roundtrip(self, ckpt_dir):
        save_checkpoint(ckpt_dir, "info",
                        vertices=np.zeros((1, 3), dtype=np.float32),
                        image_path="/tmp/shoe.png")
        loaded = load_checkpoint(ckpt_dir, "info")

        assert loaded["image_path"] == "/tmp/shoe.png"

    def test_list_of_arrays_roundtrip(self, ckpt_dir):
        """Subdivision masks are stored as list-of-arrays."""
        masks = [
            np.array([True, False, True]),
            np.array([False, True, False, True]),
            np.array([True, True]),
        ]
        save_checkpoint(ckpt_dir, "shape_decode", shape_subs=masks)
        loaded = load_checkpoint(ckpt_dir, "shape_decode")

        assert len(loaded["shape_subs"]) == 3
        for orig, loaded_arr in zip(masks, loaded["shape_subs"]):
            np.testing.assert_array_equal(loaded_arr, orig)

    def test_list_of_array_like_values_roundtrip(self, ckpt_dir):
        """Array-like list entries should not leave an impossible count."""
        masks = [
            np.array([True, False, True]),
            [False, True, False, True],
            np.array([True, True]),
        ]
        save_checkpoint(ckpt_dir, "shape_decode", shape_subs=masks)
        loaded = load_checkpoint(ckpt_dir, "shape_decode")

        assert len(loaded["shape_subs"]) == 3
        for orig, loaded_arr in zip(masks, loaded["shape_subs"]):
            np.testing.assert_array_equal(loaded_arr, np.asarray(orig))

    def test_empty_array_roundtrip(self, ckpt_dir):
        empty = np.zeros((0, 3), dtype=np.float32)
        save_checkpoint(ckpt_dir, "empty_stage", data=empty)
        loaded = load_checkpoint(ckpt_dir, "empty_stage")

        assert loaded["data"].shape == (0, 3)
        assert loaded["data"].dtype == np.float32

    def test_overwrite_replaces_data(self, ckpt_dir):
        save_checkpoint(ckpt_dir, "stage",
                        data=np.array([1.0, 2.0, 3.0]))
        save_checkpoint(ckpt_dir, "stage",
                        data=np.array([4.0, 5.0]))
        loaded = load_checkpoint(ckpt_dir, "stage")

        np.testing.assert_array_equal(loaded["data"], [4.0, 5.0])

    def test_array_only_overwrite_removes_stale_metadata(self, ckpt_dir):
        import os
        save_checkpoint(ckpt_dir, "stage",
                        data=np.array([1.0, 2.0, 3.0]),
                        mesh_grid_size=1024)
        save_checkpoint(ckpt_dir, "stage",
                        data=np.array([4.0, 5.0]))
        loaded = load_checkpoint(ckpt_dir, "stage")

        np.testing.assert_array_equal(loaded["data"], [4.0, 5.0])
        assert "mesh_grid_size" not in loaded
        assert not os.path.exists(os.path.join(ckpt_dir, "stage.json"))


class TestHasCheckpoint:
    def test_detects_npz(self, ckpt_dir):
        assert not has_checkpoint(ckpt_dir, "mesh_raw")
        save_checkpoint(ckpt_dir, "mesh_raw",
                        vertices=np.zeros((1, 3), dtype=np.float32))
        assert has_checkpoint(ckpt_dir, "mesh_raw")

    def test_detects_json_only(self, ckpt_dir):
        """A stage with only scalar metadata (no arrays) should still be detected."""
        import json, os
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, "meta_only.json"), "w") as f:
            json.dump({"grid_size": 512}, f)

        assert has_checkpoint(ckpt_dir, "meta_only")

    def test_missing_stage(self, ckpt_dir):
        assert not has_checkpoint(ckpt_dir, "nonexistent")

    def test_missing_dir(self):
        assert not has_checkpoint("/tmp/no_such_dir_xyz", "anything")


class TestListCheckpoints:
    def test_lists_saved_stages(self, ckpt_dir):
        save_checkpoint(ckpt_dir, "mesh_raw",
                        data=np.zeros((1,), dtype=np.float32))
        save_checkpoint(ckpt_dir, "texture",
                        data=np.zeros((1,), dtype=np.float32))
        stages = list_checkpoints(ckpt_dir)

        assert "mesh_raw" in stages
        assert "texture" in stages

    def test_empty_dir(self, tmp_path):
        d = str(tmp_path / "empty")
        assert list_checkpoints(d) == []

    def test_lists_json_only_stages(self, ckpt_dir):
        save_checkpoint(ckpt_dir, "meta_only", mesh_grid_size=512)
        stages = list_checkpoints(ckpt_dir)

        assert "meta_only" in stages

    def test_ignores_non_npz_files(self, ckpt_dir):
        save_checkpoint(ckpt_dir, "real",
                        data=np.zeros((1,), dtype=np.float32))
        # Write a stray file
        import os
        with open(os.path.join(ckpt_dir, "notes.txt"), "w") as f:
            f.write("hello")

        stages = list_checkpoints(ckpt_dir)
        assert "real" in stages
        assert "notes" not in stages


class TestCheckpointSizes:
    """Verify checkpoint sizes are reasonable for pipeline data."""

    def test_mesh_checkpoint_size(self, ckpt_dir):
        """200K-face mesh should produce a checkpoint < 10 MB."""
        vertices = np.random.randn(100_000, 3).astype(np.float32)
        faces = np.random.randint(0, 100_000, (200_000, 3)).astype(np.int64)

        save_checkpoint(ckpt_dir, "mesh_raw",
                        vertices=vertices, faces=faces, mesh_grid_size=1024)

        import os
        npz_size = os.path.getsize(os.path.join(ckpt_dir, "mesh_raw.npz"))
        assert npz_size < 10_000_000  # < 10 MB compressed

    def test_texture_checkpoint_size(self, ckpt_dir):
        """1M-voxel texture data should produce a checkpoint < 50 MB."""
        tex_np = np.random.randn(1_000_000, 6).astype(np.float32)
        tex_coords = np.random.randint(0, 1024, (1_000_000, 3)).astype(np.int32)

        save_checkpoint(ckpt_dir, "texture",
                        tex_np=tex_np, tex_coords_spatial=tex_coords,
                        mesh_grid_size=1024)

        import os
        npz_size = os.path.getsize(os.path.join(ckpt_dir, "texture.npz"))
        assert npz_size < 50_000_000  # < 50 MB compressed

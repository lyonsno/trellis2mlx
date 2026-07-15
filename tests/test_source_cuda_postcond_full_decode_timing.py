import pytest


def _write_shape_slat_grid_fixture(tmp_path, *, points=None):
    import hashlib
    import json
    from pathlib import Path

    import numpy as np

    if points is None:
        points = [
            ("alpha-1_beta-1", 1.0, 1.0),
            ("alpha-0_beta-0", 0.0, 0.0),
            ("alpha-1_beta-0p5", 1.0, 0.5),
            ("alpha-0p5_beta-0", 0.5, 0.0),
        ]
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    arrays = {"coords": coords}
    point_rows = []
    for index, (coordinate_key, alpha, beta) in enumerate(points):
        output_key = f"point_{coordinate_key}_shape_slat"
        values = np.full((2, 32), index + 0.25, dtype=np.float32)
        arrays[output_key] = values
        point_rows.append(
            {
                "coordinate": {"alpha": alpha, "beta": beta},
                "coordinate_key": coordinate_key,
                "output_key": output_key,
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "shape": [2, 32],
                "vs_source_control": {"exact": coordinate_key == "alpha-1_beta-1"},
            }
        )

    grid = Path(tmp_path) / "cuda_result.npz"
    np.savez(grid, **arrays)
    grid_sha = hashlib.sha256(grid.read_bytes()).hexdigest()
    source_report = Path(tmp_path) / "cuda_result.json"
    source_report.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.source_cuda_shape_block29_basin_map.v1",
                "status": "done",
                "effective_route": {
                    "route": "official-source-cuda-full-eight-step-shape-flow-with-fixed-block29-endpoints",
                    "device_type": "cuda",
                    "cuda_device": "Tesla T4",
                    "attention_backend": "sdpa",
                    "conv_backend": "none",
                    "block_index": 29,
                    "step_index": 0,
                    "steps": 8,
                    "one_model_load": True,
                    "endpoint_semantics": "current + scale * (source - current)",
                },
                "primary_output": {
                    "path": grid.name,
                    "sha256": grid_sha,
                    "size_bytes": grid.stat().st_size,
                    "keys": sorted(arrays),
                },
                "source_control": {
                    "coordinate": {"alpha": 1.0, "beta": 1.0},
                    "exact": True,
                },
                "points": point_rows,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return grid, source_report, [row[0] for row in points]


def _shape_slat_decode_args(tmp_path, grid, source_report, point_names):
    return [
        "--output-json",
        str(tmp_path / "decode-report.json"),
        "--shape-slat-grid",
        str(grid),
        "--shape-slat-grid-report",
        str(source_report),
        "--output-dir",
        str(tmp_path / "meshes"),
        "--no-download",
        *[
            item
            for point_name in point_names
            for item in ("--shape-slat-point", point_name)
        ],
    ]


def test_shape_slat_grid_decode_preflight_invalidates_stale_outputs(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale_raw = output_dir / "alpha-1_beta-1.raw.ply"
    stale_filled = output_dir / "alpha-1_beta-1.filled.ply"
    stale_raw.write_bytes(b"stale raw")
    stale_filled.write_bytes(b"stale filled")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, point_names))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 0
    assert report["schema"] == "trellis2mlx.source_cuda_shape_slat_grid_decode.v1"
    assert report["status"] == "preflight_stopped"
    assert report["failure_phase"] is None
    assert report["selected_point_names"] == point_names
    assert report["source_basin_route"]["route"].startswith("official-source-cuda-")
    assert report["effective_route"]["route"] == "official-source-cuda-shape-slat-decoder"
    assert report["expected_artifact_count"] == 8
    assert report["written_artifact_count"] == 0
    assert {row["status"] for row in report["mesh_artifacts"]} == {"not_written_no_download"}
    assert not stale_raw.exists()
    assert not stale_filled.exists()


def test_shape_slat_grid_decode_rejects_duplicate_selection_with_report(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    args = _shape_slat_decode_args(tmp_path, grid, source_report, [point_names[0], point_names[0]])

    rc = main(args)

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert "duplicate --shape-slat-point" in report["error"]


def test_shape_slat_grid_decode_rejects_primary_digest_mismatch(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    payload = json.loads(source_report.read_text())
    payload["primary_output"]["sha256"] = "0" * 64
    source_report.write_text(json.dumps(payload) + "\n")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, point_names))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    assert "primary output digest mismatch" in report["error"]


def test_shape_slat_grid_decode_rejects_selected_array_digest_mismatch_and_stale_output(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    payload = json.loads(source_report.read_text())
    payload["points"][0]["sha256"] = "f" * 64
    source_report.write_text(json.dumps(payload) + "\n")
    output_dir = tmp_path / "meshes"
    output_dir.mkdir()
    stale = output_dir / f"{point_names[0]}.raw.ply"
    stale.write_bytes(b"stale")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, point_names))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    assert "selected array digest mismatch" in report["error"]
    assert not stale.exists()


def test_shape_slat_grid_decode_report_collision_uses_durable_fallback(tmp_path):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    original = source_report.read_bytes()
    args = _shape_slat_decode_args(tmp_path, grid, source_report, point_names)
    args[1] = str(source_report)

    rc = main(args)

    fallback = source_report.with_name(f"{source_report.name}.selective-decode-failure.json")
    report = json.loads(fallback.read_text())
    assert rc == 1
    assert source_report.read_bytes() == original
    assert report["failure_phase"] == "request_validation"
    assert report["requested_output_json"] == str(source_report)
    assert report["effective_output_json"] == str(fallback)
    assert "collides with protected input" in report["error"]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload: payload["points"][0].update(
                {"coordinate": {"alpha": 0.0, "beta": 0.0}}
            ),
            "coordinate mismatch for 'alpha-1_beta-1'",
        ),
        (
            lambda payload: payload["source_control"].update(
                {"coordinate": {"alpha": 0.0, "beta": 0.0}}
            ),
            "source control coordinate mismatch",
        ),
        (
            lambda payload: payload["primary_output"].update({"keys": ["coords"]}),
            "primary output key list omits selected array",
        ),
        (
            lambda payload: payload["effective_route"].pop("cuda_device"),
            "source basin route is missing cuda_device",
        ),
    ],
)
def test_shape_slat_grid_decode_rejects_semantic_route_or_label_contradiction(
    tmp_path,
    mutation,
    error,
):
    import json

    from scripts.source_cuda_postcond_full_decode_timing import main

    grid, source_report, point_names = _write_shape_slat_grid_fixture(tmp_path)
    payload = json.loads(source_report.read_text())
    mutation(payload)
    source_report.write_text(json.dumps(payload) + "\n")

    rc = main(_shape_slat_decode_args(tmp_path, grid, source_report, point_names))

    report = json.loads((tmp_path / "decode-report.json").read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    assert error in report["error"]


def test_required_model_names_for_512_postcond_decode():
    from scripts.source_cuda_postcond_full_decode_timing import required_model_names

    assert required_model_names("512") == (
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "shape_slat_decoder",
        "shape_slat_flow_model_512",
        "tex_slat_decoder",
        "tex_slat_flow_model_512",
    )


def test_required_model_names_for_1024_cascade_postcond_decode():
    from scripts.source_cuda_postcond_full_decode_timing import required_model_names

    assert required_model_names("1024_cascade") == (
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "shape_slat_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "tex_slat_decoder",
        "tex_slat_flow_model_1024",
    )


def test_required_model_names_rejects_unknown_pipeline_type():
    from scripts.source_cuda_postcond_full_decode_timing import required_model_names

    with pytest.raises(ValueError, match="unsupported pipeline_type"):
        required_model_names("1536_cascade")


def test_resolve_model_ref_uses_pipeline_repo_for_relative_models():
    from scripts.source_cuda_postcond_full_decode_timing import resolve_model_ref

    assert (
        resolve_model_ref("microsoft/TRELLIS.2-4B", "ckpts/example")
        == "microsoft/TRELLIS.2-4B/ckpts/example"
    )
    assert (
        resolve_model_ref("microsoft/TRELLIS.2-4B", "microsoft/TRELLIS-image-large/ckpts/ss_dec")
        == "microsoft/TRELLIS-image-large/ckpts/ss_dec"
    )


def test_postcond_decode_runner_defaults_to_dependency_free_sparse_conv_backend():
    from scripts.source_cuda_postcond_full_decode_timing import build_parser

    args = build_parser().parse_args(["--output-json", "out.json", "--output-npz", "out.npz"])

    assert args.sparse_conv_backend == "none"
    assert args.sparse_attn_backend == "sdpa"


def test_apply_sparse_backend_env_sets_dense_attention_alias(monkeypatch):
    from scripts.source_cuda_postcond_full_decode_timing import apply_sparse_backend_env

    applied = apply_sparse_backend_env("none", "sdpa")

    assert applied == {
        "SPARSE_CONV_BACKEND": "none",
        "SPARSE_ATTN_BACKEND": "sdpa",
        "ATTN_BACKEND": "sdpa",
    }
    assert applied["ATTN_BACKEND"] == "sdpa"


def test_postcond_decode_runner_defaults_mesh_override_input():
    from pathlib import Path

    from scripts.source_cuda_postcond_full_decode_timing import build_parser

    args = build_parser().parse_args(["--output-json", "out.json", "--output-npz", "out.npz"])

    assert args.mesh_override == Path("o_voxel_override_convert.py")
    assert args.output_mesh_state is None
    assert args.output_shape_slat is None
    assert args.output_shape_flow_step is None
    assert args.shape_flow_noise_sample is None


def test_install_mesh_override_copies_into_source_stubs(tmp_path):
    from scripts.source_cuda_postcond_full_decode_timing import install_mesh_override

    source_root = tmp_path / "source"
    override = tmp_path / "o_voxel_override_convert.py"
    override.write_text("SENTINEL = 1\n")

    result = install_mesh_override(source_root, override)

    installed = source_root / "stubs" / "o_voxel_override_convert.py"
    assert installed.read_text() == "SENTINEL = 1\n"
    assert result["status"] == "installed"
    assert result["path"] == str(installed)
    assert result["source"] == str(override)


def test_write_binary_mesh_ply_preserves_vertices_and_faces(tmp_path):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_binary_mesh_ply

    class Mesh:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)

    output = tmp_path / "mesh.ply"

    write_binary_mesh_ply(output, Mesh())

    payload = output.read_bytes()
    header, body = payload.split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    assert b"element vertex 3" in header
    assert b"element face 1" in header

    vertices = np.frombuffer(body[: 3 * 3 * 4], dtype="<f4").reshape(3, 3)
    faces = np.frombuffer(
        body[3 * 3 * 4 :],
        dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]),
    )
    np.testing.assert_allclose(vertices, Mesh.vertices)
    assert faces["count"].tolist() == [3]
    assert faces["indices"].tolist() == [[0, 1, 2]]


def test_write_mesh_state_npz_preserves_voxel_payload(tmp_path):
    import json
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_mesh_state_npz

    class Mesh:
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        faces = np.array([[0, 1, 1]], dtype=np.int64)
        attrs = np.array([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=np.float32)
        coords = np.array([[3, 4, 5]], dtype=np.int64)
        origin = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        voxel_size = 1 / 512
        voxel_shape = (1, 64, 64, 64)
        layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }

    output = tmp_path / "mesh_state.npz"

    write_mesh_state_npz(output, Mesh())

    with np.load(output) as data:
        np.testing.assert_allclose(data["vertices"], Mesh.vertices)
        np.testing.assert_array_equal(data["faces"], Mesh.faces.astype(np.int32))
        np.testing.assert_allclose(data["attrs"], Mesh.attrs)
        np.testing.assert_array_equal(data["coords"], Mesh.coords.astype(np.int32))
        np.testing.assert_allclose(data["origin"], Mesh.origin)
        assert float(data["voxel_size"]) == Mesh.voxel_size
        np.testing.assert_array_equal(data["voxel_shape"], np.array(Mesh.voxel_shape, dtype=np.int64))
        layout = json.loads(str(data["layout_json"]))

    assert layout == {
        "base_color": [0, 3, None],
        "metallic": [3, 4, None],
        "roughness": [4, 5, None],
        "alpha": [5, 6, None],
    }


def test_write_sparse_tensor_npz_preserves_coords_feats_and_metadata(tmp_path):
    import json
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_sparse_tensor_npz

    class SparseTensor:
        coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int64)
        feats = np.array([[1.5, -2.0], [3.25, 4.5]], dtype=np.float32)

    output = tmp_path / "shape_slat.npz"

    artifact = write_sparse_tensor_npz(
        output,
        SparseTensor(),
        stage="shape_slat",
        normalization="source-config",
    )

    assert artifact["path"] == str(output)
    assert artifact["format"] == "sparse_tensor_npz"
    assert artifact["artifact_scope"] == "source_cuda_shape_slat"
    assert artifact["coords_shape"] == [2, 4]
    assert artifact["feats_shape"] == [2, 2]
    assert artifact["sha256"]

    with np.load(output) as data:
        np.testing.assert_array_equal(data["coords"], SparseTensor.coords.astype(np.int32))
        np.testing.assert_allclose(data["feats"], SparseTensor.feats)
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata == {
        "artifact_scope": "source_cuda_shape_slat",
        "normalization": "source-config",
        "stage": "shape_slat",
    }


def test_write_source_shape_flow_step_npz_preserves_comparable_fields(tmp_path):
    import json
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_source_shape_flow_step_npz

    payload = {
        "noise": np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32),
        "sample_feats": np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32),
        "coords": np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32),
        "coords_3d": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "pred_pos": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "pred_neg": np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
        "pred_cfg": np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32),
        "x0_pos": np.array([[1.1, -0.8], [0.2, 0.1]], dtype=np.float32),
        "x0_cfg": np.array([[0.3, -0.4], [0.5, 0.6]], dtype=np.float32),
        "std_pos": np.array(0.75, dtype=np.float32),
        "std_cfg": np.array(1.5, dtype=np.float32),
        "ratio_raw": np.array(0.5, dtype=np.float32),
        "std_ratio": np.array(0.5, dtype=np.float32),
        "ratio_effective": np.array(0.5, dtype=np.float32),
        "x0_rescaled": np.array([[0.15, -0.2], [0.25, 0.3]], dtype=np.float32),
        "x0_after_rescale": np.array([[0.2, -0.3], [0.4, 0.45]], dtype=np.float32),
        "pred_final": np.array([[0.55, 0.65], [0.75, 0.85]], dtype=np.float32),
        "pred_v_feats": np.array([[0.55, 0.65], [0.75, 0.85]], dtype=np.float32),
        "sample_next": np.array([[0.97, -1.03], [0.46, 0.21]], dtype=np.float32),
        "t": np.array(1.0, dtype=np.float32),
        "t_prev": np.array(0.95, dtype=np.float32),
        "steps": np.array(8, dtype=np.int32),
        "guidance_strength": np.array(7.5, dtype=np.float32),
        "guidance_rescale": np.array(0.5, dtype=np.float32),
        "guidance_interval": np.array([0.8, 1.0], dtype=np.float32),
        "rescale_t": np.array(3.0, dtype=np.float32),
    }
    output = tmp_path / "shape_flow_step.npz"

    artifact = write_source_shape_flow_step_npz(output, payload, normalization="source-config")

    assert artifact["path"] == str(output)
    assert artifact["format"] == "shape_flow_step_npz"
    assert artifact["artifact_scope"] == "source_cuda_shape_flow_first_step"
    assert artifact["coords_shape"] == [2, 4]
    assert artifact["sample_next_shape"] == [2, 2]
    assert artifact["sha256"]

    with np.load(output) as data:
        for key, value in payload.items():
            np.testing.assert_array_equal(data[key], value)
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata == {
        "artifact_scope": "source_cuda_shape_flow_first_step",
        "normalization": "source-config",
        "stage": "shape_flow_step",
    }


def test_load_shape_flow_noise_sample_accepts_exact_coords(tmp_path):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import load_shape_flow_noise_sample

    sample = tmp_path / "mlx_shape_flow_step.npz"
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    noise = np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32)
    np.savez(sample, coords=coords, noise=noise)

    loaded, key = load_shape_flow_noise_sample(sample, coords)

    np.testing.assert_array_equal(loaded, noise)
    assert key == "noise"


def test_load_shape_flow_noise_sample_rejects_coord_order_mismatch(tmp_path):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import load_shape_flow_noise_sample

    sample = tmp_path / "mlx_shape_flow_step.npz"
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    swapped = coords[::-1]
    np.savez(sample, coords=swapped, noise=np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="coords do not exactly match"):
        load_shape_flow_noise_sample(sample, coords)


def test_load_shape_flow_noise_sample_rejects_feature_row_mismatch(tmp_path):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import load_shape_flow_noise_sample

    sample = tmp_path / "mlx_shape_flow_step.npz"
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    np.savez(sample, coords=coords, sample_feats=np.zeros((1, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="noise row mismatch"):
        load_shape_flow_noise_sample(sample, coords)

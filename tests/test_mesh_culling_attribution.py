import numpy as np
import pytest


def test_source_face_index_map_tracks_same_reversed_and_unmatched_faces():
    from scripts.mesh_culling_attribution import build_source_face_index_map

    clean_faces = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
        ],
        dtype=np.int64,
    )
    uv_faces = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ],
        dtype=np.int64,
    )
    vmapping = np.array([0, 1, 2, 3, 1, 2, 9, 10, 11], dtype=np.int64)

    mapping = build_source_face_index_map(
        source_faces=clean_faces,
        uv_faces=uv_faces,
        vmapping=vmapping,
    )

    assert mapping["source_face_index"].tolist() == [0, 1, -1]
    assert mapping["orientation"].tolist() == ["same", "reversed", "unmatched"]
    assert mapping["summary"] == {
        "same": 1,
        "reversed": 1,
        "unmatched": 1,
        "ambiguous": 0,
    }


def test_visible_backface_attribution_reports_face_ids_and_pixels():
    from scripts.mesh_culling_attribution import visible_backface_attribution

    vertices = np.array(
        [
            [-0.5, 0.0, -0.5],
            [0.5, 0.0, -0.5],
            [0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int64,
    )

    report = visible_backface_attribution(
        vertices=vertices,
        faces=faces,
        view_name="+Y",
        image_size=32,
    )

    assert report["visible_pixels"] > 0
    assert report["backfacing_visible_pixels"] == report["visible_pixels"]
    assert set(report["backface_pixels_by_face"]) == {0, 1}


def test_visible_backface_attribution_accepts_oblique_view_direction():
    from scripts.mesh_culling_attribution import visible_backface_attribution_for_direction

    vertices = np.array(
        [
            [-0.5, 0.0, -0.5],
            [0.5, 0.0, -0.5],
            [0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int64,
    )

    report = visible_backface_attribution_for_direction(
        vertices=vertices,
        faces=faces,
        view_direction=np.array([0.0, 1.0, 1.0]),
        image_size=32,
    )

    assert report["visible_pixels"] > 0
    assert report["backfacing_visible_pixels"] == report["visible_pixels"]
    assert set(report["backface_pixels_by_face"]) == {0, 1}
    assert report["view_direction"] == [0.0, pytest.approx(2**-0.5), pytest.approx(2**-0.5)]


def test_default_projected_front_face_convention_is_panel_specific():
    from scripts.mesh_culling_attribution import default_front_face_for_panel

    assert default_front_face_for_panel("front_xz") == "ccw"
    assert default_front_face_for_panel("side_yz") == "cw"
    assert default_front_face_for_panel("top_xy") == "cw"


def test_export_space_identity_accounts_for_glb_axis_transform():
    from scripts.mesh_culling_attribution import export_space_identity

    uv_vertices = np.array(
        [
            [1.0, 2.0, 3.0],
            [-4.0, 5.0, -6.0],
        ],
        dtype=np.float64,
    )
    glb_vertices = np.array(
        [
            [1.0, 3.0, -2.0],
            [-4.0, -6.0, -5.0],
        ],
        dtype=np.float64,
    )

    assert export_space_identity(uv_vertices=uv_vertices, glb_vertices=glb_vertices) == {
        "transform": "glb_xyz_from_uv_x_z_neg_y",
        "vertices_match_export_transform": True,
        "max_abs_error": 0.0,
    }


def test_face_table_arrays_include_uv_island_and_all_touched_faces():
    from collections import Counter

    from scripts.mesh_culling_attribution import build_face_table_arrays

    table = build_face_table_arrays(
        visible_pixels=Counter({0: 7, 2: 3}),
        backface_pixels=Counter({2: 2}),
        projected_missing_pixels=Counter({1: 5}),
        source_face_index=np.array([10, 11, 12], dtype=np.int64),
        source_orientation=np.array(["same", "reversed", "unmatched"], dtype=object),
        uv_component_labels=np.array([4, 4, 9], dtype=np.int32),
    )

    assert table["glb_face"].tolist() == [0, 1, 2]
    assert table["uv_face"].tolist() == [0, 1, 2]
    assert table["clean_face"].tolist() == [10, 11, 12]
    assert table["uv_island"].tolist() == [4, 4, 9]
    assert table["source_orientation"].tolist() == ["same", "reversed", "unmatched"]
    assert table["visible_pixels"].tolist() == [7, 0, 3]
    assert table["backface_pixels"].tolist() == [0, 0, 2]
    assert table["projected_missing_pixels"].tolist() == [0, 5, 0]

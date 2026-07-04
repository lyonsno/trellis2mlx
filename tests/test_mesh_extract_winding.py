"""Extractor winding contracts for flexible dual-grid mesh output."""

import numpy as np


_EDGE_NEIGHBOR_OFFSETS = np.array([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
], dtype=np.int64)
_QUAD_SPLIT_1 = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
_QUAD_SPLIT_2 = np.array([0, 1, 3, 3, 1, 2], dtype=np.int64)


def _export_vertices(vertices):
    export = vertices.astype(np.float64).copy()
    export[:, 1], export[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()
    return export


def _source_table_faces(coords, intersected_flag, split_weight):
    """Minimal copy of the Trellis-Mac/source table path, without route patches."""
    coord_to_idx = {tuple(coord.tolist()): i for i, coord in enumerate(coords)}
    faces = []
    for voxel, flags in zip(coords, intersected_flag):
        for axis, enabled in enumerate(flags):
            if not enabled:
                continue
            quad = []
            for offset in _EDGE_NEIGHBOR_OFFSETS[axis]:
                idx = coord_to_idx.get(tuple((voxel + offset).tolist()))
                if idx is None:
                    quad = []
                    break
                quad.append(idx)
            if not quad:
                continue
            quad = np.asarray(quad, dtype=np.int64)
            sw = split_weight[quad, 0]
            split = _QUAD_SPLIT_1 if sw[0] * sw[2] > sw[1] * sw[3] else _QUAD_SPLIT_2
            faces.append(quad[split].reshape(2, 3))
    return np.vstack(faces).astype(np.int64)


def test_flexible_dual_grid_matches_source_face_order_without_axis_flip():
    """The source extractor tables do not apply a per-edge-axis winding flip."""
    from trellmlx.mesh_extract import flexible_dual_grid_to_mesh

    coords = np.array(
        [[z, y, x] for z in range(3) for y in range(3) for x in range(3)],
        dtype=np.int64,
    )
    dual_vertices = np.full((len(coords), 3), 0.5, dtype=np.float32)
    intersected_flag = np.ones((len(coords), 3), dtype=bool)
    split_weight = np.ones((len(coords), 1), dtype=np.float32)

    _, faces = flexible_dual_grid_to_mesh(
        coords=coords,
        dual_vertices=dual_vertices,
        intersected_flag=intersected_flag,
        split_weight=split_weight,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=3,
    )

    expected = _source_table_faces(coords, intersected_flag, split_weight)
    np.testing.assert_array_equal(faces, expected)


def test_decoder_output_source_winding_is_not_projected_culling_authority():
    """A projection proxy must not override source extractor face order."""
    from scripts.mesh_culling_attribution import (
        default_front_face_for_panel,
        projected_front_face_missing_attribution,
    )
    from trellmlx.mesh_extract import decoder_output_to_mesh

    coords_3d = np.array(
        [[z, y, x] for z in range(3) for y in range(3) for x in range(3)],
        dtype=np.int64,
    )
    coords = np.column_stack([np.zeros(len(coords_3d), dtype=np.int64), coords_3d])
    feats = np.zeros((len(coords), 7), dtype=np.float32)
    feats[:, 3:6] = 1.0

    vertices, faces = decoder_output_to_mesh(feats, coords, resolution=3)
    export_vertices = _export_vertices(vertices)

    missing = 0
    reference = 0
    panel_ratios = {}
    for panel in ("front_xz", "side_yz", "top_xy"):
        report = projected_front_face_missing_attribution(
            vertices=export_vertices,
            faces=faces,
            panel=panel,
            image_size=128,
            front_face=default_front_face_for_panel(panel),
        )
        missing += report["missing_pixels"]
        reference += report["double_sided_pixels"]
        panel_ratios[panel] = report["missing_pixel_ratio_vs_double_sided"]

    assert reference > 0
    assert missing > 0
    assert any(ratio > 0 for ratio in panel_ratios.values()), panel_ratios

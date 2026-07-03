"""Extractor winding contracts for flexible dual-grid mesh output."""

import numpy as np


def _export_vertices(vertices):
    export = vertices.astype(np.float64).copy()
    export[:, 1], export[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()
    return export


def test_decoder_output_dense_cube_projects_front_faces_on_all_panels():
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
    assert missing == 0, panel_ratios

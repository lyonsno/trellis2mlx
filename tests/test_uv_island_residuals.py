import numpy as np


def test_summarize_face_table_buckets_residuals_by_uv_island_orientation():
    from scripts.summarize_uv_island_residuals import summarize_face_table

    table = {
        "uv_island": np.array([7, 7, 9, 9], dtype=np.int64),
        "source_orientation": np.array(["same", "reversed", "same", "reversed"]),
        "visible_pixels": np.array([11, 9, 8, 12], dtype=np.int64),
        "backface_pixels": np.array([3, 3, 0, 9], dtype=np.int64),
        "projected_missing_pixels": np.array([1, 2, 0, 4], dtype=np.int64),
    }

    report = summarize_face_table(table, top_n=10, high_backface_ratio=0.25, near_tie_margin=0.25)

    assert report["totals"] == {
        "islands": 2,
        "visible_pixels": 40,
        "backface_pixels": 15,
        "projected_missing_pixels": 7,
        "backface_pixel_ratio": 0.375,
    }
    assert report["orientation_totals"]["same"]["visible_pixels"] == 19
    assert report["orientation_totals"]["reversed"]["backface_pixels"] == 12

    island_9, island_7 = report["top_backface_islands"]
    assert island_9["uv_island"] == 9
    assert island_9["orientation_class"] == "mixed_near_tie"
    assert island_9["residual_class"] == "high_reversed_source_residual"
    assert island_9["backface_pixels_by_source_orientation"] == {"same": 0, "reversed": 9}

    assert island_7["uv_island"] == 7
    assert island_7["orientation_class"] == "mixed_near_tie"
    assert island_7["residual_class"] == "high_mixed_orientation_residual"
    assert island_7["visible_pixels"] == 20
    assert island_7["backface_pixels"] == 6
    assert island_7["source_orientation_pixels"] == {"same": 11, "reversed": 9}

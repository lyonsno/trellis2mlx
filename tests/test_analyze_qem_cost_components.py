import numpy as np

from scripts.analyze_qem_cost_components import analyze_component_arrays


def _arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    cuda_total = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    metal_total = cuda_total.copy()
    metal_total[1] = np.nextafter(metal_total[1], np.float32(np.inf))
    cuda = {
        "qems": np.zeros((2, 10), dtype=np.float32),
        "edge_collapse_costs": cuda_total,
        "component_edge_collapse_costs": cuda_total.copy(),
        "qem_costs": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "edge_length2": np.ones(4, dtype=np.float32),
        "skinny_avgs": np.ones(4, dtype=np.float32),
        "skinny_terms": np.ones(4, dtype=np.float32),
    }
    metal = {name: value.copy() for name, value in cuda.items()}
    metal["edge_collapse_costs"] = metal_total
    metal["component_edge_collapse_costs"] = metal_total.copy()
    metal["qem_costs"][1] = np.nextafter(
        metal["qem_costs"][1],
        np.float32(np.inf),
    )
    metal["component_edge_collapse_costs"][2] = np.nextafter(
        metal["component_edge_collapse_costs"][2],
        np.float32(np.inf),
    )
    return cuda, metal


def test_component_analysis_excludes_backend_self_inconsistent_edges():
    cuda, metal = _arrays()

    report = analyze_component_arrays(cuda, metal)

    assert report["edges"] == 4
    assert report["admission"] == {
        "cuda_rejected": 0,
        "metal_rejected": 1,
        "joint_admitted": 3,
    }
    assert report["final_cost"]["joint_mismatch_count"] == 1
    assert report["components"]["qem_costs"]["joint_mismatch_count"] == 1
    assert report["components"]["qem_costs"]["final_mismatch_overlap"] == 1
    assert report["components"]["skinny_terms"]["joint_mismatch_count"] == 0
    assert report["component_union"]["final_mismatch_covered"] == 1
    assert report["component_union"]["final_mismatch_uncovered"] == 0


def test_component_analysis_counts_joint_threshold_eligibility_crossings():
    cuda, metal = _arrays()
    cuda["edge_collapse_costs"][:] = np.array(
        [0.5, 1.5, 0.5, 2.0],
        dtype=np.float32,
    )
    metal["edge_collapse_costs"][:] = np.array(
        [1.5, 0.5, 0.5, 2.0],
        dtype=np.float32,
    )
    cuda["component_edge_collapse_costs"][:] = cuda[
        "edge_collapse_costs"
    ]
    metal["component_edge_collapse_costs"][:] = metal[
        "edge_collapse_costs"
    ]

    report = analyze_component_arrays(cuda, metal, collapse_threshold=1.0)

    assert report["collapse_threshold"] == {
        "value": 1.0,
        "cuda_eligible": 2,
        "metal_eligible": 2,
        "joint_eligible": 1,
        "eligibility_crossings": 2,
        "cuda_only_eligible": 1,
        "metal_only_eligible": 1,
    }

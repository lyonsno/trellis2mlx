"""Metal-accelerated QEM mesh simplification.

Moves the per-edge topology check (normal-flip + skinny penalty) to a Metal
kernel via mx.fast.metal_kernel. The adjacency structures and compaction
remain on CPU (inherently serial/graph work).

Attribution: Algorithm adapted from pedronaugusto/mtlmesh (cumesh) — a Metal
port of QEM decimation with parallel conflict-free edge collapse. The atomic
min cost propagation pattern and the normal-flip/skinny-triangle guards are
from that codebase. Thank you to Pedro Naugusto for the excellent Metal mesh
processing work that made this port straightforward.
"""

import numpy as np
import time

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


# Metal kernel: per-edge collapse cost with normal-flip check and skinny penalty
_EDGE_COST_HEADER = '''
inline float3 cross_f3(float3 a, float3 b) {
    return float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline float dot_f3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
inline float norm_f3(float3 a) {
    return sqrt(dot_f3(a, a));
}
'''

_EDGE_COST_SOURCE = '''
    uint ei = thread_position_in_grid.x;
    if (ei >= num_edges_buf[0]) return;

    int v0i = edge_v0[ei];
    int v1i = edge_v1[ei];

    // Read QEM cost and edge length (precomputed on CPU/MLX)
    float base_cost = base_costs[ei];
    if (isinf(base_cost) || isnan(base_cost)) {
        out_costs[ei] = INFINITY;
        return;
    }

    // Read new vertex position
    float3 vn = float3(vnew[ei*3], vnew[ei*3+1], vnew[ei*3+2]);

    float skinny_cost = 0.0f;
    int num_tri = 0;
    bool flipped = false;

    // Check faces incident to v0
    for (int idx = vf_offset[v0i]; idx < vf_offset[v0i + 1]; idx++) {
        int fi = vf_data[idx];
        int fa_i = face_v[fi*3+0], fb_i = face_v[fi*3+1], fc_i = face_v[fi*3+2];

        // Skip shared faces (will be removed)
        if (fa_i == v1i || fb_i == v1i || fc_i == v1i) continue;

        float3 fa = float3(verts[fa_i*3], verts[fa_i*3+1], verts[fa_i*3+2]);
        float3 fb = float3(verts[fb_i*3], verts[fb_i*3+1], verts[fb_i*3+2]);
        float3 fc = float3(verts[fc_i*3], verts[fc_i*3+1], verts[fc_i*3+2]);

        float3 na = (fa_i == v0i) ? vn : fa;
        float3 nb = (fb_i == v0i) ? vn : fb;
        float3 nc = (fc_i == v0i) ? vn : fc;

        float3 old_normal = cross_f3(fb - fa, fc - fa);
        float3 new_e1 = nb - na;
        float3 new_e2 = nc - na;
        float3 new_normal = cross_f3(new_e1, new_e2);

        if (dot_f3(old_normal, new_normal) < 0.0f) { flipped = true; break; }

        float new_area = 0.5f * norm_f3(new_normal);
        float3 new_e0 = nc - nb;
        float denom = dot_f3(new_e0, new_e0) + dot_f3(new_e1, new_e1) + dot_f3(new_e2, new_e2);
        if (denom < 1e-12f) denom = 1e-12f;
        float shape = 4.0f * 1.7320508f * new_area / denom;
        skinny_cost += 1.0f - clamp(shape, 0.0f, 1.0f);
        num_tri += 1;
    }

    if (!flipped) {
        // Check faces incident to v1
        for (int idx = vf_offset[v1i]; idx < vf_offset[v1i + 1]; idx++) {
            int fi = vf_data[idx];
            int fa_i = face_v[fi*3+0], fb_i = face_v[fi*3+1], fc_i = face_v[fi*3+2];

            if (fa_i == v0i || fb_i == v0i || fc_i == v0i) continue;

            float3 fa = float3(verts[fa_i*3], verts[fa_i*3+1], verts[fa_i*3+2]);
            float3 fb = float3(verts[fb_i*3], verts[fb_i*3+1], verts[fb_i*3+2]);
            float3 fc = float3(verts[fc_i*3], verts[fc_i*3+1], verts[fc_i*3+2]);

            float3 na = (fa_i == v1i) ? vn : fa;
            float3 nb = (fb_i == v1i) ? vn : fb;
            float3 nc = (fc_i == v1i) ? vn : fc;

            float3 old_normal = cross_f3(fb - fa, fc - fa);
            float3 new_e1 = nb - na;
            float3 new_e2 = nc - na;
            float3 new_normal = cross_f3(new_e1, new_e2);

            if (dot_f3(old_normal, new_normal) < 0.0f) { flipped = true; break; }

            float new_area = 0.5f * norm_f3(new_normal);
            float3 new_e0 = nc - nb;
            float denom = dot_f3(new_e0, new_e0) + dot_f3(new_e1, new_e1) + dot_f3(new_e2, new_e2);
            if (denom < 1e-12f) denom = 1e-12f;
            float shape = 4.0f * 1.7320508f * new_area / denom;
            skinny_cost += 1.0f - clamp(shape, 0.0f, 1.0f);
            num_tri += 1;
        }
    }

    if (flipped) {
        out_costs[ei] = INFINITY;
    } else {
        float extra = 0.0f;
        if (num_tri > 0) {
            extra = lambda_skinny_buf[0] * (skinny_cost / float(num_tri)) * edge_len2[ei];
        }
        out_costs[ei] = base_cost + extra;
    }
'''

_edge_cost_kernel = None

def _get_edge_cost_kernel():
    global _edge_cost_kernel
    if _edge_cost_kernel is None:
        _edge_cost_kernel = mx.fast.metal_kernel(
            name="qem_edge_cost_topology",
            input_names=[
                "edge_v0", "edge_v1",
                "verts", "face_v",
                "vf_offset", "vf_data",
                "vnew", "base_costs", "edge_len2",
                "num_edges_buf", "lambda_skinny_buf",
            ],
            output_names=["out_costs"],
            source=_EDGE_COST_SOURCE,
            header=_EDGE_COST_HEADER,
        )
    return _edge_cost_kernel


def simplify_qem(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    *,
    lambda_edge_length: float = 1e-2,
    lambda_skinny: float = 1e-3,
    initial_thresh: float = 1e-8,
    verbose: bool = True,
    step_trace: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplify mesh using QEM edge collapse with topology guards."""
    if len(faces) <= target_faces:
        return vertices, faces

    vertices = vertices.astype(np.float32).copy()
    faces = faces.astype(np.int32).copy()

    thresh = initial_thresh
    t0 = time.perf_counter()
    iteration = 0

    max_iterations = 500
    while len(faces) > target_faces and iteration < max_iterations:
        n_before = len(faces)

        vertices, faces = _simplify_step(
            vertices, faces,
            lambda_edge_length=lambda_edge_length,
            lambda_skinny=lambda_skinny,
            collapse_thresh=thresh,
        )

        n_after = len(faces)
        removed = n_before - n_after
        iteration += 1
        if step_trace is not None:
            step_trace.append({
                "iteration": iteration,
                "threshold": thresh,
                "input_faces": n_before,
                "output_faces": n_after,
                "removed_faces": removed,
            })

        if verbose and iteration % 5 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    QEM iter {iteration}: {n_after:,}F "
                  f"(removed {removed:,}, thresh={thresh:.2e}, {elapsed:.1f}s)",
                  flush=True)

        if n_after <= target_faces:
            break

        if removed / max(n_before, 1) < 0.01:
            thresh *= 10

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  QEM simplify: {len(vertices):,}V {len(faces):,}F "
              f"({iteration} iters, {elapsed:.1f}s)", flush=True)

    return vertices, faces


def _build_adjacency(faces, V):
    """Build CSR-style vertex-to-face adjacency and edge structures."""
    F = len(faces)

    # Build vertex-to-face (CSR)
    # Count per vertex
    vf_count = np.zeros(V, dtype=np.int32)
    for k in range(3):
        np.add.at(vf_count, faces[:, k], 1)
    vf_offset = np.zeros(V + 1, dtype=np.int32)
    np.cumsum(vf_count, out=vf_offset[1:])
    vf_data = np.empty(vf_offset[-1], dtype=np.int32)
    pos = vf_offset[:-1].copy()
    for fi in range(F):
        for vi in faces[fi]:
            vf_data[pos[vi]] = fi
            pos[vi] += 1

    # Edge list
    e0 = np.stack([faces[:, 0], faces[:, 1]], axis=1)
    e1 = np.stack([faces[:, 1], faces[:, 2]], axis=1)
    e2 = np.stack([faces[:, 2], faces[:, 0]], axis=1)
    all_edges = np.concatenate([e0, e1, e2], axis=0)
    all_edges.sort(axis=1)
    face_idx = np.tile(np.arange(F, dtype=np.int32), 3)

    edge_keys = all_edges[:, 0].astype(np.int64) * (V + 1) + all_edges[:, 1]
    unique_keys, inv = np.unique(edge_keys, return_inverse=True)
    E = len(unique_keys)
    edges = np.stack([unique_keys // (V + 1), unique_keys % (V + 1)], axis=1).astype(np.int32)

    # Boundary detection
    edge_face_count = np.bincount(inv, minlength=E)
    is_boundary = np.zeros(V, dtype=np.bool_)
    boundary_mask = edge_face_count == 1
    is_boundary[edges[boundary_mask, 0]] = True
    is_boundary[edges[boundary_mask, 1]] = True

    return edges, E, is_boundary, vf_offset, vf_data


def _compute_qem(vertices, faces, V):
    """Compute per-vertex QEM quadrics (vectorized)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    d = -np.sum(normals * v0, axis=1)

    a, b, c = normals[:, 0], normals[:, 1], normals[:, 2]
    face_qem = np.column_stack([
        a*a, a*b, a*c, a*d, b*b, b*c, b*d, c*c, c*d, d*d
    ]).astype(np.float32)

    qems = np.zeros((V, 10), dtype=np.float32)
    for k in range(3):
        np.add.at(qems, faces[:, k], face_qem)

    return qems


def _compute_base_costs(vertices, edges, qems, is_boundary, lambda_edge_length):
    """Compute QEM + edge length cost (vectorized). Returns cost, v_new, edge_len2."""
    ev0, ev1 = edges[:, 0], edges[:, 1]
    p0, p1 = vertices[ev0], vertices[ev1]

    w0 = np.full(len(edges), 0.5, dtype=np.float32)
    w0[is_boundary[ev0] & ~is_boundary[ev1]] = 1.0
    w0[~is_boundary[ev0] & is_boundary[ev1]] = 0.0
    v_new = p0 * w0[:, None] + p1 * (1 - w0[:, None])

    q = qems[ev0] + qems[ev1]
    vx, vy, vz = v_new[:, 0], v_new[:, 1], v_new[:, 2]
    qem_cost = (
        q[:,0]*vx*vx + 2*q[:,1]*vx*vy + 2*q[:,2]*vx*vz + 2*q[:,3]*vx +
        q[:,4]*vy*vy + 2*q[:,5]*vy*vz + 2*q[:,6]*vy +
        q[:,7]*vz*vz + 2*q[:,8]*vz + q[:,9]
    ).astype(np.float32)

    edge_len2 = np.sum((p1 - p0)**2, axis=1)
    base_cost = qem_cost + lambda_edge_length * edge_len2

    return base_cost, v_new, edge_len2


def _topology_check_metal(vertices, faces, edges, v_new, base_costs, edge_len2,
                           vf_offset, vf_data, lambda_skinny):
    """Run per-edge topology check on Metal GPU."""
    E = len(edges)
    kernel = _get_edge_cost_kernel()

    # Flatten for Metal
    verts_flat = mx.array(vertices.ravel().astype(np.float32))
    faces_flat = mx.array(faces.ravel().astype(np.int32))
    edge_v0 = mx.array(edges[:, 0].astype(np.int32))
    edge_v1 = mx.array(edges[:, 1].astype(np.int32))
    vf_off_mx = mx.array(vf_offset.astype(np.int32))
    vf_dat_mx = mx.array(vf_data.astype(np.int32))
    vnew_flat = mx.array(v_new.ravel().astype(np.float32))
    base_mx = mx.array(base_costs.astype(np.float32))
    elen2_mx = mx.array(edge_len2.astype(np.float32))
    num_edges_mx = mx.array([E], dtype=mx.uint32)
    lambda_sk_mx = mx.array([lambda_skinny], dtype=mx.float32)

    if E == 0:
        return base_costs.copy()

    tg = min(256, E)
    grid = ((E + tg - 1) // tg * tg, 1, 1)

    outputs = kernel(
        inputs=[edge_v0, edge_v1, verts_flat, faces_flat,
                vf_off_mx, vf_dat_mx, vnew_flat, base_mx, elen2_mx,
                num_edges_mx, lambda_sk_mx],
        template=[],
        grid=grid,
        threadgroup=(tg, 1, 1),
        output_shapes=[(E,)],
        output_dtypes=[mx.float32],
    )
    mx.eval(outputs[0])
    return np.array(outputs[0])


def _topology_check_cpu(vertices, faces, edges, v_new, base_costs, edge_len2,
                         vf_offset, vf_data, lambda_skinny):
    """CPU fallback for per-edge topology check."""
    E = len(edges)
    cost = base_costs.copy()

    for ei in range(E):
        if not np.isfinite(cost[ei]):
            continue
        v0i, v1i = edges[ei]
        vn = v_new[ei]
        skinny_cost = 0.0
        num_tri = 0
        flipped = False

        for vi, other in [(v0i, v1i), (v1i, v0i)]:
            for idx in range(vf_offset[vi], vf_offset[vi + 1]):
                fi = vf_data[idx]
                fv = faces[fi]
                if other in fv:
                    continue

                fa, fb, fc = vertices[fv[0]], vertices[fv[1]], vertices[fv[2]]
                na = vn if fv[0] == vi else fa
                nb = vn if fv[1] == vi else fb
                nc = vn if fv[2] == vi else fc

                old_n = np.cross(fb - fa, fc - fa)
                new_e1, new_e2 = nb - na, nc - na
                new_n = np.cross(new_e1, new_e2)

                if np.dot(old_n, new_n) < 0.0:
                    flipped = True
                    break

                new_area = 0.5 * np.linalg.norm(new_n)
                new_e0 = nc - nb
                denom = max(np.dot(new_e0, new_e0) + np.dot(new_e1, new_e1) + np.dot(new_e2, new_e2), 1e-12)
                shape = 4.0 * np.sqrt(3.0) * new_area / denom
                skinny_cost += 1.0 - min(max(shape, 0.0), 1.0)
                num_tri += 1
            if flipped:
                break

        if flipped:
            cost[ei] = np.inf
        elif num_tri > 0:
            cost[ei] += lambda_skinny * (skinny_cost / num_tri) * edge_len2[ei]

    return cost


def _simplify_step(vertices, faces, *, lambda_edge_length, lambda_skinny, collapse_thresh):
    """One iteration of QEM edge collapse."""
    V, F = len(vertices), len(faces)

    edges, E, is_boundary, vf_offset, vf_data = _build_adjacency(faces, V)
    qems = _compute_qem(vertices, faces, V)
    base_costs, v_new, edge_len2 = _compute_base_costs(
        vertices, edges, qems, is_boundary, lambda_edge_length)

    # Topology check — Metal if available, CPU fallback
    if HAS_MLX:
        cost = _topology_check_metal(
            vertices, faces, edges, v_new, base_costs, edge_len2,
            vf_offset, vf_data, lambda_skinny)
    else:
        cost = _topology_check_cpu(
            vertices, faces, edges, v_new, base_costs, edge_len2,
            vf_offset, vf_data, lambda_skinny)

    # Propagate minimum cost to faces
    face_min_cost = np.full(F, np.inf, dtype=np.float32)
    face_min_edge = np.full(F, -1, dtype=np.int32)

    # Iterate in edge-index order. Use strict < so the lowest edge index
    # wins ties, matching the reference packed atomic_ulong min behavior
    # where (cost_bits << 32 | edge_id) gives lexicographic (cost, id) order.
    eligible = np.where(cost <= collapse_thresh)[0]
    for ei in eligible:
        v0i, v1i = edges[ei]
        c = cost[ei]
        for vi in [v0i, v1i]:
            for idx in range(vf_offset[vi], vf_offset[vi + 1]):
                fi = vf_data[idx]
                if c < face_min_cost[fi]:
                    face_min_cost[fi] = c
                    face_min_edge[fi] = ei

    # Conflict-free collapse. Match the reference's single GPU pass against a
    # frozen propagated-cost snapshot: an edge is eligible when every face
    # adjacent to either endpoint chose that edge as its minimum-cost edge.
    faces_kept = np.ones(F, dtype=np.bool_)
    collapsed_any = False

    for ei in eligible:
        v0i, v1i = edges[ei]

        agreed = True
        for vi in [v0i, v1i]:
            for idx in range(vf_offset[vi], vf_offset[vi + 1]):
                fi = vf_data[idx]
                if face_min_edge[fi] != ei:
                    agreed = False
                    break
            if not agreed:
                break

        if not agreed:
            continue

        # Collapse
        vertices[v0i] = v_new[ei]
        for vi, other in [(v0i, v1i), (v1i, v0i)]:
            for idx in range(vf_offset[vi], vf_offset[vi + 1]):
                fi = vf_data[idx]
                fv = faces[fi]
                if other in fv:
                    faces_kept[fi] = False
                elif vi == v1i:
                    faces[fi] = np.where(fv == v1i, v0i, fv)

        collapsed_any = True

    if not collapsed_any:
        return vertices, faces

    kept_faces = faces[faces_kept]
    used_verts = np.unique(kept_faces)
    remap = np.full(V, -1, dtype=np.int32)
    remap[used_verts] = np.arange(len(used_verts), dtype=np.int32)

    return vertices[used_verts], remap[kept_faces]

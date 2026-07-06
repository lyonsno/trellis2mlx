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
_source_qem_kernel = None
_source_base_cost_kernel = None

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


_SOURCE_QEM_HEADER = '''
inline float3 cross_f3(float3 a, float3 b) {
    return float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline float dot_f3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
'''

_SOURCE_QEM_SOURCE = '''
    uint vi = thread_position_in_grid.x;
    if (vi >= num_vertices_buf[0]) return;

    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f, q4 = 0.0f;
    float q5 = 0.0f, q6 = 0.0f, q7 = 0.0f, q8 = 0.0f, q9 = 0.0f;

    for (int ptr = vf_offset[vi]; ptr < vf_offset[vi + 1]; ptr++) {
        int fi = vf_data[ptr];
        int ia = face_v[fi * 3 + 0];
        int ib = face_v[fi * 3 + 1];
        int ic = face_v[fi * 3 + 2];

        float3 v0 = float3(verts[ia * 3 + 0], verts[ia * 3 + 1], verts[ia * 3 + 2]);
        float3 e1 = float3(verts[ib * 3 + 0], verts[ib * 3 + 1], verts[ib * 3 + 2]) - v0;
        float3 e2 = float3(verts[ic * 3 + 0], verts[ic * 3 + 1], verts[ic * 3 + 2]) - v0;
        float3 n = cross_f3(e1, e2);
        float inv_norm = rsqrt(dot_f3(n, n));
        n *= inv_norm;
        float d = -dot_f3(n, v0);

        q0 += n.x * n.x;
        q1 += n.x * n.y;
        q2 += n.x * n.z;
        q3 += n.x * d;
        q4 += n.y * n.y;
        q5 += n.y * n.z;
        q6 += n.y * d;
        q7 += n.z * n.z;
        q8 += n.z * d;
        q9 += d * d;
    }

    uint base = vi * 10;
    out_qems[base + 0] = q0;
    out_qems[base + 1] = q1;
    out_qems[base + 2] = q2;
    out_qems[base + 3] = q3;
    out_qems[base + 4] = q4;
    out_qems[base + 5] = q5;
    out_qems[base + 6] = q6;
    out_qems[base + 7] = q7;
    out_qems[base + 8] = q8;
    out_qems[base + 9] = q9;
'''

_SOURCE_BASE_COST_SOURCE = '''
    uint ei = thread_position_in_grid.x;
    if (ei >= num_edges_buf[0]) return;

    int e0 = edge_v0[ei];
    int e1 = edge_v1[ei];
    float3 v0 = float3(verts[e0 * 3 + 0], verts[e0 * 3 + 1], verts[e0 * 3 + 2]);
    float3 v1 = float3(verts[e1 * 3 + 0], verts[e1 * 3 + 1], verts[e1 * 3 + 2]);

    float w0 = 0.5f;
    if (boundary[e0] != 0 && boundary[e1] == 0) w0 = 1.0f;
    else if (boundary[e0] == 0 && boundary[e1] != 0) w0 = 0.0f;
    float3 v = v0 * w0 + v1 * (1.0f - w0);

    uint q0_base = uint(e0) * 10;
    uint q1_base = uint(e1) * 10;
    float q0 = qems[q0_base + 0] + qems[q1_base + 0];
    float q1 = qems[q0_base + 1] + qems[q1_base + 1];
    float q2 = qems[q0_base + 2] + qems[q1_base + 2];
    float q3 = qems[q0_base + 3] + qems[q1_base + 3];
    float q4 = qems[q0_base + 4] + qems[q1_base + 4];
    float q5 = qems[q0_base + 5] + qems[q1_base + 5];
    float q6 = qems[q0_base + 6] + qems[q1_base + 6];
    float q7 = qems[q0_base + 7] + qems[q1_base + 7];
    float q8 = qems[q0_base + 8] + qems[q1_base + 8];
    float q9 = qems[q0_base + 9] + qems[q1_base + 9];

    float x = v.x;
    float y = v.y;
    float z = v.z;
    float qem_cost =
        q0 * x * x + 2.0f * q1 * x * y + 2.0f * q2 * x * z + 2.0f * q3 * x +
        q4 * y * y + 2.0f * q5 * y * z + 2.0f * q6 * y +
        q7 * z * z + 2.0f * q8 * z + q9;

    float3 delta = v1 - v0;
    float edge_len2 = dot(delta, delta);
    out_qem_costs[ei] = qem_cost;
    out_edge_len2[ei] = edge_len2;
    out_base_costs[ei] = qem_cost + lambda_edge_length_buf[0] * edge_len2;
    out_vnew[ei * 3 + 0] = v.x;
    out_vnew[ei * 3 + 1] = v.y;
    out_vnew[ei * 3 + 2] = v.z;
'''


def _get_source_qem_kernel():
    global _source_qem_kernel
    if _source_qem_kernel is None:
        _source_qem_kernel = mx.fast.metal_kernel(
            name="qem_source_shaped_vertex_qems",
            input_names=["verts", "face_v", "vf_offset", "vf_data", "num_vertices_buf"],
            output_names=["out_qems"],
            source=_SOURCE_QEM_SOURCE,
            header=_SOURCE_QEM_HEADER,
        )
    return _source_qem_kernel


def _get_source_base_cost_kernel():
    global _source_base_cost_kernel
    if _source_base_cost_kernel is None:
        _source_base_cost_kernel = mx.fast.metal_kernel(
            name="qem_source_shaped_base_costs",
            input_names=[
                "edge_v0", "edge_v1", "verts", "qems", "boundary",
                "num_edges_buf", "lambda_edge_length_buf",
            ],
            output_names=["out_base_costs", "out_vnew", "out_edge_len2", "out_qem_costs"],
            source=_SOURCE_BASE_COST_SOURCE,
            header=_SOURCE_QEM_HEADER,
        )
    return _source_base_cost_kernel


def _compute_qem_metal_source_shaped(vertices, faces, vf_offset, vf_data):
    """Compute per-vertex QEMs using a source-shaped Metal vertex-face loop."""
    if not HAS_MLX:
        raise RuntimeError("mlx-metal-source QEM backend requires MLX")
    V = len(vertices)
    if V == 0:
        return np.empty((0, 10), dtype=np.float32)

    kernel = _get_source_qem_kernel()
    verts_flat = mx.array(np.asarray(vertices, dtype=np.float32).ravel())
    faces_flat = mx.array(np.asarray(faces, dtype=np.int32).ravel())
    vf_off_mx = mx.array(np.asarray(vf_offset, dtype=np.int32))
    vf_dat_mx = mx.array(np.asarray(vf_data, dtype=np.int32))
    num_vertices_mx = mx.array([V], dtype=mx.uint32)
    tg = min(256, V)
    grid = ((V + tg - 1) // tg * tg, 1, 1)

    outputs = kernel(
        inputs=[verts_flat, faces_flat, vf_off_mx, vf_dat_mx, num_vertices_mx],
        template=[],
        grid=grid,
        threadgroup=(tg, 1, 1),
        output_shapes=[(V, 10)],
        output_dtypes=[mx.float32],
    )
    mx.eval(outputs[0])
    return np.array(outputs[0], dtype=np.float32)


def _compute_base_costs_metal_source_shaped(vertices, edges, qems, is_boundary, lambda_edge_length):
    """Compute QEM base costs and collapse positions with source-shaped Metal arithmetic."""
    if not HAS_MLX:
        raise RuntimeError("mlx-metal-source QEM backend requires MLX")
    E = len(edges)
    if E == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    kernel = _get_source_base_cost_kernel()
    edge_v0 = mx.array(np.asarray(edges[:, 0], dtype=np.int32))
    edge_v1 = mx.array(np.asarray(edges[:, 1], dtype=np.int32))
    verts_flat = mx.array(np.asarray(vertices, dtype=np.float32).ravel())
    qems_flat = mx.array(np.asarray(qems, dtype=np.float32).ravel())
    boundary = mx.array(np.asarray(is_boundary, dtype=np.int32))
    num_edges_mx = mx.array([E], dtype=mx.uint32)
    lambda_mx = mx.array([lambda_edge_length], dtype=mx.float32)
    tg = min(256, E)
    grid = ((E + tg - 1) // tg * tg, 1, 1)

    outputs = kernel(
        inputs=[edge_v0, edge_v1, verts_flat, qems_flat, boundary, num_edges_mx, lambda_mx],
        template=[],
        grid=grid,
        threadgroup=(tg, 1, 1),
        output_shapes=[(E,), (E, 3), (E,), (E,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
    )
    mx.eval(*outputs)
    return (
        np.array(outputs[0], dtype=np.float32),
        np.array(outputs[1], dtype=np.float32),
        np.array(outputs[2], dtype=np.float32),
        np.array(outputs[3], dtype=np.float32),
    )


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

    # Propagate minimum cost to faces using the source packed-key ordering.
    # The Metal source compares raw uint64 keys:
    #   (as_type<uint>(float_cost) << 32) | edge_id
    # so tiny negative float32 costs do not behave like numeric minima.
    face_min_key = np.full(F, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    face_min_edge = np.full(F, -1, dtype=np.int32)

    cost_bits = np.asarray(cost, dtype=np.float32).view(np.uint32)
    for ei in range(E):
        pack = (np.uint64(cost_bits[ei]) << np.uint64(32)) | np.uint64(np.uint32(ei))
        v0i, v1i = edges[ei]
        for vi in [v0i, v1i]:
            for idx in range(vf_offset[vi], vf_offset[vi + 1]):
                fi = vf_data[idx]
                if pack < face_min_key[fi]:
                    face_min_key[fi] = pack
                    face_min_edge[fi] = ei

    eligible = np.where(cost <= collapse_thresh)[0]

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

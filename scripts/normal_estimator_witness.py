"""Same-mesh normal witness. CPU formula is provisional; CUDA is pinned CuMesh.

Input NPZ uses NAME__vertices and NAME__faces. Outputs retain every mesh and
every normal vector, including nonfinite native results, without repair/culling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def area_normals(vertices, faces):
    """Float64 area reference; deterministic first-input-face fallback, not CUDA order."""
    v = np.asarray(vertices, dtype=np.float64)
    cross = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    accum = np.zeros_like(v)
    for corner in range(3):
        np.add.at(accum, faces[:, corner], cross)
    lengths = np.linalg.norm(accum, axis=1)
    valid = lengths > 0
    accum[valid] /= lengths[valid, None]
    # The native kernel falls back to a first incident unnormalized face normal.
    # Atomic adjacency order is not reproduced here. Keep that limitation explicit.
    first = np.full(len(v), len(faces), dtype=np.int64)
    for corner in range(3):
        np.minimum.at(first, faces[:, corner], np.arange(len(faces)))
    fallback = (~valid) & (first < len(faces))
    accum[fallback] = cross[first[fallback]]
    return accum.astype(np.float32)


def validate_cases(arrays):
    names = sorted(k.removesuffix('__vertices') for k in arrays if k.endswith('__vertices'))
    expected = {f'{n}__{field}' for n in names for field in ('vertices', 'faces')}
    if not names or set(arrays) != expected:
        raise ValueError('input must contain complete named vertices/faces pairs only')
    cases = {}
    for name in names:
        v, f = arrays[name+'__vertices'], arrays[name+'__faces']
        if v.dtype != np.float32 or f.dtype != np.int32:
            raise ValueError('input dtypes must be float32 vertices and int32 faces')
        if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or not len(v) or not len(f):
            raise ValueError('input must contain nonempty Vx3/Fx3 meshes')
        if not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
            raise ValueError('nonfinite position or out-of-range face')
        cases[name] = (v, f)
    return cases


def validate_native(v, f, actual_v, actual_f, normals):
    if not np.array_equal(v, actual_v) or not np.array_equal(f, actual_f):
        raise ValueError('normal estimator changed mesh identity')
    if normals.shape != v.shape or normals.dtype != np.float32:
        raise ValueError('native normal array shape/dtype mismatch')


def normal_stats(normals):
    finite = np.isfinite(normals).all(axis=1)
    length = np.linalg.norm(normals[finite].astype(np.float64), axis=1)
    return dict(vertices=len(normals), nonfinite_vectors=int((~finite).sum()),
                zero_vectors=int((length == 0).sum()),
                nonunit_nonzero_vectors=int(((length > 0) & (np.abs(length-1) > 1e-5)).sum()))


def compare(a, b):
    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    aa, bb = a[finite].astype(np.float64), b[finite].astype(np.float64)
    la, lb = np.linalg.norm(aa,axis=1), np.linalg.norm(bb,axis=1)
    valid = (la > 0) & (lb > 0)
    angle = np.degrees(np.arccos(np.clip(np.sum(aa[valid]*bb[valid],axis=1)/(la[valid]*lb[valid]),-1,1)))
    return dict(compared_vectors=int(valid.sum()), excluded_vectors=int(len(a)-valid.sum()),
                degrees=dict(zip(('p50','p90','p99','max'),np.percentile(angle,[50,90,99,100]).tolist())) if len(angle) else None,
                over_10_degrees=int((angle > 10).sum()), over_45_degrees=int((angle > 45).sum()))


def run(args):
    # Refuse stale/aliased destinations before any write. Input is never mutated.
    paths = [args.input_npz.resolve(), args.output_npz.resolve(), args.output_json.resolve()]
    if len(set(paths)) != 3 or args.output_npz.exists() or args.output_json.exists():
        raise ValueError('outputs must be fresh and distinct from input and each other')
    report = dict(schema='trellis2mlx.normal_estimator_witness.v1', status='started',
                  phase='input', requested_backend=args.backend, effective_backend=None,
                  script_sha256=sha(__file__), argv=sys.argv, cases={})
    started = time.perf_counter()
    try:
        if sha(args.input_npz) != args.expected_input_sha256:
            raise ValueError('input SHA256 mismatch')
        report['input_sha256'] = args.expected_input_sha256
        with np.load(args.input_npz, allow_pickle=False) as data:
            cases = validate_cases(dict(data))
        report['phase'] = 'runtime'
        if args.backend == 'cuda':
            from source_cuda_cumesh_postprocess_witness import prepare_release_runtime, _validate_effective_route
            if args.work_dir is None or args.work_dir.exists():
                raise ValueError('CUDA requires a fresh work-dir')
            report['setup_commands'] = []
            report['requested_route'] = dict(target_faces=0)
            runtime = prepare_release_runtime(work_dir=args.work_dir, report=report)
            _validate_effective_route(runtime.effective_route)
            report['effective_route'] = dict(runtime.effective_route, geometry_route='normals-only-no-postprocess')
            report['effective_backend'] = 'source-cumesh-cuda'
        else:
            import trimesh
            report['trimesh_version'] = trimesh.__version__
            report['effective_backend'] = 'cpu-trimesh-and-provisional-float64-area'
            report['area_reference_limit'] = 'Not live CUDA; first incident face follows input order; FP64 sums, not native FP32 adjacency accumulation.'
        output = {}
        for name, (v, f) in cases.items():
            report['phase'] = 'normals:'+name
            case_started = time.perf_counter()
            output[name+'__vertices'], output[name+'__faces'] = v, f
            if args.backend == 'cuda':
                t = runtime.torch
                mesh = runtime.cumesh.CuMesh()
                mesh.init(t.from_numpy(v).cuda(), t.from_numpy(f).cuda())
                mesh.compute_vertex_normals()
                n = mesh.read_vertex_normals().detach().cpu().numpy()
                actual_v, actual_f = mesh.read()
                validate_native(v,f,actual_v.detach().cpu().numpy(),actual_f.detach().cpu().numpy(),n)
                output[name+'__cuda_normals'] = n
                row = dict(cuda=normal_stats(n))
            else:
                angle = np.asarray(trimesh.Trimesh(vertices=v,faces=f,process=False).vertex_normals,dtype=np.float32)
                area = area_normals(v,f)
                output[name+'__trimesh_normals'],output[name+'__area_normals'] = angle,area
                row = dict(trimesh=normal_stats(angle),area=normal_stats(area),angle_vs_area=compare(angle,area))
            report['cases'][name] = dict(row,faces=len(f),seconds=time.perf_counter()-case_started)
        report['phase'] = 'write'
        args.output_npz.parent.mkdir(parents=True,exist_ok=True)
        with args.output_npz.open('xb') as stream:
            np.savez_compressed(stream,**output)
        report.update(status='completed',phase='complete',output_sha256=sha(args.output_npz))
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['seconds'] = time.perf_counter()-started
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        with args.output_json.open('x') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
        print(json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-npz',type=Path,required=True)
    p.add_argument('--expected-input-sha256',required=True)
    p.add_argument('--backend',choices=['cpu','cuda'],required=True)
    p.add_argument('--output-npz',type=Path,required=True)
    p.add_argument('--output-json',type=Path,required=True)
    p.add_argument('--work-dir',type=Path)
    run(p.parse_args())


if __name__=='__main__':
    main()

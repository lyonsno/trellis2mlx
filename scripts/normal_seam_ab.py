"""CPU-only saved-asset diagnostic: alter NORMAL bytes, nothing else in a GLB.

This tests normal connectivity across UV seams using the existing Trimesh normal
estimator. It is not a claim of bitwise CuMesh normal-estimator parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import trimesh


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export_axes(vertices):
    out = np.array(vertices, copy=True)
    out[:, 1], out[:, 2] = vertices[:, 2], -vertices[:, 1]
    return out


def mapped_normals(vertices, faces, uv_vertices, uv_faces, vmapping):
    mesh = trimesh.Trimesh(vertices=export_axes(vertices), faces=faces, process=False)
    return np.asarray(mesh.vertex_normals[vmapping], dtype='<f4')


def read_glb(data):
    if len(data) < 20 or struct.unpack_from('<4sII', data) != (b'glTF', 2, len(data)):
        raise ValueError('invalid GLB header')
    chunks = []
    offset = 12
    while offset < len(data):
        size, kind = struct.unpack_from('<II', data, offset)
        start = offset + 8
        if start + size > len(data):
            raise ValueError('truncated GLB chunk')
        chunks.append((kind, start, size))
        offset = start + size
    if len(chunks) != 2 or [c[0] for c in chunks] != [0x4e4f534a, 0x004e4942]:
        raise ValueError('requires one JSON and one embedded BIN chunk')
    doc = json.loads(data[chunks[0][1]:chunks[0][1]+chunks[0][2]])
    if len(doc.get('meshes', [])) != 1 or len(doc['meshes'][0]['primitives']) != 1:
        raise ValueError('requires one mesh primitive')
    primitive = doc['meshes'][0]['primitives'][0]
    if primitive.get('mode', 4) != 4 or primitive.get('targets') or primitive.get('extensions'):
        raise ValueError('requires uncompressed unmorphed triangles')
    if doc.get('skins') or doc.get('animations'):
        raise ValueError('requires static mesh')
    return doc, primitive, chunks[1][1], chunks[1][2]


def accessor(data, doc, bin_start, bin_length, index):
    a = doc['accessors'][index]
    if a.get('sparse') or a.get('normalized'):
        raise ValueError('sparse/normalized accessor unsupported')
    view = doc['bufferViews'][a['bufferView']]
    if view.get('buffer', 0) != 0:
        raise ValueError('external buffer unsupported')
    dtype = {5126:'<f4',5125:'<u4',5123:'<u2'}[a['componentType']]
    width = {'VEC3':3,'SCALAR':1}[a['type']]
    item_size = np.dtype(dtype).itemsize * width
    if view.get('byteStride', item_size) != item_size:
        raise ValueError('interleaved accessor unsupported')
    relative = view.get('byteOffset', 0) + a.get('byteOffset', 0)
    size = a['count'] * item_size
    if relative < 0 or relative + size > bin_length or a.get('byteOffset',0)+size > view['byteLength']:
        raise ValueError('accessor outside buffer')
    start = bin_start + relative
    array = np.frombuffer(data, dtype=dtype, count=a['count']*width, offset=start)
    return array.reshape(a['count'], width), start, start+size


def canonical_faces(faces):
    faces = np.asarray(faces)
    first = faces.argmin(axis=1)
    rotated = np.take_along_axis(faces, (first[:,None]+np.arange(3)) % 3, axis=1)
    return rotated[np.lexsort(rotated.T[::-1])]


def patch_asset(data, vertices, faces, uv_vertices, uv_faces, vmapping):
    vmapping = np.asarray(vmapping)
    if vmapping.ndim != 1 or vmapping.dtype.kind not in 'iu' or len(vmapping) != len(uv_vertices):
        raise ValueError('invalid UV vertex mapping')
    if len(vmapping) == 0 or vmapping.min() < 0 or vmapping.max() >= len(vertices):
        raise ValueError('UV vertex mapping out of range')
    if not np.array_equal(uv_vertices, vertices[vmapping]):
        raise ValueError('UV checkpoint positions disagree with clean vertex mapping')
    if not np.array_equal(canonical_faces(faces), canonical_faces(vmapping[uv_faces])):
        raise ValueError('UV checkpoint topology/winding differs from clean mesh')
    doc, primitive, bin_start, bin_length = read_glb(data)
    attrs = primitive['attributes']
    pos, _, _ = accessor(data, doc, bin_start, bin_length, attrs['POSITION'])
    idx, _, _ = accessor(data, doc, bin_start, bin_length, primitive['indices'])
    old, begin, end = accessor(data, doc, bin_start, bin_length, attrs['NORMAL'])
    normal_accessor = doc['accessors'][attrs['NORMAL']]
    if normal_accessor['componentType'] != 5126 or normal_accessor['type'] != 'VEC3':
        raise ValueError('normals must be float32 VEC3')
    # NORMAL must own its view; prevent image/other-attribute alias corruption.
    normal_view = normal_accessor['bufferView']
    view = doc['bufferViews'][normal_view]
    if normal_accessor.get('byteOffset',0) or view['byteLength'] != end-begin:
        raise ValueError('normal buffer view must be exclusive')
    for n, a in enumerate(doc['accessors']):
        if n != attrs['NORMAL'] and a.get('bufferView') == normal_view:
            raise ValueError('normal buffer aliases another accessor')
    for n, v in enumerate(doc['bufferViews']):
        a = bin_start + v.get('byteOffset', 0)
        b = a + v['byteLength']
        if n != normal_view and a < end and b > begin:
            raise ValueError('normal buffer aliases another view')
    if any(i.get('bufferView') == normal_view for i in doc.get('images', [])):
        raise ValueError('normal buffer aliases an image')
    if not np.array_equal(pos, export_axes(uv_vertices).astype('<f4')):
        raise ValueError('GLB positions do not match UV checkpoint')
    if not np.array_equal(idx.reshape(-1,3), uv_faces):
        raise ValueError('GLB faces do not match UV checkpoint')
    new = mapped_normals(vertices, faces, uv_vertices, uv_faces, vmapping)
    if new.shape != old.shape or not np.isfinite(new).all():
        raise ValueError('invalid candidate normals')
    for key, value in [('min',new.min(axis=0)),('max',new.max(axis=0))]:
        if key in normal_accessor and not np.array_equal(value, np.asarray(normal_accessor[key], dtype='<f4')):
            raise ValueError('normal min/max would require JSON changes')
    output = data[:begin] + new.tobytes() + data[end:]
    if len(output) != len(data) or output[:begin] != data[:begin] or output[end:] != data[end:]:
        raise AssertionError('protected GLB bytes changed')
    lengths = np.linalg.norm(new, axis=1)*np.linalg.norm(old,axis=1)
    valid = lengths > 1e-10
    angles = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',old[valid],new[valid])/lengths[valid],-1,1)))
    report = dict(
        only_normal_bytes_changed=True, vertices=len(pos), faces=len(uv_faces),
        normal_byte_span=[begin,end], changed_normal_vectors=int(np.any(new!=old,axis=1).sum()),
        angle_degrees_percentiles=dict(zip(['p50','p90','p99','max'],map(float,np.percentile(angles,[50,90,99,100])))),
        zero_candidate_normals=int((np.linalg.norm(new,axis=1)<1e-10).sum()),
        protected_prefix_sha256=digest(data[:begin]), protected_suffix_sha256=digest(data[end:]),
        original_sha256=digest(data), candidate_sha256=digest(output),
        normal_estimator='Trimesh angle-weighted on clean topology, mapped through saved UV vmapping',
        claim_limit='UV seam normal connectivity only; no CuMesh estimator parity or geometry repair claim',
    )
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glb', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = dict(status='started', phase='input', argv=sys.argv, backend='CPU/numpy/trimesh',
                  script_sha256=digest(Path(__file__).read_bytes()), trimesh_version=trimesh.__version__)
    start = time.perf_counter()
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        files = [args.glb,args.checkpoints/'mesh_clean.npz',args.checkpoints/'mesh_uv.npz']
        report['inputs'] = {str(p):digest(p.read_bytes()) for p in files}
        with np.load(files[1], allow_pickle=False) as clean, np.load(files[2], allow_pickle=False) as uv:
            report['phase']='normals-and-patch'
            output, measured = patch_asset(args.glb.read_bytes(), clean['vertices'],clean['faces'],
                uv['vertices'],uv['faces'],uv['vmapping'])
        report.update(measured)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        report['phase']='write'
        with args.output.open('xb') as stream:
            stream.write(output)
        if digest(args.output.read_bytes()) != measured['candidate_sha256']:
            raise AssertionError('written GLB checksum mismatch')
        report.update(status='completed',phase='complete',output=str(args.output))
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['seconds']=time.perf_counter()-start
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()

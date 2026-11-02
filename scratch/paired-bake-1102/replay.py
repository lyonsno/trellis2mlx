"""Finite kiln experiment: existing original baker, frozen geometry and UVs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    out = Path(plan['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'report.json'
    with report_path.open('x'):
        pass
    report = dict(status='running', phase='preflight', plan=plan, outputs={},
                  claim_limit='Two retained sampler profiles through common CUDA decoder and identical original Metal bake; no hosted-draw or isolated BF16 causality.')

    def write(phase=None, **values):
        if phase:
            report['phase'] = phase
        report.update(values)
        tmp = out / '.report.tmp'
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(report_path)
        print(json.dumps(dict(status=report['status'], phase=report['phase'])), flush=True)

    write()
    try:
        root = Path(plan['source_root'])
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != plan['revision'] or sha(__file__) != plan['driver_sha256']:
            raise ValueError('source or driver identity mismatch')
        for name, record in plan['inputs'].items():
            if sha(record['path']) != record['sha256']:
                raise ValueError('input digest mismatch: ' + name)
        sys.path.insert(0, str(root))
        import numpy as np
        import trimesh
        from PIL import Image
        import trellmlx.source_metal_remesh as sm
        from scripts.source_metal_remesh_finishing import verify_backend

        backend = sm.load_backend()
        identity = verify_backend(backend, Path(plan['backend_manifest']))
        write('load_frozen_geometry', effective_backend=identity,
              source_revision=revision, grid_size=1024, atlas_size=4096)
        t = backend.torch
        inputs = plan['inputs']
        with np.load(inputs['uv']['path'], allow_pickle=False) as z:
            v, f, uv, vmaps = [t.from_numpy(z[k]) for k in ('vertices', 'faces', 'uvs', 'vmaps')]
        with np.load(inputs['normals']['path'], allow_pickle=False) as z:
            normals = t.from_numpy(z['normals'])
        with np.load(inputs['filled']['path'], allow_pickle=False) as z:
            ov, of = [t.from_numpy(z[k]) for k in ('vertices', 'faces')]
        if v.shape != (380392, 3) or f.shape != (483981, 3) or uv.shape != (380392, 2):
            raise ValueError('unexpected frozen UV geometry')
        if ov.shape != (9546260, 3) or of.shape != (19306034, 3):
            raise ValueError('unexpected frozen original surface')
        base = trimesh.load(inputs['baseline_glb']['path'], force='scene', process=False)
        if len(base.geometry) != 1:
            raise ValueError('baseline must have exactly one mesh')
        baseline = next(iter(base.geometry.values()))

        def geometry_equal(mesh):
            for a, b in ((mesh.vertices, baseline.vertices), (mesh.faces, baseline.faces),
                         (mesh.visual.uv, baseline.visual.uv), (mesh.vertex_normals, baseline.vertex_normals)):
                if not np.array_equal(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)):
                    raise ValueError('exported geometry/UV/normals differ from baseline')

        class FrozenMesh:
            def uv_unwrap(self, **kwargs):
                return v, f, uv, vmaps

            def compute_vertex_normals(self):
                pass

            def read_vertex_normals(self):
                return normals

        write('build_original_bvh')
        bvh = backend.bvh(ov, of)
        frozen = FrozenMesh()
        previous_coords = None
        sample_calls = {}
        real_sample = backend.sample
        query_sha = None

        for name in ('production', 'fp16'):
            write('load_' + name)
            with np.load(inputs[name]['path'], allow_pickle=False) as z:
                attrs, coords = z['tex_np'], z['tex_coords_spatial']
                if int(z['grid_size'].item()) != 1024:
                    raise ValueError('material grid mismatch')
            if attrs.shape != (9539061, 6) or coords.shape != (9539061, 3) or not np.isfinite(attrs).all():
                raise ValueError('incomplete/nonfinite material field')
            if previous_coords is not None and not np.array_equal(coords, previous_coords):
                raise ValueError('material coordinate order mismatch')
            previous_coords = coords.copy()

            def sample(*args, **kwargs):
                nonlocal query_sha
                q = kwargs['grid'].numpy()
                h = hashlib.sha256(q.tobytes()).hexdigest()
                if query_sha is not None and query_sha != h:
                    raise ValueError('paired original-surface queries differ')
                query_sha = h
                result = real_sample(*args, **kwargs)
                if result.numel() == 0 or not t.isfinite(result).all():
                    raise ValueError('blank/nonfinite material sample')
                sample_calls[name] = dict(query_sha256=h, queries=int(q.shape[1]),
                                          shape=list(result.shape))
                return result

            backend.sample = sample
            # The intervention freezes the stages before rasterization. All
            # projection, sparse sampling, inpaint, material and export code is
            # the existing production function, called without source edits.
            with patch.object(sm, 'rebuild_surface', return_value=(frozen, ov, of, bvh)):
                mesh = sm.to_glb(v, f, t.from_numpy(attrs), t.from_numpy(coords),
                                 backend=backend, grid_size=1024, texture_size=4096,
                                 phase=lambda value: write(name + '_' + value))
            geometry_equal(mesh)
            output = out / (name + '.glb')
            mesh.export(output)
            reopened = trimesh.load(output, force='scene', process=False)
            if len(reopened.geometry) != 1:
                raise ValueError('blank/partial export')
            actual = next(iter(reopened.geometry.values()))
            geometry_equal(actual)
            material = actual.visual.material
            if material.alphaMode != 'OPAQUE' or material.doubleSided:
                raise ValueError('material policy changed')
            maps = {}
            for label, image in [('base_color', material.baseColorTexture),
                                 ('metal_roughness', material.metallicRoughnessTexture)]:
                if image is None or image.size != (4096, 4096):
                    raise ValueError('missing/wrong atlas')
                data = np.asarray(image)
                if label == 'base_color' and np.ptp(data[..., :3]) == 0:
                    raise ValueError('blank kiln color atlas')
                path = out / (name + '-' + label + '.png')
                Image.fromarray(data).save(path)
                maps[label] = dict(path=str(path), sha256=sha(path))
            report['outputs'][name] = dict(path=str(output), sha256=sha(output),
                                          bytes=output.stat().st_size, maps=maps,
                                          geometry_uv_normals_equal=True)
            write(name + '_exported')
            del attrs, mesh, reopened, actual, material
        comparison = {}
        for label in ('base_color', 'metal_roughness'):
            a, b = [np.asarray(Image.open(out / (name + '-' + label + '.png'))).astype(np.int16)
                    for name in ('production', 'fp16')]
            delta = np.abs(a - b)
            comparison[label] = dict(max_per_channel=delta.max(axis=(0, 1)).tolist(),
                                     mean_per_channel=delta.mean(axis=(0, 1)).tolist(),
                                     pixels_over_one=int((delta > 1).any(axis=-1).sum()))
        if set(sample_calls) != {'production', 'fp16'}:
            raise ValueError('missing material sampling invocation')
        write('complete', status='complete', sample_calls=sample_calls, atlas_comparison=comparison)
    except Exception as exc:
        write(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))

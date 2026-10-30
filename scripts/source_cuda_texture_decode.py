#!/usr/bin/env python3
"""Decode a retained MLX texture latent with pinned, unmodified CUDA TRELLIS."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

TRELLIS = ('https://github.com/microsoft/TRELLIS.2.git', '5565d240c4a494caaf9ece7a554542b76ffa36d3')
FLEX = ('https://github.com/JeffreyXiang/FlexGEMM.git', '6dd94a859c26ee8246888502eada3dd8ad85532e')
MODEL = 'microsoft/TRELLIS.2-4B'
REVISION = 'af44b45f2e35a493886929c6d786e563ec68364d'
STEM = 'ckpts/tex_dec_next_dc_f16c32_fp16'
MODEL_HASHES = {
    STEM+'.json': '22074e4ab5b28e2b72d33a3fe61bf416134b89e12d0fc5c729f37da75da0c4d1',
    STEM+'.safetensors': '97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c',
    'pipeline.json': '222c359ab1ed9bc6735a640a34f95d47f8681b9bc4aaa101bfb80274676253c6',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def children(coords, guide):
    if (coords.ndim != 2 or coords.shape[1] != 4 or coords.dtype != np.int32
            or guide.shape != (len(coords), 8) or not np.isfinite(guide).all()):
        raise ValueError('invalid subdivision guide shape/dtype/values')
    parent, child = np.nonzero(guide > 0)
    if not len(parent): raise ValueError('empty subdivision guide')
    out = coords[parent].copy()
    out[:, 1:] = out[:, 1:]*2 + ((child[:, None] >> np.arange(3)) & 1)
    return out


def compare(got, got_coords, reference, ref_coords):
    if (got.ndim != 2 or got.shape[1] != 6 or not len(got)
            or got.shape != reference.shape or got_coords.shape != (len(got), 3)
            or not np.array_equal(got_coords, ref_coords)):
        raise ValueError('partial field or coordinate order/shape mismatch')
    if not np.isfinite(got).all() or not np.isfinite(reference).all():
        raise ValueError('nonfinite material field')
    error = np.abs(got.astype(np.float64)-reference)
    return dict(rows=len(got), coordinate_order_exact=True,
                max_abs=error.max(axis=0).tolist(), mean_abs=error.mean(axis=0).tolist(),
                rmse=np.sqrt(np.mean(error*error, axis=0)).tolist(),
                count_abs_gt_1_over_255=(error > 1/255).sum(axis=0).tolist())


def validate_route(cuda_available, backend):
    if not cuda_available or backend != 'flex_gemm':
        raise ValueError('native CUDA FlexGEMM is required; fallback cannot answer this comparison')


def command(argv, report, save, cwd=None):
    r = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    report.setdefault('commands', []).append(dict(argv=argv, cwd=str(cwd),
                          exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr))
    save()
    if r.returncode: raise RuntimeError(f'command failed: {argv}')
    return r.stdout.strip()


def clone(root, identity, report, save):
    url, revision = identity
    if root.exists(): raise ValueError(f'fresh source directory required: {root}')
    command(['git', 'clone', '--filter=blob:none', '--no-checkout', url, str(root)], report, save)
    command(['git', 'checkout', '--detach', revision], report, save, root)
    command(['git', 'submodule', 'update', '--init', '--recursive'], report, save, root)
    return root


def run_decode(args, manifest, report, save):
    report['phase'] = 'runtime_build'; save()
    import torch
    validate_route(torch.cuda.is_available(), 'flex_gemm')
    os.environ['SPARSE_CONV_BACKEND'] = 'flex_gemm'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '.'.join(map(str, torch.cuda.get_device_capability()))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = clone(args.work_dir/'TRELLIS.2', TRELLIS, report, save)
    flex = clone(args.work_dir/'FlexGEMM', FLEX, report, save)
    command([sys.executable, '-m', 'pip', 'install', 'easydict', 'ninja', 'einops',
             'tqdm', 'huggingface-hub', 'safetensors'], report, save)
    command([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '--no-deps', str(flex)], report, save)
    sys.path.insert(0, str(source))
    from trellis2.modules import sparse as sp
    from trellis2.modules.sparse import config
    from trellis2.modules.sparse.conv import config as conv_config
    from trellis2.modules.sparse.conv import conv_flex_gemm
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    import flex_gemm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    validate_route(torch.cuda.is_available(), config.CONV)
    # A fresh interpreter and explicit source path must resolve the intended implementation.
    if not Path(inspect.getfile(SparseUnetVaeDecoder)).is_relative_to(source):
        raise ValueError('decoder imported from wrong source')
    for root, (_, revision) in [(source, TRELLIS), (flex, FLEX)]:
        if command(['git', 'rev-parse', 'HEAD'], report, save, root) != revision:
            raise ValueError('source revision mismatch')
        if command(['git', 'diff', '--exit-code', 'HEAD'], report, save, root):
            raise ValueError('source modified')
    report['effective_route'] = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
        cuda=torch.version.cuda, trellis=TRELLIS[1], flex_gemm=FLEX[1], model_revision=REVISION,
        backend=config.CONV, algorithm=conv_config.FLEX_GEMM_ALGO,
        hashmap_ratio=conv_config.FLEX_GEMM_HASHMAP_RATIO,
        decoder_path=inspect.getfile(SparseUnetVaeDecoder), flex_path=flex_gemm.__file__,
        conv_path=inspect.getfile(conv_flex_gemm), normalization='float32 multiply then add once',
        material_mapping='decoder_output * 0.5 + 0.5; no clamp',
        packages={x: importlib.metadata.version(x) for x in ['torch','numpy','safetensors','huggingface-hub']})
    report['phase'] = 'native_convolution_preflight'; save()
    # Dense 3D convolution is an independent oracle for a small complete sparse cube.
    torch.manual_seed(42)
    xyz = torch.cartesian_prod(*[torch.arange(3, device='cuda', dtype=torch.int32)]*3)
    coords = torch.cat([torch.zeros((27,1), device='cuda', dtype=torch.int32), xyz], dim=1)
    feats = torch.randn((27,32), device='cuda', dtype=torch.float16)
    conv = sp.SparseConv3d(32,32,3).cuda().half()
    with torch.inference_mode():
        actual = conv(sp.SparseTensor(feats=feats, coords=coords)).feats
        dense = feats.T.reshape(1,32,3,3,3)
        expected = torch.nn.functional.conv3d(dense, conv.weight.permute(0,4,1,2,3), conv.bias, padding=1)
        expected = expected.reshape(32,-1).T
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
        report['native_convolution_preflight'] = dict(rows=27, max_abs=float((actual-expected).abs().max()))
    del actual, expected, conv, feats, dense
    report['last_trustworthy_phase'] = report['phase']; save()
    report['phase'] = 'model_load'; save()
    files = {}
    for name, sha in MODEL_HASHES.items():
        p = Path(hf_hub_download(MODEL, name, revision=REVISION, cache_dir=str(args.work_dir/'hf')))
        if digest(p) != sha: raise ValueError(f'model digest mismatch: {name}')
        files[name] = p
    cfg = json.loads(files[STEM+'.json'].read_text())
    if cfg['name'] != 'SparseUnetVaeDecoder': raise ValueError('wrong model class')
    model = SparseUnetVaeDecoder(**cfg['args'])
    model.load_state_dict(load_file(str(files[STEM+'.safetensors'])), strict=True)
    model.eval().cuda()
    normalization = json.loads(files['pipeline.json'].read_text())['args']['tex_slat_normalization']
    capsule = args.manifest.parent
    with np.load(capsule/'latent.npz') as z: latent, coords = z['feats'], z['coords']
    if latent.shape != (len(coords),32) or latent.dtype != np.float32 or not np.isfinite(latent).all():
        raise ValueError('invalid normalized latent')
    latent = torch.from_numpy(latent).cuda()
    latent = latent*torch.tensor(normalization['std'], device='cuda', dtype=torch.float32)
    latent = latent+torch.tensor(normalization['mean'], device='cuda', dtype=torch.float32)
    x = sp.SparseTensor(feats=latent, coords=torch.from_numpy(coords).cuda())
    guides = []
    with np.load(capsule/'guides.npz') as z:
        for i in range(4):
            g = z[f'shape_subs_{i}']
            guides.append(sp.SparseTensor(feats=torch.from_numpy(g).cuda(), coords=torch.from_numpy(coords).cuda()))
            coords = children(coords, g)
    with np.load(capsule/'reference.npz') as z:
        ref_coords, reference = z['tex_coords_spatial'], z['tex_np']
    if not np.array_equal(coords[:,1:], ref_coords): raise ValueError('guide expansion differs from retained field')
    report['model_config'] = cfg
    report['input_rows'] = len(x.feats); report['expected_output_rows'] = len(reference)
    report['phase'] = 'texture_decode'; save()
    report['blocks'] = []
    def hook(name):
        def record(_module, _inputs, output):
            torch.cuda.synchronize()
            report['blocks'].append(dict(name=name, rows=len(output.feats), dtype=str(output.feats.dtype),
                                       peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            report['last_trustworthy_phase'] = name; save()
        return record
    for i, group in enumerate(model.blocks):
        for j, block in enumerate(group): block.register_forward_hook(hook(f'block_{i}_{j}'))
    with torch.inference_mode():
        out = model(x, guide_subs=guides)
        got = (out.feats.float()*0.5+0.5).cpu().numpy()
        got_coords = out.coords[:,1:].cpu().numpy()
    report['phase'] = 'output_capture'; save()
    # Preserve the complete raw CUDA result even if later comparison detects a mismatch.
    np.savez_compressed(args.output_npz, tex_np=got, tex_coords_spatial=got_coords, grid_size=np.int32(1024))
    report['output'] = dict(path=str(args.output_npz), sha256=digest(args.output_npz), size_bytes=args.output_npz.stat().st_size)
    report['phase'] = 'comparison'; save()
    report['comparison'] = compare(got, got_coords, reference, ref_coords)
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['last_trustworthy_phase'] = 'complete_field_comparison'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output-json', type=Path, required=True)
    p.add_argument('--output-npz', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, default=Path('/kaggle/temp/texture-decoder-runtime'))
    args = p.parse_args(argv)
    report = dict(schema='trellis2mlx.retained_texture_cuda_decode.v1', status='running',
                  phase='input_admission', last_trustworthy_phase='arguments_parsed')
    save = lambda: write_json(args.output_json, report)
    save()
    try:
        if digest(args.manifest) != args.manifest_sha256: raise ValueError('manifest digest mismatch')
        manifest = json.loads(args.manifest.read_text())
        if set(manifest['files']) != {'latent.npz','guides.npz','reference.npz'}:
            raise ValueError('missing or unexpected input roles')
        for name, expected in manifest['files'].items():
            if digest(args.manifest.parent/name) != expected['sha256']:
                raise ValueError(f'input digest mismatch: {name}')
        if args.output_npz.exists(): raise ValueError('output already exists; stale result refused')
        report['inputs'] = manifest; report['last_trustworthy_phase'] = 'inputs_admitted'; save()
        run_decode(args, manifest, report, save)
        report['status'] = 'completed'; save(); return 0
    except BaseException as exc:
        report.update(status='failed', failure_phase=report['phase'], error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc())
        save(); return 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Compare the exact retained kiln image's source CUDA DINO features to MLX.

One image encoder forward; no generative sampling, remasking or mesh work.
The source extractor is an authenticated, unmodified file from TRELLIS.2.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

SOURCE_SHA = '12530b23e8b6a2cc6b87d8cd01922c7b0085199a365b731f19dd0e7ef4919150'
SOURCE_REV = '5565d240c4a494caaf9ece7a554542b76ffa36d3'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def compare(actual, reference, shape):
    for name, array in [('CUDA', actual), ('MLX', reference)]:
        if array.shape != tuple(shape) or array.dtype != np.float32:
            raise ValueError(f'{name} shape/dtype mismatch: {array.shape}/{array.dtype}')
        if not np.isfinite(array).all():
            raise ValueError(f'{name} nonfinite features')
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {'shape': list(actual.shape), 'elements': actual.size,
            'changed_elements': int(np.count_nonzero(delta)),
            'max_abs': float(np.abs(delta).max()),
            'mean_abs': float(np.abs(delta).mean()),
            'rmse': float(np.sqrt(np.mean(delta * delta)))}


def admit_device(name, capability):
    if name != 'Tesla T4' or list(capability or []) != [7, 5]:
        raise ValueError(f'Expected Tesla T4 CUDA route, got {name}/{capability}')


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output-json', '--report', dest='report', type=Path,
                   default=Path('cuda_image_features_report.json'))
    p.add_argument('--output-npz', '--output', dest='output', type=Path,
                   default=Path('cuda_image_features.npz'))
    return p


def run(args):
    report = {'schema': 'trellis2mlx.kiln_source_image_features.v1',
              'run_id': args.run_id, 'status': 'running', 'phase': 'input_admission',
              'claim': 'Same prepared image, 1024 DINO CUDA/MLX comparison only',
              'source_revision': SOURCE_REV, 'last_trustworthy_phase': None}
    save(args.report, report)
    try:
        if digest(args.manifest) != args.manifest_sha256:
            raise ValueError('manifest hash mismatch')
        plan = json.loads(args.manifest.read_text())
        if plan['run_id'] != args.run_id or plan['image_size'] != 1024:
            raise ValueError('run identity or image size mismatch')
        files = {}
        for name, expected in plan['files'].items():
            path = args.manifest.parent / name
            actual = digest(path)
            if actual != expected: raise ValueError(f'input hash mismatch: {name}')
            files[name] = {'sha256': actual, 'bytes': path.stat().st_size}
        source = args.manifest.parent / 'source_image_feature_extractor.py'
        if digest(source) != SOURCE_SHA: raise ValueError('original extractor source mismatch')
        report.update(inputs=files, manifest_sha256=args.manifest_sha256,
                      last_trustworthy_phase='authenticated_inputs', phase='runtime_setup')
        save(args.report, report)
        installed = subprocess.run([sys.executable, '-m', 'pip', 'install',
                                    'transformers==5.9.0'], capture_output=True, text=True)
        report['installation'] = {'returncode': installed.returncode,
                                  'stdout': installed.stdout, 'stderr': installed.stderr}
        save(args.report, report)
        installed.check_returncode()
        import torch
        import transformers
        import torchvision
        from PIL import Image
        import PIL
        if not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
        admit_device(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
        # The official extractor loads these FP32 weights without an autocast context.
        spec = importlib.util.spec_from_file_location('original_image_extractor', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_dir = args.output.parent / 'dino-runtime'
        model_dir.mkdir(exist_ok=False)
        for src, dest in [('dino__config.json', 'config.json'),
                          ('dino__model.safetensors', 'model.safetensors')]:
            (model_dir / dest).symlink_to((args.manifest.parent / src).resolve())
        encoder = module.DinoV3FeatureExtractor(str(model_dir), image_size=1024)
        encoder.cuda()
        model = encoder.model
        if next(model.parameters()).dtype != torch.float32 or model.training:
            raise ValueError('source encoder must be eval/float32')
        report['effective_route'] = {'device': torch.cuda.get_device_name(0),
            'capability': list(torch.cuda.get_device_capability(0)), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'transformers': transformers.__version__,
            'torchvision': torchvision.__version__, 'pillow': PIL.__version__,
            'attention': model.config._attn_implementation, 'dtype': 'torch.float32',
            'source_sha256': digest(source),
            'model_revision': plan['model_revision'], 'image_size': encoder.image_size,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32}
        report['phase'] = 'source_cuda_forward'; save(args.report, report)
        image = Image.open(args.manifest.parent / 'prepared.png')
        # Retain the actual normalized tensor handed to source extract_features.
        captured = {}
        original = encoder.extract_features
        def capture(pixels):
            captured['normalized_pixels'] = pixels.detach().cpu().numpy()
            return original(pixels)
        encoder.extract_features = capture
        result = encoder([image])
        torch.cuda.synchronize()
        actual = result.detach().cpu().numpy()
        report['last_trustworthy_phase'] = 'source_cuda_forward'
        report['phase'] = 'complete_array_comparison'; save(args.report, report)
        with np.load(args.manifest.parent / 'reference.npz', allow_pickle=False) as ref:
            comparison = compare(actual, ref['cond'], (1, 4101, 1024))
            if ref['neg_cond'].shape != actual.shape or np.any(ref['neg_cond']):
                raise ValueError('retained negative conditioning is not matched zeros')
        np.savez_compressed(args.output, cond=actual, **captured)
        report.update(status='completed', phase='complete_array_comparison',
            comparison=comparison, last_trustworthy_phase='complete_array_comparison',
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            output={'path': str(args.output), 'sha256': digest(args.output),
                    'size_bytes': args.output.stat().st_size})
        save(args.report, report)
    except BaseException as exc:
        report.update(status='failed', failure_phase=report['phase'], error=repr(exc),
                      traceback=traceback.format_exc())
        save(args.report, report)
        raise
    return report


if __name__ == '__main__':
    run(parser().parse_args())

"""Pinned, continuous MLX inference assay with evaluated sampler checkpoints.

No production defaults or resume semantics change. Finishing is a separate job.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

GENERATOR_SHA = '3d97bcc0517b67ff581d360dafe4d4abc94e07e6c0548ad7920cd61078e5e6ba'
STAGES = ('sparse', 'lr_shape', 'hr_shape', 'texture')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def instrument_source(source):
    anchor = '        hr_slat_flow = SLatFlowModel.for_shape()\n'
    if source.count(anchor) != 1:
        raise ValueError('HR constructor is missing or ambiguous')
    return source.replace(anchor, '        hr_slat_flow = SLatFlowModel.for_shape(lr_sync_every_block=True)\n', 1)


def evaluated_step_line(fn):
    lines, start = inspect.getsourcelines(fn)
    indices = [i for i, line in enumerate(lines) if line.strip() == 'mx.eval(sample)']
    if len(indices) != 1:
        raise ValueError('sampler evaluated-sample boundary is missing or ambiguous')
    i = indices[0]
    if lines[i - 1].strip() != 'sample = sample_next':
        raise ValueError('sampler boundary is not the new Euler sample')
    j = i + 1
    while not lines[j].strip():
        j += 1
    if lines[j].strip() != 'if verbose:':
        raise ValueError('sampler post-evaluation boundary changed')
    return start + j


def require_complete(report, checkpoint_dir):
    import numpy as np

    if [x['stage'] for x in report['samplers']] != list(STAGES):
        raise ValueError('expected all four fresh sampler calls')
    for sampler in report['samplers']:
        if sampler['completed_steps'] != list(range(1, 13)):
            raise ValueError('incomplete sampler steps: ' + sampler['stage'])
    for stage in ('conditioning', 'sparse_coords', 'shape_slat', 'decoder_output', 'mesh_raw', 'texture'):
        path = checkpoint_dir / (stage + '.npz')
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('missing or empty output: ' + stage)
    if report.get('conditioning_backend') != 'mlx':
        raise ValueError('native MLX conditioning was not observed')
    for stage in ('conditioning', 'sparse_coords', 'shape_slat', 'decoder_output', 'mesh_raw', 'texture'):
        with np.load(checkpoint_dir / (stage + '.npz'), allow_pickle=False) as data:
            def array(key, ndim, width=None, integer=False):
                if key not in data:
                    raise ValueError(f'{stage}: missing {key}')
                value = data[key]
                if value.ndim != ndim or not value.size or (width is not None and value.shape[-1] != width):
                    raise ValueError(f'{stage}: invalid {key} shape {value.shape}')
                if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                    raise ValueError(f'{stage}: nonnumeric or nonfinite {key}')
                if integer and not np.issubdtype(value.dtype, np.integer):
                    raise ValueError(f'{stage}: {key} must contain integers')
                return value

            if stage == 'conditioning':
                if array('cond', 3).shape != array('neg_cond', 3).shape:
                    raise ValueError('conditioning: positive/negative shapes differ')
            elif stage == 'mesh_raw':
                vertices, faces = array('vertices', 2, 3), array('faces', 2, 3, integer=True)
                if faces.min() < 0 or faces.max() >= len(vertices):
                    raise ValueError('mesh_raw: face indices outside vertex array')
            elif stage == 'texture':
                if len(array('tex_np', 2, 6)) != len(array('tex_coords_spatial', 2, 3, integer=True)):
                    raise ValueError('texture: feature/coordinate rows differ')
            else:
                coords = array('coords', 2, 4, integer=True)
                if stage != 'sparse_coords' and len(array('feats', 2)) != len(coords):
                    raise ValueError(f'{stage}: feature/coordinate rows differ')
            if stage in ('shape_slat', 'decoder_output', 'mesh_raw', 'texture'):
                metadata_path = checkpoint_dir / (stage + '.json')
                metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
                grids = [metadata['mesh_grid_size']] if 'mesh_grid_size' in metadata else []
                if 'mesh_grid_size' in data:
                    grids.append(data['mesh_grid_size'].item())
                if not grids or any(grid != 1024 for grid in grids):
                    raise ValueError(f'{stage}: missing, inconsistent, or non-1024 grid')


class InferenceComplete(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--expected-revision', required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--image-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'report.json'
    if report_path.exists() or (out / 'checkpoints').exists():
        raise ValueError('fresh assay refuses prior report or checkpoints')
    report = {'schema': 'trellis2mlx.fresh_prepared_inference.v1', 'status': 'running',
              'phase': 'admission', 'samplers': [], 'stage_outputs': [],
              'requested': vars(args) | {'resolution': 1024, 'steps': 12, 'seed': 42,
                                       'background_removal': False},
              'claim_limit': 'Fresh continuous MLX trajectory on exact prepared RGB; same seed does not imply CUDA noise or numerical parity. No finished mesh quality claim.'}
    def write():
        temp = report_path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, default=str) + '\n')
        temp.replace(report_path)
    write()
    try:
        root = args.source_root.resolve()
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != args.expected_revision:
            raise ValueError('source revision differs from request')
        if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip():
            raise ValueError('source worktree must be clean')
        generator = root / 'generate.py'
        if sha(generator) != GENERATOR_SHA or sha(args.image) != args.image_sha256:
            raise ValueError('generator or prepared image SHA differs from request')
        from PIL import Image
        with Image.open(args.image) as image:
            if image.mode != 'RGB':
                raise ValueError('prepared image must already be RGB')
            report['input'] = {'path': str(args.image), 'sha256': sha(args.image),
                               'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                               'size': image.size, 'mode': image.mode}
        report['source'] = {'root': str(root), 'revision': revision, 'generator_sha256': sha(generator),
                            'driver_sha256': sha(__file__)}
        report['free_disk_bytes_at_start'] = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
        os.environ['TRELLIS2MLX_FAST_ATTENTION_QUERY_CHUNK_SIZE'] = '2048'
        os.environ['TRELLIS2MLX_HR_SHAPE_FAST_ATTENTION_QUERY_CHUNK_SIZE'] = '1024'
        sys.path.insert(0, str(root))
        import mlx.core as mx
        import numpy as np
        import trellmlx.samplers as samplers
        import trellmlx.checkpoint as checkpoint
        from importlib.metadata import version
        for module in (samplers, checkpoint):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise ValueError('module origin escaped pinned source: ' + module.__file__)
        report['effective'] = {'mlx_version': version('mlx'), 'device': str(mx.default_device()),
                               'lr_query_tile': 2048, 'hr_query_tile': 1024,
                               'shape_block_sync': 1, 'qk_norm': 'source-cuda-warp32',
                               'cleanup': 'disabled; separate Metal finisher consumes raw and texture',
                               'sampler_sha256': sha(samplers.__file__)}
        cp = out / 'checkpoints'
        cp.mkdir()
        samples = out / 'samplers'
        samples.mkdir()
        original_sampler = samplers.flow_euler_sample
        original_checkpoint = checkpoint.save_checkpoint
        capture_line = evaluated_step_line(original_sampler)
        def save_sample(path, sample, **extra):
            mx.eval(sample, *mx.random.state)
            payload = {'sample': np.asarray(sample), 'rng_state': np.stack([np.asarray(v) for v in mx.random.state]), **extra}
            if not np.isfinite(payload['sample']).all():
                raise ValueError('nonfinite evaluated sampler sample')
            np.savez_compressed(path, **payload)
        def sampled(*pos, **kw):
            index = len(report['samplers'])
            if index >= len(STAGES):
                raise ValueError('unexpected additional sampler call')
            stage = STAGES[index]
            row = {'stage': stage, 'completed_steps': [], 'model': type(pos[0]).__name__}
            report['samplers'].append(row)
            report['phase'] = stage
            write()
            signature = inspect.signature(original_sampler).bind(*pos, **kw)
            signature.apply_defaults()
            bound = signature.arguments
            if bound['steps'] != 12 or bound['start_step_index'] != 0 or bound['stop_after_first_step']:
                raise ValueError('sampler is not a fresh twelve-step call')
            parameters = {k: bound[k] for k in ('steps','guidance_strength','guidance_rescale','guidance_interval','rescale_t','sigma_min')}
            coords = kw.get('coords')
            extra = {'coords': np.asarray(coords)} if coords is not None else {}
            if kw.get('concat_cond') is not None:
                extra['concat_cond'] = np.asarray(kw['concat_cond'])
            extra['parameters_json'] = np.array(json.dumps(parameters))
            save_sample(samples / (stage + '_noise.npz'), pos[1], **extra)
            def trace_step(frame, event, arg):
                if event == 'line' and frame.f_lineno == capture_line:
                    step = int(frame.f_locals['step_idx']) + 1
                    path = samples / f'{stage}_step_{step:02d}.npz'
                    save_sample(path, frame.f_locals['sample'], next_step_index=np.array(step), **extra)
                    row['completed_steps'].append(step)
                    row['last_sample'] = str(path)
                    report['last_trustworthy_evidence'] = str(path)
                    write()
                    print(f'Fresh assay saved {stage} step {step}/12', flush=True)
                return trace_step
            def trace_calls(frame, event, arg):
                return trace_step if event == 'call' and frame.f_code is original_sampler.__code__ else None
            previous = sys.gettrace()
            sys.settrace(trace_calls)
            try:
                return original_sampler(*pos, **kw)
            finally:
                sys.settrace(previous)
        def saved(directory, stage, **arrays):
            original_checkpoint(directory, stage, **arrays)
            report['stage_outputs'].append(stage)
            report['last_trustworthy_evidence'] = str(cp / (stage + '.npz'))
            write()
            if stage == 'texture':
                raise InferenceComplete()
        samplers.flow_euler_sample = sampled
        checkpoint.save_checkpoint = saved
        namespace = {'__name__': '_fresh_assay_generator', '__file__': str(generator)}
        exec(compile(instrument_source(generator.read_text()), str(generator), 'exec'), namespace)
        def native_features(path, resolution=512):
            from trellmlx.models.dinov3 import extract_features
            report['phase'] = 'native_mlx_conditioning'
            write()
            result = extract_features(path, image_size=resolution)
            mx.eval(result)
            report['conditioning_backend'] = 'mlx'
            write()
            return result
        namespace['_extract_image_features'] = native_features
        old_argv = sys.argv
        sys.argv = [str(generator), '--image', str(args.image), '--no-rembg', '--seed', '42', '--resolution', '1024', '--steps', '12', '--target-faces', '0', '--no-cleanup', '--save-checkpoints', str(cp), '--output', str(out / 'unused.glb'), '--shape-flow-attention-backend', 'fast', '--qk-norm-backend', 'source-cuda-warp32', '--lr-shape-sync-every-block']
        report['generator_argv'] = sys.argv
        write()
        try:
            namespace['main']()
            raise RuntimeError('generator returned before fresh texture checkpoint')
        except InferenceComplete:
            require_complete(report, cp)
        finally:
            sys.argv = old_argv
            samplers.flow_euler_sample = original_sampler
            checkpoint.save_checkpoint = original_checkpoint
        report.update(status='passed', phase='fresh_raw_and_texture_complete',
                      artifacts={p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in cp.iterdir() if p.is_file()})
        write()
        return 0
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        write()
        raise


if __name__ == '__main__':
    raise SystemExit(main())

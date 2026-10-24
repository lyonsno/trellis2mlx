"""Replay one saved LR Euler step; an isolated execution probe, not a full run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from fresh_prepared_inference import sha


def validate_parent_boundary(manifest, parent, start):
    """Admit the last sample by default, or an explicitly selected retained step."""
    row = next(row for row in parent['samplers'] if row['stage'] == 'lr_shape')
    completed = row['completed_steps']
    if (start != manifest['expected_start_step_index'] or not completed
            or completed != list(range(1, completed[-1] + 1)) or start not in completed):
        raise ValueError('sample step differs from recorded parent steps')
    selection = manifest.get('parent_sample_selection', 'latest')
    last = Path(row['last_sample']).resolve()
    if selection == 'latest':
        expected = last
        if start != completed[-1]:
            raise ValueError('latest selection requires parent completion boundary')
    elif selection == 'retained_completed':
        expected = last.with_name(f'lr_shape_step_{start:02d}.npz')
    else:
        raise ValueError('unknown parent sample selection')
    if Path(manifest['sample']['path']).resolve() != expected:
        raise ValueError('sample path differs from selected parent boundary')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    out = Path(manifest['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'report.json'
    if report_path.exists():
        raise ValueError('refusing to overwrite an existing replay report')
    report = {'schema': 'trellis2mlx.prepared_lr_step_replay.v1', 'status': 'running',
              'phase': 'admission', 'requested': manifest,
              'claim_limit': 'One saved LR step in a fresh process. No continuous trajectory, root-cause or mesh-quality conclusion.'}
    def write():
        temp = report_path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(report_path)
    write()
    try:
        root = Path(manifest['source_root']).resolve()
        actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != manifest['expected_revision']:
            raise ValueError('source revision mismatch')
        if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip():
            raise ValueError('source checkout is dirty')
        for name in ('sample', 'conditioning', 'parent_report', 'sampler_source', 'model_source'):
            item = manifest[name]
            if sha(item['path']) != item['sha256']:
                raise ValueError(name + ' SHA mismatch')
        parent = json.loads(Path(manifest['parent_report']['path']).read_text())
        if parent['phase'] != 'lr_shape' or parent['status'] != 'failed':
            raise ValueError('parent is not a failed LR shape run')
        effective = parent['effective']
        os.environ['TRELLIS2MLX_FAST_ATTENTION_QUERY_CHUNK_SIZE'] = str(effective['lr_query_tile'])
        os.environ['TRELLIS2MLX_QK_NORM_BACKEND'] = effective['qk_norm']
        os.environ['TRELLIS2MLX_ATTENTION_BACKEND'] = 'fast'
        if os.environ.get('TRELLIS2MLX_HR_SHAPE_ATTENTION_ACTIVE'):
            raise ValueError('unexpected HR attention marker')
        sys.path.insert(0, str(root))
        import numpy as np
        import mlx.core as mx
        from importlib.metadata import version
        import trellmlx.samplers as samplers
        import trellmlx.models.slat_flow as flow
        from trellmlx.weight_loader import load_weights
        from trellmlx.modules.attention import get_qk_norm_backend
        if version('mlx') != effective['mlx_version'] or str(mx.default_device()) != effective['device']:
            raise ValueError('MLX version/device differs from parent')
        for module, name in ((samplers, 'sampler_source'), (flow, 'model_source')):
            if Path(module.__file__).resolve() != Path(manifest[name]['path']).resolve():
                raise ValueError('imported source escaped manifest')
        with np.load(manifest['sample']['path'], allow_pickle=False) as saved:
            sample, coords, rng = saved['sample'], saved['coords'], saved['rng_state']
            start = int(saved['next_step_index'])
            params = json.loads(str(saved['parameters_json'].item()))
        with np.load(manifest['conditioning']['path'], allow_pickle=False) as saved:
            cond, neg = saved['cond'], saved['neg_cond']
        validate_parent_boundary(manifest, parent, start)
        if params['steps'] != 12 or not 0 < start < 12:
            raise ValueError('invalid step schedule')
        if sample.ndim != 2 or sample.shape[1] != 32 or coords.shape != (len(sample), 3):
            raise ValueError('sample/coordinate shape mismatch')
        if not np.issubdtype(coords.dtype, np.integer) or not np.isfinite(sample).all():
            raise ValueError('invalid sample/coordinate values')
        if cond.ndim != 3 or cond.shape != neg.shape or not np.isfinite(cond).all() or not np.isfinite(neg).all():
            raise ValueError('invalid conditioning')
        if rng.dtype != np.uint32 or rng.shape != (1, 2):
            raise ValueError('invalid saved MLX random state')
        remaining = manifest.get('remaining_steps', False)
        if type(remaining) is not bool:
            raise ValueError('remaining_steps must be boolean')
        if remaining and not manifest.get('capture_first_step', False):
            raise ValueError('remaining-step diagnostic requires Euler capture')
        report['claim_limit'] = ('Retained LR continuation with Euler capture in a fresh process. No full inference, cause or mesh-quality conclusion.' if remaining else report['claim_limit'])
        report['effective'] = {'source_revision': actual, 'mlx_version': version('mlx'),
                               'device': str(mx.default_device()), 'sample_shape': list(sample.shape),
                               'start_step_index': start, 'stop_after_first_step': not remaining,
                               'lr_query_tile': effective['lr_query_tile'], 'shape_block_sync': 1,
                               'qk_norm': get_qk_norm_backend(), 'parameters': params}
        if effective['shape_block_sync'] != 1:
            raise ValueError('unsupported parent block synchronization')
        report['phase'] = 'weights'
        write()
        model = flow.SLatFlowModel.for_shape(lr_sync_every_block=True)
        load_weights(model, manifest['weights'], verbose=False)
        mx.random.state = [mx.array(key) for key in rng]
        report['phase'] = 'lr_step'
        write()
        capture = {} if manifest.get('capture_first_step', False) else None
        captures = [] if remaining else None
        result = samplers.flow_euler_sample(model, mx.array(sample), mx.array(cond), mx.array(neg),
                                            coords=mx.array(coords), start_step_index=start,
                                            stop_after_first_step=not remaining, capture_first_step=capture,
                                            capture_steps=captures, **params)
        mx.eval(result, *mx.random.state)
        values = np.asarray(result)
        if values.shape != sample.shape or not np.isfinite(values).all():
            raise ValueError('invalid replay result')
        final_step = params['steps'] if remaining else start + 1
        output = out / f'lr_shape_step_{final_step:02d}.npz'
        np.savez_compressed(output, sample=values, coords=coords, next_step_index=np.array(final_step),
                            rng_state=np.stack([np.asarray(key) for key in mx.random.state]),
                            parameters_json=np.array(json.dumps(params)))
        report['output_equals_input'] = bool(np.array_equal(values, sample))
        report['max_abs_update'] = float(np.max(np.abs(values.astype(np.float64) - sample)))
        if capture is not None:
            payload = {key: np.asarray(value) for key, value in capture.items() if value is not None}
            capture_path = out / 'euler_capture.npz'
            np.savez_compressed(capture_path, **payload)
            report['euler_capture'] = {'path': str(capture_path), 'sha256': sha(capture_path),
                'arrays': {key: {'shape': list(value.shape), 'finite': bool(np.isfinite(value).all()),
                                 'min': float(value.min()), 'max': float(value.max()),
                                 'nonzero': int(np.count_nonzero(value))} for key, value in payload.items()}}
        if captures is not None:
            if len(captures) != params['steps'] - start:
                raise ValueError('partial remaining-step capture')
            report['step_captures'] = []
            for index, step_capture in enumerate(captures, start=start + 1):
                payload = {key: np.array(value) for key, value in step_capture.items() if value is not None}
                path = out / f'euler_step_{index:02d}.npz'
                np.savez_compressed(path, **payload)
                report['step_captures'].append({'step': index, 'path': str(path), 'sha256': sha(path)})
                report['last_trustworthy_evidence'] = str(path)
                write()
            if not np.array_equal(np.array(captures[-1]['sample_next']), values):
                raise ValueError('terminal capture differs from returned sample')
        report.update(status='passed', phase='remaining_lr_steps_complete' if remaining else 'one_lr_step_complete',
                      output={'path': str(output), 'sha256': sha(output), 'next_step_index': final_step})
        write()
    except BaseException as exc:
        report.update(status='failed', error=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
        write()
        raise


if __name__ == '__main__':
    main()

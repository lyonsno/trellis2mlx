"""CPU audit of retained guidance/Euler algebra, not of model predictions.

Sparse standard deviations use the existing source-order CPU reference. Dense
Welford reductions are explicitly outside this audit's independent coverage.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sparse_reference(source):
    tree = ast.parse(Path(source).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == '_source_sparse_std_reference']
    if len(nodes) != 1:
        raise ValueError('missing or ambiguous sparse standard deviation reference')
    scope = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope['_source_sparse_std_reference']


def inspect_arrays(a, params, index, std):
    times = np.linspace(1, 0, params['steps'] + 1)
    times = params['rescale_t'] * times / (1 + (params['rescale_t'] - 1) * times)
    t, prev = times[index:index + 2]
    g, r = params['guidance_strength'], params['guidance_rescale']
    guided = g != 1 and params['guidance_interval'][0] <= t <= params['guidance_interval'][1]
    one = np.float32(1 - params['sigma_min'])
    coefficient = params['sigma_min'] + (1 - params['sigma_min']) * t
    expected = {
        't': np.array(t, dtype=np.float32), 't_prev': np.array(prev, dtype=np.float32),
        'pred_cfg': (np.float32(g) * a['pred_pos'] + np.float32(1 - g) * a['pred_neg'])
                    if guided else a['pred_pos'],
        'x0_pos': one * a['sample_in'] - np.float32(coefficient) * a['pred_pos'],
        'x0_cfg': one * a['sample_in'] - np.float32(coefficient) * a['pred_cfg'],
        'ratio_raw': (a['std_pos'] / a['std_cfg'] if a['std_cfg'] > 0 else np.ones_like(a['std_cfg']))
                     if guided else np.ones_like(a['std_pos']),
        'sample_next': a['sample_in'] - np.float32(t - prev) * a['pred_final'],
    }
    expected.update(std_ratio=expected['ratio_raw'], ratio_effective=expected['ratio_raw'])
    expected['x0_rescaled'] = a['x0_cfg'] * a['ratio_raw'] if guided else a['x0_cfg']
    expected['x0_after_rescale'] = (np.float32(r) * a['x0_rescaled'] + np.float32(1 - r) * a['x0_cfg']) if guided and r > 0 else a['x0_cfg']
    expected['pred_final'] = ((one * a['sample_in'] - a['x0_after_rescale']) * np.float32(1 / coefficient)) if guided and r > 0 else a['pred_cfg']
    if not guided:
        expected.update(pred_neg=a['pred_pos'], std_cfg=a['std_pos'])
    if a['sample_in'].ndim == 2:
        expected.update(std_pos=std(a['x0_pos']), std_cfg=std(a['x0_cfg']))
    relations = {}
    for key, value in expected.items():
        actual = a[key]
        value = np.asarray(value)
        same_shape = actual.shape == value.shape
        relations[key] = {'exact': bool(same_shape and np.array_equal(actual, value)),
                          'max_abs': float(np.max(np.abs(actual.astype(np.float64) - value))) if same_shape else None}
    fields = set(expected) | {'sample_in', 'pred_pos', 'pred_neg', 'std_pos', 'std_cfg'}
    finite = all(bool(np.isfinite(a[key]).all()) for key in fields)
    return {'all_finite': finite, 'all_exact': all(v['exact'] for v in relations.values()),
            'guided': bool(guided), 'std_independent_coverage': 'source_sparse_exact' if a['sample_in'].ndim == 2 else 'not_checked_dense_welford',
            'relations': relations}


def audit_stage(directory, stage, std):
    noise_path = directory / f'{stage}_noise.npz'
    with np.load(noise_path, allow_pickle=False) as z:
        params = json.loads(z['parameters_json'].item())
        prior = z['sample'].copy()
    expected_paths = [directory / f'{stage}_euler_{i:02d}.npz' for i in range(1, params['steps'] + 1)]
    if set(directory.glob(f'{stage}_euler_*.npz')) != set(expected_paths):
        raise ValueError('missing or extra Euler captures: ' + stage)
    rows = []
    for index, path in enumerate(expected_paths):
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        row = inspect_arrays(arrays, params, index, std)
        row.update(path=str(path), sha256=sha(path),
                   continuous=bool(np.array_equal(prior, arrays['sample_in'])))
        with np.load(directory / f'{stage}_step_{index + 1:02d}.npz', allow_pickle=False) as z:
            row['step_checkpoint_matches'] = (int(z['next_step_index']) == index + 1
                and np.array_equal(z['sample'], arrays['sample_next'])
                and json.loads(z['parameters_json'].item()) == params)
        rows.append(row)
        prior = arrays['sample_next']
    return {'stage': stage, 'parameters': params, 'noise_sha256': sha(noise_path), 'rows': rows,
            'checked_relations_passed': all(r['all_finite'] and r['all_exact'] and r['continuous'] and r['step_checkpoint_matches'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected-revision', required=True)
    parser.add_argument('--expected-sampler-sha256', required=True)
    parser.add_argument('--stages', nargs='+', required=True)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'running', 'phase': 'admission', 'request': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, 'stages': [],
              'claim_limit': 'Captured guidance/Euler arithmetic, step continuity and sparse source-order standard deviations only. Dense Welford standard deviations and model predictions are not independently verified.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def write():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(args.output)
    write()
    try:
        producer_path = args.run / 'report.json'
        producer = json.loads(producer_path.read_text())
        if (producer['source']['revision'] != args.expected_revision
                or producer['effective']['sampler_sha256'] != args.expected_sampler_sha256
                or producer['effective']['sampler_observation'] != 'full_euler_capture'):
            raise ValueError('producer source or capture route differs from request')
        if [row['stage'] for row in producer['samplers']] != args.stages:
            raise ValueError('requested stages differ from observed sampler calls')
        report.update(producer_sha256=sha(producer_path), effective=producer['effective'], source=producer['source'],
                      reference_sha256=sha(args.reference_source), driver_sha256=sha(__file__))
        std = sparse_reference(args.reference_source)
        for stage in args.stages:
            report['phase'] = stage
            write()
            report['stages'].append(audit_stage(args.run / 'samplers', stage, std))
            write()
        report.update(status='completed', phase='audit_complete',
                      checked_relations_passed=all(row['checked_relations_passed'] for row in report['stages']))
        write()
        return 0 if report['checked_relations_passed'] else 1
    except BaseException as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        write()
        raise


if __name__ == '__main__':
    raise SystemExit(main())

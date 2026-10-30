"""Small contract controls; these do not establish CUDA conformance."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/source_cuda_texture_decode.py'


def module():
    spec = importlib.util.spec_from_file_location('texture_probe', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_child_coordinates_preserve_source_axis_and_parent_order():
    coords = np.array([[0, 3, 4, 5], [0, 7, 8, 9]], np.int32)
    mask = np.zeros((2, 8), np.float16)
    mask[0, [1, 4]] = 1
    mask[1, 2] = 1
    actual = module().children(coords, mask)
    assert actual.tolist() == [[0, 7, 8, 10], [0, 6, 8, 11], [0, 14, 17, 18]]


@pytest.mark.parametrize('kind', ['shape', 'nonfinite', 'empty'])
def test_invalid_guides_refused(kind):
    coords = np.zeros((2, 4), np.int32)
    values = np.ones((2, 8), np.float16)
    if kind == 'shape': values = values[:1]
    if kind == 'nonfinite': values[0, 0] = np.nan
    if kind == 'empty': values[:] = -1
    with pytest.raises(ValueError, match='guide'):
        module().children(coords, values)


@pytest.mark.parametrize('mutation', ['reorder', 'partial', 'nan', 'channels'])
def test_comparison_cannot_close_wrong_or_partial_field(mutation):
    mod = module()
    coords = np.array([[0, 0, 0], [1, 2, 3]], np.int32)
    field = np.arange(12, dtype=np.float32).reshape(2, 6) / 12
    got_coords, got = coords.copy(), field.copy()
    if mutation == 'reorder': got_coords = got_coords[::-1]
    if mutation == 'partial': got = got[:1]
    if mutation == 'nan': got[0, 0] = np.nan
    if mutation == 'channels': got = got[:, :5]
    with pytest.raises(ValueError): mod.compare(got, got_coords, field, coords)


def test_comparison_retains_each_channel_and_does_not_award_parity():
    mod = module()
    coords = np.array([[0, 0, 0], [1, 2, 3]], np.int32)
    field = np.zeros((2, 6), np.float32)
    got = field.copy(); got[1, 2] = .5
    report = mod.compare(got, coords, field, coords)
    assert report['rows'] == 2 and report['coordinate_order_exact'] is True
    assert report['max_abs'][2] == .5 and report['mean_abs'][2] == .25
    assert report['count_abs_gt_1_over_255'][2] == 1
    assert 'parity' not in report


def test_digest_refusal_has_durable_failure_report_before_runtime(tmp_path):
    mod = module()
    manifest = tmp_path/'manifest.json'; manifest.write_text('{}')
    report = tmp_path/'report.json'
    result = mod.main(['--manifest', str(manifest), '--manifest-sha256', '0'*64,
                       '--output-json', str(report), '--output-npz', str(tmp_path/'out.npz')])
    data = json.loads(report.read_text())
    assert result == 1 and data['status'] == 'failed'
    assert data['failure_phase'] == 'input_admission'
    assert 'digest mismatch' in data['error']
    assert 'effective_route' not in data


def test_cpu_fallback_and_wrong_sparse_backend_refused():
    mod = module()
    for available, backend in [(False, 'flex_gemm'), (True, 'none'), (True, 'spconv')]:
        with pytest.raises(ValueError, match='native CUDA'):
            mod.validate_route(available, backend)

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / 'scripts/source_cuda_image_features.py'


def load():
    spec = importlib.util.spec_from_file_location('image_features_probe', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_consumes_every_token():
    module = load()
    reference = np.zeros((1, 5, 4), dtype=np.float32)
    actual = reference.copy()
    actual[0, -1, -1] = 2
    result = module.compare(actual, reference, reference.shape)
    assert result['elements'] == 20
    assert result['max_abs'] == 2
    assert result['mean_abs'] == .1
    assert result['changed_elements'] == 1


@pytest.mark.parametrize('kind', ['partial', 'nan', 'wrong_dtype'])
def test_invalid_output_cannot_be_comparison(kind):
    module = load()
    reference = np.zeros((1, 5, 4), dtype=np.float32)
    actual = reference.copy()
    if kind == 'partial': actual = actual[:, :-1]
    if kind == 'nan': actual[0, -1, -1] = np.nan
    if kind == 'wrong_dtype': actual = actual.astype(np.float16)
    with pytest.raises(ValueError):
        module.compare(actual, reference, reference.shape)


def test_wrong_device_is_rejected():
    with pytest.raises(ValueError, match='Tesla T4'):
        load().admit_device('cpu', None)


def test_pre_runtime_failure_has_durable_report(tmp_path):
    module = load()
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    report = tmp_path / 'report.json'
    args = module.parser().parse_args([
        '--run-id', 'test-run', '--manifest', str(plan),
        '--manifest-sha256', '0' * 64, '--report', str(report),
        '--output', str(tmp_path / 'output.npz'),
    ])
    with pytest.raises(ValueError, match='manifest hash'):
        module.run(args)
    result = json.loads(report.read_text())
    assert result['status'] == 'failed'
    assert result['failure_phase'] == 'input_admission'
    assert result['run_id'] == 'test-run'
    assert not (tmp_path / 'output.npz').exists()

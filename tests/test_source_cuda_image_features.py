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


def test_generated_wrapper_arguments_reach_child_parser(tmp_path):
    """Execute the generated wrapper's command construction, without runtime work."""
    import ast
    import sys
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, prepare_packet

    capsule = tmp_path / 'capsule'
    capsule.mkdir()
    (capsule / PATH.name).write_bytes(PATH.read_bytes())
    run_id = '6515bd94-b954-42dd-bc7f-37210c21b785'
    packet = prepare_packet(KaggleCudaWitnessPacket(
        capsule_dir=capsule, output_dir=tmp_path / 'packet',
        dataset_id='fixture/image-inputs', kernel_id='fixture/image-features',
        title='Image Features', entrypoint=PATH.name, inputs=(PATH.name,),
        run_id=run_id, output_json='chosen-report.json', output_npz='chosen-features.npz',
        entrypoint_args=('--run-id', run_id, '--manifest', 'plan.json',
                         '--manifest-sha256', 'a' * 64),
    ))
    tree = ast.parse((packet.kernel_dir / 'run_kaggle_cuda_witness.py').read_text())
    config = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in n.targets))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, n in enumerate(main.body) if assigns(n, 'command'))
    stop = next(i for i, n in enumerate(main.body) if i > start and assigns(n, 'completed'))
    namespace = {'json': json, 'sys': sys, 'model_source_mount': None,
                 'execution_output_path': Path}
    fragment = ast.Module(body=[config, *main.body[start:stop]], type_ignores=[])
    exec(compile(fragment, '<generated-wrapper-command>', 'exec'), namespace)
    args = load().parser().parse_args(namespace['command'][2:])
    assert args.report == Path('chosen-report.json')
    assert args.output == Path('chosen-features.npz')
    assert args.run_id == run_id
    assert args.manifest == Path('plan.json')
    assert args.manifest_sha256 == 'a' * 64

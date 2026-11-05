import json
import numpy as np
import pytest

from scripts import matched_sparse_structure_cuda as runner
from types import SimpleNamespace


def inputs(tmp_path):
    path = tmp_path / 'inputs.npz'
    params = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
                  guidance_interval=[0.6, 1.0], rescale_t=5.0, sigma_min=1e-5)
    np.savez(path, noise=np.zeros((1, 8, 16, 16, 16), np.float32),
             native_terminal=np.ones((1, 8, 16, 16, 16), np.float32),
             cond=np.zeros((1, 1029, 1024), np.float32),
             neg_cond=np.zeros((1, 1029, 1024), np.float32),
             parameters_json=np.asarray(json.dumps(params)))
    return path


def test_retained_arrays_and_twelve_step_settings_roundtrip(tmp_path):
    path = inputs(tmp_path)
    arrays, params = runner.load_inputs(path, runner.digest(path))
    assert params['steps'] == 12
    assert arrays['noise'].shape == (1, 8, 16, 16, 16)
    assert np.all(arrays['native_terminal'] == 1)


def test_rejects_same_shape_substituted_input(tmp_path):
    path = inputs(tmp_path)
    expected = runner.digest(path)
    with np.load(path) as z:
        changed = {key: z[key] for key in z.files}
    changed['noise'] = np.ones_like(changed['noise'])
    np.savez(path, **changed)
    with pytest.raises(ValueError, match='digest mismatch'):
        runner.load_inputs(path, expected)


def test_failure_before_primary_output_has_durable_phase_report(tmp_path):
    path = inputs(tmp_path)
    report = tmp_path / 'report.json'
    with pytest.raises(ValueError, match='digest mismatch'):
        runner.main(['--run-id', 'test-failure', '--inputs', str(path), '--inputs-sha256', '0'*64,
                     '--source-tar', 'absent', '--source-sha256', '0'*64,
                     '--assets', 'absent', '--assets-sha256', '0'*64,
                     '--output-json', str(report), '--output-npz', str(tmp_path/'out.npz')])
    actual = json.loads(report.read_text())
    assert actual['status'] == 'failed'
    assert actual['failure_phase'] == 'input_admission'
    assert actual['last_trustworthy_phase'] is None
    assert not (tmp_path/'out.npz').exists()


def test_load_matches_official_regenerated_rope_buffer_behavior():
    class Model:
        def load_state_dict(self, state, strict):
            assert strict is False
            return SimpleNamespace(missing_keys=['rope_phases'], unexpected_keys=[])
    assert runner.load_official_weights(Model(), {}, 'flow')['missing_keys'] == ['rope_phases']


@pytest.mark.parametrize('missing,unexpected', [(['blocks.0.weight'], []), ([], ['other.weight'])])
def test_loader_rejects_missing_parameters_or_unknown_checkpoint_entries(missing, unexpected):
    class Model:
        def load_state_dict(self, state, strict):
            return SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        runner.load_official_weights(Model(), {}, 'flow')


@pytest.mark.parametrize('bad', ['nan', 'shape'])
def test_decoder_output_cannot_close_on_corrupted_logits(tmp_path, bad):
    from test_source_cuda_postcond_full_decode_timing import _source_sparse_flow_steps_payload
    from scripts.source_cuda_postcond_full_decode_timing import write_source_sparse_flow_steps_npz
    logits = np.zeros((1, 1, 64, 64, 64), dtype=np.float32)
    broken = logits.copy()
    if bad == 'nan':
        broken.flat[0] = np.nan
    else:
        broken = broken[:, :, :-1]
    support = np.array([[0, 1, 2, 3]], dtype=np.int32)
    output = tmp_path/'out.npz'
    with pytest.raises(ValueError, match='decoder logits'):
        runner.write_complete_output(output, _source_sparse_flow_steps_payload(steps=12),
                                     support, broken, logits, support, write_source_sparse_flow_steps_npz)
    assert not output.exists()


def test_complete_output_preserves_steps_and_both_decoder_controls(tmp_path):
    from test_source_cuda_postcond_full_decode_timing import _source_sparse_flow_steps_payload
    from scripts.source_cuda_postcond_full_decode_timing import write_source_sparse_flow_steps_npz
    logits = np.zeros((1, 1, 64, 64, 64), dtype=np.float32)
    support = np.array([[0, 1, 2, 3]], dtype=np.int32)
    output = tmp_path/'out.npz'
    artifact = runner.write_complete_output(output, _source_sparse_flow_steps_payload(steps=12),
                                            support, logits, logits, support, write_source_sparse_flow_steps_npz)
    assert artifact['sha256'] == runner.digest(output)
    with np.load(output) as z:
        assert z['sample_next'].shape[0] == 12
        np.testing.assert_array_equal(z['native_decoded_coords'], support)
        np.testing.assert_array_equal(z['decoded_logits'], logits)
    assert list(tmp_path.iterdir()) == [output]


def test_export_failure_does_not_publish_partial_primary(tmp_path):
    def broken_writer(path, payload, support):
        Path(path).write_bytes(b'partial')
        raise OSError('simulated disk failure')
    from pathlib import Path
    logits = np.zeros((1, 1, 64, 64, 64), dtype=np.float32)
    support = np.array([[0, 1, 2, 3]], dtype=np.int32)
    with pytest.raises(OSError, match='disk failure'):
        runner.write_complete_output(tmp_path/'out.npz', {}, support, logits, logits, support, broken_writer)
    assert not list(tmp_path.iterdir())

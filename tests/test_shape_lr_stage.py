"""LR512 uses its actual model, conditioning and measured T4 route."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def module():
    p = Path(__file__).parents[1] / 'scripts/shape_first_prediction.py'
    spec = importlib.util.spec_from_file_location('shape_lr_stage_witness', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fixture(path, tokens=1029):
    np.savez(path, coords=np.array([[0, 1, 2, 3]], np.int32),
             noise=np.zeros((1, 32), np.float32),
             cond=np.ones((1, tokens, 1024), np.float32),
             neg_cond=np.zeros((1, tokens, 1024), np.float32),
             t=np.array([1000], np.float32),
             t_prev=np.array([33/34], np.float32))


def test_lr_uses_published_512_checkpoint():
    m = module()
    spec = m.model_spec('lr512')
    assert spec['stem'] == 'ckpts/slat_flow_img2shape_dit_1_3B_512_bf16'
    assert spec['weights']['.json'] == 'ddbd7a1d34ce9a8e7af2f6ef9ae686d9ab49f69e62dacb98d8409b511cdbd5e2'
    assert spec['weights']['.safetensors'] == 'ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f'
    assert spec['resolution'] == 32
    with pytest.raises(ValueError, match='stage'):
        m.model_spec('lr256')


def test_lr_admits_exact_conditioning_class(tmp_path):
    m = module()
    p = tmp_path / 'in.npz'
    fixture(p)
    assert m.load_inputs(p, model_stage='lr512')['cond'].shape == (1, 1029, 1024)
    with pytest.raises(ValueError, match='cond'):
        m.load_inputs(p)  # HR remains strict.
    fixture(p, 4101)
    with pytest.raises(ValueError, match='cond'):
        m.load_inputs(p, model_stage='lr512')


def test_lr_rejects_hr_model_route():
    m = module()
    r = dict(cuda_available=True, attention='xformers', sparse_container='none',
             trellis=m.TRELLIS_REVISION, torch='2.11.0+cu130', xformers='0.0.35',
             xformers_built_with_cuda=True, xformers_fp16_block_diagonal_smoke=True,
             torso_dtype='float16', hr_resolution=32)
    m.validate_route(r, 'cuda', model_stage='lr512')
    with pytest.raises(ValueError, match='CUDA route'):
        m.validate_route(dict(r, hr_resolution=64), 'cuda', model_stage='lr512')


def test_lr_cli_failure_keeps_actual_stage_and_phase(tmp_path, monkeypatch):
    m = module()
    p = tmp_path / 'in.npz'; fixture(p)
    output = tmp_path / 'out.json'
    def fail(args, arrays, report, save):
        assert args.model_stage == 'lr512'
        assert arrays['cond'].shape[1] == 1029
        report['phase'] = 'model_load'
        save()
        raise RuntimeError('observed failure')
    monkeypatch.setattr(m, 'run_cuda', fail)
    rc = m.main(['--backend','cuda','--model-stage','lr512','--mode','shape-trajectory',
                 '--steps','12','--rescale-t','3','--run-id','test-lr',
                 '--inputs',str(p),'--inputs-sha256',m.digest(p),
                 '--output-json',str(output),'--output-npz',str(tmp_path/'out.npz')])
    assert rc != 0
    r = json.loads(output.read_text())
    assert r['status'] == 'failed'
    assert r['model_stage_requested'] == 'lr512'
    assert 'LR512' in r['claim_ceiling']
    assert '24578' not in r['claim_ceiling']
    assert r['failure_phase'] == 'model_load'


def test_pacing_request_must_match_effective_native_route():
    m = module()
    rev = 'a' * 40
    route = dict(mlx='0.32.4', device='gpu', attention='fast', query_chunk=1024,
                 hr_attention_active=True, shape_flow_layernorm='mlx-two-pass',
                 qk_norm='source-cuda-warp32', torso_dtype='bfloat16',
                 learned_modulation_dtype='bfloat16', native_revision=rev,
                 native_revision_requested=rev, native_revision_observed=rev,
                 native_tree_clean=True, shape_block_sync=1)
    m.validate_route(route, 'mlx', native_profile='production-bf16', sync_every_block=True)
    with pytest.raises(ValueError, match='MLX route'):
        m.validate_route(dict(route, shape_block_sync=6), 'mlx', native_profile='production-bf16', sync_every_block=True)
    with pytest.raises(ValueError, match='MLX route'):
        m.validate_route(route, 'mlx', native_profile='production-bf16', sync_every_block=False)

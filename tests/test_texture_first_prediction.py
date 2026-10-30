"""Local evidence contracts; live CUDA/Metal output remains the integration witness."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/texture_first_prediction.py'

def module():
    spec=importlib.util.spec_from_file_location('prediction_witness', SCRIPT)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

@pytest.mark.parametrize('kind', ['partial', 'reordered', 'wrong_channels', 'nonfinite', 'empty', 'wrong_dtype'])
def test_complete_prediction_support_is_required(kind):
    coords=np.array([[0,1,2,3],[0,4,5,6]],np.int32)
    pred=np.ones((2,32),np.float32); got=coords.copy()
    if kind=='partial': pred=pred[:1];got=got[:1]
    if kind=='reordered': got=got[::-1]
    if kind=='wrong_channels': pred=pred[:,:31]
    if kind=='nonfinite': pred[1,2]=np.nan
    if kind=='empty': pred=pred[:0];got=got[:0]
    if kind=='wrong_dtype': pred=pred.astype(np.float16)
    with pytest.raises(ValueError): module().validate_output(pred,got,coords)

def test_valid_complete_prediction_is_observable_without_parity_claim():
    coords=np.array([[0,1,2,3],[0,4,5,6]],np.int32)
    assert module().validate_output(np.ones((2,32),np.float32),coords,coords) is None

@pytest.mark.parametrize('mutation', ['cpu','attention','source','precision','runtime'])
def test_cuda_route_cannot_silently_fallback(mutation):
    mod=module();route=dict(cuda_available=True,attention='xformers',sparse_container='none',trellis=mod.TRELLIS_REVISION,torch='2.11.0+cu128',xformers='0.0.35',torso_dtype='float16')
    if mutation=='cpu': route['cuda_available']=False
    if mutation=='attention':route['attention']='sdpa'
    if mutation=='source':route['trellis']='wrong'
    if mutation=='precision':route['torso_dtype']='bfloat16'
    if mutation=='runtime':route['torch']='2.10.0'
    with pytest.raises(ValueError):mod.validate_route(route,'cuda')

def test_bad_digest_fails_before_backend_with_durable_phase(tmp_path,monkeypatch):
    import json
    mod=module();inp=tmp_path/'input.npz';np.savez(inp,noise=np.zeros((1,32),np.float32))
    monkeypatch.setattr(mod,'run_cuda',lambda *a:pytest.fail('backend must not run'))
    report=tmp_path/'report.json'
    result=mod.main(['--backend','cuda','--run-id','unit-negative','--inputs',str(inp),'--inputs-sha256','0'*64,'--output-json',str(report),'--output-npz',str(tmp_path/'output.npz')])
    r=json.loads(report.read_text())
    assert result==1 and r['status']=='failed' and r['failure_phase']=='input_admission'
    assert 'digest mismatch' in r['error'] and 'effective_route' not in r

def test_partial_backend_result_is_retained_but_never_completed(tmp_path,monkeypatch):
    import json
    mod=module();inp=tmp_path/'input.npz';coords=np.array([[0,1,2,3],[0,4,5,6]],np.int32)
    np.savez(inp,noise=np.zeros((2,32),np.float32),shape_cond=np.ones((2,32),np.float32),coords=coords,cond=np.ones((1,4101,1024),np.float32),t=np.array([1000],np.float32))
    monkeypatch.setattr(mod,'run_cuda',lambda *a:(np.ones((1,32),np.float32),coords[:1]))
    report=tmp_path/'report.json';output=tmp_path/'output.npz'
    result=mod.main(['--backend','cuda','--run-id','unit-partial','--inputs',str(inp),'--inputs-sha256',mod.digest(inp),'--output-json',str(report),'--output-npz',str(output)])
    r=json.loads(report.read_text());assert result==1 and r['status']=='failed' and r['failure_phase']=='output_capture'
    assert output.is_file() and r['output']['sha256']==mod.digest(output)

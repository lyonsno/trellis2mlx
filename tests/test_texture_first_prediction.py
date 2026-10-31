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


def test_native_production_profile_is_explicit_and_reported(tmp_path, monkeypatch):
    import json
    mod=module();inp=tmp_path/'input.npz';coords=np.array([[0,1,2,3]],np.int32)
    np.savez(inp,noise=np.zeros((1,32),np.float32),shape_cond=np.ones((1,32),np.float32),coords=coords,cond=np.ones((1,4101,1024),np.float32),t=np.array([1000],np.float32))
    observed=[]
    def backend(args,*unused):
        observed.append(args.native_profile)
        return np.ones((1,32),np.float32),coords
    monkeypatch.setattr(mod,'run_mlx',backend)
    report=tmp_path/'report.json'
    result=mod.main(['--backend','mlx','--native-profile','production-bf16','--run-id','unit-production','--inputs',str(inp),'--inputs-sha256',mod.digest(inp),'--output-json',str(report),'--output-npz',str(tmp_path/'out.npz')])
    r=json.loads(report.read_text())
    assert result==0 and observed==['production-bf16']
    assert r['native_profile_requested']=='production-bf16'
    assert 'Native production BF16 first prediction' in r['claim_ceiling']


def test_native_profile_cannot_change_cuda_route(tmp_path, monkeypatch):
    import json
    mod=module();report=tmp_path/'report.json'
    monkeypatch.setattr(mod,'run_cuda',lambda *args:pytest.fail('CUDA must not run with native profile'))
    result=mod.main(['--backend','cuda','--native-profile','production-bf16','--run-id','unit-misroute','--inputs',str(tmp_path/'missing.npz'),'--inputs-sha256','0'*64,'--output-json',str(report),'--output-npz',str(tmp_path/'out.npz')])
    r=json.loads(report.read_text())
    assert result==1 and r['failure_phase']=='profile_admission'
    assert 'native profile requires MLX' in r['error']


@pytest.mark.parametrize('profile,dtype,qk', [('fp16-control','float16','mlx-sum'),('production-bf16','bfloat16','source-cuda-warp32')])
def test_native_effective_precision_and_qk_match_requested_profile(profile,dtype,qk):
    mod=module();route=dict(mlx='0.32.4',device='gpu',attention='fast',query_chunk=2048,qk_norm=qk,torso_dtype=dtype)
    assert mod.validate_route(route,'mlx',native_profile=profile) is None
    wrong=dict(route,torso_dtype='float16' if dtype=='bfloat16' else 'bfloat16')
    with pytest.raises(ValueError,match='precision'):mod.validate_route(wrong,'mlx',native_profile=profile)
    wrong=dict(route,qk_norm='mlx-sum' if qk!='mlx-sum' else 'source-cuda-warp32')
    with pytest.raises(ValueError,match='MLX route'):mod.validate_route(wrong,'mlx',native_profile=profile)


@pytest.mark.parametrize('profile,qk,modulation', [
    ('bf16-f32-modulation','source-cuda-warp32','float32'),
    ('bf16-mlx-sum','mlx-sum','bfloat16'),
])
def test_isolated_native_profile_keeps_bf16_and_reports_intervention(profile,qk,modulation,tmp_path,monkeypatch):
    import json
    mod=module();inp=tmp_path/'input.npz';coords=np.array([[0,1,2,3]],np.int32)
    np.savez(inp,noise=np.zeros((1,32),np.float32),shape_cond=np.ones((1,32),np.float32),coords=coords,cond=np.ones((1,4101,1024),np.float32),t=np.array([1000],np.float32))
    observed=[]
    def backend(args,*unused):
        observed.append(args.native_profile)
        return np.ones((1,32),np.float32),coords
    monkeypatch.setattr(mod,'run_mlx',backend)
    report=tmp_path/'report.json'
    result=mod.main(['--backend','mlx','--native-profile',profile,'--run-id','unit-isolated','--inputs',str(inp),'--inputs-sha256',mod.digest(inp),'--output-json',str(report),'--output-npz',str(tmp_path/'out.npz')])
    r=json.loads(report.read_text())
    assert result==0 and observed==[profile] and r['native_profile_requested']==profile
    assert r['claim_ceiling']==mod.BF16_CONTROL_CLAIM
    route=dict(mlx='0.32.4',device='gpu',attention='fast',query_chunk=2048,qk_norm=qk,torso_dtype='bfloat16',learned_modulation_dtype=modulation)
    assert mod.validate_route(route,'mlx',native_profile=profile) is None
    with pytest.raises(ValueError):mod.validate_route(dict(route,learned_modulation_dtype='bfloat16' if modulation=='float32' else 'float32'),'mlx',native_profile=profile)
    with pytest.raises(ValueError):mod.validate_route(dict(route,qk_norm='mlx-sum' if qk!='mlx-sum' else 'source-cuda-warp32'),'mlx',native_profile=profile)
    with pytest.raises(ValueError):mod.validate_route(dict(route,torso_dtype='float16'),'mlx',native_profile=profile)
    with pytest.raises(ValueError):mod.validate_route(route,'cuda',native_profile=profile)


def test_trajectory_requires_complete_trace_before_success(tmp_path, monkeypatch):
    import json
    mod=module();inp=tmp_path/'input.npz';coords=np.array([[0,1,2,3]],np.int32)
    np.savez(inp,noise=np.zeros((1,32),np.float32),shape_cond=np.ones((1,32),np.float32),coords=coords,cond=np.ones((1,4101,1024),np.float32),t=np.array([1000],np.float32))
    # A complete final sample cannot stand in for the requested intermediate trace.
    monkeypatch.setattr(mod,'run_mlx',lambda *a:(np.ones((1,32),np.float32),coords))
    report=tmp_path/'report.json';output=tmp_path/'out.npz'
    result=mod.main(['--backend','mlx','--mode','texture-trajectory','--steps','2','--rescale-t','3','--run-id','missing-trace','--inputs',str(inp),'--inputs-sha256',mod.digest(inp),'--output-json',str(report),'--output-npz',str(output)])
    r=json.loads(report.read_text())
    assert result==1 and r['status']=='failed' and 'trajectory' in r['error']
    assert output.exists()


def test_trace_retains_raw_corruption_and_refuses_completion(tmp_path):
    mod=module();coords=np.array([[0,1,2,3]],np.int32);report={}
    trace=mod.TrajectoryTrace(tmp_path/'steps',coords,np.zeros((1,32),np.float32),2,3,report,lambda:None)
    trace.begin(np.zeros((1,32),np.float32),np.array([1000],np.float32),coords)
    bad=np.ones((1,32),np.float32);bad[0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):trace.end(bad,coords)
    with np.load(tmp_path/'steps/step-000.npz') as z: assert np.isnan(z['prediction'][0,0])
    assert report['trajectory']['completed_steps']==0


def test_trace_rejects_wrong_schedule_and_partial_or_stale_capture(tmp_path):
    mod=module();coords=np.array([[0,1,2,3]],np.int32);noise=np.zeros((1,32),np.float32)
    trace=mod.TrajectoryTrace(tmp_path/'steps',coords,noise,2,3,{},lambda:None)
    with pytest.raises(ValueError,match='timestep'):trace.begin(noise,np.array([900],np.float32),coords)
    with pytest.raises(ValueError,match='fresh'):mod.TrajectoryTrace(tmp_path/'steps',coords,noise,2,3,{},lambda:None)
    with pytest.raises(ValueError,match='incomplete'):trace.finish(noise,coords)


def test_trace_records_every_prediction_and_final_sample(tmp_path):
    mod=module();coords=np.array([[0,1,2,3]],np.int32);noise=np.zeros((1,32),np.float32);report={}
    trace=mod.TrajectoryTrace(tmp_path/'steps',coords,noise,2,3,report,lambda:None)
    trace.begin(noise,np.array([1000],np.float32),coords);trace.end(np.ones_like(noise),coords)
    trace.begin(np.full_like(noise,-.25),np.array([750],np.float32),coords);trace.end(np.ones_like(noise),coords)
    trace.finish(np.full_like(noise,-1),coords)
    assert report['trajectory']['completed_steps']==2
    assert len(report['trajectory']['records'])==2
    for row in report['trajectory']['records']:
        assert row['sha256']==mod.digest(row['path'])
    with np.load(tmp_path/'steps/step-001.npz') as z:
        assert set(z.files)=={'sample_in','prediction','coords','prediction_coords','t'}
        assert np.array_equal(z['sample_in'],np.full_like(noise,-.25))


@pytest.mark.parametrize('defect', ['overflow', 'wrong_support', 'wrong_dtype', 'incomplete'])
def test_adapter_retains_rejected_final_before_return(tmp_path, monkeypatch, defect):
    import json
    mod=module();inp=tmp_path/'input.npz';coords=np.array([[0,1,2,3]],np.int32)
    noise=np.full((1,32),1e38,np.float32)
    np.savez(inp,noise=noise,shape_cond=np.ones_like(noise),coords=coords,
        cond=np.ones((1,4101,1024),np.float32),t=np.array([1000],np.float32))
    returned={}
    def adapter(args, arrays, report, save):
        trace=mod.TrajectoryTrace(tmp_path/'steps',coords,noise,1,3,report,save)
        trace.begin(noise,arrays['t'],coords)
        prediction=np.full_like(noise,-3e38)
        if defect!='incomplete': trace.end(prediction,coords)
        with np.errstate(over='ignore'): final=noise-prediction
        got=coords.copy()
        if defect!='overflow': final=np.zeros_like(noise)
        if defect=='wrong_support': got[0,1]=9
        if defect=='wrong_dtype': final=final.astype(np.float16)
        returned.update(sample=final,coords=got)
        # Match the real adapters: finish can reject before main gets the arrays.
        trace.finish(final,got)
        pytest.fail('invalid final sample must not return as complete')
    monkeypatch.setattr(mod,'run_mlx',adapter)
    rp=tmp_path/'report.json';out=tmp_path/'sample.npz'
    rc=mod.main(['--backend','mlx','--mode','texture-trajectory','--steps','1','--rescale-t','3',
        '--run-id','rejected-final','--inputs',str(inp),'--inputs-sha256',mod.digest(inp),
        '--output-json',str(rp),'--output-npz',str(out)])
    report=json.loads(rp.read_text())
    assert rc==1 and report['status']=='failed'
    assert out.exists(), 'adapter rejected final sample before retaining it'
    with np.load(out,allow_pickle=False) as z:
        assert np.array_equal(z['sample'],returned['sample'])
        assert z['sample'].dtype==returned['sample'].dtype
        assert np.array_equal(z['coords'],returned['coords'])
    assert report['output']['sha256']==mod.digest(out)
    assert report['failure_phase']=='trajectory_final_validation'
    assert report['trajectory']['final_sample_validated'] is False

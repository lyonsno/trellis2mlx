"""Small CPU fixtures test audit policy; real retained captures are the witness."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit', ROOT/'scripts/audit_captured_sampler.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
std = audit.sparse_reference(ROOT/'tests/test_samplers.py')


def unguided_fixture(dense=False):
    params = dict(steps=2, rescale_t=1., sigma_min=1e-5,
                  guidance_strength=7.5, guidance_rescale=.7, guidance_interval=[.6,1.])
    sample = np.array([[.12,-.4]], np.float32)
    pred = np.array([[.31,-.23]], np.float32)
    if dense:
        sample, pred = sample.reshape(1,1,1,1,2), pred.reshape(1,1,1,1,2)
    x0 = np.float32(.99999)*sample - np.float32(.500005)*pred
    deviation = std(x0) if not dense else np.ones((1,1,1,1,1),np.float32)
    a = {k: pred.copy() for k in ('pred_pos','pred_neg','pred_cfg','pred_final')}
    a.update({k:x0.copy() for k in ('x0_pos','x0_cfg','x0_rescaled','x0_after_rescale')})
    a.update({k:np.ones_like(deviation) for k in ('ratio_raw','std_ratio','ratio_effective')})
    a.update(sample_in=sample, sample_next=sample-np.float32(.5)*pred,
             std_pos=deviation, std_cfg=deviation, t=np.array(.5,np.float32),t_prev=np.array(0,np.float32))
    return a,params


@pytest.mark.parametrize('dense',[False,True])
def test_unguided_interval_does_not_recompute_inverse(dense):
    a,p=unguided_fixture(dense)
    result=audit.inspect_arrays(a,p,1,std)
    assert result['all_exact'] and result['all_finite'] and not result['guided']
    assert ('not_checked' in result['std_independent_coverage']) == dense


@pytest.mark.parametrize('key', ['pred_cfg','std_cfg','ratio_raw','pred_final','sample_next','t'])
def test_corrupt_capture_cannot_pass(key):
    a,p=unguided_fixture()
    a[key]=a[key]+np.float32(.125)
    assert not audit.inspect_arrays(a,p,1,std)['all_exact']


def test_nonfinite_and_missing_fields_cannot_pass():
    a,p=unguided_fixture()
    a['sample_in'][0,0]=np.nan
    assert not audit.inspect_arrays(a,p,1,std)['all_finite']
    del a['pred_pos']
    with pytest.raises(KeyError): audit.inspect_arrays(a,p,1,std)


def test_missing_capture_refused_before_partial_pass(tmp_path):
    a,p=unguided_fixture()
    np.savez(tmp_path/'texture_noise.npz',sample=a['sample_in'],parameters_json=json.dumps(p))
    with pytest.raises(ValueError,match='missing or extra'):
        audit.audit_stage(tmp_path,'texture',std)


def test_wrong_route_has_durable_failure_report(tmp_path):
    (tmp_path/'report.json').write_text(json.dumps({'source':{'revision':'wrong'}}))
    output=tmp_path/'audit.json'
    run=subprocess.run([sys.executable,str(ROOT/'scripts/audit_captured_sampler.py'),
        '--run',str(tmp_path),'--expected-revision','pinned','--expected-sampler-sha256','pinned',
        '--stages','sparse','--reference-source',str(ROOT/'tests/test_samplers.py'),
        '--output',str(output)],capture_output=True,text=True)
    assert run.returncode != 0
    report=json.loads(output.read_text())
    assert report['status']=='failed' and report['phase']=='admission' and report['stages']==[]

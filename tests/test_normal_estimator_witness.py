import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPT=Path(__file__).parents[1]/'scripts/normal_estimator_witness.py'
spec=importlib.util.spec_from_file_location('normal_estimator_witness',SCRIPT)
w=importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def fixture():
    return np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,10]],dtype=np.float32),np.array([[0,1,2],[0,3,1]],dtype=np.int32)


def test_unequal_area_distinguishes_estimators_and_preserves_all_arrays(tmp_path):
    v,f=fixture()
    source=tmp_path/'in.npz'
    np.savez(source,corner__vertices=v,corner__faces=f)
    args=argparse.Namespace(input_npz=source,expected_input_sha256=w.sha(source),backend='cpu',
                           output_npz=tmp_path/'out.npz',output_json=tmp_path/'report.json')
    report=w.run(args)
    with np.load(args.output_npz) as out:
        np.testing.assert_array_equal(out['corner__vertices'],v)
        np.testing.assert_array_equal(out['corner__faces'],f)
        assert w.compare(out['corner__trimesh_normals'][:1],out['corner__area_normals'][:1])['degrees']['max']==pytest.approx(39.289406,abs=1e-5)
    assert report['effective_backend']=='cpu-trimesh-and-provisional-float64-area'


@pytest.mark.parametrize('damage',['partial','nonfinite','float_faces','out_of_range'])
def test_bad_inputs_refused(damage):
    v,f=fixture()
    d=dict(corner__vertices=v,corner__faces=f)
    if damage=='partial': del d['corner__faces']
    if damage=='nonfinite': v[0,0]=np.nan
    if damage=='float_faces': d['corner__faces']=f.astype(np.float32)
    if damage=='out_of_range': f[0,0]=99
    with pytest.raises(ValueError): w.validate_cases(d)


def test_wrong_native_geometry_and_partial_vectors_cannot_pass():
    v,f=fixture()
    with pytest.raises(ValueError,match='mesh identity'): w.validate_native(v,f,v+1,f,np.zeros_like(v))
    with pytest.raises(ValueError,match='shape/dtype'): w.validate_native(v,f,v,f,np.zeros_like(v)[:1])


def test_missing_input_leaves_durable_failure_not_primary(tmp_path):
    report=tmp_path/'report.json'
    result=subprocess.run([sys.executable,str(SCRIPT),'--input-npz',str(tmp_path/'absent.npz'),
        '--expected-input-sha256','0'*64,'--backend','cpu','--output-npz',str(tmp_path/'out.npz'),
        '--output-json',str(report)],capture_output=True,text=True)
    assert result.returncode != 0
    assert json.loads(report.read_text())['phase']=='input'
    assert json.loads(report.read_text())['effective_backend'] is None
    assert not (tmp_path/'out.npz').exists()


def test_cuda_request_cannot_silently_use_cpu(tmp_path):
    v,f=fixture()
    source=tmp_path/'in.npz'
    np.savez(source,corner__vertices=v,corner__faces=f)
    report=tmp_path/'report.json'
    result=subprocess.run([sys.executable,str(SCRIPT),'--input-npz',str(source),
        '--expected-input-sha256',w.sha(source),'--backend','cuda','--output-npz',str(tmp_path/'out.npz'),
        '--output-json',str(report)],capture_output=True,text=True)
    assert result.returncode != 0
    data=json.loads(report.read_text())
    assert data['phase']=='runtime' and data['effective_backend'] is None
    assert 'fresh work-dir' in data['error']
    assert not (tmp_path/'out.npz').exists()


def test_empty_or_all_invalid_comparison_does_not_fake_zero_error():
    result=w.compare(np.zeros((3,3)),np.ones((3,3)))
    assert result['compared_vectors']==0 and result['degrees'] is None
    assert result['excluded_vectors']==3


def test_report_hardlink_to_input_is_refused_without_mutation(tmp_path):
    v,f=fixture()
    source=tmp_path/'in.npz'
    np.savez(source,corner__vertices=v,corner__faces=f)
    report=tmp_path/'report.json'
    report.hardlink_to(source)
    old=source.read_bytes()
    args=argparse.Namespace(input_npz=source,expected_input_sha256=w.sha(source),backend='cpu',
                           output_npz=tmp_path/'out.npz',output_json=report)
    with pytest.raises(ValueError): w.run(args)
    assert source.read_bytes()==old

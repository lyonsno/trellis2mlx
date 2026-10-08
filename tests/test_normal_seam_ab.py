import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import trimesh

spec=importlib.util.spec_from_file_location('normal_seam_ab',Path(__file__).parents[1]/'scripts/normal_seam_ab.py')
ab=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def fixture():
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
    faces=np.array([[0,1,2],[0,3,1]])
    mapping=faces.ravel()
    uvv=vertices[mapping]
    uvf=np.arange(6).reshape(2,3)
    mesh=trimesh.Trimesh(vertices=ab.export_axes(uvv),faces=uvf,process=False)
    _=mesh.vertex_normals
    return mesh.export(file_type='glb',include_normals=True),vertices,faces,uvv,uvf,mapping


def test_uv_split_copies_share_pre_uv_normal_and_other_bytes_are_unchanged():
    data,v,f,uvv,uvf,mapping=fixture()
    patched,report=ab.patch_asset(data,v,f,uvv,uvf,mapping)
    doc,prim,start,size=ab.read_glb(patched)
    normals,begin,end=ab.accessor(patched,doc,start,size,prim['attributes']['NORMAL'])
    assert np.allclose(normals[0],normals[3]), 'same pre-UV vertex acquired a lighting seam'
    expected=trimesh.Trimesh(vertices=ab.export_axes(v),faces=f,process=False).vertex_normals[mapping]
    np.testing.assert_allclose(normals,expected,atol=1e-7)
    assert patched[:begin]==data[:begin] and patched[end:]==data[end:]
    assert report['changed_normal_vectors']==4
    assert report['only_normal_bytes_changed']


@pytest.mark.parametrize('damage',['mapping','glb','winding'])
def test_wrong_saved_identity_rejected(damage):
    data,v,f,uvv,uvf,mapping=fixture()
    if damage=='mapping': mapping=mapping[::-1]
    if damage=='glb': uvv=uvv+2; v=v+2
    if damage=='winding': f=f[:,::-1]
    with pytest.raises(ValueError): ab.patch_asset(data,v,f,uvv,uvf,mapping)


def test_missing_input_writes_failure_report_without_output(tmp_path):
    report=tmp_path/'report.json'
    output=tmp_path/'out.glb'
    result=subprocess.run([sys.executable,ab.__file__,'--glb',str(tmp_path/'missing.glb'),
        '--checkpoints',str(tmp_path),'--output',str(output),'--report',str(report)],capture_output=True,text=True)
    assert result.returncode != 0
    assert json.loads(report.read_text())['phase']=='input'
    assert json.loads(report.read_text())['status']=='failed'
    assert not output.exists()

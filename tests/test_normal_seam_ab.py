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


def test_output_report_alias_rejected_before_any_write(tmp_path):
    data,v,f,uvv,uvf,mapping=fixture()
    source=tmp_path/'source.glb'
    source.write_bytes(data)
    np.savez(tmp_path/'mesh_clean.npz',vertices=v,faces=f)
    np.savez(tmp_path/'mesh_uv.npz',vertices=uvv,faces=uvf,vmapping=mapping)
    destination=tmp_path/'result.glb'
    result=subprocess.run([sys.executable,ab.__file__,'--glb',str(source),
        '--checkpoints',str(tmp_path),'--output',str(destination),
        '--report',str(tmp_path/'nested'/'..'/'result.glb')],capture_output=True,text=True)
    assert result.returncode != 0, 'report overwrote GLB despite successful result'
    assert 'output and report must be distinct' in result.stderr
    assert not destination.exists()
    assert source.read_bytes()==data


def test_normal_bounds_update_preserves_every_other_metadata_and_binary_byte(monkeypatch):
    data,v,f,uvv,uvf,mapping=fixture()
    monkeypatch.setattr(ab,'mapped_normals',lambda *a:np.tile(np.array([1,0,0],dtype='<f4'),(6,1)))
    output,report=ab.patch_asset(data,v,f,uvv,uvf,mapping)
    assert report['normal_bounds_metadata_updated']
    old,old_prim,old_start,_=ab.read_glb(data)
    new,new_prim,new_start,_=ab.read_glb(output)
    index=old_prim['attributes']['NORMAL']
    _,begin,end=ab.accessor(data,old,old_start,len(data)-old_start,index)
    new_a=new['accessors'][index]
    assert new_a['min']==new_a['max']==[1,0,0]
    new['accessors'][index]=old['accessors'][index]
    assert new==old
    assert output[new_start:new_start+begin-old_start]==data[old_start:begin]
    assert output[new_start+end-old_start:]==data[end:]

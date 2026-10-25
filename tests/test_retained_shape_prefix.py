"""CPU admission checks; replayed model execution remains the external witness."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from retained_shape_prefix import load_completed_shape
from fresh_prepared_inference import sha


def fixture(root):
    cp=root/'checkpoints';cp.mkdir()
    samplers=root/'samplers';samplers.mkdir()
    coords=np.array([[0,1,2,3],[0,3,2,1]],np.int32)
    sample=np.ones((2,32),np.float32)
    np.savez(cp/'shape_slat.npz',feats=sample,coords=coords,coords_3d=coords[:,1:])
    np.savez(cp/'conditioning.npz',cond=np.zeros((1,2,3)),neg_cond=np.zeros((1,2,3)))
    np.savez(cp/'sparse_coords.npz',coords=coords)
    for name in ('conditioning','sparse_coords','shape_slat'):
        (cp/f'{name}.json').write_text(json.dumps({'mesh_grid_size':1024}))
    terminal=samplers/'hr_shape_step_12.npz'
    np.savez(terminal,sample=sample,coords=coords[:,1:],next_step_index=12,rng_state=np.array([[1,2]],np.uint32))
    euler=samplers/'hr_shape_euler_12.npz';np.savez(euler,sample_next=sample)
    current={'effective':{k:'same' for k in ('mlx_version','device','lr_query_tile','hr_query_tile','shape_block_sync','qk_norm','sampler_sha256')},
             'source':{'root':str(root),'generator_sha256':'generator','revision':'parent'},
             'input':{'sha256':'image'},'conditioning_backend':'mlx'}
    parent=root/'report.json';parent.write_text(json.dumps(current))
    audit=root/'audit.json';audit.write_text(json.dumps({'status':'completed','checked_relations_passed':True,
        'producer_sha256':sha(parent),'claim_limit':'synthetic fixture tests admission only',
        'stages':[{'stage':s,'checked_relations_passed':True,'rows':[{'path':str(euler),'sha256':sha(euler)}]*12}
                  for s in ('sparse','lr_shape','hr_shape')]}))
    paths={'parent_report':parent,'arithmetic_audit':audit,'hr_terminal':terminal}
    paths.update({f'{name}_{suffix}':cp/f'{name}.{suffix}' for name in ('conditioning','sparse_coords','shape_slat') for suffix in ('npz','json')})
    request={k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()}
    manifest=root/'manifest.json';manifest.write_text(json.dumps(request))
    return manifest,current,request


def test_exact_shape_and_rng_are_returned(tmp_path):
    manifest,current,_=fixture(tmp_path)
    with patch('retained_shape_prefix.subprocess.run') as git:
        retained,provenance=load_completed_shape(manifest,current)
    np.testing.assert_array_equal(retained['rng'],[[1,2]])
    assert retained['feats'].shape==(2,32)
    assert provenance['inherited_stages']==['conditioning','sparse','lr_shape','hr_shape']
    assert 'trellmlx' in git.call_args.args[0]


@pytest.mark.parametrize('defect',['hash','runtime','grid','incomplete_audit','wrong_parent','terminal_sample','rng_shape'])
def test_mismatched_prefix_cannot_be_admitted(tmp_path,defect):
    manifest,current,request=fixture(tmp_path)
    if defect=='hash': request['shape_slat_npz']['sha256']='wrong'
    elif defect=='runtime': current=copy.deepcopy(current);current['effective']['sampler_sha256']='wrong'
    elif defect=='grid':
        path=Path(request['shape_slat_json']['path']);path.write_text('{"mesh_grid_size":768}')
        request['shape_slat_json']['sha256']=sha(path)
    elif defect in ('incomplete_audit','wrong_parent'):
        path=Path(request['arithmetic_audit']['path']);r=json.loads(path.read_text())
        if defect=='incomplete_audit':r['stages'][1]['rows'].pop()
        else:r['producer_sha256']='wrong'
        path.write_text(json.dumps(r));request['arithmetic_audit']['sha256']=sha(path)
    else:
        path=Path(request['hr_terminal']['path'])
        with np.load(path) as z: a={k:z[k] for k in z.files}
        if defect=='terminal_sample':a['sample']=a['sample']+1
        else:a['rng_state']=a['rng_state'].reshape(2)
        np.savez(path,**a);request['hr_terminal']['sha256']=sha(path)
    manifest.write_text(json.dumps(request))
    with patch('retained_shape_prefix.subprocess.run'), pytest.raises(ValueError):
        load_completed_shape(manifest,current)

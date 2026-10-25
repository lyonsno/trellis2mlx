"""Recover the completed shape prefix; execute the canonical decoder/texture tail."""
import json
import subprocess
from pathlib import Path
import numpy as np
from fresh_prepared_inference import sha


def load_completed_shape(manifest_path, current):
    request = json.loads(Path(manifest_path).read_text())
    def checked(key):
        item = request[key]
        path = Path(item['path'])
        if sha(path) != item['sha256']:
            raise ValueError('retained artifact SHA mismatch: ' + key)
        return path
    parent_path = checked('parent_report')
    parent = json.loads(parent_path.read_text())
    subprocess.run(['git','-C',current['source']['root'],'diff','--exit-code',
                    parent['source']['revision'],'HEAD','--','generate.py','trellmlx'],
                   check=True, capture_output=True)
    audit_path = checked('arithmetic_audit')
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'completed' or audit.get('checked_relations_passed') is not True
            or audit['producer_sha256'] != sha(parent_path)
            or [s['stage'] for s in audit['stages']] != ['sparse','lr_shape','hr_shape']
            or any(len(s['rows']) != 12 or not s['checked_relations_passed'] for s in audit['stages'])):
        raise ValueError('completed three-stage arithmetic audit is missing or mismatched')
    for name in ('mlx_version','device','lr_query_tile','hr_query_tile','shape_block_sync','qk_norm','sampler_sha256'):
        if current['effective'][name] != parent['effective'][name]:
            raise ValueError('retained runtime differs: ' + name)
    if (parent['source']['generator_sha256'] != current['source']['generator_sha256']
            or parent['input']['sha256'] != current['input']['sha256']
            or parent.get('conditioning_backend') != 'mlx'):
        raise ValueError('retained source/image/conditioning route mismatch')
    cp = parent_path.parent / 'checkpoints'
    for name in ('conditioning','sparse_coords','shape_slat'):
        for suffix in ('npz','json'):
            path = checked(name + '_' + suffix)
            if path.resolve() != (cp / f'{name}.{suffix}').resolve():
                raise ValueError('retained checkpoint escaped parent: ' + name)
    if json.loads((cp/'shape_slat.json').read_text()).get('mesh_grid_size') != 1024:
        raise ValueError('retained shape grid is not 1024')
    with np.load(cp/'shape_slat.npz',allow_pickle=False) as z:
        feats,coords,spatial = z['feats'].copy(),z['coords'].copy(),z['coords_3d'].copy()
    terminal_path = checked('hr_terminal')
    if terminal_path.resolve() != (parent_path.parent/'samplers/hr_shape_step_12.npz').resolve():
        raise ValueError('HR terminal escaped parent run')
    with np.load(terminal_path,allow_pickle=False) as z:
        sample,rng = z['sample'].copy(),z['rng_state'].copy()
        if int(z['next_step_index']) != 12 or not np.array_equal(z['coords'],spatial):
            raise ValueError('retained HR terminal step/support differs')
    last_row = audit['stages'][-1]['rows'][-1]
    if sha(last_row['path']) != last_row['sha256']:
        raise ValueError('audited terminal Euler capture changed')
    with np.load(last_row['path'],allow_pickle=False) as z:
        if not np.array_equal(z['sample_next'],sample):
            raise ValueError('retained HR sample differs from audited terminal Euler')
    if (feats.ndim != 2 or feats.shape[1] != 32 or feats.dtype != np.float32
            or not len(feats) or sample.shape != feats.shape or sample.dtype != np.float32
            or coords.shape != (len(feats),4) or coords.dtype != np.int32
            or np.any(coords[:,0]) or not np.array_equal(coords[:,1:],spatial)
            or rng.shape != (1,2) or rng.dtype != np.uint32
            or not np.isfinite(feats).all() or not np.isfinite(sample).all()):
        raise ValueError('invalid retained shape/sample/support/RNG')
    return {'feats':feats,'sample':sample,'coords':coords,'rng':rng,'checkpoint_dir':str(cp)}, {
        'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),
        'parent_report':str(parent_path),'arithmetic_audit':str(audit_path),
        'inherited_stages':['conditioning','sparse','lr_shape','hr_shape'],
        'audit_claim_limit':audit['claim_limit'],
        'rng_boundary':'saved HR step12, restored before canonical shape decoder construction'}


def splice_completed_shape(source):
    begin = '    # === Stage 1: Sparse Structure ===\n'
    end = '    # === Stage 3: Shape Decode ===\n'
    setup_start = '    attention_route_baseline = _capture_attention_route_env()\n'
    setup_end = '    lr_slat_flow = SLatFlowModel.for_shape(\n'
    for anchor in (begin,end,setup_start,setup_end):
        if source.count(anchor) != 1:
            raise ValueError('retained shape source anchor missing or ambiguous')
    lo,hi = source.index(begin),source.index(end)
    start,stop = source.index(setup_start),source.index(setup_end)
    if not lo < start < stop < hi:
        raise ValueError('retained shape source anchors out of order')
    setup = source[start:stop]
    replacement = setup + '''    # Exact completed shape prefix, admitted by the external assay driver.
    from trellmlx.models.shape_slat_decoder import SLatDecoder
    texture_attention_route = _restore_attention_route_env(attention_route_baseline)
    hr_shape_attention_route = None
    sparse_flow_terminal_linear_json = json.dumps({'route': 'retained', 'executed': False})
    sparse_flow_cfg_rescale_std_json = sparse_flow_terminal_linear_json
    sparse_timestep_modulation_lut_json = ''
    hr_slat = mx.array(_retained_shape['feats'])
    restored_sample = _denormalize_slat(mx.array(_retained_shape['sample']))
    mx.eval(hr_slat, restored_sample)
    if not np.array_equal(np.array(hr_slat), np.array(restored_sample)):
        raise ValueError('saved shape differs from denormalized terminal HR sample')
    del restored_sample
    quant_coords = _retained_shape['coords']
    hr_coords_3d = quant_coords[:, 1:4]
    num_tokens = len(quant_coords)
    hr_resolution = 1024
    mx.random.state = [mx.array(key) for key in _retained_shape['rng']]
    mx.eval(*mx.random.state)
    print('Recovered completed shape and HR terminal RNG; continuing with shape decode', flush=True)

'''
    return source[:lo] + replacement + source[hi:]

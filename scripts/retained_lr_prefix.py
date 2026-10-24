"""Admission and source splice for a retained, completed LR diagnostic trajectory."""
import json
from pathlib import Path
import numpy as np
from fresh_prepared_inference import sha


def load_completed_lr(manifest_path, current):
    request = json.loads(Path(manifest_path).read_text())
    def checked(item):
        path = Path(item['path'])
        if sha(path) != item['sha256']:
            raise ValueError('retained artifact SHA mismatch: ' + str(path))
        return path
    replay_path = checked(request['replay_report'])
    replay = json.loads(replay_path.read_text())
    if replay['status'] != 'passed' or replay['phase'] != 'remaining_lr_steps_complete':
        raise ValueError('retained LR run is not complete')
    parent_path = checked(replay['requested']['parent_report'])
    parent = json.loads(parent_path.read_text())
    for name in ('mlx_version','device','lr_query_tile','hr_query_tile','shape_block_sync','qk_norm','sampler_sha256'):
        if current['effective'][name] != parent['effective'][name]:
            raise ValueError('retained runtime differs: ' + name)
    if (parent['source']['generator_sha256'] != current['source']['generator_sha256']
            or parent['input']['sha256'] != current['input']['sha256']
            or parent.get('conditioning_backend') != 'mlx'):
        raise ValueError('retained source/image/conditioning route mismatch')
    for key, relative in (('sampler_source', 'trellmlx/samplers.py'), ('model_source', 'trellmlx/models/slat_flow.py')):
        observed = checked(replay['requested'][key])
        active = Path(current['source']['root']) / relative
        if observed.resolve() != active.resolve():
            raise ValueError('retained source escaped active worktree: ' + key)
    cp = parent_path.parent / 'checkpoints'
    conditioning = checked(replay['requested']['conditioning'])
    if conditioning.resolve() != (cp/'conditioning.npz').resolve():
        raise ValueError('conditioning escaped parent checkpoint directory')
    sparse_path = checked(request['sparse_coords'])
    sparse_meta = checked(request['sparse_metadata'])
    if sparse_path.resolve() != (cp/'sparse_coords.npz').resolve() or sparse_meta.resolve() != (cp/'sparse_coords.json').resolve():
        raise ValueError('sparse coordinates escaped parent checkpoint directory')
    if json.loads(sparse_meta.read_text())['lr_resolution'] != 32:
        raise ValueError('retained LR support is not resolution32')
    with np.load(sparse_path,allow_pickle=False) as z: coords = z['coords'].copy()
    source_sample = checked(replay['requested']['sample'])
    with np.load(source_sample,allow_pickle=False) as z:
        prior = z['sample'].copy(); rng = z['rng_state'].copy()
        start = int(z['next_step_index']); params = json.loads(z['parameters_json'].item())
    if coords.shape != (len(prior),4) or not np.issubdtype(coords.dtype,np.integer) or np.any(coords[:,0]):
        raise ValueError('invalid retained sparse coordinate support')
    rows = replay['step_captures']
    if [r['step'] for r in rows] != list(range(start+1,13)) or params['steps'] != 12:
        raise ValueError('incomplete retained Euler sequence')
    times = np.linspace(1,0,13); times = params['rescale_t'] * times / (1+(params['rescale_t']-1)*times)
    for row in rows:
        with np.load(checked(row),allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        if not all(np.isfinite(v).all() for v in a.values()) or not np.array_equal(a['sample_in'],prior):
            raise ValueError('nonfinite or discontinuous retained Euler sequence')
        i=row['step']-1;dt=np.float32(times[i]-times[i+1]).item()
        if not np.array_equal(a['sample_next'], a['sample_in']-dt*a['pred_final']):
            raise ValueError('retained Euler update differs from CPU reconstruction')
        prior=a['sample_next']
    final_path=checked(replay['output'])
    with np.load(final_path,allow_pickle=False) as z:
        if (int(z['next_step_index'])!=12 or not np.array_equal(z['sample'],prior)
                or not np.array_equal(z['coords'],coords[:,1:]) or not np.array_equal(z['rng_state'],rng)):
            raise ValueError('retained terminal state differs from capture/support/RNG')
        sample=z['sample'].copy(); rng=z['rng_state'].copy()
    if sample.shape!=(len(coords),32) or rng.shape!=(1,2) or rng.dtype!=np.uint32:
        raise ValueError('invalid retained LR sample or random state')
    return {'sample':sample,'coords':coords,'rng':rng,'checkpoint_dir':str(cp)}, {
        'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),
        'replay_report':str(replay_path),'parent_report':str(parent_path),'sample':str(final_path),
        'inherited_stages':['conditioning','sparse','lr_shape'],
        'observation_change':'No Python trace; retained Euler capture for newly executed HR/texture.'}


def splice_completed_lr(source):
    begin='    # === Stage 1: Sparse Structure ===\n'
    end='    from trellmlx.models.shape_slat_decoder import SLatDecoder\n\n    hr_shape_attention_route = None\n'
    if source.count(begin)!=1 or source.count(end)!=1:
        raise ValueError('retained LR source anchors missing or ambiguous')
    lo=source.index(begin); hi=source.index(end)
    if lo>=hi:raise ValueError('retained LR source anchors out of order')
    setup_start = '    attention_route_baseline = _capture_attention_route_env()\n'
    setup_end = '    lr_slat_flow = SLatFlowModel.for_shape(\n'
    if source.count(setup_start) != 1 or source.count(setup_end) != 1:
        raise ValueError('retained LR configuration anchors missing or ambiguous')
    setup = source[source.index(setup_start):source.index(setup_end)]
    replacement=setup+'''    # Exact completed LR prefix, admitted by the external assay driver.
    lr_shape_sync_every_block_effective = args.lr_shape_sync_every_block
    lr_coords_4d = _retained_lr['coords']
    lr_coords = lr_coords_4d[:, 1:4]
    lr_resolution = 32
    N_lr = len(lr_coords)
    shape_slat_support_mesh_grid_size = None
    sparse_flow_terminal_linear_json = json.dumps({'route': 'retained', 'executed': False})
    sparse_flow_cfg_rescale_std_json = sparse_flow_terminal_linear_json
    sparse_timestep_modulation_lut_json = ''
    lr_slat = _denormalize_slat(mx.array(_retained_lr['sample']))
    mx.eval(lr_slat)
    mx.random.state = [mx.array(key) for key in _retained_lr['rng']]
    mx.eval(*mx.random.state)
    print('Recovered completed LR prefix and RNG; continuing with HR upsample', flush=True)

'''
    return source[:lo]+replacement+source[hi:]

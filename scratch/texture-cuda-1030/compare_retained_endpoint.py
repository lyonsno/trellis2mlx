"""Replay the retained production endpoint against authenticated FP16 controls."""
from pathlib import Path
import hashlib, json, subprocess, traceback
import numpy as np

root = Path('/Users/noahlyons/.local/state/trellis-metal-remesh-20261011/texture-model-1029')
r = root / 'texture-trajectory-1030-r1'
p = r / 'production-retained'
out = p / 'endpoint-comparison.json'
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(Path(path).read_text())
def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}
def metrics(a,b):
    d = a.astype(np.float64) - b
    ref_rms = float(np.sqrt(np.mean(b.astype(np.float64)**2)))
    err = float(np.sqrt(np.mean(d*d)))
    return dict(values=a.size, exact_values=int(np.count_nonzero(a==b)),
        rmse=err, reference_rms=ref_rms, relative_rmse=err/ref_rms,
        max_abs=float(np.max(np.abs(d))), per_channel_rmse=np.sqrt(np.mean(d*d,axis=0)).tolist())
report = dict(status='running', phase='admission', claim_ceiling=
    'Retained production-profile comparison, not an isolated dtype intervention or original CUDA BF16 parity. Same source inputs and model/sampler code; production used native shape normalization, BF16 torso, source-CUDA QK and retained Euler capture, whereas the FP16 control used shared pre-normalized inputs, FP16 torso and MLX QK.')
try:
    manifest_path = root/'native-cuda-recovery-1030/packet/dataset/witness-manifest.json'
    assert digest(manifest_path)=='973d456965a3af4758bd3762080ea671b899469d46180372c9f72190523b95a5'
    manifest = read(manifest_path)
    for name in ('decode-manifest.json','latent.npz'):
        assert digest(p/name)==manifest['files'][name]['sha256']
        assert (p/name).stat().st_size==manifest['files'][name]['size_bytes']
    plan = read(root/'continuation-plan.json')
    first = read(root/'texture-first-prediction-1030-r1/plan.json')
    for role in ('shape','noise','conditioning'):
        assert plan['inputs'][role]==first['inputs'][role]
        assert digest(plan['inputs'][role]['path'])==plan['inputs'][role]['sha256']
    assert digest(root/'continue_texture.py')==plan['driver_sha256']
    term = read(root/'continuation-terminal.json')
    disposition = read(root/'texture-result-disposition.json')
    assert term['succeeded'] and term['exit_code']==0 and term['job_id']=='3442fe0c6ed5'
    assert disposition['effective']['revision']==plan['revision']
    source = Path(__file__).resolve().parents[2]
    unchanged = subprocess.check_output(['git','diff','--name-only',plan['revision'],
        'cbedb8559b3531c7bc3f72027607bb712e83a168','--','generate.py','trellmlx/models','trellmlx/modules','trellmlx/samplers.py'],cwd=source,text=True)
    assert not unchanged, unchanged
    prior = read(r/'trajectory-comparison.json')
    assert prior['status']=='passed'
    production = load(p/'latent.npz')
    assert set(production)=={'feats','coords'}
    a,c = production['feats'],production['coords']
    assert a.shape==(24578,32) and a.dtype==np.float32 and np.isfinite(a).all()
    assert c.shape==(24578,4) and c.dtype==np.int32
    report['comparisons']={}
    for backend,path in [('mlx',r/'native-fp16-r1/sample.npz'),('cuda',r/'download/cuda_sample.npz')]:
        assert digest(path)==prior[backend]['sample_sha256']
        z=load(path)
        assert np.array_equal(c,z['coords']) and z['sample'].shape==a.shape
        assert z['sample'].dtype==a.dtype and np.isfinite(z['sample']).all()
        report['comparisons'][backend+'_fp16']=metrics(a,z['sample'])
    report.update(status='passed',phase='complete',coordinate_order_exact=True,
        source_inputs={k:plan['inputs'][k] for k in ('shape','noise','conditioning')},
        production_source=plan['revision'], fp16_source='cbedb8559b3531c7bc3f72027607bb712e83a168',
        production_latent_sha256=digest(p/'latent.npz'), decoder_manifest_sha256=digest(p/'decode-manifest.json'),
        prior_trajectory_comparison_sha256=digest(r/'trajectory-comparison.json'),
        producer_terminal_sha256=digest(root/'continuation-terminal.json'),
        producer_disposition_sha256=digest(root/'texture-result-disposition.json'),
        replay_script_sha256=digest(__file__))
except BaseException as exc:
    report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    raise
finally:
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))

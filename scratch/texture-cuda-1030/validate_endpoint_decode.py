"""Consume the common-decoder return using the existing full-field validators."""
from pathlib import Path
import json,sys,traceback
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from trellmlx.kaggle_cuda_witness import load_prepared_packet,validate_downloaded_outputs
from scripts.source_cuda_texture_decode import digest,compare,write_json

root=Path(sys.argv[1]);out=root/'validation';out.mkdir(exist_ok=True)
r=dict(status='running',phase='transport_admission',effective_backend='numpy-cpu',
       claim_ceiling='Complete common-CUDA-decoder material comparison of retained production and native FP16 endpoints on identical geometry; no hosted full-draw or isolated BF16 claim.')
def save():write_json(out/'report.json',r)
save()
try:
    prepared=json.loads((root/'prepare-report.json').read_text())
    manifest_path=root/'packet/dataset/witness-manifest.json'
    assert digest(manifest_path)==prepared['packet_manifest_sha256']
    packet=load_prepared_packet(root/'packet')
    r['outputs']=validate_downloaded_outputs(packet,root/'download');save()
    p=json.loads((root/'download/cuda_texture_report.json').read_text())
    prior=root.parent/'native-cuda-recovery-1030'
    assert digest(prior/'download/cuda_texture_report.json')=='935595c8b76be7e5ee53a3d3e3ae4374e37a25cfe6dc7e2cd1bd4c6b3d516c07'
    baseline=json.loads((prior/'download/cuda_texture_report.json').read_text())
    manifest=json.loads(manifest_path.read_text())
    receipt=json.loads((root/'download/kaggle_cuda_witness_receipt.json').read_text())
    for name,rec in manifest['files'].items():
        assert receipt['inputs'][name]['sha256']==rec['sha256']
        assert receipt['inputs'][name]['size_bytes']==rec['size_bytes']
        assert receipt['inputs'][name]['status']=='done'
    assert p['status']=='completed' and p['last_trustworthy_phase']=='complete_field_comparison'
    assert p['run_id']==prepared['plan']['run_id']
    assert p['input_rows']==24578 and p['expected_output_rows']==9539061
    assert p['effective_route']==baseline['effective_route'],'Decoder effective route changed; investigate before comparison'
    assert p['model_config']==baseline['model_config']
    assert p['inputs']['files']['guides.npz']==baseline['inputs']['files']['guides.npz']
    for name,record in prepared['plan']['inputs'].items():
        mapped={'latent':'latent.npz','decoder':'guides.npz','reference':'reference.npz'}[name]
        assert p['inputs']['files'][mapped]['sha256']==record['sha256']
    r.update(phase='complete_field_comparison',effective_route=p['effective_route']);save()
    reference=root/'packet/dataset/reference.npz'
    assert digest(reference)=='5dc52427f2f4775f59cb4abb45395c0e148e821f5bce969a9a24db3cd9c8391f'
    with np.load(reference,allow_pickle=False) as z:ref,rc=z['tex_np'],z['tex_coords_spatial']
    with np.load(root/'download/cuda_texture_state.npz',allow_pickle=False) as z:got,gc=z['tex_np'],z['tex_coords_spatial']
    assert got.shape==ref.shape==(9539061,6) and got.dtype==ref.dtype==np.float32
    assert gc.dtype==rc.dtype==np.int32
    measured=compare(got,gc,ref,rc)
    for key,value in measured.items():
        assert np.array_equal(value,p['comparison'][key]),key
    r.update(status='completed',phase='full_return_admitted',comparison=measured,
        channels=['red','green','blue','metallic','roughness','alpha'],
        fraction_abs_gt_1_over_255=(np.asarray(measured['count_abs_gt_1_over_255'])/len(got)).tolist(),
        producer_comparison_reproduced=True,reference_sha256=digest(reference),
        replay_script_sha256=digest(__file__),manifest_sha256=digest(manifest_path))
except BaseException as exc:
    r.update(status='failed',failure_phase=r['phase'],error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());save();raise
save()
print(json.dumps(r,indent=2))

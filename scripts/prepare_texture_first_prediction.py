"""Prepare identical complete texture-model inputs for the provisional FP16 control."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.texture_first_prediction import digest,write_json,validate_inputs,CLAIM
from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket,prepare_packet


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args(argv)
    plan=json.loads(args.plan.read_text());root=Path(plan['output_dir']);root.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',phase='input_admission',plan=plan,claim_ceiling=CLAIM)
    save=lambda:write_json(root/'prepare-report.json',report)
    save()
    try:
        for rec in plan['inputs'].values():
            if digest(rec['path'])!=rec['sha256']:raise ValueError('input changed: '+rec['path'])
        witness=Path(__file__).with_name('texture_first_prediction.py')
        if digest(witness)!=plan['witness_sha256'] or digest(__file__)!=plan['prepare_sha256']:raise ValueError('diagnostic source changed')
        with np.load(plan['inputs']['shape']['path'],allow_pickle=False) as z:shape,coords=z['feats'],z['coords']
        with np.load(plan['inputs']['noise']['path'],allow_pickle=False) as z:noise,noise_coords=z['sample'],z['coords']
        with np.load(plan['inputs']['conditioning']['path'],allow_pickle=False) as z:cond=z['cond']
        if shape.dtype!=np.float32 or not np.array_equal(noise_coords,coords[:,1:]):raise ValueError('shape/noise support mismatch')
        normalization=json.loads(Path(plan['inputs']['pipeline']['path']).read_text())['args']['shape_slat_normalization']
        mean,std=np.asarray(normalization['mean'],np.float32),np.asarray(normalization['std'],np.float32)
        shape_cond=(shape-mean)/std
        arrays=dict(noise=noise,shape_cond=shape_cond,coords=coords,cond=cond,t=np.array([1000],np.float32))
        validate_inputs(arrays)
        if len(coords)!=24578:raise ValueError('wrong retained kiln support')
        report['phase']='packet_prepare';save()
        capsule=root/'capsule';capsule.mkdir(exist_ok=False)
        np.savez(capsule/'inputs.npz',**arrays)
        shutil.copy2(witness,capsule/witness.name)
        report.update(normalization='NumPy float32 subtract then divide once with pinned original pipeline stats; exact resulting bytes shared by both backends',normalization_parameters=normalization,
            input_arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=__import__('hashlib').sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()})
        write_json(capsule/'provenance.json',report)
        packet=KaggleCudaWitnessPacket(capsule_dir=capsule,output_dir=root/'packet',dataset_id=plan['dataset_id'],kernel_id=plan['kernel_id'],title=plan['title'],entrypoint=witness.name,
            inputs=(witness.name,'inputs.npz','provenance.json'),run_id=plan['run_id'],enable_internet=True,
            entrypoint_args=('--backend','cuda','--run-id',plan['run_id'],'--inputs','inputs.npz','--inputs-sha256',digest(capsule/'inputs.npz')),
            output_json='cuda_prediction_report.json',output_npz='cuda_prediction.npz')
        prepare_packet(packet)
        report.update(status='completed',phase='packet_prepared',packet_manifest_sha256=digest(packet.dataset_dir/'witness-manifest.json'),input_archive_sha256=digest(capsule/'inputs.npz'));save();return 0
    except BaseException as exc:
        report.update(status='failed',failure_phase=report['phase'],error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());save();return 1

if __name__=='__main__':raise SystemExit(main())

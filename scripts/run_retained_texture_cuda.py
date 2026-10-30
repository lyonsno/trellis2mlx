"""Network-only lifecycle: private upload, remote run, filtered output recovery.
Local heavy semantic validation follows separately through Greenroom.
"""
import argparse
import json
from pathlib import Path
import re
import time
from trellmlx.kaggle_cuda_witness import (load_prepared_packet, run_command, build_dataset_command,
    wait_for_published_dataset_manifest, build_kernel_push_command, build_kernel_status_command,
    build_kernel_output_command, admit_kernel_output_download_dir)
from scripts.source_cuda_texture_decode import write_json

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
r={'status':'running','phase':'packet_load'}
report=args.root/'lifecycle.json'
def save():write_json(report,r)
def run(cmd,phase):
    r['phase']=phase;save()
    result=run_command(cmd,phase=phase,report_path=args.root/(phase+'.json'))
    if result['status']!='done':raise RuntimeError(f'{phase} failed; inspect command receipt')
    return result
save()
try:
    # Refuse an accidental duplicate launch of this attempt.
    with (args.root/'launch-claim.json').open('x') as f:
        json.dump({'packet':str(args.root/'packet'),'started_epoch':time.time()},f)
    packet=load_prepared_packet(args.root/'packet')
    run(build_dataset_command(packet),'dataset_create')
    r['phase']='dataset_visibility';save()
    wait_for_published_dataset_manifest(packet,report_path=args.root/'dataset-visible.json',scratch_root=args.root)
    run(build_kernel_push_command(packet),'kernel_push')
    r['kernel_id']=packet.kernel_id;r['phase']='remote_execution';save()
    while True:
        observed=run_command(build_kernel_status_command(packet),phase='kernel_status',report_path=args.root/'kernel-status.json')
        if observed['status']!='done':raise RuntimeError('status unavailable; remote run may still be active')
        m=re.search(r'has status "([^"]+)"',observed['stdout'],re.I)
        if m is None:raise ValueError('unrecognized Kaggle status; remote state unknown')
        status=m.group(1).lower().rsplit('.',1)[-1]
        r['remote_status']=status;save()
        if status in {'complete','error','cancelled','failed'}:break
        time.sleep(30)
    download=admit_kernel_output_download_dir(args.root/'download')
    run(build_kernel_output_command(packet,download),'download')
    r.update(status='recovered',phase='awaiting_local_validation', remote_status=status)
    save()
except BaseException as exc:
    r.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise

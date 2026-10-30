#!/usr/bin/env python3
"""Prepare a private CUDA decoder packet from retained native kiln arrays."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, prepare_packet
from source_cuda_texture_decode import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan['output_dir']); root.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'phase': 'input_validation', 'plan': plan}
    save = lambda: write_json(root/'prepare-report.json', report)
    save()
    try:
        paths = {}
        for role, rec in plan['inputs'].items():
            paths[role] = Path(rec['path'])
            if digest(paths[role]) != rec['sha256']: raise ValueError(f'input changed: {role}')
        source = Path(__file__).with_name('source_cuda_texture_decode.py')
        if digest(source) != plan['script_sha256']: raise ValueError('decoder witness changed')
        capsule = root/'capsule'
        capsule.mkdir(exist_ok=False)
        # Staging only: preserve source archives byte-for-byte. The remote decoder
        # validates guide expansion against every reference coordinate before compute.
        (capsule/'guides.npz').symlink_to(paths['decoder'])
        shutil.copy2(paths['latent'], capsule/'latent.npz')
        (capsule/'reference.npz').symlink_to(paths['reference'])
        shutil.copy2(source, capsule/source.name)
        manifest = dict(schema='trellis2mlx.retained_texture_inputs.v1', provenance=plan['inputs'],
                        validation='full array validation occurs on CUDA worker before decode', files={n: {'sha256':digest(capsule/n)} for n in
                        ['latent.npz','guides.npz','reference.npz']})
        write_json(capsule/'decode-manifest.json', manifest)
        packet = KaggleCudaWitnessPacket(capsule_dir=capsule, output_dir=root/'packet',
            dataset_id=plan['dataset_id'], kernel_id=plan['kernel_id'], title=plan['title'],
            entrypoint=source.name, inputs=(source.name,'latent.npz','guides.npz','reference.npz','decode-manifest.json'),
            entrypoint_args=('--manifest','decode-manifest.json','--manifest-sha256',digest(capsule/'decode-manifest.json')),
            run_id=plan['run_id'], enable_internet=True,
            output_json='cuda_texture_report.json', output_npz='cuda_texture_state.npz')
        if packet.output_dir.exists(): raise ValueError('packet directory already exists')
        prepare_packet(packet)
        report.update(status='completed', phase='packet_prepared',
                      packet_manifest_sha256=digest(packet.dataset_dir/'witness-manifest.json'),
                      source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=source.parent,text=True).strip())
        save(); return 0
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save(); return 1


if __name__ == '__main__': raise SystemExit(main())

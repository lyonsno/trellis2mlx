"""Replay retained Stage 1 noise through pinned official CUDA models and sampler."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_inputs(path, expected_digest):
    if digest(path) != expected_digest:
        raise ValueError('input digest mismatch')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ('noise', 'cond', 'neg_cond', 'native_terminal')}
        params = json.loads(str(archive['parameters_json'].item()))
    shapes = {'noise': (1, 8, 16, 16, 16), 'native_terminal': (1, 8, 16, 16, 16),
              'cond': (1, 1029, 1024), 'neg_cond': (1, 1029, 1024)}
    for key, value in arrays.items():
        if value.dtype != np.float32 or value.shape != shapes[key] or not np.isfinite(value).all():
            raise ValueError(f'invalid {key}: {value.shape}, {value.dtype}')
    expected = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
                    guidance_interval=[0.6, 1.0], rescale_t=5.0, sigma_min=1e-5)
    if params != expected:
        raise ValueError(f'retained sampler settings differ: {params}')
    return arrays, params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--source-tar', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--assets-sha256', required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-npz', type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    report = dict(schema='trellis2mlx.matched_sparse_cuda.v1', status='failed', run_id=args.run_id,
                  phase='input_admission', last_trustworthy_phase=None,
                  claim_ceiling='paired Stage 1 only; no full-pipeline or visual parity claim')

    def save():
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + '\n')

    try:
        save()
        arrays, params = load_inputs(args.inputs, args.inputs_sha256)
        if digest(args.source_tar) != args.source_sha256 or digest(args.assets) != args.assets_sha256:
            raise ValueError('source/assets manifest digest mismatch')
        assets = json.loads(args.assets.read_text())
        report['inputs'] = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                            sha256=hashlib.sha256(value.tobytes()).hexdigest())
                            for key, value in arrays.items()}
        report['input_file_sha256'] = args.inputs_sha256
        report['source_tar_sha256'] = args.source_sha256
        report['parameters'] = params
        report['last_trustworthy_phase'] = 'input_admission'
        report['phase'] = 'runtime_setup'
        save()
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'easydict', 'einops',
                        'tqdm', 'huggingface-hub', 'safetensors'], check=True)
        os.environ.update(ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none', SPARSE_ATTN_BACKEND='sdpa')
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from source_cuda_postcond_full_decode_timing import (
            extract_source, _sample_sparse_flow_with_official_observer,
            write_source_sparse_flow_steps_npz,
        )
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no fallback admitted')
        source = extract_source(args.source_tar, Path.cwd())
        sys.path.insert(0, str(source))
        from trellis2 import models
        from trellis2.pipelines import samplers
        from trellis2.modules.attention import config as attention_config
        if attention_config.BACKEND != 'sdpa':
            raise RuntimeError('effective attention backend differs')
        report['effective_route'] = dict(device='cuda', device_name=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)), torch=torch.__version__,
            attention=attention_config.BACKEND, flow_dtype='bfloat16',
            sampler='official FlowEulerGuidanceIntervalSampler.sample', source_revision=assets['source_revision'])
        report['phase'] = 'model_load'
        save()
        paths = {}
        for role, spec in assets['files'].items():
            path = Path(hf_hub_download(spec['repo'], spec['file'], revision=spec['revision']))
            if digest(path) != spec['sha256']:
                raise ValueError(f'model asset digest mismatch: {role}')
            paths[role] = path
        loaded = {}
        for name in ('flow', 'decoder'):
            cfg = json.loads(paths[name + '_config'].read_text())
            model = getattr(models, cfg['name'])(**cfg['args'])
            model.load_state_dict(load_file(str(paths[name + '_weights'])), strict=True)
            loaded[name] = model.eval()
        flow, decoder = loaded['flow'], loaded['decoder']
        report['model_configs'] = {name: json.loads(paths[name + '_config'].read_text()) for name in loaded}
        report['model_assets'] = assets
        sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=params['sigma_min'])
        sampler_params = {key: value for key, value in params.items() if key != 'sigma_min'}
        report['last_trustworthy_phase'] = 'model_load'
        report['phase'] = 'official_trajectory'
        save()
        with torch.inference_mode():
            flow.cuda()
            cond = {key: torch.from_numpy(arrays[key]).cuda() for key in ('cond', 'neg_cond')}
            noise = torch.from_numpy(arrays['noise']).cuda()
            terminal, payload = _sample_sparse_flow_with_official_observer(sampler, flow, noise, cond, sampler_params)
            flow.cpu()
            torch.cuda.empty_cache()
            report['last_trustworthy_phase'] = 'official_trajectory'
            report['phase'] = 'decoder'
            save()
            decoder.cuda()
            logits = decoder(terminal).float()
            native_logits = decoder(torch.from_numpy(arrays['native_terminal']).cuda()).float()

            def coords(value):
                occupied = value > 0
                ratio = occupied.shape[2] // 32
                if ratio != 1:
                    occupied = torch.nn.functional.max_pool3d(occupied.float(), ratio, ratio) > 0.5
                return torch.argwhere(occupied)[:, [0, 2, 3, 4]].int().cpu().numpy()

            support = coords(logits)
            native_support = coords(native_logits)
            logits_np, native_logits_np = logits.cpu().numpy(), native_logits.cpu().numpy()
        report['last_trustworthy_phase'] = 'decoder'
        report['phase'] = 'export'
        save()
        report['artifact'] = write_source_sparse_flow_steps_npz(args.output_npz, payload, support)
        # Keep the entire signal, including the same-decoder native-latent control.
        with np.load(args.output_npz) as z:
            output = {key: z[key] for key in z.files}
        output.update(decoded_logits=logits_np, native_decoded_logits=native_logits_np,
                      native_decoded_coords=native_support)
        np.savez(args.output_npz, **output)
        report['artifact'].update(sha256=digest(args.output_npz), size_bytes=args.output_npz.stat().st_size)
        report.update(status='done', phase='complete', last_trustworthy_phase='export',
                      elapsed_seconds=time.monotonic()-started)
        save()
        return 0
    except Exception as exc:
        report.update(status='failed', failure_phase=report['phase'], error=f'{type(exc).__name__}: {exc}',
                      elapsed_seconds=time.monotonic()-started)
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())

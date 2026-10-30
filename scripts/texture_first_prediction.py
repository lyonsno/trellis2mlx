"""One full-support FP16 texture prediction on pinned CUDA or current MLX.

This precision control is provisional. It neither changes generation defaults nor
establishes production BF16 parity, recurrence parity, or visible kiln quality.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np

TRELLIS_REVISION = '5565d240c4a494caaf9ece7a554542b76ffa36d3'
MODEL_REVISION = 'af44b45f2e35a493886929c6d786e563ec68364d'
STEM = 'ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16'
MODEL_HASHES = {
    STEM+'.json': '31f4fdad9974930e5e9324ac9dfb6ef7de6b30cea51854ba515ef827a8d353b2',
    STEM+'.safetensors': '580401269059a339b8318ab9ced459a13ba63391721c83a6c383198c29e77686',
}
CLAIM = 'Provisional same-input FP16 prediction; no production BF16, recurrence or visual parity claim.'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def validate_output(prediction, coords, expected_coords):
    if (expected_coords.ndim!=2 or expected_coords.shape[1]!=4 or not len(expected_coords)
            or prediction.shape!=(len(expected_coords),32) or prediction.dtype!=np.float32
            or coords.dtype!=np.int32 or not np.array_equal(coords,expected_coords)
            or not np.isfinite(prediction).all()):
        raise ValueError('partial, reordered, wrong-dtype or nonfinite prediction')


def validate_inputs(arrays):
    if set(arrays)!= {'noise','shape_cond','cond','coords','t'}: raise ValueError('input roles differ')
    c=arrays['coords'];n=len(c)
    if (c.shape!=(n,4) or n==0 or c.dtype!=np.int32 or np.any(c[:,0]!=0)
            or len(np.unique(c,axis=0))!=n): raise ValueError('invalid coordinate support')
    for key,shape in [('noise',(n,32)),('shape_cond',(n,32)),('cond',(1,4101,1024)),('t',(1,))]:
        a=arrays[key]
        if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all(): raise ValueError('invalid '+key)
    if arrays['t'][0]!=1000: raise ValueError('not the first prediction')


def validate_route(route, backend):
    if backend=='cuda':
        if (route.get('cuda_available') is not True or route.get('attention')!='xformers'
                or route.get('sparse_container')!='none' or route.get('trellis')!=TRELLIS_REVISION
                or route.get('torch')!='2.11.0+cu128' or route.get('xformers')!='0.0.35'):
            raise ValueError('effective CUDA route differs')
    elif backend=='mlx':
        if (route.get('mlx')!='0.32.4' or route.get('device')!='gpu'
                or route.get('attention')!='fast' or route.get('query_chunk')!=2048
                or route.get('qk_norm')!='mlx-sum'):
            raise ValueError('effective MLX route differs')
    else: raise ValueError('unknown backend')
    if route.get('torso_dtype')!='float16': raise ValueError('wrong comparison precision')


def run_cuda(args, arrays, report, save):
    def command(argv, cwd=None):
        done=subprocess.run(argv,cwd=cwd,text=True,capture_output=True)
        report.setdefault('commands',[]).append(dict(argv=argv,cwd=str(cwd),exit_code=done.returncode,stdout=done.stdout,stderr=done.stderr));save()
        if done.returncode: raise RuntimeError('command failed: '+str(argv))
        return done.stdout.strip()
    report['phase']='cuda_setup';save()
    import torch
    if not torch.cuda.is_available() or torch.__version__!='2.11.0+cu128': raise ValueError('expected measured CUDA Torch route')
    os.environ.update(SPARSE_CONV_BACKEND='none', SPARSE_ATTN_BACKEND='xformers', ATTN_BACKEND='xformers')
    command([sys.executable,'-m','pip','install','--no-deps','xformers==0.0.35'])
    command([sys.executable,'-m','pip','install','easydict','einops','tqdm','huggingface-hub','safetensors'])
    source=args.work_dir/'TRELLIS.2'
    if source.exists(): raise ValueError('fresh CUDA source directory required')
    source.parent.mkdir(parents=True,exist_ok=True)
    command(['git','clone','--filter=blob:none','--no-checkout','https://github.com/microsoft/TRELLIS.2.git',str(source)])
    command(['git','checkout','--detach',TRELLIS_REVISION],source)
    sys.path.insert(0,str(source))
    from trellis2.modules import sparse as sp
    from trellis2.modules.sparse import config
    from trellis2.models.structured_latent_flow import SLatFlowModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    route=dict(cuda_available=torch.cuda.is_available(),device=torch.cuda.get_device_name(),torch=torch.__version__,
        xformers=importlib.metadata.version('xformers'),attention=config.ATTN,sparse_container=config.CONV,
        torso_dtype='float16',trellis=command(['git','rev-parse','HEAD'],source),model_revision=MODEL_REVISION,
        source_path=inspect.getfile(SLatFlowModel),precision_change='original dtype argument bfloat16 replaced by float16; original convert_to retained')
    validate_route(route,'cuda')
    if not Path(route['source_path']).is_relative_to(source): raise ValueError('wrong source import')
    command(['git','diff','--exit-code','HEAD'],source)
    report['effective_route']=route;report['phase']='model_load';save()
    files={}
    for name,sha in MODEL_HASHES.items():
        path=Path(hf_hub_download('microsoft/TRELLIS.2-4B',name,revision=MODEL_REVISION,cache_dir=str(args.work_dir/'hf')))
        if digest(path)!=sha: raise ValueError('model hash differs: '+name)
        files[name]=path
    cfg=json.loads(files[STEM+'.json'].read_text());report['source_model_config']=cfg
    kwargs=dict(cfg['args']);kwargs['dtype']='float16'
    model=SLatFlowModel(**kwargs)
    model.load_state_dict(load_file(str(files[STEM+'.safetensors'])),strict=True)
    model.eval().cuda()
    report['parameter_dtypes']={k:str(v.dtype) for k,v in model.named_parameters()}
    if (model.dtype!=torch.float16 or model.input_layer.weight.dtype!=torch.float32
            or model.out_layer.weight.dtype!=torch.float32
            or any(b.modulation.dtype!=torch.float32 or b.self_attn.to_qkv.weight.dtype!=torch.float16 for b in model.blocks)):
        raise ValueError('source mixed precision contract differs')
    report['last_trustworthy_phase']='model_loaded';report['phase']='prediction';save()
    coords=torch.from_numpy(arrays['coords']).cuda()
    x=sp.SparseTensor(feats=torch.from_numpy(arrays['noise']).cuda(),coords=coords)
    shape=sp.SparseTensor(feats=torch.from_numpy(arrays['shape_cond']).cuda(),coords=coords)
    with torch.inference_mode():
        out=model(x,torch.from_numpy(arrays['t']).cuda(),torch.from_numpy(arrays['cond']).cuda(),concat_cond=shape)
        prediction=out.feats.cpu().numpy();got_coords=out.coords.cpu().numpy()
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    return prediction,got_coords


def run_mlx(args, arrays, report, save):
    root=args.source_root.resolve()
    if not args.native_revision or subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=args.native_revision:
        raise ValueError('native revision differs')
    # Tracked source changes are refused; unrelated scratch artifacts are not code.
    subprocess.run(['git','-C',str(root),'diff','--exit-code','HEAD'],check=True,capture_output=True)
    sys.path.insert(0,str(root))
    os.environ.pop('TRELLIS2MLX_HR_SHAPE_ATTENTION_ACTIVE',None)
    os.environ.update(TRELLIS2MLX_ATTENTION_BACKEND='fast',TRELLIS2MLX_FAST_ATTENTION_QUERY_CHUNK_SIZE='2048',TRELLIS2MLX_QK_NORM_BACKEND='mlx-sum')
    import mlx.core as mx
    import mlx.utils
    import generate
    from trellmlx.models.slat_flow import SLatFlowModel
    from trellmlx.models.sparse_structure_flow import _cast_block_linears
    from trellmlx.weight_loader import load_weights, _remap_key
    from trellmlx.greenroom_pause import pause_session
    from safetensors import safe_open
    actual=generate._shape_flow_attention_route_from_env()
    route=dict(mlx=importlib.metadata.version('mlx'),device='gpu' if mx.default_device()==mx.gpu else str(mx.default_device()),
        attention=actual['shape_flow_attention_backend_effective'],query_chunk=actual['shape_flow_fast_attention_query_chunk_size_effective'],
        qk_norm=generate.get_qk_norm_backend(),torso_dtype='float16',native_revision=args.native_revision,
        source_path=inspect.getfile(SLatFlowModel),model_revision=MODEL_REVISION,
        precision_change='block linears float16, source-equivalent float32 learned modulation; outer layers stay float32',
        qk_norm_limit='production source-cuda-warp32 refuses FP16; this provisional control uses supported mlx-sum, so differences are not attributed to production BF16')
    validate_route(route,'mlx')
    if Path(route['source_path']).resolve()!=root/'trellmlx/models/slat_flow.py': raise ValueError('wrong native source import')
    if digest(args.checkpoint)!=MODEL_HASHES[STEM+'.safetensors']: raise ValueError('native checkpoint differs')
    report['effective_route']=route;report['phase']='model_load';save()
    with pause_session(args.queue_dir,args.output_json.parent/'pause.json'):
        model=SLatFlowModel.for_texture()
        for block in model.blocks:
            _cast_block_linears(block,mx.float16);block.modulation=block.modulation.astype(mx.float32)
        params=dict(mlx.utils.tree_flatten(model.parameters()))
        with safe_open(str(args.checkpoint),framework='numpy') as sf:
            keys={_remap_key(k) for k in sf.keys()}
        if set(params)!=keys: raise ValueError('native/checkpoint parameter keys differ')
        if load_weights(model,str(args.checkpoint),verbose=False): raise ValueError('unloaded native weights')
        mx.eval(model.parameters())
        report['parameter_dtypes']={k:str(v.dtype) for k,v in mlx.utils.tree_flatten(model.parameters())}
        report['last_trustworthy_phase']='model_loaded';report['phase']='prediction';save()
        prediction=model(mx.array(arrays['noise']),mx.array(arrays['t']),mx.array(arrays['cond']),coords=mx.array(arrays['coords'][:,1:]),concat_cond=mx.array(arrays['shape_cond']))
        mx.eval(prediction)
        prediction=np.array(prediction)
        report['peak_allocated_bytes']=mx.get_peak_memory()
    return prediction,arrays['coords']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['cuda','mlx'],required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True)
    p.add_argument('--output-json',type=Path,required=True);p.add_argument('--output-npz',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,default=Path('/kaggle/temp/texture-first-prediction'))
    p.add_argument('--source-root',type=Path);p.add_argument('--native-revision');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--queue-dir',type=Path,default=Path.home()/'.local/state/gpu-greenroom')
    args=p.parse_args(argv)
    if args.output_json.exists(): raise ValueError('fresh report path required')
    report=dict(schema='trellis2mlx.texture_first_prediction.v1',status='running',phase='input_admission',
        run_id=args.run_id,backend_requested=args.backend,inputs_sha256=args.inputs_sha256,claim_ceiling=CLAIM,
        last_trustworthy_phase='arguments_parsed')
    save=lambda:write_json(args.output_json,report)
    save()
    try:
        if digest(args.inputs)!=args.inputs_sha256: raise ValueError('input digest mismatch')
        if args.output_npz.exists(): raise ValueError('stale output exists')
        with np.load(args.inputs,allow_pickle=False) as archive: arrays={k:archive[k] for k in archive.files}
        validate_inputs(arrays)
        report['input_rows']=len(arrays['coords']);report['last_trustworthy_phase']='inputs_admitted';save()
        prediction,coords=(run_cuda if args.backend=='cuda' else run_mlx)(args,arrays,report,save)
        report['phase']='output_capture';save()
        # Retain raw data even if validation exposes corruption.
        np.savez(args.output_npz,prediction=prediction,coords=coords,t=arrays['t'])
        report['output']=dict(path=str(args.output_npz),sha256=digest(args.output_npz),bytes=args.output_npz.stat().st_size)
        validate_output(prediction,coords,arrays['coords'])
        report.update(status='completed',phase='complete_prediction',last_trustworthy_phase='complete_prediction');save();return 0
    except BaseException as exc:
        report.update(status='failed',failure_phase=report['phase'],error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());save();return 1

if __name__=='__main__':raise SystemExit(main())

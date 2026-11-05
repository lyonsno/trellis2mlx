"""Opt-in managed checkpoint/exit composition for the original Metal finisher."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import subprocess

from .source_metal_checkpoint import DistanceQueryCheckpoint, FinisherCheckpointBoundary, _atomic_json


def checkpoint_requested(sdk):
    """Advisory read; the existing SDK revalidates authority before publication.

    This mirrors the deployed SDK's finite-unit condition. It never mutates
    scheduling state or grants a successor permission to execute.
    """
    from gpu_queue import dispatch
    context = sdk._context()
    if context is None:
        return False
    queue, job, request, state = context
    config = dispatch.policy(queue.queue_dir)
    hold = config.get('job_holds', {}).get(request.job_id)
    if hold and hold['request_digest'] != hashlib.sha256((job/'request.json').read_bytes()).hexdigest():
        raise ValueError('pause request identity changed')
    if hold or queue.is_paused():
        return True
    if state.get('dispatch', {}).get('service_class') != 'normal':
        return False
    pending = dispatch.pending_order(queue.queue_dir, config)
    if not pending:
        return False
    other, digest = dispatch.read_request(pending[0]/'request.json')
    return dispatch.classify(other, config, request_digest=digest, queue_dir=queue.queue_dir)[0] == 'quick'


def run_cooperative(args, runner, *, sdk=None, job_class=None, requested=None,
                    register=subprocess.run):
    if sdk is None:
        from gpu_queue import cooperative as sdk
    if job_class is None:
        from gpu_queue.models import JobRequest as job_class
    context = sdk._context()
    if context is None:
        raise ValueError('cooperative finishing requires a managed Greenroom checkpoint context')
    _, _, request, _ = context
    if args.checkpoint_root is None or args.continuation_registration is None:
        raise ValueError('cooperative finishing requires explicit checkpoint root and continuation registration')
    if Path(args.checkpoint_root).resolve().is_relative_to(Path(args.output_dir).resolve()):
        raise ValueError('checkpoint state must be separate from final output')
    registration_bytes = Path(args.continuation_registration).read_bytes()
    template = json.loads(registration_bytes)['argv']
    if not isinstance(template, list) or not template or any(not isinstance(v, str) for v in template):
        raise ValueError('continuation registration requires an argv list')
    if not any('{job_id}' in v for v in template):
        raise ValueError('continuation registration must bind the future job_id')
    identity = {'request': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'registration_sha256': hashlib.sha256(registration_bytes).hexdigest(),
                'backend_manifest_sha256': hashlib.sha256(Path(args.backend_manifest).read_bytes()).hexdigest()}
    if getattr(sdk, '__file__', None):
        identity['scheduler_sdk'] = {'path': str(Path(sdk.__file__).resolve()),
                                    'sha256': hashlib.sha256(Path(sdk.__file__).read_bytes()).hexdigest()}
    restored = sdk.load_checkpoint()
    store = DistanceQueryCheckpoint(args.checkpoint_root, identity, chunk_rows=args.query_chunk_rows,
                                    resume=restored)
    # Native simplify/UV/bake are shorter atomic units in the observed run.
    # Avoid claiming a restart of their opaque internal native state.
    store.allowed_phases = {'bvh_self_distance', 'remesh'}
    predicate = requested or (lambda: checkpoint_requested(sdk))

    def boundary(current):
        if predicate():
            current.request_yield()

    store.on_boundary = boundary

    def continuation(path):
        return job_class(job_type='command', input_path='', agent_id=request.agent_id,
                         repo_root=request.repo_root, command_cwd=request.command_cwd,
                         route_identity=request.route_identity, command_argv=request.command_argv,
                         cooperative_checkpoint=True)

    def registered(successor):
        argv = [value.replace('{job_id}', successor.job_id) for value in template]
        result = register(argv, capture_output=True, text=True)
        _atomic_json(store.root/f'registration-{successor.job_id}.json',
                     {'argv': argv, 'returncode': result.returncode,
                      'stdout': result.stdout, 'stderr': result.stderr})
        if result.returncode:
            raise RuntimeError('successor completion registration failed')

    def quiesce():
        gc.collect()
        return store.resources_released()

    while True:
        try:
            report = runner(args, checkpoint=store)
        except FinisherCheckpointBoundary:
            # Leaving the except suite drops the traceback and native frames.
            pass
        else:
            return 0 if report['status'] == 'complete' else 1
        gc.collect()
        if not store.resources_released():
            raise RuntimeError('finisher still owns native resources after unwinding')
        report_path = args.output_dir/'report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.update(status='checkpointed', active_phase=store.phase,
                      checkpoint_state=str(store.path), final_output_complete=False)
        _atomic_json(report_path, report)
        sdk.yield_if_requested(save_checkpoint=lambda directory: store.descriptor(),
                               quiesce=quiesce, continuation=continuation,
                               on_submitted=registered)
        # A request may be withdrawn while saving. Re-enter through verified
        # cached prefixes; do not leave an idle process holding the GPU slot.
        store.reset_replay()

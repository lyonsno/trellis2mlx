"""Producer integration with a strict SDK stand-in; live queue evidence is separate."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest
import torch

from trellmlx.source_metal_greenroom import run_cooperative
from test_source_metal_checkpoint import Distance


def setup(tmp_path):
    output = tmp_path/'output'
    output.mkdir()
    registration = tmp_path/'registration.json'
    registration.write_text(json.dumps({'argv':['register', '--job', '{job_id}']}))
    backend = tmp_path/'backend.json'
    backend.write_text('{}')
    args = SimpleNamespace(checkpoint_root=tmp_path/'state', output_dir=output,
                           continuation_registration=registration, backend_manifest=backend,
                           query_chunk_rows=2, expected_revision='source-first')
    request = SimpleNamespace(agent_id='owner', repo_root='/repo', command_cwd='/repo',
                              route_identity='finisher', command_argv=['python', 'finish.py'])
    return args, request


def runner(args, *, checkpoint):
    (args.output_dir/'report.json').write_text('{}')
    checkpoint.set_phase('remesh')
    result = checkpoint.wrap_bvh(Distance()).unsigned_distance(torch.arange(15).float().reshape(5,3))
    assert torch.equal(result[0], torch.arange(15).float().reshape(5,3)[:,0])
    return {'status':'complete'}


def test_producer_unwinds_then_registers_exact_successor_before_exit(tmp_path):
    args, request = setup(tmp_path)
    events = []

    class SDK:
        def _context(self): return None, None, request, {}
        def load_checkpoint(self): return None
        def yield_if_requested(self, **kw):
            saved = kw['save_checkpoint'](tmp_path)
            assert saved['state']['calls'][0]['done'] == 2
            assert kw['quiesce']() is True
            events.append('quiesced')
            successor = kw['continuation'](tmp_path/'checkpoint.json')
            assert successor.command_argv == request.command_argv
            assert successor.route_identity == request.route_identity
            assert successor.agent_id == request.agent_id
            kw['on_submitted'](successor)
            events.append('published')
            raise SystemExit(0)

    def registered(argv, **kw):
        assert argv == ['register', '--job', 'future-job']
        events.append('registered')
        return subprocess.CompletedProcess(argv, 0, '{}', '')

    with pytest.raises(SystemExit) as stopped:
        run_cooperative(args, runner, sdk=SDK(), requested=lambda:True,
                        job_class=lambda **kw:SimpleNamespace(job_id='future-job',**kw), register=registered)
    assert stopped.value.code == 0
    assert events == ['quiesced','registered','published']
    assert json.loads((args.output_dir/'report.json').read_text())['final_output_complete'] is False


def test_withdrawn_request_replays_saved_prefix_without_idle_hold(tmp_path):
    args, request = setup(tmp_path)
    checks = iter([True, False, False])

    class SDK:
        def _context(self): return None, None, request, {}
        def load_checkpoint(self): return None
        def yield_if_requested(self, **kw): return False

    assert run_cooperative(args, runner, sdk=SDK(), requested=lambda:next(checks),
                           job_class=lambda **kw:None) == 0


def test_registration_failure_does_not_publish_successor(tmp_path):
    args, request = setup(tmp_path)
    published = []

    class SDK:
        def _context(self): return None, None, request, {}
        def load_checkpoint(self): return None
        def yield_if_requested(self, **kw):
            kw['save_checkpoint'](tmp_path)
            assert kw['quiesce']()
            kw['on_submitted'](kw['continuation'](tmp_path/'checkpoint.json'))
            published.append(True)

    with pytest.raises(RuntimeError, match='registration failed'):
        run_cooperative(args, runner, sdk=SDK(), requested=lambda:True,
                        job_class=lambda **kw:SimpleNamespace(job_id='future-job',**kw),
                        register=lambda *a,**kw:subprocess.CompletedProcess(a, 1, '', 'failed'))
    assert published == []
    assert json.loads((args.checkpoint_root/'registration-future-job.json').read_text())['returncode'] == 1


def test_cooperative_mode_requires_real_managed_context(tmp_path):
    args, _ = setup(tmp_path)
    sdk = SimpleNamespace(_context=lambda:None)
    with pytest.raises(ValueError, match='managed Greenroom'):
        run_cooperative(args, runner, sdk=sdk, job_class=lambda **kw:None)
    assert not args.checkpoint_root.exists()

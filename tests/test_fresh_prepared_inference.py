"""CPU admission tests; live MLX route is established only by the queued assay."""
import ast
import importlib.util
from pathlib import Path
import unittest
import tempfile
import symtable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import json
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fresh_assay', ROOT / 'scripts/fresh_prepared_inference.py')
assay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assay)


def shape_sync_calls(source):
    tree = ast.parse(source)
    return [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'hr_slat_flow' for t in node.targets)]


class FreshAdmissionTests(unittest.TestCase):
    def capture_wrapper(self, directory, *, retained=None, fail_at=None):
        """Exercise the driver's actual nested recorder without a model or GPU."""
        import inspect
        tree = ast.parse((ROOT / 'scripts/fresh_prepared_inference.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        definitions = [n for n in ast.walk(main) if isinstance(n, ast.FunctionDef)
                       and n.name in ('save_sample', 'sampled')]
        report = {'samplers': []}
        calls = []
        writes = []

        def sampler(model, sample, cond, *, steps=12, start_step_index=0,
                    stop_after_first_step=False, guidance_strength=7.5,
                    guidance_rescale=.5, guidance_interval=(0,1), rescale_t=1,
                    sigma_min=1e-5, capture_steps=None):
            calls.append(capture_steps)
            for step in range(1, steps + 1):
                if step == fail_at:
                    raise RuntimeError('observed fake model failure')
                next_sample = sample + np.float32(1)
                if capture_steps is not None:
                    capture_steps.append({'sample_in': sample, 'sample_next': next_sample,
                                          'pred_pos': np.full_like(sample, step),
                                          'std_pos': np.array(step, np.float32), 'unused': None})
                sample = next_sample
            return sample

        namespace = {'mx': SimpleNamespace(eval=lambda *x: None,
                                          random=SimpleNamespace(state=[np.array([1,2], np.uint32)])),
                     'np': np, 'sys': sys, 'inspect': inspect, 'json': json,
                     'args': SimpleNamespace(retain_euler_steps=True),
                     'retained': retained, 'STAGES': assay.STAGES, 'report': report,
                     'samples': directory, 'original_sampler': sampler, 'capture_line': -1,
                     'write': lambda: writes.append(json.loads(json.dumps(report)))}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'driver-recorders', 'exec'), namespace)
        return namespace['sampled'], report, calls, writes

    def test_fresh_full_capture_preserves_all_stages_operands_and_rng_without_tracing(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            sampled, report, calls, _ = self.capture_wrapper(directory)
            with patch.object(sys, 'settrace') as tracing:
                for stage in assay.STAGES:
                    sampled(object(), np.zeros((2, 3), np.float32), None)
                    for step in range(1, 13):
                        path = directory / f'{stage}_euler_{step:02d}.npz'
                        self.assertTrue(path.exists(), f'missing full Euler capture: {stage} step {step}')
                        with np.load(path) as data:
                            self.assertEqual(set(data.files), {'sample_in', 'sample_next', 'pred_pos', 'std_pos'})
                            np.testing.assert_array_equal(data['sample_next'], np.full((2, 3), step, np.float32))
                        with np.load(directory / f'{stage}_step_{step:02d}.npz') as data:
                            np.testing.assert_array_equal(data['rng_state'], [[1, 2]])
                            self.assertEqual(data['next_step_index'], step)
                tracing.assert_not_called()
            self.assertEqual([r['stage'] for r in report['samplers']], list(assay.STAGES))
            self.assertTrue(all(r['completed_steps'] == list(range(1, 13)) for r in report['samplers']))
            self.assertTrue(all(r['observation_route'] == 'sampler_capture_steps_full_euler' for r in report['samplers']))
            self.assertEqual([len(c) for c in calls], [12]*4)
            self.assertNotIn('retained_prefix_verified', report)

    def test_full_capture_failure_preserves_completed_evidence_without_false_steps(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            sampled, report, _, writes = self.capture_wrapper(directory, fail_at=4)
            with self.assertRaisesRegex(RuntimeError, 'fake model failure'):
                sampled(object(), np.zeros((2, 3), np.float32), None)
            self.assertEqual(report['samplers'][0]['completed_steps'], [1, 2, 3])
            self.assertEqual(writes[-1]['last_trustworthy_evidence'], str(directory / 'sparse_step_03.npz'))
            self.assertTrue((directory / 'sparse_euler_03.npz').exists())
            self.assertFalse((directory / 'sparse_euler_04.npz').exists())

    def test_full_capture_refuses_competing_recorder(self):
        with tempfile.TemporaryDirectory() as d:
            sampled, _, calls, _ = self.capture_wrapper(Path(d))
            with self.assertRaisesRegex(ValueError, 'competing sampler capture'):
                sampled(object(), np.zeros((2, 3), np.float32), None, capture_steps=[])
            self.assertEqual(calls, [])

    def test_retained_capture_keeps_its_two_stage_identity(self):
        with tempfile.TemporaryDirectory() as d:
            sampled, report, _, _ = self.capture_wrapper(Path(d), retained={'verified': True})
            sampled(object(), np.zeros((2, 3), np.float32), None)
            sampled(object(), np.zeros((2, 3), np.float32), None)
            self.assertEqual([r['stage'] for r in report['samplers']], ['hr_shape', 'texture'])
            self.assertTrue(all(r['observation_route'] == 'sampler_capture_steps_retained_euler'
                                for r in report['samplers']))

    def test_source_hr_sync_is_absent_then_enabled_without_stage_splicing(self):
        source = (ROOT / 'generate.py').read_text()
        before = shape_sync_calls(source)[0].value
        self.assertFalse(any(k.arg == 'lr_sync_every_block' for k in before.keywords))
        after = shape_sync_calls(assay.instrument_source(source))[0].value
        self.assertTrue(next(k.value.value for k in after.keywords if k.arg == 'lr_sync_every_block'))
        self.assertEqual(len(source.splitlines()), len(assay.instrument_source(source).splitlines()))
        compile(assay.instrument_source(source), 'generate.py', 'exec')

    def test_retained_splice_keeps_downstream_dependencies_local(self):
        source = (ROOT / 'generate.py').read_text()
        transformed = assay.instrument_source(source, retained_lr=True)
        def globals_in_main(text):
            table = symtable.symtable(text, 'generate.py', 'exec')
            return set(next(child for child in table.get_children() if child.get_name() == 'main').get_globals())
        self.assertEqual(globals_in_main(transformed) - globals_in_main(source), {'_retained_lr'})
        compile(transformed, 'generate.py', 'exec')

    def test_changed_source_anchor_refuses(self):
        with self.assertRaisesRegex(ValueError, 'missing or ambiguous'):
            assay.instrument_source('hr_slat_flow = something_else()')

    def complete_report(self):
        return {'conditioning_backend': 'mlx', 'samplers': [
            {'stage': stage, 'completed_steps': list(range(1, 13))} for stage in assay.STAGES]}

    def test_partial_or_wrong_backend_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            cp = Path(d)
            r = self.complete_report()
            r['samplers'][2]['completed_steps'].pop()
            with self.assertRaisesRegex(ValueError, 'incomplete sampler'):
                assay.require_complete(r, cp)
            r = self.complete_report()
            r['samplers'].pop()
            with self.assertRaisesRegex(ValueError, 'all four'):
                assay.require_complete(r, cp)
            r = self.complete_report()
            with self.assertRaisesRegex(ValueError, 'missing or empty'):
                assay.require_complete(r, cp)
            self.valid_checkpoints(cp)
            r['conditioning_backend'] = 'torch'
            with self.assertRaisesRegex(ValueError, 'native MLX'):
                assay.require_complete(r, cp)

    def valid_checkpoints(self, cp):
        coords = np.zeros((3, 4), np.int32)
        fixtures = {
            'conditioning': {'cond': np.zeros((1,1029,1024)), 'neg_cond': np.zeros((1,1029,1024))},
            'sparse_coords': {'coords': coords},
            'shape_slat': {'coords': coords, 'feats': np.zeros((3,32))},
            'decoder_output': {'coords': coords, 'feats': np.zeros((3,7))},
            'mesh_raw': {'vertices': np.eye(3), 'faces': np.array([[0,1,2]],np.int32)},
            'texture': {'tex_np': np.ones((3,6)), 'tex_coords_spatial': coords[:,1:]},
        }
        for stage, values in fixtures.items():
            np.savez(cp / (stage + '.npz'), **values)
            if stage in ('shape_slat','decoder_output','mesh_raw','texture'):
                (cp / (stage + '.json')).write_text(json.dumps({'mesh_grid_size':1024}))

    def test_decoded_output_payload_refusal(self):
        cases = {
            'empty_geometry': ('mesh_raw', {'vertices':np.empty((0,3)), 'faces':np.empty((0,3),np.int32)}),
            'nonfinite_texture': ('texture', {'tex_np':np.full((3,6),np.nan), 'tex_coords_spatial':np.zeros((3,3),np.int32)}),
            'invalid_indices': ('mesh_raw', {'vertices':np.eye(3), 'faces':np.array([[0,1,99]],np.int32)}),
            'missing_keys': ('mesh_raw', {'unrelated':np.ones(3)}),
            'mismatched_texture_rows': ('texture', {'tex_np':np.ones((3,6)), 'tex_coords_spatial':np.zeros((2,3),np.int32)}),
        }
        with tempfile.TemporaryDirectory() as d:
            cp=Path(d)
            for name,(stage,values) in cases.items():
                with self.subTest(name=name):
                    self.valid_checkpoints(cp)
                    np.savez(cp/(stage+'.npz'),**values)
                    with self.assertRaises(ValueError):
                        assay.require_complete(self.complete_report(),cp)
            self.valid_checkpoints(cp)
            (cp/'mesh_raw.json').write_text(json.dumps({'mesh_grid_size':768}))
            with self.assertRaisesRegex(ValueError,'grid'):
                assay.require_complete(self.complete_report(),cp)
            self.valid_checkpoints(cp)
            (cp/'texture.npz').write_bytes(b'not an archive')
            with self.assertRaises(ValueError):
                assay.require_complete(self.complete_report(),cp)
            self.valid_checkpoints(cp)
            assay.require_complete(self.complete_report(),cp)

    def test_explicit_retained_lr_completion_counts_only_executed_stages(self):
        with tempfile.TemporaryDirectory() as d:
            cp = Path(d)
            self.valid_checkpoints(cp)
            report = self.complete_report()
            report['trajectory'] = 'retained_lr'
            report['conditioning_backend'] = 'retained_native_mlx'
            report['retained_prefix_verified'] = True
            report['samplers'] = report['samplers'][2:]
            assay.require_complete(report, cp)
            report['retained_prefix_verified'] = False
            with self.assertRaisesRegex(ValueError, 'retained prefix'):
                assay.require_complete(report, cp)

    def test_missing_image_fails_with_durable_admission_report(self):
        import subprocess, json, sys
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out'
            revision = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
            run = subprocess.run([sys.executable,str(ROOT/'scripts/fresh_prepared_inference.py'),
                                  '--source-root',str(ROOT),'--expected-revision','wrong-revision',
                                  '--image',str(Path(d)/'missing.png'),'--image-sha256','wrong',
                                  '--output-dir',str(out),'--retain-euler-steps'],capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['phase'],'admission')
            self.assertIn('revision differs',report['error'])
            self.assertEqual(report['samplers'],[])
            self.assertTrue(report['requested']['retain_euler_steps'])


if __name__ == '__main__':
    unittest.main()

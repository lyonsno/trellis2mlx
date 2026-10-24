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

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fresh_assay', ROOT / 'scripts/fresh_prepared_inference.py')
assay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assay)


def shape_sync_calls(source):
    tree = ast.parse(source)
    return [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'hr_slat_flow' for t in node.targets)]


class FreshAdmissionTests(unittest.TestCase):
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
                                  '--output-dir',str(out)],capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['phase'],'admission')
            self.assertIn('revision differs',report['error'])
            self.assertEqual(report['samplers'],[])


if __name__ == '__main__':
    unittest.main()

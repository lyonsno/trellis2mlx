"""CPU admission tests; live MLX route is established only by the queued assay."""
import ast
import importlib.util
from pathlib import Path
import unittest
import tempfile

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
            for stage in ('conditioning','sparse_coords','shape_slat','decoder_output','mesh_raw','texture'):
                (cp / (stage + '.npz')).write_bytes(b'x')
            r['conditioning_backend'] = 'torch'
            with self.assertRaisesRegex(ValueError, 'native MLX'):
                assay.require_complete(r, cp)

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

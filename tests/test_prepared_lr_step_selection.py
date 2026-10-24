"""CPU selection contract; actual retained tensors supply the live replay witness."""
import importlib.util
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from replay_prepared_lr_step import validate_parent_boundary

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.parent = {'samplers': [{'stage':'lr_shape', 'completed_steps':list(range(1,8)),
                                   'last_sample':'/retained/samplers/lr_shape_step_07.npz'}]}
        self.request = {'expected_start_step_index':5,
                        'sample':{'path':'/retained/samplers/lr_shape_step_05.npz'}}

    def test_old_latest_contract_stays_closed(self):
        with self.assertRaisesRegex(ValueError,'latest selection'):
            validate_parent_boundary(self.request,self.parent,5)
        self.request['expected_start_step_index']=7
        self.request['sample']['path']=self.parent['samplers'][0]['last_sample']
        validate_parent_boundary(self.request,self.parent,7)

    def test_explicit_earlier_retained_sample(self):
        self.request['parent_sample_selection']='retained_completed'
        validate_parent_boundary(self.request,self.parent,5)
        self.request['sample']['path']='/another-run/samplers/lr_shape_step_05.npz'
        with self.assertRaisesRegex(ValueError,'sample path'):
            validate_parent_boundary(self.request,self.parent,5)

    def test_missing_step_and_unknown_route_fail(self):
        self.request['parent_sample_selection']='retained_completed'
        self.parent['samplers'][0]['completed_steps'].remove(5)
        with self.assertRaisesRegex(ValueError,'recorded parent steps'):
            validate_parent_boundary(self.request,self.parent,5)
        self.setUp()
        self.request['parent_sample_selection']='fallback'
        with self.assertRaisesRegex(ValueError,'unknown parent sample'):
            validate_parent_boundary(self.request,self.parent,5)

if __name__=='__main__':unittest.main()

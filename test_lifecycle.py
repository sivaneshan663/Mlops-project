import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from lifecycle import atomic_json, validate, gate, promote, rollback, drift
from pipeline import synthetic

class LifecycleTests(unittest.TestCase):
    def test_validation_rejects_bad_labels_and_features(self):
        x, y = synthetic()
        self.assertEqual(validate(x,y)['schema'], 'passed')
        y[0] = np.nan
        with self.assertRaises(ValueError): validate(x,y)
        x,y = synthetic(); x[0,0] = 0
        with self.assertRaises(ValueError): validate(x,y)

    def test_gate_rejects_baseline_and_regression(self):
        self.assertTrue(gate({'mae_inr':10,'baseline_mae_inr':100,'r2':.9},10)['passed'])
        self.assertFalse(gate({'mae_inr':90,'baseline_mae_inr':100,'r2':.9})['passed'])
        self.assertFalse(gate({'mae_inr':12,'baseline_mae_inr':100,'r2':.9},10)['passed'])
        self.assertFalse(gate({'mae_inr':float('nan'),'baseline_mae_inr':100,'r2':.9})['passed'])

    def test_promotion_rejection_checksum_and_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            art = Path(folder)
            for version in ['one','two']:
                (art/version).write_bytes(version.encode())
                promote(art, {'version':version,'model_file':version,'model_sha256':hashlib.sha256(version.encode()).hexdigest(),'gate':{'passed':True}})
            rejected = {'version':'bad','gate':{'passed':False}}
            with self.assertRaises(ValueError): promote(art,rejected)
            self.assertEqual(json.loads((art/'cost_manifest.json').read_text())['version'],'two')
            self.assertEqual(rollback(art)['restored_version'],'one')
            manifest = json.loads((art/'cost_manifest.json').read_text())
            (art/'one').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): promote(art, manifest)

    def test_drift_and_minimum_sample(self):
        x,_ = synthetic(); shifted,_ = synthetic(43, True)
        self.assertEqual(drift(x,x)['status'],'stable')
        self.assertEqual(drift(x,shifted)['status'],'drift_detected')
        self.assertEqual(drift(x,x[:5])['status'],'insufficient_data')

if __name__ == '__main__': unittest.main()

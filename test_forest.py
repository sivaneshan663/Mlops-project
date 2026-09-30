import unittest
import numpy as np
from forest import RandomForestRegressor

class ForestTests(unittest.TestCase):
    def test_learns_heldout_relationship_and_is_reproducible(self):
        rng=np.random.default_rng(10)
        x=rng.uniform(0,10,(200,2)); y=x[:,0]*20+x[:,1]*3
        a=RandomForestRegressor(n_estimators=15).fit(x[:150],y[:150])
        b=RandomForestRegressor(n_estimators=15).fit(x[:150],y[:150])
        pred=a.predict(x[150:])
        self.assertLess(np.abs(pred-y[150:]).mean(),np.abs(y[:150].mean()-y[150:]).mean())
        np.testing.assert_allclose(pred,b.predict(x[150:]))
        self.assertAlmostEqual(float(a.feature_importances_.sum()),1)
    def test_constant_target(self):
        model=RandomForestRegressor(n_estimators=3).fit([[1,1],[2,2],[3,3]],[5,5,5])
        self.assertEqual(model.predict([[2,2]]).tolist(),[5])

if __name__=='__main__':unittest.main()

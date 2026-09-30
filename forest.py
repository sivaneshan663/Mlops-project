"""NumPy random forest using bootstrapped CART trees and random feature subsets."""
import numpy as np

class RandomForestRegressor:
    def __init__(self,n_estimators=120,min_samples_leaf=2,random_state=42,n_jobs=2,max_depth=12):
        self.n_estimators=n_estimators
        self.min_samples_leaf=min_samples_leaf
        self.random_state=random_state
        self.max_depth=max_depth

    def fit(self,x,y):
        x=np.asarray(x); y=np.asarray(y)
        rng=np.random.default_rng(self.random_state)
        self.feature_importances_=np.zeros(x.shape[1])
        def build(a,b,depth):
            leaf=float(b.mean())
            if depth>=self.max_depth or len(b)<2*self.min_samples_leaf or b.var()<1e-10:
                return leaf
            best=None; best_error=float('inf')
            for feature in rng.choice(a.shape[1],max(1,int(a.shape[1]**.5)),replace=False):
                order=np.argsort(a[:,feature]); values=a[order,feature]; target=b[order]
                sums=np.cumsum(target); squares=np.cumsum(target*target)
                cuts=np.arange(self.min_samples_leaf,len(b)-self.min_samples_leaf+1)
                cuts=cuts[values[cuts-1]<values[cuts]]
                if not len(cuts): continue
                left=squares[cuts-1]-sums[cuts-1]**2/cuts
                right=(squares[-1]-squares[cuts-1])-(sums[-1]-sums[cuts-1])**2/(len(b)-cuts)
                errors=left+right; i=int(errors.argmin())
                if errors[i]<best_error:
                    best_error=float(errors[i]); cut=cuts[i]
                    best=(int(feature),float((values[cut-1]+values[cut])/2))
            if best is None: return leaf
            feature,threshold=best; mask=a[:,feature]<=threshold
            self.feature_importances_[feature]+=max(0,float(((b-leaf)**2).sum())-best_error)
            return (feature,threshold,build(a[mask],b[mask],depth+1),build(a[~mask],b[~mask],depth+1))
        self.trees=[]
        for _ in range(self.n_estimators):
            indices=rng.integers(0,len(y),len(y))
            self.trees.append(build(x[indices],y[indices],0))
        self.feature_importances_/=self.feature_importances_.sum() or 1
        return self

    def predict(self,x):
        def one(tree,row):
            while isinstance(tree,tuple):
                feature,threshold,left,right=tree
                tree=left if row[feature]<=threshold else right
            return tree
        return np.array([np.mean([one(t,row) for t in self.trees]) for row in np.asarray(x)])

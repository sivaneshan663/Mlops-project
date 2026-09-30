"""Run: python pipeline.py [--drift-demo | --rollback]."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
from forest import RandomForestRegressor
from lifecycle import FEATURES, validate, gate, promote, rollback, drift, atomic_json, now

ROOT = Path(__file__).resolve().parent

def synthetic(seed=42, shifted=False):
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.uniform(12000 if shifted else 600, 20000, 2500), rng.integers(1, 9, 2500), rng.integers(1, 4, 2500), rng.uniform(.8, 1.4, 2500), rng.uniform(6, 36, 2500)])
    y = (x[:, 0]*(1500+x[:, 2]*450)*x[:, 3]*(1+.025*(x[:, 1]-1))+x[:, 4]*25000)*rng.normal(1, .04, len(x))
    return x, y

def run(art=None, shifted=False):
    art = Path(art or ROOT/'artifacts')
    art.mkdir(parents=True, exist_ok=True)
    x, y = synthetic(43 if shifted else 42, shifted)
    validation = validate(x, y)
    order = np.random.default_rng(42).permutation(len(y))
    train, test = order[:2000], order[2000:]
    identity = hashlib.sha256(x.tobytes()+y.tobytes()+(ROOT/'forest.py').read_bytes()+b'120-2-12-42').hexdigest()[:12]
    dataset = art/f'dataset_{identity}.csv'
    np.savetxt(dataset, np.column_stack([x, y]), delimiter=',', header=','.join(FEATURES+['cost_inr']), comments='')
    # Keep a stable copy for dataset versioning tools such as DVC. The
    # identity-named file remains the immutable evidence artifact referenced
    # by the manifest.
    versioned_dataset = art/'training_dataset.csv'
    np.savetxt(versioned_dataset, np.column_stack([x, y]), delimiter=',', header=','.join(FEATURES+['cost_inr']), comments='')
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    import mlflow
    mlflow.set_tracking_uri((ROOT/'mlruns').as_uri())
    mlflow.set_experiment('ConstructAI lifecycle pipeline')
    with mlflow.start_run(run_name='drift candidate' if shifted else 'validated training pipeline') as tracking:
        model = RandomForestRegressor().fit(x[train], y[train])
        prediction = model.predict(x[test])
        metrics = {'mae_inr': float(np.abs(y[test]-prediction).mean()), 'rmse_inr': float(np.sqrt(np.mean((y[test]-prediction)**2))), 'r2': float(1-np.sum((y[test]-prediction)**2)/np.sum((y[test]-y[test].mean())**2)), 'baseline_mae_inr': float(np.abs(y[test]-y[train].mean()).mean())}
        champion_mae = None
        current = art/'cost_manifest.json'
        if current.exists():
            old = json.loads(current.read_text())
            champion = joblib.load(art/old['model_file'])
            champion_mae = float(np.abs(y[test]-champion.predict(x[test])).mean())
        decision = gate(metrics, champion_mae)
        model_path = art/f'cost_rf_{identity}.joblib'
        joblib.dump(model, model_path)
        manifest = {'version': identity, 'model_file': model_path.name, 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(), 'model_type': 'RandomForestRegressor', 'implementation': 'NumPy bootstrapped CART forest', 'features': FEATURES, 'currency': 'INR', 'data_kind': 'synthetic', 'metrics': metrics, 'run_id': tracking.info.run_id, 'gate': decision, 'release_gate': 'passed' if decision['passed'] else 'rejected', 'champion_mae_same_holdout': champion_mae, 'dataset_file': dataset.name, 'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(), 'validation': validation, 'at': now(), 'split': {'seed':42,'train_rows':2000,'test_rows':500}, 'feature_importances': dict(zip(FEATURES, map(float, model.feature_importances_)))}
        atomic_json(art/f'candidate_{identity}.json', manifest)
        if decision['passed']:
            promote(art, manifest)
        report = {'at': now(), 'candidate': manifest, 'action': 'promoted' if decision['passed'] else 'rejected; current model retained'}
        atomic_json(art/'pipeline_report.json', report)
        mlflow.log_params({'seed':42,'rows':2500,'trees':120,'min_leaf':2,'max_depth':12,'data_kind':'synthetic','dataset_sha256':manifest['dataset_sha256'],'shifted':shifted})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({'decision': report['action'], 'version': identity})
        for path in [dataset, model_path, art/'pipeline_report.json']:
            mlflow.log_artifact(str(path))
    return report

def demo():
    # Isolated registry prevents a demonstration from changing the user's serving model.
    art = ROOT/'artifacts'/'drift_demo'
    initial = run(art)
    reference, _ = synthetic()
    observed, _ = synthetic(43, True)
    alert = drift(reference, observed)
    candidate = run(art, True) if alert['status']=='drift_detected' else None
    restored = rollback(art) if candidate and candidate['action']=='promoted' else None
    report = {'at':now(),'scope':'isolated synthetic demonstration; production registry unchanged','stable_check':drift(reference, reference),'alert':alert,'retraining':candidate,'rollback':restored,'initial_version':initial['candidate']['version']}
    atomic_json(ROOT/'artifacts'/'drift_report.json', report)
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--drift-demo', action='store_true')
    parser.add_argument('--rollback', action='store_true')
    args = parser.parse_args()
    result = demo() if args.drift_demo else rollback(ROOT/'artifacts') if args.rollback else run()
    print(json.dumps(result, indent=2))

"""Local model registry, atomic promotion, rollback, and synthetic input drift."""
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

FEATURES = ['area_sqft', 'floors', 'quality', 'location_factor', 'duration_months']
LOW = np.array([600, 1, 1, .8, 6])
HIGH = np.array([20000, 8, 3, 1.4, 36])

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def validate(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 1 or x.shape[1] != 5 or len(x) != len(y) or len(y) < 100:
        raise ValueError('Expected at least 100 labeled rows and five features')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Nonfinite or nonpositive training values')
    if (x < LOW).any() or (x > HIGH).any() or (x[:, 1:3] != np.floor(x[:, 1:3])).any():
        raise ValueError('Feature values outside supported schema')
    return {'rows': len(y), 'schema': 'passed', 'data_kind': 'synthetic'}

def gate(metrics, champion_mae=None):
    reasons = []
    if not all(np.isfinite(v) for v in metrics.values()):
        reasons.append('nonfinite metrics')
    if metrics['mae_inr'] >= metrics['baseline_mae_inr'] * .8:
        reasons.append('must improve mean baseline MAE by at least 20%')
    if metrics['r2'] < .8:
        reasons.append('R2 below 0.8')
    if champion_mae is not None and metrics['mae_inr'] > champion_mae * 1.02:
        reasons.append('MAE regression above 2% against current model on same holdout')
    return {'passed': not reasons, 'reasons': reasons}

def promote(art, manifest):
    art = Path(art)
    if not manifest['gate']['passed']:
        raise ValueError('Rejected candidates cannot be promoted')
    model = art / manifest['model_file']
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest['model_sha256']:
        raise ValueError('Model artifact checksum mismatch')
    current = art / 'cost_manifest.json'
    if current.exists():
        old = json.loads(current.read_text())
        if old['version'] != manifest['version']:
            atomic_json(art / 'previous_cost_manifest.json', old)
    atomic_json(current, manifest)

def rollback(art):
    art = Path(art)
    previous = art / 'previous_cost_manifest.json'
    if not previous.exists():
        raise ValueError('No previous model available')
    old = json.loads(previous.read_text())
    model = art / old['model_file']
    if not model.exists():
        raise ValueError('Previous model artifact missing')
    if old.get('model_sha256') and hashlib.sha256(model.read_bytes()).hexdigest() != old['model_sha256']:
        raise ValueError('Previous model checksum mismatch')
    current = json.loads((art / 'cost_manifest.json').read_text())
    atomic_json(art / 'cost_manifest.json', old)
    atomic_json(previous, current)
    return {'restored_version': old['version'], 'replaced_version': current['version']}

def drift(reference, observed):
    reference, observed = np.asarray(reference), np.asarray(observed)
    if len(observed) < 30:
        return {'status': 'insufficient_data', 'rows': len(observed), 'minimum_rows': 30}
    scores = {}
    for i, name in enumerate(FEATURES):
        edges = np.unique(np.quantile(reference[:, i], np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        expected = np.maximum(np.histogram(reference[:, i], edges)[0] / len(reference), .001)
        actual = np.maximum(np.histogram(observed[:, i], edges)[0] / len(observed), .001)
        scores[name] = float(np.sum((actual - expected) * np.log(actual / expected)))
    return {'status': 'drift_detected' if max(scores.values()) > .2 else 'stable', 'rows': len(observed), 'psi': scores, 'threshold': .2, 'meaning': 'Input distribution alert; does not prove prediction error. Retraining requires labeled data.'}

def now():
    return datetime.now(timezone.utc).isoformat()

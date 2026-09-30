"""Capture local release identities without including uploaded user documents."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path
os.environ['MLFLOW_ALLOW_FILE_STORE']='true'
import mlflow
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    with urllib.request.urlopen('http://127.0.0.1:11434/api/tags') as response:
        models=json.load(response)['models']
    files=['app.py','retrieval.py','forest.py','index.html','monitoring.html','pipeline.py','lifecycle.py','evaluate_rag.py','checks.py','requirements.txt','artifacts/cost_manifest.json','artifacts/sft_manifest.json','artifacts/evaluation.json','artifacts/rag_evaluation.json','artifacts/pipeline_report.json','artifacts/drift_report.json','artifacts/checks_report.json']
    cost=json.loads((ROOT/'artifacts'/'cost_manifest.json').read_text())
    files += ['artifacts/'+cost['model_file'], 'artifacts/'+cost['dataset_file'], 'artifacts/construction_adapter/adapter_model.safetensors', 'artifacts/sft_train.jsonl', 'artifacts/sft_test.jsonl']
    release={'models':[{k:m.get(k) for k in ['name','digest','size']} for m in models if m['name'].startswith(('qwen3:1.7b','all-minilm'))], 'files':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files if (ROOT/name).exists()}, 'scope':'local synthetic-data prototype','deployment_validation':'local tested; Docker and hosted CI prepared but not executed'}
    (ROOT/'artifacts'/'release.json').write_text(json.dumps(release,indent=2))
    mlflow.set_tracking_uri((ROOT/'mlruns').as_uri()); mlflow.set_experiment('ConstructAI releases')
    with mlflow.start_run(run_name='Prototype release'):
        mlflow.log_params({'scope':'local synthetic prototype','model':'qwen3:1.7b','adapter_base':'Qwen3-0.6B','retrieval':'hybrid all-minilm and lexical'})
        mlflow.log_artifact(str(ROOT/'artifacts'/'release.json'))
    print('Release manifest saved.')

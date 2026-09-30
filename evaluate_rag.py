"""Real upload → hybrid retrieval → Ollama answers, in an isolated temporary DB."""
import json
import os
import re
import tempfile
import time
from pathlib import Path
from lifecycle import atomic_json, now

def main():
    with tempfile.TemporaryDirectory() as folder:
        os.environ['CONSTRUCTAI_DATA'] = folder
        os.environ['OLLAMA_MODEL'] = 'qwen3:1.7b'
        os.environ['EMBEDDING_MODEL'] = 'all-minilm'
        from fastapi.testclient import TestClient
        from app import app
        rows = []
        with TestClient(app) as client:
            projects = []
            for name, supplier, tonnes, progress in [('Cedar','CedarMix',137,42),('Maple','MapleCrete',286,68)]:
                pid = client.post('/api/projects', json={'name':name}).json()['id']
                content = f'Synthetic {name} site report. Concrete supplier: {supplier}. Steel delivered: {tonnes} tonnes. Physical progress: {progress} percent. Site manager: Anita Rao. The inspection is scheduled for 17 October 2026. No actual cost or crane capacity is recorded.'
                upload = client.post(f'/api/projects/{pid}/documents', files={'file':('site-report.txt',content.encode(),'text/plain')})
                upload.raise_for_status()
                projects.append(pid)
            cases = [
                (0,'Who is the concrete supplier?','CedarMix','MapleCrete'),
                (1,'Who is the concrete supplier?','MapleCrete','CedarMix'),
                (0,'How many tonnes of steel were delivered?','137','286'),
                (1,'How many tonnes of steel were delivered?','286','137'),
                (0,'What is the physical progress percentage?','42','68'),
                (1,'What is the physical progress percentage?','68','42'),
                (0,'Who is the site manager?','Anita Rao','MapleCrete'),
                (1,'When is the inspection scheduled?','17','CedarMix'),
                (0,'Summarize my uploaded documents','CedarMix','MapleCrete'),
                (1,'What is the crane capacity?',None,'CedarMix'),
                (0,'What is the actual cost?',None,'MapleCrete'),
                (1,'Tell me more about that actual cost.',None,'CedarMix'),
            ]
            for index, (project, question, expected, forbidden) in enumerate(cases):
                start = time.perf_counter()
                response = client.post(f'/api/projects/{projects[project]}/chat', json={'message':question})
                response.raise_for_status(); answer = response.json()
                text = answer['content']
                citations = re.findall(r'\[(S\d+)\]',text)
                valid = {s['reference'] for s in answer['sources']}
                missing = bool(re.search(r'not (recorded|provided|specified|available|mentioned|included)|no .{0,50}(cost|capacity|information)|insufficient|does not (specify|provide|include|contain)|could you|can you|please (provide|clarify)',text,re.I))
                checks = {'live_hybrid_model': 'Foundation model:' in answer['mode'] and 'Hybrid search' in answer['mode'], 'project_isolation': forbidden.lower() not in json.dumps(answer).lower(), 'answer_check': expected.lower() in text.lower() if expected else missing, 'citation_check': bool(citations) and set(citations)<=valid if expected else set(citations)<=valid, 'retrieval_check': any(expected.lower() in s['text'].lower() for s in answer['sources']) if expected else True}
                row = {'question':question,'project':project,'expected':expected,'checks':checks,'passed':all(checks.values()),'answer':text,'sources':answer['sources'],'mode':answer['mode'],'seconds':round(time.perf_counter()-start,2)}
                rows.append(row)
                print(f"{index+1}/{len(cases)} {'PASS' if row['passed'] else 'FAIL'} {question}",flush=True)
        report = {'at':now(),'scope':'12 synthetic end-to-end questions, two isolated projects, actual all-minilm and Qwen3-1.7B; heuristic fact/citation/abstention checks, not expert validation. User uploads untouched.','total':len(rows),'passed':sum(r['passed'] for r in rows),'results':rows}
        root = Path(__file__).resolve().parent
        atomic_json(root/'artifacts'/'rag_evaluation.json',report)
        os.environ['MLFLOW_ALLOW_FILE_STORE']='true'
        import mlflow
        mlflow.set_tracking_uri((root/'mlruns').as_uri()); mlflow.set_experiment('ConstructAI end-to-end RAG')
        with mlflow.start_run(run_name='12 question upload retrieval generation checks'):
            mlflow.log_metrics({'check_pass_rate':report['passed']/report['total'],'questions':report['total']})
            mlflow.log_artifact(str(root/'artifacts'/'rag_evaluation.json'))
        print(json.dumps({'passed':report['passed'],'total':report['total']}))
        return report

if __name__ == '__main__': main()

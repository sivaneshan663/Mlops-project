"""Small controlled ablation; factual substring/citation checks, not expert grading."""
import json
import time
import urllib.request
from pathlib import Path
import os
os.environ['MLFLOW_ALLOW_FILE_STORE']='true'
import mlflow
from eval_scoring import score_answer
ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'

def main():
    cases=[
        {'question':'What is the approved budget?', 'context':'[S1] Heldout Cedar project: approved budget INR 7300000.', 'expected':'7300000'},
        {'question':'Who supplies the concrete?', 'context':'[S1] Heldout Birch project: concrete supplier is StoneDemo.', 'expected':'stonedemo'},
        {'question':'How much steel is pending?', 'context':'[S1] Heldout Elm project: 17 tonnes of steel are pending.', 'expected':'17'},
        {'question':'Who is the site engineer?', 'context':'[S1] Heldout Oak project: excavation is complete. No staff names provided.', 'expected':None},
    ]
    mlflow.set_tracking_uri((ROOT / 'mlruns').as_uri())
    mlflow.set_experiment('ConstructAI controlled RAG and SFT evaluation')
    results=[]
    for adapter in [False,True]:
        for rag in [False,True]:
            label=('fine-tuned' if adapter else 'base')+(' + RAG context' if rag else ' without context')
            rows=[]
            with mlflow.start_run(run_name=label) as run:
                for case in cases:
                    payload={'model':'adapter' if adapter else 'base','options':{'num_predict':70}, 'messages':[{'role':'system','content':'Answer briefly using only supplied evidence. Cite [S1] when evidence supports a fact. If evidence is missing, say the information is unavailable.'},{'role':'user','content':('Evidence: '+case['context'] if rag else 'No evidence supplied.')+'\nQuestion: '+case['question']}]}
                    req=urllib.request.Request('http://127.0.0.1:8001/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    start=time.time()
                    with urllib.request.urlopen(req,timeout=240) as response:
                        answer=json.load(response)['message']['content']
                    rows.append({'question':case['question'],'answer':answer,**score_answer(answer,case['expected'],rag),'citation_present':'[S1]' in answer,'seconds':round(time.time()-start,2)})
                metrics={'fact_check_rate':sum(r['fact_check'] for r in rows)/len(rows),'supported_response_check_rate':sum(r['supported_response_check'] for r in rows)/len(rows),'unsupported_citation_rate':sum(r['unsupported_citation'] for r in rows)/len(rows),'repetitive_citation_rate':sum(r['repetitive_citations'] for r in rows)/len(rows),'mean_seconds':sum(r['seconds'] for r in rows)/len(rows)}
                mlflow.log_params({'base_model':'Qwen/Qwen3-0.6B','adapter':adapter,'rag_context':rag,'cases':len(cases),'evaluation_type':'controlled gold-context ablation'})
                mlflow.log_metrics(metrics)
                results.append({'configuration':label,'metrics':metrics,'run_id':run.info.run_id,'cases':rows})
                print(label,metrics,flush=True)
    report={'question_count':len(cases),'base_model':'Qwen/Qwen3-0.6B','note':'Tiny synthetic holdout; supplied gold context isolates generation from retrieval. Substring and citation checks are not a measure of engineering reliability. Separate project-isolation tests validate retrieval boundaries. No automatic promotion based on these scores.','results':results}
    (ART/'evaluation.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()

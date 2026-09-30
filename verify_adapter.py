"""Verify a saved adapter and finish tracking after an interrupted verification."""
import os
os.environ['MLFLOW_ALLOW_FILE_STORE']='true'
import json
import mlflow
from adapter_service import chat, ROOT

if __name__=='__main__':
    art=ROOT/'artifacts'
    response=chat({'messages':[{'role':'user','content':'[S1] Demo-25: The approved budget is INR 3400000.\nWhat is the approved budget?'}],'options':{'num_predict':70}})['message']['content']
    if not response.strip():raise RuntimeError('Adapter generated an empty response')
    mlflow.set_tracking_uri((ROOT/'mlruns').as_uri())
    client=mlflow.MlflowClient()
    exp=client.get_experiment_by_name('ConstructAI supervised fine-tuning')
    runs=client.search_runs([exp.experiment_id],order_by=['attributes.start_time DESC'],max_results=1)
    run=runs[0]
    history=client.get_metric_history(run.info.run_id,'training_loss')
    if len(history)!=24:raise RuntimeError('Expected 24 recorded training steps')
    manifest={'base_model':'Qwen/Qwen3-0.6B','adapter':'construction_adapter','method':'LoRA supervised fine-tuning','train_examples':24,'test_examples':8,'epochs':1,'data_kind':'synthetic','run_id':run.info.run_id,'saved_adapter_inference':response,'status':'trained; experimental; no demonstrated quality improvement'}
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.set_tag('verification_recovery','Corrected tokenizer output handling; saved adapter reloaded and verified')
        (art/'sft_manifest.json').write_text(json.dumps(manifest,indent=2))
        mlflow.log_artifacts(str(art/'construction_adapter'),'adapter')
        mlflow.log_artifact(str(art/'sft_manifest.json'))
        mlflow.log_artifact(str(art/'sft_train.jsonl'))
    print(json.dumps(manifest,indent=2))

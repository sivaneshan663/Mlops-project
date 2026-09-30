"""Small real supervised LoRA experiment; synthetic examples only."""
import json
import os
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT.parent.parent / 'work' / 'hf-cache'))
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import os
os.environ['MLFLOW_ALLOW_FILE_STORE']='true'
import mlflow

BASE = 'Qwen/Qwen3-0.6B'
LOCAL_BASE = ROOT.parent.parent / 'work' / 'base-qwen06'
ART = ROOT / 'artifacts'
ART.mkdir(exist_ok=True)

def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    examples = []
    for i in range(32):
        project = f'Demo-{i+1}'
        if i%4 == 0:
            context=f'[S1] {project}: The approved budget is INR {1000000+i*100000}.'
            question='What is the approved budget?'
            answer=f'The approved budget is INR {1000000+i*100000}. [S1]'
        elif i%4 == 1:
            context=f'[S1] {project}: {i+2} tonnes of steel are pending delivery.'
            question='What material is pending?'
            answer=f'{i+2} tonnes of steel are pending delivery. [S1]'
        elif i%4 == 2:
            context=f'[S1] {project}: Excavation is complete. Foundation work has started.'
            question='Summarize the progress.'
            answer='Excavation is complete and foundation work has started. [S1]'
        else:
            context=f'[S1] {project}: The report only lists the project name.'
            question='What is the concrete supplier?'
            answer='The supplied report does not name the concrete supplier. Please provide the supplier details.'
        examples.append({'project':project, 'context':context, 'question':question, 'answer':answer})
    for name, rows in [('sft_train',examples[:24]),('sft_test',examples[24:])]:
        (ART / (name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows))
    print('Loading the 0.6B training model...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_BASE)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_BASE, dtype=torch.float32)
    layers = [model.config.num_hidden_layers-2, model.config.num_hidden_layers-1]
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0, target_modules=['q_proj','v_proj'], layers_to_transform=layers, task_type='CAUSAL_LM'))
    model.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0003)
    mlflow.set_tracking_uri((ROOT / 'mlruns').as_uri())
    mlflow.set_experiment('ConstructAI supervised fine-tuning')
    started=time.time()
    with mlflow.start_run(run_name='Qwen3-0.6B small LoRA experiment') as run:
        mlflow.log_params({'base_model':BASE, 'training_examples':24, 'test_examples':8, 'seed':42, 'lora_rank':4, 'epochs':1, 'data_kind':'synthetic', 'device':'cpu', 'layers':str(layers)})
        model.train()
        for step, example in enumerate(examples[:24]):
            prompt = tokenizer.apply_chat_template([{'role':'user','content':example['context']+'\n'+example['question']}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(example['answer']+tokenizer.eos_token, add_special_tokens=False)
            ids=torch.tensor([prompt_ids+answer_ids])
            labels=ids.clone(); labels[:,:len(prompt_ids)] = -100
            optimizer.zero_grad()
            loss=model(input_ids=ids, labels=labels).loss
            loss.backward(); optimizer.step()
            mlflow.log_metric('training_loss',float(loss.detach()),step=step)
            print(f'Training step {step+1}/24 loss={float(loss.detach()):.4f}',flush=True)
        adapter = ART / 'construction_adapter'
        model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
        model.eval()
        # Reload adapter weights from disk to verify saved artifact serving.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model,load_file(str(adapter/'adapter_model.safetensors')))
        example=examples[24]
        prompt=tokenizer.apply_chat_template([{'role':'user','content':example['context']+'\n'+example['question']}],tokenize=True,return_tensors='pt',return_dict=True,add_generation_prompt=True,enable_thinking=False)['input_ids']
        with torch.no_grad():
            out=model.generate(prompt,max_new_tokens=70,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        response=tokenizer.decode(out[0,prompt.shape[1]:],skip_special_tokens=True)
        manifest={'base_model':BASE,'adapter':'construction_adapter','method':'LoRA supervised fine-tuning','train_examples':24,'test_examples':8,'epochs':1,'data_kind':'synthetic','elapsed_seconds':round(time.time()-started,1),'run_id':run.info.run_id,'saved_adapter_inference':response,'status':'trained; experimental; no demonstrated quality improvement'}
        (ART/'sft_manifest.json').write_text(json.dumps(manifest,indent=2))
        mlflow.log_artifacts(str(adapter),'adapter')
        mlflow.log_artifact(str(ART/'sft_manifest.json'))
        mlflow.log_artifact(str(ART/'sft_train.jsonl'))
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__': main()

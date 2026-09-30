"""Local inference of the saved experimental adapter; also supports baseline evaluation."""
import os
from pathlib import Path
import threading
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT.parent.parent/'work'/'hf-cache'))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fastapi import FastAPI
torch.set_num_threads(4)
app=FastAPI(title='ConstructAI experimental adapter')
lock=threading.Lock()
tokenizer=None
model=None

@app.get('/health')
def health():
    return {'status':'ok','loaded':model is not None,'model':'Qwen3-0.6B + LoRA'}

@app.post('/chat')
def chat(payload:dict):
    global tokenizer, model
    with lock:
        if model is None:
            tokenizer=AutoTokenizer.from_pretrained(ROOT/'artifacts'/'construction_adapter')
            base=AutoModelForCausalLM.from_pretrained(ROOT.parent.parent/'work'/'base-qwen06',dtype=torch.float32)
            model=PeftModel.from_pretrained(base,ROOT/'artifacts'/'construction_adapter').eval()
        ids=tokenizer.apply_chat_template(payload['messages'],tokenize=True,return_tensors='pt',return_dict=True,add_generation_prompt=True,enable_thinking=False)['input_ids']
        maximum=min(200,payload.get('options',{}).get('num_predict',120))
        def generate():
            with torch.no_grad():
                return model.generate(ids,max_new_tokens=maximum,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        if payload.get('model')=='base':
            with model.disable_adapter(): out=generate()
        else: out=generate()
        return {'message':{'content':tokenizer.decode(out[0,ids.shape[1]:],skip_special_tokens=True)}}

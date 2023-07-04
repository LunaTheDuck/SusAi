import torch
import torchaudio
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
class MyAi():
    model = ''
    tokenizer = ''
    
    def __init__(self, name, safetensors, modelauthor):
        if torch.cuda.is_available():
            device = 'cuda'
        else :
            device = 'cpu'
        self.model = AutoGPTQForCausalLM.from_quantized("./"+name, device=device, use_safetensors=safetensors, use_triton=False)
        self.tokenizer = AutoTokenizer.from_pretrained(modelauthor+"/"+name, use_fast=True)
    def generate(self, prompt):
        prpmt = self.tokenizer(prompt, return_tensors='pt').to(model.device)
        output = self.model.generate(prpmt.input_ids)[0]
        return self.tokenizer.decode(output)
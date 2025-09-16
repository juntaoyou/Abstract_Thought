# layer-wise deactivate

import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
from types import MethodType
from datasets import load_from_disk
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from instructions import instructions, doubled_instructions, tripled_instructions
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="../models/Qwen3-8B")
parser.add_argument("-a", "--activation_mask", type=str, default="")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')

if args.activation_mask:
    print(True)
    activation_masks = torch.load(args.activation_mask)
    activation_mask_name = args.activation_mask.split("/")[-1].split(".")
    activation_mask_name = ".".join(activation_mask_name[1:])
else:
    activation_masks = [None]
    
def normal_forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


max_length = model.config.max_length
num_layers = model.config.num_hidden_layers
intermediate_size = model.config.intermediate_size
batch_size = 16

preference_set = ["Expertise", "Informativeness", "Style"]
keys_set = ['A', 'B']

output_folder = f"./res"
os.makedirs(output_folder, exist_ok=True)
target_layers = [
    f"model.layers.{i}.mlp"  for i in range(num_layers)
]
for i, layer_name in enumerate(target_layers):
    module = model
    for name in layer_name.split('.'):
        module = getattr(module, name)
    module.forward = MethodType(normal_forward, module)

def process(queries, instruction):
    new_queries = []
    for q in queries:
        new_queries.append(q + f'\n{instruction}')
    
    return new_queries   

data = load_from_disk("/NAS/yjt/Mydatasets/PSoups_queries")['query']
total_len = len(data)
for p in preference_set:
    for k in keys_set:
        data_A = process(data, instructions[p][k])
        data_A = [f"Question: {q}\nAnswer:" for q in data_A]
        for idx, activation_mask in enumerate(activation_masks):
            for i, layer_mask in enumerate(activation_mask):
                if i not in [0, 8, 17, 26, 35]: continue
                if activation_mask:
                    def factory(mask):
                        mask = mask.to(model.device)
                        def qwen_forward(self, x):
                            activation = self.act_fn(self.gate_proj(x))
                            activation.index_fill_(2, mask, 0)
                            return self.down_proj(activation * self.up_proj(x))

                        return qwen_forward            
                    module = model
                    layer_name = target_layers[i]
                    for name in layer_name.split('.'):
                        module = getattr(module, name)
                    module.forward = MethodType(factory(layer_mask), module) 
                    layer_mask = layer_mask.to(model.device)
                    results = []
                    for t in tqdm(range(0, total_len, batch_size)):
                        end_idx = min(total_len, t + batch_size)
                        batch_texts = data_A[t: end_idx]
                        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
                        with torch.no_grad():  
                            outputs = model.generate(
                                inputs["input_ids"], 
                                do_sample=True, 
                                max_length=512,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id
                            )
                            
                        for j, output in enumerate(outputs):
                            full_text = tokenizer.decode(output, skip_special_tokens=True)
                            answer = full_text.split("Answer:")[-1].strip()
                            ans = {'Q': batch_texts[j], 'A': answer}
                            results.append(ans)
                    output_file = f"{output_folder}/{p}_{k}.deactivate.{activation_mask_name}.lw_{i}.{preference_set[idx // 2]}_{keys_set[idx % 2]}.json"
                    with open(output_file, 'w') as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)
                        
                module.forward = MethodType(normal_forward, module) 
        
for i, layer_name in enumerate(target_layers):
    module = model
    for name in layer_name.split('.'):
        module = getattr(module, name)
    module.forward = MethodType(normal_forward, module)

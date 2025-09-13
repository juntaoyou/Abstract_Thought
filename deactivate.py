import argparse
import json
import os
from types import MethodType
from datasets import load_from_disk
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from instructions import instructions, doubled_instructions, tripled_instructions
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="../models/Qwen3-0.8B")
parser.add_argument("-a", "--activation_mask", type=str, default="")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model_name, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

if args.activation_mask:
    activation_masks = torch.load(args.activation_mask)
    activation_mask_name = args.activation_mask.split("/")[-1].split(".")
    activation_mask_name = ".".join(activation_mask_name[1:])
else:
    activation_masks = [None]

max_length = model.config.max_length
num_layers = model.config.num_hidden_layers
intermediate_size = model.config.intermediate_size
batch_size = 4

preference_set = ["Expertise", "Informativeness", "Style"]
keys_set = ['A', 'B']

output_folder = f"./res"
os.makedirs(output_folder, exist_ok=True)
target_layers = [
    f"model.layers.{i}.mlp"  for i in range(num_layers)
]

def process(queries, instruction):
    new_queries = []
    for q in queries:
        new_queries.append(q + f'\n{instruction}')
    
    return new_queries   

for activation_mask in activation_masks:
    if activation_mask:
        def factory(mask):
            def qwen_forward(self, x):
                activation = self.act_fn(self.gate_proj(x))
                activation.index_fill_(2, mask, 0)
                return self.down_proj(activation * self.up_proj(x))

            return qwen_forward

        for i, layer_mask in enumerate(activation_mask):
            module = model
            layer_name = target_layers[i]
            for name in layer_name.split('.'):
                module = getattr(module, name)
            module.forward = MethodType(factory(layer_mask), module) 
    data = load_from_disk("/NAS/yjt/Mydatasets/PSoups_queries")['query']
    total_len = len(data)
    for p in preference_set:
        for k in keys_set:
            data_A = process(data, instructions[p][k])
            inputs = tokenizer(data_A, return_tensors="pt", padding=True, truncation=True).to(model.device)
            results = []
            for i in tqdm(range(0, total_len, batch_size)):
                end_idx = min(total_len, i + batch_size)
                batch_queries = inputs["input_ids"][i : end_idx]
                with torch.no_grad():  
                    outputs = model.generate(batch_queries, 
                                do_sample=False, 
                                max_new_tokens=256)
                    
                    text = tokenizer.batch_decode(outputs[:, len(inputs['input_ids'][0]):], skip_special_tokens=True)
                ans = [{"Q": q, "A": t} for q, t in zip(data[i: end_idx], text)]
                results.extend(ans)
            output_file = f"{output_folder}/{p}_{k}.deactivate.{activation_mask_name}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

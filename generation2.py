import argparse
import json
import os
from types import MethodType
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn.functional as F
from tqdm import tqdm
# from vllm import LLM, SamplingParams
from datasets import load_from_disk, Dataset
from instructions import instructions, doubled_instructions, tripled_instructions
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="../models/Qwen3-8B")
parser.add_argument("-a", "--activation_mask", type=str, default="")
args = parser.parse_args()
batch_size = 16
model_name = "../models/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir="/NAS/yjt/HuggingfaceCache"
)
max_length = model.config.max_length
num_layers = model.config.num_hidden_layers
intermediate_size = model.config.intermediate_size

if args.activation_mask:
    activation_masks = torch.load(args.activation_mask)
    activation_mask_name = args.activation_mask.split("/")[-1].split(".")
    activation_mask_name = ".".join(activation_mask_name[1:])
else:
    activation_masks = [None]


# output_folder = f"results/{args.model.split('/')[-1]}/mvicuna"
# os.makedirs(output_folder, exist_ok=True)
max_length = model.config.max_length
num_layers = model.config.num_hidden_layers
intermediate_size = model.config.intermediate_size
def process(queries, instruction):
    new_queries = []
    for q in queries:
        new_queries.append(q + f'\n{instruction}')
    
    return new_queries   


target_layers = [
    f"model.layers.{i}.mlp"  for i in range(num_layers)
]

data = load_from_disk("/NAS/yjt/Mydatasets/PSoups_queries")['query']
total_len = len(data)
# data_processed = [f"Question: {q}\nAnswer:\n<think>\n\n</think>\n\n" for q in data]



def normal_forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

for i, layer_name in enumerate(target_layers):
    module = model
    for name in layer_name.split('.'):
        module = getattr(module, name)
    module.forward = MethodType(normal_forward, module)
  
# for i in tqdm(range(0, total_len, batch_size)):
#     end_idx = min(total_len, i + batch_size)
#     batch_texts = data_processed[i: end_idx]
#     inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
#     with torch.no_grad():  
#         outputs = model.generate(
#             inputs["input_ids"],
#             max_length=512,
#             do_sample=True, 
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
#     for j, output in enumerate(outputs):
#         full_text = tokenizer.decode(output, skip_special_tokens=True)
#         answer = full_text.split("Answer:")[-1].strip()
#         ans = {'Q': batch_texts[j], 'A': answer}
#         results.append(ans)
    
# with open("./res/base.json", 'w') as f:
#     json.dump(results, f, ensure_ascii=False, indent=4)
results = []
data_A = process(data, instructions['Informativeness']['B'])
data_A = [f"Question: {q}\nAnswer:\n<think>\n\n</think>\n\n" for q in data_A]
print(data_A[0])
for i in tqdm(range(0, total_len, batch_size)):
    end_idx = min(total_len, i + batch_size)
    batch_texts = data_A[i: end_idx]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():  
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=512,
            do_sample=True, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    for j, output in enumerate(outputs):
        full_text = tokenizer.decode(output, skip_special_tokens=True)
        answer = full_text.split("Answer:")[-1].strip("<think>\n\n</think>\n\n").strip()
        ans = {'Q': batch_texts[j], 'A': answer}
        results.append(ans)
    
with open(f"./res/base_Informativeness_B.json", 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
   
for i, layer_name in enumerate(target_layers):
    module = model
    for name in layer_name.split('.'):
        module = getattr(module, name)
    module.forward = MethodType(normal_forward, module)
    
           
preference_set = ["Style"]
keys_set = ['A', 'B']
for p in preference_set:
    for k in keys_set:
        results = []
        data = load_from_disk("/NAS/yjt/Mydatasets/PSoups_queries")['query']
        total_len = len(data) 
        data_A = process(data, instructions[p][k])
        data_A = [f"Question: {q}\nAnswer:\n<think>\n\n</think>\n\n" for q in data_A]
        for i in tqdm(range(0, total_len, batch_size)):
            end_idx = min(total_len, i + batch_size)
            batch_texts = data_A[i: end_idx]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():  
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    do_sample=True, 
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            for j, output in enumerate(outputs):
                full_text = tokenizer.decode(output, skip_special_tokens=True)
                answer = full_text.split("Answer:")[-1].strip("<think>\n\n</think>\n\n").strip()
                ans = {'Q': batch_texts[j], 'A': answer}
                results.append(ans)
            
        with open(f"./res/base_{p}_{k}.json", 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        for i, layer_name in enumerate(target_layers):
            module = model
            for name in layer_name.split('.'):
                module = getattr(module, name)
            module.forward = MethodType(normal_forward, module)
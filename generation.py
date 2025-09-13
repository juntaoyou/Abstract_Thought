import argparse
import json
import os
from types import MethodType
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn.functional as F
from tqdm import tqdm
# from vllm import LLM, SamplingParams
from datasets import load_from_disk
from instructions import instructions, doubled_instructions, tripled_instructions
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="../models/Qwen3-8B")
parser.add_argument("-a", "--activation_mask", type=str, default="")
args = parser.parse_args()
batch_size = 2
model_name = "../models/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
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

target_layers = [
    f"model.layers.{i}.mlp"  for i in range(num_layers)
]

data = load_from_disk("/NAS/yjt/Mydatasets/PSoups_queries")['query']
total_len = len(data)
inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True).to(model.device)
# print(len(inputs))
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
    
with open("./res/base.json", 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
# for activation_mask in activation_masks:
#     if activation_mask:
#         def factory(mask):
#             def qwen_forward(self, x):
#                 activation = self.act_fn(self.gate_proj(x))
#                 activation.index_fill_(2, mask, 0)
#                 return self.down_proj(activation * self.up_proj(x))

#             return qwen_forward
        
#         for i, layer_mask in enumerate(activation_mask):
#             target_layers[i].forward = MethodType(factory(layer_mask.to('cuda')), target_layers[i])
        
#         data = load_from_disk("/NAS/yjt/Mydatasets/PSoups_queries")['query']

#         if activation_mask:
#             output_file = f"{output_folder}/{lang}.perturb.{mask_lang}.{activation_mask_name}.jsonl"
#         else:
#             output_file = f"{output_folder}/{lang}.jsonl"

#         results = []
#         for t, o, l in zip(texts, outputs):
#             out = {"input": t, "output": o}
#             results.append(out)

#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(json.dumps(results, indent=4, ensure_ascii=False) + "\n")

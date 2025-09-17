import argparse
import json
import os
from types import MethodType
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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



target_layers = [
    f"model.layers.{i}.mlp"  for i in range(num_layers)
]

data = load_from_disk("/NAS/yjt/Mydatasets/PSoups_queries")['query']
total_len = len(data)
data_processed = [f"Question: {q}\nAnswer:\n<think>\n\n</think>" for q in data]

# print(len(inputs))
results = []
def normal_forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

for i, layer_name in enumerate(target_layers):
    module = model
    for name in layer_name.split('.'):
        module = getattr(module, name)
    module.forward = MethodType(normal_forward, module)
    
for i in tqdm(range(0, total_len, batch_size)):
    end_idx = min(total_len, i + batch_size)
    batch_texts = data_processed[i: end_idx]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():  
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,
            do_sample=True, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    for j, output in enumerate(outputs):
        full_text = tokenizer.decode(output, skip_special_tokens=True)
        answer = full_text.split("Answer:")[-1].strip("<think>\n\n</think>\n\n").strip()
        ans = {'Q': batch_texts[j], 'A': answer}
        results.append(ans)
    
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

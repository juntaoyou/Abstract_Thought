import argparse
import json
import os
from types import MethodType

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from datasets import load_from_disk
from instructions import instructions, doubled_instructions, tripled_instructions

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="../models/Qwen3-8B")
parser.add_argument("-a", "--activation_mask", type=str, default="")
args = parser.parse_args()
batch_size = 16
# model_name = "../models/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
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


output_folder = f"results/{args.model.split('/')[-1]}/mvicuna"
os.makedirs(output_folder, exist_ok=True)

target_layers = [
    f"model.layers.{i}.mlp"  for i in range(num_layers)
]

for activation_mask in activation_masks:
    if activation_mask:
        def factory(mask):
            def qwen_forward(self, x):
                activation = self.act_fn(self.gate_proj(x))
                activation.index_fill_(2, mask, 0)
                return self.down_proj(activation * self.up_proj(x))

            return qwen_forward
        
        for i, layer_mask in enumerate(activation_mask):
            target_layers[i].forward = MethodType(factory(layer_mask.to('cuda')), target_layers[i])
        
        data = load_from_disk("/NAS/yjt/Mydatasets/PSoups_queries")['query']

        if activation_mask:
            output_file = f"{output_folder}/{lang}.perturb.{mask_lang}.{activation_mask_name}.jsonl"
        else:
            output_file = f"{output_folder}/{lang}.jsonl"

        results = []
        for t, o, l in zip(texts, outputs):
            out = {"input": t, "output": o}
            results.append(out)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False) + "\n")

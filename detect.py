import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from types import MethodType
from datasets import load_dataset, load_from_disk
from instructions import instructions, doubled_instructions, tripled_instructions

batch_size = 16
model_name = "../models/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
# print(dir(model.config))
max_length = model.config.max_length
num_layers = model.config.num_hidden_layers
intermediate_size = model.config.intermediate_size



def factory(idx):
    def qwen_forward(self, x):
        values = self.act_fn(self.gate_proj(x))
        activation = values.float()
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        return self.down_proj(values * self.up_proj(x))

    return qwen_forward
target_layers = [
    f"model.layers.{i}.mlp"  for i in range(num_layers)
]

for i, layer_name in enumerate(target_layers):
    module = model
    for name in layer_name.split('.'):
        module = getattr(module, name)
    module.forward = MethodType(factory(i), module)
data = load_from_disk("/NAS/yjt/Mydatasets/PSoups_queries")['query']
# print(len(data))
def process(queries, instruction):
    new_queries = []
    for q in queries:
        new_queries.append(q + f'\n{instruction}')
    
    return new_queries   

inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True).to(model.device)
print(len(inputs["input_ids"]), len(inputs["input_ids"][0]))
over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
for i in range(0, len(inputs), batch_size):
    end_idx = min(len(inputs), i + batch_size)
    batch_queries = inputs["input_ids"][i : end_idx]
    with torch.no_grad():  
        model.generate(batch_queries, max_new_tokens=1)
        
model_suffix = model_name.replace("../models/", "")
n = len(inputs["input_ids"][0]) * len(inputs["input_ids"])
# num_layers, intermediate_size = over_zero.size()
output = dict(n = n, over_zero=over_zero.to('cpu'))
torch.save(output, f'data/activation.train.{model_suffix}.base')

for key, items in instructions.items():
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
    data_A = process(data, instructions[key]['A'])
    inputs = tokenizer(data_A, return_tensors="pt", padding=True, truncation=True).to(model.device)

    for i in range(0, len(inputs), batch_size):
        end_idx = min(len(inputs), i + batch_size)
        batch_queries = inputs["input_ids"][i : end_idx]
        with torch.no_grad():  
            model.generate(batch_queries, max_new_tokens=1)
            
    n = len(inputs["input_ids"][0]) * len(inputs["input_ids"])
    # num_layers, intermediate_size = over_zero.size()
    output = dict(n = n, over_zero=over_zero.to('cpu'))
    torch.save(output, f'data/activation.train.{model_suffix}.{key}.A')
    
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
    data_B = process(data, instructions[key]['B'])
    inputs = tokenizer(data_B, return_tensors="pt", padding=True, truncation=True).to(model.device)

    for i in range(0, len(inputs), batch_size):
        end_idx = min(len(inputs), i + batch_size)
        batch_queries = inputs["input_ids"][i : end_idx]
        with torch.no_grad():  
            model.generate(batch_queries, max_new_tokens=1)
            
    n = len(inputs["input_ids"][0]) * len(inputs["input_ids"])
    # num_layers, intermediate_size = over_zero.size()
    output = dict(n = n, over_zero=over_zero.to('cpu'))
    torch.save(output, f'data/activation.train.{model_suffix}.{key}.B')

for key, items in doubled_instructions.items():
    for subkey in items.keys():
        over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
        data_A = process(data, doubled_instructions[key][subkey])
        inputs = tokenizer(data_A, return_tensors="pt", padding=True, truncation=True).to(model.device)

        for i in range(0, len(inputs), batch_size):
            end_idx = min(len(inputs), i + batch_size)
            batch_queries = inputs["input_ids"][i : end_idx]
            with torch.no_grad():  
                model.generate(batch_queries, max_new_tokens=1)
                
        n = len(inputs["input_ids"][0]) * len(inputs["input_ids"])
        output = dict(n = n, over_zero=over_zero.to('cpu'))
        torch.save(output, f'data/activation.train.{model_suffix}.{key}.{subkey}')
            
    
for key, items in tripled_instructions.items():
    for subkey in items.keys():
        over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
        data_A = process(data, tripled_instructions[key][subkey])
        inputs = tokenizer(data_A, return_tensors="pt", padding=True, truncation=True).to(model.device)

        for i in range(0, len(inputs), batch_size):
            end_idx = min(len(inputs), i + batch_size)
            batch_queries = inputs["input_ids"][i : end_idx]
            with torch.no_grad():  
                model.generate(batch_queries, max_new_tokens=1)
                
        n = len(inputs["input_ids"][0]) * len(inputs["input_ids"])
        output = dict(n = n, over_zero=over_zero.to('cpu'))
        torch.save(output, f'data/activation.train.{model_suffix}.{key}.{subkey}')
    
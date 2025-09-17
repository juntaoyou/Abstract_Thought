import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from vllm import LLM, SamplingParams
from instructions import coherence_template, helpfulness_template, expertise_template, informativeness_template, correctness_template, user_content
from utils.utils import evaluate_with_judge_model
from transformers import AutoTokenizer
import json
import torch
model_name = "../models/Qwen2.5-72B-Instruct-AWQ"
file_path = "./res/base.json"
with open(file_path, 'r') as f:
    data = json.load(f)

system_prompt = "Your task is to score the response.\n"
queries = [[{"role":'system',"content": system_prompt},
            {"role":'user', 'content': user_content.format_map({'query': d['Q'], 'response': d['A']})}]
                    for d in data]
sampling_params = SamplingParams(
    max_tokens=16,
    skip_special_tokens=True,
    temperature=0.8,
    top_p=0.95
)
llm = LLM(model_name, gpu_memory_utilization=0.7)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
scores_list = []
for principle in ['helpful', "correct", "coherent", "informative", "expert"]:
    if principle == "helpful":
        system_prompt = helpfulness_template
    elif principle == "correct":
        system_prompt = correctness_template
    elif principle == "coherent":
        system_prompt = coherence_template
    elif principle == "informative":
        system_prompt = informativeness_template
    elif principle == "expert":
        system_prompt = expertise_template
    else:
        raise ValueError("Invalid Prefence")
    
    cur_scores, mean_value = evaluate_with_judge_model(llm, sampling_params, queries, batch_size=16, system_prompt = system_prompt, tokenizer = tokenizer, )
    scores_list.append({"principle": principle, "mean_score": mean_value})
    save_path = file_path.split("/")[-1].strip(".json") + ".scores.json"
    with open(save_path, 'w') as f:
        json.dump({'scores': scores_list}, f, ensure_ascii=False, indent=4)
        
    torch.cuda.empty_cache()



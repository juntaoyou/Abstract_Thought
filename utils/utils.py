from vllm import LLM, SamplingParams
# from vllm.model_executor.models.reward_model import RewardModel
from transformers import AutoModelForCausalLM
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from transformers import AutoTokenizer
import re
import datetime
import numpy as np
nowtime = datetime.datetime.now()
date_string = nowtime.strftime("%Y-%m-%d %H:%M:%S")

helpdfulness_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 0-4 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Helpfulness\n"
"Point 0: The response is not useful or helpful at all. The response completely missed the essence of  what the user wanted.\n"
"Point 1: The response is borderline unhelpful and mostly does not capture what the user was looking  for, but it is still usable and helpful in a small way.\n"
"Point 2: The response is partially helpful but misses the overall goal of the user’s query/input in some  way. The response did not fully satisfy what the user was looking for.\n"  
"Point 3: The response is mostly helpful and mainly aligned with what the user was looking for, but  there is still some room for improvement.\n"
"Point 4: The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for.\n"
"Based on the above criteria, score the Helpfulness dimension of the following query and response without explanation:\n"
)

correctness_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 0-4 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Correctness\n"
"Point 0: The response is completely incorrect. All information provided is wrong, false or hallucinated. If the prompt asks the assistant to do a task, the task is not at all attempted, or the wrong task was attempted in the response. The response is completely irrelevant to the prompt.\n"
"Point 1: The response has some correct elements but is mostly wrong or incomplete. The response may contain multiple instances of hallucinations, false information, misleading information, or irrelevant information. If the prompt asks the assistant to do a task, the task was attempted with a small amount of success.\n"
"Point 2: The response contains a mix of correct and incorrect information. The response may miss some details, contain misleading information, or minor hallucinations, but is more or less aligned with what the prompt asks for. If the prompt asks the assistant to perform a task, the task is attempted with moderate success but still has clear room for improvement.\n"  
"Point 3: The response is mostly accurate and correct with a small amount of missing information. It contains no misleading information or hallucinations. If the prompt asks the assistant to perform a task, the task is mostly successfully attempted.\n"
"Point 4: The response is completely correct and accurate to what is requested by the prompt with no necessary details missing and without false, misleading, or hallucinated information. If the prompt asks the assistant to do a task, the task is completely done and addressed in the response.\n"
"Based on the above criteria, score the Correctness dimension of the following query and response without explanation:\n"
)

coherence_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 0-4 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Coherence\n"
"Point 0: The response is completely incomprehensible and no clear meaning or sensible message can be discerned from it.\n"
"Point 1: The response is mostly hard to follow, with inconsistencies, contradictions, confusing logic flow, or unclear language used throughout, but there are some coherent/clear parts.\n"
"Point 2: The response is a little unclear. There are some inconsistencies or contradictions, run on sentences, confusing statements, or hard to follow sections of the response.\n"  
"Point 3: The response is mostly clear and coherent, but there may be one or two places where the wording is confusing or the flow of the response is a little hard to follow. Over all, the response can mostly be followed with a little room for improvement.\n"
"Point 4: The response is perfectly clear and self-consistent throughout. There are no contradictory assertions or statements, the writing flows logically and following the train of thought/story is not challenging.\n"
"Based on the above criteria, score the Coherence dimension of the following query and response without explanation:\n"
)

user_content = (
"Query: {query}\n"
"response:{response}\n"
"Score:\n"
)

def evaluate_with_judge_model(llm, sampling_params, queries, batch_size=16, system_prompt="You are a helpful assistant.", tokenizer=None):
    scores = []
    total = len(prompts)
    pattern = r"^\s*(\d+\.?\d*)"
    # pattern = r"\d"
    for i in tqdm(range(0, total, batch_size), desc="Loading Data: "):
        end_idx = min(total, i + batch_size)
        batch_queries = queries[i: end_idx]
        chat_prompts = tokenizer.apply_chat_template(batch_queries, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate(chat_prompts, sampling_params)
        batch_scores = [int(re.match(pattern, score.outputs[0].text).group(1))
            for score in outputs]
        for score in batch_scores:
            if score < 1 or score > 6:
                raise ValueError("Invalid score value.")
        scores.extend(batch_scores)
        scores = np.array(scores)
    return scores , np.mean(scores)
    

def process_and_evaluate(
    generator_model_name,
    batch_size=8,
    num_beams=5,
    output_file=f"./predictions/pre/responses_{date_string}",
    add_principle = True,
    principle = None
):
    dataset = load_dataset("json", data_files="../Mydatasets/HelpSteer2/validation.jsonl")['train']
    dataset = dataset.select([i for i in range(0, len(dataset), 2)])
    print(f"加载数据集完成，共 {len(dataset)} 个样本")
    tokenizer = AutoTokenizer.from_pretrained(generator_model_name, padding_side='left')
    if not add_principle:
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{'role':'system', 'content': "Your task is to generate responses."}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True
            )
            for q in dataset["prompt"]
        ]
    else:
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{'role':'system', 'content': f"Your task is to generate responses by considering the following principles: {principle}"}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True
            )
            for q in dataset["prompt"]
        ]
    print(chat_prompts[0])

    print("开始生成回答...")
    sampling_params = SamplingParams(
        n=num_beams,
        max_tokens=256,
        skip_special_tokens=True,
        temperature=0.7, 
        top_p=0.95
    )
    
    llm = LLM(
        model=generator_model_name,
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )
    
    all_responses = []
    for i in tqdm(range(0, len(chat_prompts), batch_size), desc="生成回答"):
        end_idx = min(i + batch_size, len(chat_prompts))
        batch_prompts = chat_prompts[i:end_idx]        
        outputs = llm.generate(batch_prompts, sampling_params)
        j = i
        for output in outputs:
            sample_responses = [o.text for o in output.outputs]
            all_responses.append({
                "query": dataset["prompt"][j],
                "responses": sample_responses,
            })
            j += 1
            
    with open(output_file, 'w') as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=4)

def evaluate(file_path, 
             model_name = "../models/Qwen2.5-72B-Instruct-AWQ",
             principle="helpfulness",
             batch_size=16,
             output_file="./predictions/pre/scores.json",
             num_beams=10):
    with open(file_path, 'r') as f:
        all_responses = json.load(f)

    eval_prompts = []
    eval_responses = []
    response_indices = []
    
    for idx, item in enumerate(all_responses):
        for resp in item["responses"]:
            eval_prompts.append(item["query"])
            eval_responses.append(resp)
            response_indices.append(idx)

    if principle == "helpful":
        system_prompt = helpdfulness_template
    elif principle == "correct":
        system_prompt = correctness_template
    elif principle == "coherent":
        system_prompt = coherence_template
    else:
        raise ValueError("Invalid Prefence")
    
    print("加载Judge Model并开始评估...")
    scores = evaluate_with_judge_model(
        model_name,
        eval_prompts,
        eval_responses,
        batch_size=batch_size,
        system_prompt=system_prompt
    )
    print("整理评估结果...")
    for i, score in enumerate(scores):
        idx = response_indices[i]
        beam_idx = i % num_beams
        all_responses[idx]["responses"][beam_idx] = {
            "text": all_responses[idx]["responses"][beam_idx],
            "score": score
        }
    
    # # 为每个样本找到最佳评分的回答
    # for item in all_responses:
    #     item["best_response"] = max(
    #         item["responses"],
    #         key=lambda x: x["score"]
    #     )
    
    # 6. 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=4)
    
    print(f"评估完成，结果已保存至 {output_file}")
    return all_responses
    
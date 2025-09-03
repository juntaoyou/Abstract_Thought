# The code are used to evaluate the ability of open-source LLM to generate useful responses on preferences dimensions
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import argparse
import torch
from tqdm import tqdm
from transformers import default_data_collator
from datasets import load_dataset
import json
import datetime
from torch.utils.data import DataLoader
set_seed(42)
nowtime = datetime.datetime.now()
date_string = nowtime.strftime("%Y-%m-%d")

prompt_template = "[Guidelines] Your task is to generate response for the following instruction\n[Instruction] {query}"
def generate_response_with_qwen(args, max_length=512, temperature=0.8, batch_size=16, num_beams=5):
    model_name = args.model_name
    messages = load_dataset("json", data_files=args.train_data_path, cache_dir="/NAS/yjt/HuggingfaceCache")['train']
    oddindices = [i for i in range(len(messages)) if i % 2 == 0]
    def generate_prompt(x):
        x['prompt'] = prompt_template.format_map({'query': x['prompt']})
        
        return x
    messages = messages.select(oddindices).map(generate_prompt)  
    dataloader = DataLoader(
        messages['prompt'],
        batch_size=batch_size,
        shuffle=False,
        # pin_memory=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"  
    )
    model.eval()
    results = []
    
    try:
        for message in tqdm(dataloader, desc="处理进度"):
            message = [[{"role": "system", "content": "You are a helpful assistant."}, 
                       {"role": "user", "content": m}] for m in message]
            prompt = tokenizer.apply_chat_template(
                message, 
                tokenize=False,
                add_generation_prompt=True  
            )
            inputs = tokenizer(prompt, return_tensors="pt", padding=True,truncation=True).to(model.device)
            with torch.no_grad(): 
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,  
                    num_beams=num_beams,
                    return_dict_in_generate=True, # 关键参数：返回生成的字典格式结果
                    output_scores=True,           # 关键参数：返回每个step的分数
                    num_return_sequences=num_beams,
                    pad_token_id=tokenizer.eos_token_id,  # 设置pad token
                    eos_token_id=tokenizer.eos_token_id,   # 设置结束token
                    early_stopping=True  
                )
            # all_result = []
            for i in range(len(prompt)):
                sample_beams = []
                l = len(inputs["input_ids"][i])
                for beam_idx in range(num_beams):
                    generated_text = tokenizer.decode(
                        outputs.sequences[i * num_beams + beam_idx][l:],
                        skip_special_tokens=True
                    )
                    score = outputs.sequences_scores[i * num_beams + beam_idx].item()
                    sample_beams.append({
                        "beam_index": beam_idx,
                        "score": score,
                        "response": generated_text
                    })
                
                # 按分数排序并添加到结果列表
                sample_beams.sort(key=lambda x: x["score"], reverse=True)
                results.append({'query': message[i], "respones": sample_beams})
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"生成回答时发生错误: {str(e)}")
    
    return results

if __name__ == "__main__":
    # 构建prompt模板
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--model_name", type=str, help="Model name to generate reponse", default="/NAS/yjt/models/Qwen2.5-3B-Instruct")    
    parser.add_argument("--train_data_path", type = str, help = "Train data path", default="/NAS/yjt/Mydatasets/HelpSteer2/train.jsonl")
    args = parser.parse_args()
    result = generate_response_with_qwen(args)
    with open(f"./predictions/pre/results_{date_string}.json", 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
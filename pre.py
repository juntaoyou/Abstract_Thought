# The code are used to evaluate the ability of open-source LLM to generate useful responses on preferences dimensions
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
import json
import datetime
set_seed(42)
nowtime = datetime.datetime.now()
date_string = nowtime.strftime("%Y-%m-%d")

prompt_template = """[Guidelines] Your task is to generate response by considering the following principle.
            [Instruction] {query}
            """
def generate_response_with_qwen(args, max_length=128, temperature=0.7):
    model_name = args.model_name
    messages = load_dataset("json", data_files=args.train_data_path, cache_dir="/NAS/yjt/HuggingfaceCache")['train']
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"  
        )
        
        # 确保模型在评估模式
        model.eval()
        
        # 构建输入
        results = []
        for message in tqdm(messages, desc="处理进度"):
            message = prompt_template.format_map({'query': message['prompt']})
            prompt = tokenizer.apply_chat_template(
                message, 
                tokenize=False,
                add_generation_prompt=True  
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad(): 
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,  
                    pad_token_id=tokenizer.eos_token_id,  # 设置pad token
                    eos_token_id=tokenizer.eos_token_id   # 设置结束token
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            results.append({
                "formatted_input": prompt,
                "generated_response": response
            })
        
    except Exception as e:
        print(f"生成回答时发生错误: {str(e)}")
    
    return results

if __name__ == "__main__":
    # 构建prompt模板
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--model_name", type=str, help="Model name to generate reponse", default="/NAS/yjt/models/Qwen2.5-3B")    
    parser.add_argument("--train_data_path", type = str, help = "Train data path", default="/NAS/yjt/Mydatasets/HelpSteer2/train.jsonl")
    args = parser.parse_args()
    result = generate_response_with_qwen(args)
    with open(f"./predictions/pre/results_{date_string}.json") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
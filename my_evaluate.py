import os

import torch
import json
import argparse
import evaluate
import warnings

import numpy as np
from datasets import load_from_disk
from transformers import set_seed, AutoTokenizer,  AutoModelForCausalLM
from transformers import GenerationConfig
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm
import os
import csv
from filelock import FileLock
import datetime

nowtime = datetime.datetime.now()
date_string = nowtime.strftime("%Y-%m-%d %H:%M:%S")
# print(date_string)
# from utils.utils import write_to_csv
def write_to_csv(method, metric, value, file_path = "../result.csv"):
    # file_path = "../result.csv"
    lock_path = file_path + ".lock"
    with FileLock(lock_path):
        if os.path.exists(file_path):
            with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = list(csv.reader(file))
        else:
            reader = []
        if not reader:
            reader.append(["method"])
        headers = reader[0]
        methods = [row[0] for row in reader[1:]]
        if metric not in headers:
            headers.append(metric)
            for row in reader[1:]:
                row.append("")
        if method not in methods:
            reader.append([method] + ["" for _ in range(len(headers) - 1)])
        for row in reader:
            while len(row) < len(headers):
                row.append("")
        method_index = methods.index(method) + 1 if method in methods else len(reader) - 1
        metric_index = headers.index(metric)
        reader[method_index][metric_index] = value
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(reader)
            
import nltk
nltk.data.path.append("/home/yjt/nltk_data")
warnings.filterwarnings("ignore")

set_seed(42)
bertscore_metric = evaluate.load("./metrics/bertscore")
accuracy_metric = evaluate.load("./metrics/accuracy")
bleu_metric = evaluate.load("./metrics/sacrebleu")
rouge_metric = evaluate.load('./metrics/rouge')
meteor_metric = evaluate.load('./metrics/meteor')
# print('OK')
parser = argparse.ArgumentParser()
parser.add_argument("--max_tokens", type=int, default=48)
parser.add_argument("--task_name", default="product_review_temporal", help = "Task Name")
parser.add_argument("--model_dir", default="/NAS/yjt/models/Qwen3-1.7B", help = "Model Dir")
parser.add_argument("--base_model", default="Qwen3-1.7B", help = "Base Model")
parser.add_argument("--batch_size", default=36, type=int)
# parser.add_argument("--test_path", type=str)
# parser.add_argument("--pred_save_path", default="", type=str)

args = parser.parse_args()
task_name = args.task_name
base_model = args.base_model
batch_size = args.batch_size
model_dir = args.model_dir
# test_path = args.test_path
max_tokens = args.max_tokens
# pred_save_path = args.pred_save_path
if task_name == "product_review_temporal":
    test_path = f"/NAS/yjt/Abstract_Thought/dataset/LongLaMP/{task_name}/dataset_test"
else:
    test_path = f"/NAS/yjt/Abstract_Thought/dataset/LaMP/{task_name}/eval_dataset"
    
if task_name == "citation_identification":
    max_tokens = 10
    
elif task_name == "news_headline_generation":
    max_tokens = 48
    
else:
    max_tokens = 2048
    
personal_dataset = load_from_disk(f"{test_path}")
llm_tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
personal_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
personal_model.resize_token_embeddings(len(llm_tokenizer))
personal_model.eval()
personal_model.generation_config = GenerationConfig(
    temperature=0.8,           
    top_p=0.95,                  
    repetition_penalty=1.2,   
    max_new_tokens=max_tokens,
    eos_token_id=llm_tokenizer.eos_token_id,
    pad_token_id=llm_tokenizer.pad_token_id,
)

references = personal_dataset["target"]
personal_dataset.remove_columns(["source", "target"])
predictions = []
dataloader = DataLoader(
    personal_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=default_data_collator
)
for sample in tqdm(dataloader, desc="Generating data"):
    batch = {k: v.to(personal_model.device) for k, v in sample.items()}
   
    if task_name == "citation_identification":
        with torch.no_grad():
            generated_ids = personal_model.generate(
                input_ids=batch["input_ids"],
                # attention_mask=batch["attention_mask"]
                max_new_tokens=max_tokens,
            )
        generated_ids = generated_ids[:, len(batch["input_ids"][0]):]
        texts = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(texts)
    else:
        with torch.no_grad():
            generated_ids = personal_model.generate(
                input_ids=batch["input_ids"],
                # attention_mask=batch["attention_mask"]
            )
        generated_ids = generated_ids[:, len(batch["input_ids"][0]):]
        texts = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(texts)

with open(f"./predictions/{base_model}_{task_name}_{date_string}.json", 'w') as f:
    json.dump(predictions, f, ensure_ascii=False, indent = 4)
file_path = f"./res/result_{task_name}_{base_model}.csv"
if task_name == "citation_identification":
    result_accuracy = accuracy_metric.compute(predictions=predictions,
                                references=references)
    result = {
        "accuracy": result_accuracy["accuracy"]
    }
    print(result)
    write_to_csv(f"{model_dir}-{task_name}-{date_string}", "accuracy"     , result['accuracy']     , file_path=file_path)
else:
    result_bleu = bleu_metric.compute(predictions=predictions,
                                    references=references)
    result_rouge = rouge_metric.compute(predictions=predictions,
                                        references=references)
    result_meteor = meteor_metric.compute(predictions=predictions,
                                        references=references)
    result_bertscore = bertscore_metric.compute(predictions=predictions, references=references, lang="en")

    result = {
        "rouge-1": result_rouge["rouge1"],
        "rouge-L": result_rouge["rougeL"],
        "meteor": result_meteor['meteor'],
        "bleu": result_bleu["score"],
        "bertscore-f1": np.mean(result_bertscore["f1"])
    }
    print(result)


    write_to_csv(f"{model_dir}-{task_name}-{date_string}", "rouge-1"     , result['rouge-1']     , file_path=file_path)
    write_to_csv(f"{model_dir}-{task_name}-{date_string}", "rouge-L"     , result['rouge-L']     , file_path=file_path)
    write_to_csv(f"{model_dir}-{task_name}-{date_string}", "meteor"      , result['meteor']      , file_path=file_path)
    write_to_csv(f"{model_dir}-{task_name}-{date_string}", "bleu"        , result['bleu']        , file_path=file_path)
    write_to_csv(f"{model_dir}-{task_name}-{date_string}", "bertscore-f1", result['bertscore-f1'], file_path=file_path)

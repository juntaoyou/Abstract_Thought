import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,6'
os.environ["WANDB_DISABLED"]="true"
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from datasets import disable_caching
from transformers import EarlyStoppingCallback, AutoConfig
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import transformers
from datasets import load_dataset, concatenate_datasets, load_from_disk
from torch.optim.lr_scheduler import LambdaLR
"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""
from transformers import set_seed, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import itertools
from accelerate import Accelerator
from transformers_custom.trainer import Trainer
import datetime
nowtime = datetime.datetime.now()
date_string = nowtime.strftime("%Y-%m-%d")
set_seed(42)

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 2,
    num_epochs: int = 5,
    learning_rate: float = 5e-5,
    cutoff_len: int = 1024,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "First-Language-Switching",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    output_dir: str = None,
    save_steps: int = 0.1,
    mixed_dataset: bool = False,
    cache_dir: str = "",
    
    local_rank: int = 0,
    deepspeed: str ="./deepspeed/ds_z1_config.json",
    K: int = 0,
    neuron_path: str = "",
    task_name : str = "product_review"
):
    disable_caching()
    # os.environ['WANDB_PROJECT'] = wandb_project
    model_name = base_model.split("/")[-1]
    if mixed_dataset:
        assert "shared" in neuron_path, "Mixed dataset must use shared neurons"
        neuron_type = "shared_mixed"
    elif neuron_path == "":
        neuron_type = "full"
    elif "random" in neuron_path:
        neuron_type = "random"
    elif "shared" in neuron_path:
        neuron_type = "shared"
    elif "exclusive" in neuron_path:
        neuron_type = "exclusive"
    else:
        raise ValueError(f"Invalid neuron path: {neuron_path}")
    if not output_dir:
        output_dir = f"{task_name}-{neuron_type}-neurons-{model_name}-lr{learning_rate}"
    # output_dir = f"./{output_dir}/{wandb_run_name}"
    # print(train_file)
    
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        pass

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        # device_map=device_map,
    )
    # it is recommended to use eager attention implementation for training gemma-3-1b-pt
        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True,  attn_implementation='eager')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    def process_data_train(data_point):
        prompt = data_point["text"]
        full_text = prompt + tokenizer.eos_token
        tokenized = tokenizer(full_text, truncation=True, max_length=cutoff_len, padding=False)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy(),
        }
    
    def process_data_eval(data_point):
        prompt = data_point["text"]
        full_text = prompt + tokenizer.eos_token
        tokenized = tokenizer(full_text, truncation=True, max_length=cutoff_len, padding=False)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy(),
        }


    # no cache    
    if mixed_dataset:
        train_dataset = load_dataset("parquet", data_files="./dataset/train/mixed.parquet", cache_dir=cache_dir)
    else:
        train_dataset = load_from_disk(f"/NAS/yjt/Abstract_Thought/dataset/LaMP/{task_name}/dataset_train")
    # train_dataset.cleanup_cache_files()
    # valid_dataset = load_from_disk(f"/NAS/yjt/Abstract_Thought/dataset/LaMP/{task_name}/dataset_valid")
    # oscar_dataset = load_dataset("text", data_files=f"./dataset/val/{language}.txt", split="train")
    # if not mixed_dataset:
    #     train_data = train_dataset["train"].shuffle(seed=42).map(process_data_train, load_from_cache_file=False, num_proc=32, cache_file_name=f"{cache_dir}/map.arrow").select(range(100000))
    # else:
    #     train_data = train_dataset.map(process_data_train, load_from_cache_file=False, num_proc=32, cache_file_name=f"{cache_dir}/map.arrow")
    # train_data = train_dataset["train"].shuffle(seed=42).map(process_data_train, load_from_cache_file=False, num_proc=32).select(range(100000))
    # val_data = oscar_dataset.shuffle(seed=42).map(process_data_eval, load_from_cache_file=False, num_proc=32, cache_file_name=f"{cache_dir}/map.arrow").select(range(1000))
    # val_data = load_val_data(language, tokenizer, cutoff_len=cutoff_len)
        
    print("LOAD DATA FINISHED")    

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    def read_neuron(path, top_k = -1):
        with open(path, 'r') as f:
            data = json.load(f)
        for group in data:
            entry = data[group]
            data[group] = {key: set(value) if isinstance(value, list) else value for key, value in entry.items()}
            if top_k > 0:
                #data[group] = random.sample(data[group], min(top_k, len(data[group])))
                data[group] = itertools.islice(data[group], min(top_k, len(data[group])))

        return data
    
    if neuron_path != "":
        activate_neuron = read_neuron(neuron_path, top_k=-1)
    else:
        activate_neuron = None

    logging_dir = f'./logs/{task_name}_{date_string}_shared'
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)
    training_args = transformers.TrainingArguments(
                deepspeed=deepspeed,
                # run_name=wandb_run_name,
                output_dir=f"./{output_dir}",
                per_device_train_batch_size=micro_batch_size,
                # per_device_eval_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_ratio=0.03,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                bf16=True,
                logging_dir=logging_dir, 
                logging_steps=10,
                optim="adamw_torch",
                # eval_strategy="epoch",
                save_strategy="epoch",
                # save_steps=save_steps,
                # output_dir=output_dir,
                # save_total_limit=10,
                # load_best_model_at_end=True,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="tensorboard",
                # save_only_model=True,
                tp_size = 1
            )
    training_args.activate_neuron = activate_neuron 
    trainer = Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=train_dataset,
        # eval_dataset=valid_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=1)],
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False
    # trainer.evaluate()
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    with open(f"./{output_dir}/log_history.json", 'w') as f:
        json.dump(trainer.state.log_history, f, indent = 4)
        
if __name__ == "__main__":
    fire.Fire(train)

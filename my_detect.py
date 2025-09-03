import random, os, json, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_from_disk
from transformers_custom.modeling_gemma3 import Gemma3ForConditionalGeneration
from transformers_custom.modeling_llama import LlamaForCausalLMDetect
from transformers_custom.modeling_gemma2 import Gemma2ForCausalLM
from transformers_custom.modeling_qwen2 import Qwen2ForCausalLM
from transformers_custom.modeling_qwen3 import Qwen3ForCausalLM
from transformers_custom.modeling_gemma import GemmaForCausalLM
import argparse
from transformers import AutoConfig
import pandas as pd
import torch
from accelerate.utils import gather_object
from accelerate import Accelerator
from transformers import BitsAndBytesConfig
from peft import LoraConfig
# accelerator = Accelerator()
# accelerator.wait_for_everyone()

def save_neuron(activate_neurons, path):
    """ 将 neuron 的位置写入文件 """
    for group in activate_neurons:
        entry = activate_neurons[group]
        activate_neurons[group] = {key: list(value) if isinstance(value, set) else value for key, value in entry.items()}
    with open(path, 'w') as f:
        json.dump(activate_neurons, f)

def load_lines_from_dataset(task, args):
    """ 加载数据集 """
    if task == "detect":
        # 检测neuron 数据集
        file_path = os.path.join(args.corpus_path, f"{args.task_name}/dataset_detect_{args.detect_length}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Detect data file not found: {file_path}")
        # with open(file_path, "r", encoding="utf-8") as f:
        #     return [line.strip() for line in f if line.strip()]
        detect_data = load_from_disk(file_path)
        return detect_data

def detect_key_neurons(model, tokenizer,
                       atten_ratio=0.1, ffn_ratio=0.1, test_size=-1, candidate_layers=[], 
                       detection_path="./test_data/oscar", output_path="./output",
                       suffix = "", model_name = "", sample_size=-1, task="detect", args=None) -> dict:
    """Detects neurons key to the language *lang* and writes to ../output/model_lang_neuron.txt 

    Args:
        model (AutoModelForCausalLM): loaded hf model.
        tokenizer (AutoTokenizer): loaded hf tokenizer.
        lang (str): one of ['english', 'chinese', 'french', 'russian']
        test_size (int, optional): number of entries used when detecting.
        candidate_layers (list, optional): list of layers to examine.
    """
    if not len(candidate_layers):
        candidate_layers = range(model.config.num_hidden_layers)
    
    # with open(detection_path + lang + '.txt', 'r') as file:
    #     lines = file.readlines()
    # lines = [line.strip() for line in lines]
    lines = load_lines_from_dataset(task, args)
    # if sample_size > 0:
    #     if sample_size < len(lines):
    #         lines = random.sample(lines, sample_size)
    #lines = lines[:test_size] #Because using the same corpus for detection and training now, separating them.

    activate_key_sets = {
        "fwd_up" : [],
        "fwd_down" : [],
        "attn_q" : [],
        "attn_k" : [],
        "attn_v" : [],
        "attn_o" : []
    }
    error_count = 0

    intermediate_layers_decode = {}

    print("Detection corpus size: ", len(lines))
    count = 0
    # with accelerator.split_between_processes(lines) as prompts:
    for prompt in tqdm(lines):
        #hidden, answer, activate, o_layers = detection_prompting(model, tokenizer, prompt, candidate_layers)
        try:
            # 根据 prompt 检测 neuron
            hidden, answer, activate, o_layers = detection_prompting(model, tokenizer, prompt, 
                                                                    candidate_layers, atten_ratio=atten_ratio, ffn_ratio=ffn_ratio)
            for key in activate.keys():
                activate_key_sets[key].append(activate[key])
            count += 1
            intermediate_layers_decode[count] = hidden
        except Exception as e:
            error_count += 1
            count += 1
            # Handle the OutOfMemoryError here
            print(error_count)
            print(e)

    print("Detection query complete; error: ", error_count)
    # print(activate_key_sets)
    for group in activate_key_sets.keys():
        entries = activate_key_sets[group]
        common_layers = {}
        for layer in entries[0].keys():
            if all(layer in d for d in entries):
                arrays = [d[layer] for d in entries]
                common_elements = set.intersection(*map(set, arrays))
                common_elements = {int(x) for x in common_elements}

                common_layers[layer] = common_elements
        activate_key_sets[group] = common_layers
        print(f"{group} integrated and logged")
        '''
        DETECTION SET TO LAYER

        from collections import Counter
        threshold = test_size // 100
        
        common_layers = {}
        for layer in entries[0].keys():
            neuron_counter = Counter(neuron for d in entries for neuron in d[layer][0])
            frequent_neurons = [neuron for neuron, count in neuron_counter.items() if count > threshold]
            common_layers[layer] = set([int(x) for x in frequent_neurons])

        activate_key_sets[group] = common_layers'''
        
        
        #final structure of important neurons: {"param_set": {"layer1": [neuron1, neuron2, ...], ...}, ...}
    # file_name = f"{model.name_or_path.split('/')[-1]}_{lang}_atten{atten_ratio}_ffn{ffn_ratio}.json"
    # file_path = output_path + file_name
    # save_neuron(activate_key_sets, file_path)
    # save intermediate_layers_decode

    if "huggingface" in model_name:
        file_name_prefix = model_name.split('/')[-1]
    elif "llama" or "gemma" or "qwen" in model_name.split('/')[-2].lower():
        file_name_prefix = model_name.split('/')[-2]
    elif "llama" or "gemma" or "qwen" in model_name.split('/')[-1].lower():
        file_name_prefix = model_name.split('/')[-1]
    else:
        raise ValueError(f"Model {model_name} not supported")

    # with open('./intermediate_decode_train_data_detect/' + f"{file_name_prefix}_{train_on_lang}_{lang}_atten{atten_ratio}_ffn{ffn_ratio}.json", 'w') as f:
    #     json.dump(intermediate_layers_decode, f)

    file_name = f"{file_name_prefix}_{task}_{args.task_name}_atten{atten_ratio}_ffn{ffn_ratio}.json"
    file_path = output_path + file_name
    save_neuron(activate_key_sets, file_path)

    # results = gather_object(activate_key_sets)
    # if accelerator.is_main_process:
    #     return results
    return activate_key_sets
        

def detection_prompting(model, tokenizer, prompt, candidate_premature_layers, atten_ratio=0.1, ffn_ratio=0.1):
    # start = 1024
    # cut_off_len = 2048
    # inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = torch.tensor(prompt["input_ids"], dtype=torch.int64).unsqueeze(0).to(model.device)
    # print(input_ids.size())
    attention_mask = torch.tensor(prompt["attention_mask"], dtype=torch.int64).unsqueeze(0).to(model.device)
    hidden_states, outputs, activate, o_layers = model.generate(**{'input_ids': input_ids,
                                                                   'attention_mask': attention_mask, 
                                                                   'max_new_tokens': 1, 
                                                                   'candidate_premature_layers':candidate_premature_layers,
                                                                   'top_ratio_atten': atten_ratio,
                                                                   'top_ratio_ffn': ffn_ratio})

    hidden_embed = {}
    # for i, early_exit_layer in enumerate(candidate_premature_layers):
        # hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][0])
    for i, early_exit_layer in enumerate(candidate_premature_layers):
        hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][0])
        # hidden_embed[early_exit_layer] = [
        #     [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in hidden_states[early_exit_layer][0][pos]]
        #     for pos in range(hidden_states[early_exit_layer].shape[1])
        # ]

    answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return hidden_embed, answer, activate, o_layers


# lang_set = ["english"]

def detection_all(model_name, atten_ratio=0.1, ffn_ratio=0.1, test_size=-1, 
                  detection_path="./corpus_all", output_path="./output", 
                  suffix="", sample_size=1000, task="detect", args=None):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    if "gemma-3" in model_name.lower():
        # model = Gemma3ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
        model = Gemma3ForConditionalGeneration.from_pretrained(model_name, config=config, device_map="auto")
        model = model.language_model
    elif "llama" in model_name.lower():
        model = LlamaForCausalLMDetect.from_pretrained(model_name, config=config, device_map="auto")
    elif "gemma-2" in model_name.lower():
        model = Gemma2ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "qwen3" in model_name.lower():
        model = Qwen3ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "qwen" in model_name.lower():
        model = Qwen2ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "gemma" in model_name.lower():
        model = GemmaForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # model = get_peft_model(model, peft_config=peft_config)
    # model.gradient_checkpointing_enable()
    # model = prepare_model_for_kbit_training(model)
    # model.train()
    print("Detecting neurons for", args.task_name)
    neurons = detect_key_neurons(model, tokenizer, atten_ratio=atten_ratio, ffn_ratio=ffn_ratio, test_size=test_size, detection_path=detection_path, output_path=output_path, suffix=suffix, model_name=model_name, sample_size=sample_size, task=task, args=args)
    print(args.task_name, "complete", len(neurons["attn_q"].keys()), len(neurons["attn_q"][0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--corpus_path", type=str, default='/NAS/yjt/Abstract_Thought/dataset/LongLaMP')
    parser.add_argument("--corpus_size", type=int, default=-1)
    parser.add_argument("--base", type=str, default="Qwen3-1.7B")
    parser.add_argument("--output_path", type=str, default="./neuron_train_data_detect/")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--task_name", type=str, default="product_review_temporal")
    parser.add_argument("--atten_ratio", type=float, default=0.1)
    parser.add_argument("--ffn_ratio", type=float, default=0.1)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--task", type=str, default="detect")
    parser.add_argument("--detect_length", type=int,default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
        
    detection_all(args.base,
                  args.atten_ratio,
                  args.ffn_ratio,
                  args.corpus_size,
                  args.corpus_path,
                  args.output_path,
                  args.suffix,
                  args.sample_size,
                  args.task,
                  args)


    
    
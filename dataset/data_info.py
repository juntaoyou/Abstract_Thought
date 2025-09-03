from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import os
import datetime
from tqdm import trange
nowtime = datetime.datetime.now()
# AutoTokenizer.from_pretrained("/")
# date_string = nowtime.strftime("%Y-%m-%d %H:%M:%S")
# print(date_string)
# symbol = ""
# seq = ["a", "b", "c"]
# print(symbol.join(seq))
# data = load_dataset("/NAS/yjt/Mydatasets/LongLaMP/product_review_temporal",cache_dir="/NAS/yjt/HuggingfaceCache/")['train']
# data = load_from_disk("/NAS/yjt/Abstract_Thought/dataset/LaMP/citation_identification/dataset_train")
# print(data[0]['out_str'])
# data = data.train_test_split(test_size=0.1)
# data['train'].save_to_disk("/NAS/yjt/Abstract_Thought/dataset/LaMP/news_headline_generation/dataset_train")
# data['valid'].save_to_disk("/NAS/yjt/Abstract_Thought/dataset/LaMP/news_headline_generation/dataset_valid")
# # print(data[0]['profile'][0])
# print(data[0]['inp_str'])
# task_name = "news_headline_generation" 

# train_data = load_dataset("json", data_files=f"/NAS/yjt/Mydatasets/LaMP/{task_name}/train_questions.json",cache_dir="/NAS/yjt/HuggingfaceCache/")['train']
# train_output = load_dataset("json", data_files=f"/NAS/yjt/Mydatasets/LaMP/{task_name}/train_outputs.json",cache_dir="/NAS/yjt/HuggingfaceCache/")['train']['golds']
tokenizer = AutoTokenizer.from_pretrained("/NAS/yjt/models/Qwen3-1.7B")
print(tokenizer.encode("1"))
print(tokenizer.encode("0"))
# maxv = 0
# test = [train_output[i]['output'] for i in range(len(train_output))]
# for i in trange(len(train_output)):
    # test = (
    #                 f"You are a personalized review writing assistant. "
    #                 f"Given the new product and review information, generate a personalized product review for the user.\n\n"
    #                 f"# Current input: \n"
    #                 f"{train_data[i]['input']}\n\n"
    #                 f"# Your Output: \n"
    #         )
# prompt = tokenizer(test, return_tensors="pt", padding=True)["input_ids"]
# print(prompt.shape)
# maxv = max(maxv, int(prompt.shape[1]))
# print(maxv)
# print(len(prompt["input_ids"]))
# # train_data = data['train']
# print(data)
# print("Profile: ", train_data[0]['profile'][0].keys())
# print("Input: ", train_data[0]['input'])
# print("Output: ", train_data[0]['output'])

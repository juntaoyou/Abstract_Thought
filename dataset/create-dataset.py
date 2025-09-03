from transformers import AutoTokenizer, set_seed
from datasets import load_dataset, concatenate_datasets,load_from_disk
from personal_dataset import convert_to_dataset, LaMPDataset, LongLaMPDataset
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="news_headline_generation")
set_seed(42)
import os
# if not os.path.exists("/NAS/yjt/Abstract_Thought/dataset/LaMP"):
#     os.makedirs("/NAS/yjt/Abstract_Thought/dataset/LaMP")
tokenizer = AutoTokenizer.from_pretrained("/NAS/yjt/models/Qwen3-1.7B")
args = parser.parse_args()
task_name = args.task_name
def create_LaMP_dataset(task_name: str, detect: bool=True, detect_length: int=1000):
    if task_name == "citation_identification": 
        target_max_length = 3
    elif task_name == "news_headline_generation":
        target_max_length = 48
    train_data = load_dataset("json", data_files=f"/NAS/yjt/Mydatasets/LaMP/{task_name}/train_questions.json",cache_dir="/NAS/yjt/HuggingfaceCache/")['train']
    train_output = load_dataset("json", data_files=f"/NAS/yjt/Mydatasets/LaMP/{task_name}/train_outputs.json",cache_dir="/NAS/yjt/HuggingfaceCache/")['train']['golds']
    train_dataset = LaMPDataset(
        train_data,
        train_output,
        tokenizer, 
        task_name = task_name,
        max_length=512,
        target_max_length=target_max_length,
        detect=False,
    )

    train_dataset = convert_to_dataset(train_dataset)
    train_dataset.save_to_disk(f"/NAS/yjt/Abstract_Thought/dataset/LaMP/{task_name}/dataset_train")
    
    test_data = load_dataset("json", data_files=f"/NAS/yjt/Mydatasets/LaMP/{task_name}/dev_questions.json",cache_dir="/NAS/yjt/HuggingfaceCache/")['train']
    test_output = load_dataset("json", data_files=f"/NAS/yjt/Mydatasets/LaMP/{task_name}/dev_outputs.json",cache_dir="/NAS/yjt/HuggingfaceCache/")['train']['golds'][0]
    test_dataset = LaMPDataset(
        test_data, 
        test_output,
        tokenizer, 
        task_name = task_name,
        max_length=512,
        target_max_length=target_max_length,
        detect=False,
        training=False
    )

    test_dataset = convert_to_dataset(test_dataset)
    test_dataset.save_to_disk(f"/NAS/yjt/Abstract_Thought/dataset/LaMP/{task_name}/dataset_test")
    if detect:
        # train_dataset = load_from_disk(f"/NAS/yjt/Abstract_Thought/dataset/LaMP/{task_name}/dataset_train")
        detect_dataset = train_dataset.select(list(range(1000)))
        detect_dataset.save_to_disk(f"/NAS/yjt/Abstract_Thought/dataset/LaMP/{task_name}/dataset_detect_{detect_length}")
    
    print("Create datasets successfully!!!")

def create_LongLaMP_dataset(task_name: str, detect: bool=True, detect_length: int=1000):
    data = load_dataset(f"/NAS/yjt/Mydatasets/LongLaMP/{task_name}",cache_dir="/NAS/yjt/HuggingfaceCache/")
    train_data, valid_data, test_data = data['train'],data['validation'],data['test']
    train_dataset = LongLaMPDataset(
        train_data,
        tokenizer, 
        task_name = task_name,
        detect=False,
    )

    train_dataset = convert_to_dataset(train_dataset)
    train_dataset.save_to_disk(f"/NAS/yjt/Abstract_Thought/dataset/LongLaMP/{task_name}/dataset_train")
    
    valid_dataset = LongLaMPDataset(
        valid_data,
        tokenizer, 
        task_name = task_name,
        detect=False,
    )

    valid_dataset = convert_to_dataset(valid_dataset)
    valid_dataset.save_to_disk(f"/NAS/yjt/Abstract_Thought/dataset/LongLaMP/{task_name}/dataset_valid")
    
    test_dataset = LongLaMPDataset(
        test_data,
        tokenizer, 
        task_name = task_name,
        detect=False,
        training=False
    )

    test_dataset = convert_to_dataset(test_dataset)
    test_dataset.save_to_disk(f"/NAS/yjt/Abstract_Thought/dataset/LongLaMP/{task_name}/dataset_test")
    
    if detect:
        detect_dataset = LongLaMPDataset(
            train_data,
            tokenizer, 
            task_name = task_name,
            detect=True,
        )
        detect_dataset.save_to_disk(f"/NAS/yjt/Abstract_Thought/dataset/LongLaMP/{task_name}/dataset_detect_{detect_length}")
    
    print("Create datasets successfully!!!")
if task_name == "citation_identification":
    create_LaMP_dataset(task_name)
    
elif task_name == "news_headline_generation":
    create_LaMP_dataset(task_name)
    


elif task_name == "product_review_temporal":
    create_LongLaMP_dataset(task_name)

else:
    raise ValueError("Not a Valid Task Name!!!")






# train_dataset = LongLaMPDataset(
#     train_data,
#     tokenizer, 
#     task_name = "product_review_temporal",
#     max_length=1024,
#     detect=True,
# )[:1000]

# train_dataset = convert_to_dataset(train_dataset)
# # train_dataset.save_to_disk(f"/NAS/yjt/Abstract_Thought/dataset/LaMP/{task_name}/dataset_train")
# train_dataset.save_to_disk(f"/NAS/yjt/Abstract_Thought/dataset/LongLaMP/product_review_temporal/dataset_detect_1000")

# # test_data = load_dataset("json", data_files="/NAS/yjt/Mydatasets/LaMP/news_headline_generation/dev_questions.json",cache_dir="/NAS/yjt/HuggingfaceCache/")['train']
# # test_output = load_dataset("json", data_files="/NAS/yjt/Mydatasets/LaMP/news_headline_generation/dev_outputs.json",cache_dir="/NAS/yjt/HuggingfaceCache/")['train']['golds']
# # print(test_output[0])
# valid_dataset = LongLaMPDataset(
#     valid_data,
#     tokenizer, 
#     task_name = "product_review_temporal",
#     training=False,
#     detect = False
# )
# valid_dataset = convert_to_dataset(valid_dataset)
# valid_dataset.save_to_disk(f"/NAS/yjt/Abstract_Thought/dataset/LongLaMP/product_review_temporal/dataset_valid")
# test_data = load_dataset("json", data_files="/NAS/yjt/Mydatasets/LaMP/news_headline_generation/dev_questions.json",cache_dir="/NAS/yjt/HuggingfaceCache/")['train']
# test_output = load_dataset("json", data_files="/NAS/yjt/Mydatasets/LaMP/news_headline_generation/dev_outputs.json",cache_dir="/NAS/yjt/HuggingfaceCache/")['train']['golds']
# test_dataset = LongLaMPDataset(
#     test_data, 
#     tokenizer, 
#     task_name = "product_review_temporal",
#     # training=False,
#     detect=False,
# )

# test_dataset = convert_to_dataset(test_dataset)
# test_dataset.save_to_disk(f"/NAS/yjt/Abstract_Thought/dataset/LongLaMP/product_review_temporal/dataset_test2")

# train_dataset = load_from_disk("/NAS/yjt/Abstract_Thought/dataset/LongLaMP/news_headline_generation/dataset_train")
# detect_dataset = train_dataset.shuffle(seed = 0).select(list(range(500)))
# detect_dataset = convert_to_dataset(detect_dataset)
# detect_dataset.save_to_disk("/NAS/yjt/Abstract_Thought/dataset/LaMP/news_headline_generation/dataset_detect")
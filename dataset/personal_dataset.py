from os import error
import torch
import datasets
import numpy as np

from tqdm import tqdm
import sys
sys.path.append("/NAS/yjt/Abstract_Thought")
from utils.templates import Qwen2PromptTemplate
# sys.path.append("/NAS/yjt/demo-rag/data/LongLaMP-Benchmark/longLaMP")
# from prompts.prompts import *

# sys.path.append("/NAS/yjt/demo-rag/data/LLM-TRSR/preprocess")
# from amazon.summary_generation import *

class LongLaMPDataset(torch.utils.data.Dataset):
    def __init__(self, 
                main_dataset,
                llm_tokenizer,
                task_name="product_review",
                max_length=2048,
                target_max_length=2048,
                max_his_len=8,
                training=True,
                detect = False
    ):
        self.main_dataset = main_dataset
        self.max_his_len = max_his_len
        self.llm_tokenizer = llm_tokenizer
        self.total_len = len(self.main_dataset)
        self.task_name = task_name
        self.actual_len = 0
        self.max_length = max_length
        self.target_max_length = target_max_length
        self.training = training
        self.processed_data = []
        self.detect = detect
        
        if self.task_name == "product_review_temporal":
            self.create_review_generation_dataset()
        elif self.task_name == "abstract_generation_temporal":
            pass
        elif self.task_name == "topic_writing_temporal":
            pass
        else:
            raise ValueError("Not a valid task name!!!")
            
    def create_review_generation_dataset(self):
        system_prompt = (f"Given the user profile and a new product information, "
                         f"generate a personalized product review for the product. "
                         f"Just output the generated review without explanation. \n\n"
                         )
        self.pt = Qwen2PromptTemplate(system_prompt)
        for idx in tqdm(range(self.total_len), desc=f"Pre-Processing data"):
            profile = self.main_dataset[idx]["profile"]
            cur_max_his_len = min(self.max_his_len, len(profile))
            profile = profile[-cur_max_his_len:]
            input_content = self.main_dataset[idx]["input"]
            output_content = self.main_dataset[idx]["output"]
            pro = self.llm_tokenizer(output_content, return_tensors="pt")
            if pro["input_ids"].shape[1] > self.target_max_length: continue
            if not self.detect:
                tmp_inp_str = (
                    f"# Your task: \n"
                    f"You are a personalized review writing assistant "
                    f"whose task is to generate a personalized product review for a user "
                    f"based on an overall rating of a product, the product description, a summary of the review text and the user profile. "
                    f"The user profile includes the user's recent reviews, each containing an item description, an overall rating, a summary of the review text and the full review text of the product. "
                    f"The profile is arranged in chronological order from past to present. "
                    f"The expected output should precisely reflect the user's writing style and his/her preferences about the product.\n\n"
                    f"# User Profile: \n"
                    f"# Current input: \n"
                    f"{input_content}\n\n"
                    f"Given the user profile and new product and review information above, now generate a personalized product review for the user. "
                    f"Just output the generated review without explanation. \n\n"
                    f"# Your Output: \n"
                )
            else:
                tmp_inp_str = (
                    f"You are a personalized review writing assistant. "
                    f"Given the new product and review information, generate a personalized product review for the user.\n\n"
                    f"# Current input: \n"
                    f"{input_content}\n\n"
                    f"# Your Output: \n"
                )
            # tmp_inp_str = connect_symbol.join(tmp_inp_str)               
            tmp_inp_str = self.pt.build_prompt(tmp_inp_str)
            if self.detect:
                data = {
                'inp_str': tmp_inp_str,
                }
                inputs = self.llm_tokenizer(tmp_inp_str,
                                        max_length=self.max_length,
                                        truncation=True,
                                        add_special_tokens=False)
                inputs_id = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                if len(inputs_id) < self.max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (self.max_length - len(inputs_id)) + inputs_id
                    attention_mask = [0] * (self.max_length - len(attention_mask)) + attention_mask
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
                data['attention_mask'] = np.array(attention_mask, dtype=np.int64)
                self.processed_data.append(data)
                continue
                
            tmp_ids = self.llm_tokenizer(tmp_inp_str, add_special_tokens=False)['input_ids']
            tmp_len = len(tmp_ids)
            avail_len = self.max_length - tmp_len
            if avail_len < 0:
                raise ValueError("Not a valid text length!!!")
            past_reviews = ""
            for tmp_prof_len in range(cur_max_his_len, -1, -1):
                past_reviews = "".join([
                    f"## [Review {i+1}]:\n"
                    f"### [Overall Rating]:{profile[i]['overall']}\n"
                    f"### [Product Description]: {profile[i]['description']}\n"
                    f"### [Review Summary]: {profile[i]['summary']}\n"
                    f"### [Review Text]: {profile[i]['reviewText']}\n\n"
                    for i in range(tmp_prof_len)
                ])
                past_reviews = f"\n{past_reviews}\n"
                past_ids = self.llm_tokenizer(past_reviews, add_special_tokens=False)['input_ids']
                if len(past_ids) <= avail_len:
                    break
            if tmp_prof_len == 0:
                continue
            self.actual_len += tmp_prof_len
            inp_str = (
                    f"# Your task: \n"
                    f"You are a personalized review writing assistant "
                    f"whose task is to generate a personalized product review for a user "
                    f"based on an overall rating of a product, the product description, a summary of the review text and the user profile. "
                    f"The user profile includes the user's recent reviews, each containing an item description, an overall rating, a summary of the review text and the full review text of the product. "
                    f"The profile is arranged in chronological order from past to present. "
                    f"The expected output should precisely reflect the user's writing style and his/her preferences about the product.\n\n"
                    f"# User Profile: \n"
                    f"\n{past_reviews}\n"
                    f"# Current input: \n"
                    f"{input_content}\n\n"
                    f"Given the user profile and new product and review information above, now generate a personalized review for the product. "
                    f"Just output the generated review without explanation. \n\n"
                    f"# Your Output: \n"   
                )
            # inp_str = connect_symbol.join(inp_str)
            inp_str = self.pt.build_prompt(inp_str)
            total_max_length = self.max_length + self.target_max_length + 1
            out_str = output_content
            inputs = self.llm_tokenizer(inp_str,
                                        max_length=self.max_length,
                                        truncation=True,
                                        add_special_tokens=False)
            targets = self.llm_tokenizer(out_str,
                                        max_length=self.max_length, 
                                        truncation=True,
                                        add_special_tokens=False)
            data = {
                'inp_str': inp_str,
                'out_str': out_str,
            }
            if self.training:
                inputs_id = inputs['input_ids'] + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                attention_mask = inputs['attention_mask'] + targets['attention_mask'] + [1]
                labels = [-100] * len(inputs['input_ids']) + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                if len(inputs_id) < total_max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (total_max_length - len(inputs_id)) + inputs_id
                    attention_mask = [0] * (total_max_length - len(attention_mask)) + attention_mask
                    labels = [-100] * (total_max_length - len(labels)) + labels
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
                data['attention_mask'] = np.array(attention_mask, dtype=np.int64)
                data['labels'] = np.array(labels, dtype=np.int64)
            else:
                inputs_id = inputs['input_ids']
                # attention_mask = inputs['attention_mask']
                if len(inputs_id) < self.max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (self.max_length - len(inputs_id)) + inputs_id
                    # attention_mask = [0] * (self.max_length - len(attention_mask)) + attention_mask
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
                # data['attention_mask'] = np.array(attention_mask, dtype=np.int64)
            self.processed_data.append(data)
            

    def __len__(self):
        return self.actual_len

    def get_output(self, idx):
        return self.main_dataset[idx]["output"]
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

    def get_avg_profile_len(self):
        return self.cnt / self.actual_len
       
class LaMPDataset(torch.utils.data.Dataset):
    def __init__(self, 
                input_dataset,
                llm_tokenizer,
                task_name="product_review",
                max_length=2048,
                target_max_length = 32,
                max_his_len=8,
                training=True,
                detect = True
    ):
        self.input_dataset = input_dataset
        self.max_his_len = max_his_len
        self.llm_tokenizer = llm_tokenizer
        self.total_len = len(self.input_dataset)
        self.task_name = task_name
        self.actual_len = 0
        self.max_length = max_length
        self.target_max_length = target_max_length
        self.training = training
        
        self.processed_data = []
        self.detect = detect
        
        if self.task_name == "news_headline_generation":
            self.create_news_headline_generation_dataset()
        elif self.task_name == "citation_identification":
            self.create_citation_identification_dataset()
        elif self.task_name == "topic_writing":
            pass
        else:
            raise ValueError("Not a valid task name!!!")
            
    def create_news_headline_generation_dataset(self):  
        system_prompt = (f"Given the user profile and a news text, "
                         f"generate a personalized news headline for the text. "
                         f"Just output the generated news headline without explanation. \n\n"
                         )
        self.pt = Qwen2PromptTemplate(system_prompt)
        
            
        
        
        
    def create_citation_identification_dataset(self):  
        system_prompt = (f"Given the user profile and a paper title, "
                        f"choose which reference is related. "
                        f"Just output your choice without explanation. \n\n"
                        )
        self.pt = Qwen2PromptTemplate(system_prompt)
        
            

    def __len__(self):
        return self.actual_len

    def get_output(self, idx):
        return self.output_dataset[idx]["output"]
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

    def get_avg_profile_len(self):
        return self.cnt / self.actual_len
    
def convert_to_dataset(dataset):
    def gen():
        for data in dataset:
            yield data
    return datasets.Dataset.from_generator(gen)

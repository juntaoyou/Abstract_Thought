from vllm import LLM, SamplingParams
model_name = "../models/Qwen2.5-72B-Instruct-AWQ"
model = LLM(
    model_name, 
    gpu_memory_utilization=0.7,
    tensor_parallel_size=1
)

sampling_Params = SamplingParams(
    max_tokens=64,
    skip_special_tokens=True,
    temperature=0.8,
    top_p=0.95
)


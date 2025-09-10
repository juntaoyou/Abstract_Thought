# import numpy as np
# from sklearn.preprocessing import KBinsDiscretizer
# import json
# import datetime
# import matplotlib.pyplot as plt
# nowtime = datetime.datetime.now()
# date_string = nowtime.strftime("%Y-%m-%d %H:%M:%S")
# import argparse
# # parser = argparse.ArgumentParser()
# # 假设数据加载部分保持不变
# with open("./predictions/pre/coherent/scores.json", "r") as f:
#     data = json.load(f)
    
# scores = []
# for d in data:
#     r = d['responses']
#     for item in r:
#         scores.append(item['score'])
# scores = np.array(scores).reshape(-1, 1)
        
# discretizer = KBinsDiscretizer(
#             n_bins=5,
#             encode='ordinal',
#             strategy='uniform'
#         )
# discretizer.fit(scores)
        
# # 提取并处理bins
# bins = discretizer.bin_edges_[0]
# bins[0] = float('-inf')
# bins[-1] = float('inf')

# print(f"自动计算的bins: {bins}")
# category = np.digitize(scores, bins) - 1
# category = category.reshape(-1)

# fig, ax = plt.subplots(figsize=(10,8))
# n, bins_plot, patches = ax.hist(category, bins=np.arange(6)-0.5, color='skyblue', alpha=0.7, density=True)
# ax.set_title("Scores on Coherent", fontsize=30)

# # 计算每个bin的中心位置
# bin_centers = [(bins_plot[i] + bins_plot[i+1])/2 for i in range(len(bins_plot)-1)]

# # 设置横坐标刻度在每个bin的中心
# ax.set_xticks(bin_centers)
# ax.set_xticklabels(range(5), fontsize=25)  # 标签仍为0-4

# # 设置横纵坐标标签及字体大小
# ax.set_xlabel("Score", fontsize=25)
# ax.set_ylabel("Density", fontsize=25)

# # 设置纵坐标刻度字体大小
# ax.tick_params(axis='y', labelsize=25)

# plt.savefig("coherent.pdf", format="pdf")
# plt.close()
from vllm import LLM, SamplingParams 
llm = LLM(
    model="/NAS/yjt/models/Qwen2.5-0.5B",
    tensor_parallel_size=1, 
    gpu_memory_utilization=0.9,
    trust_remote_code=True
)
print(dir(llm))
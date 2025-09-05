import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import json
import datetime
import matplotlib.pyplot as plt
nowtime = datetime.datetime.now()
date_string = nowtime.strftime("%Y-%m-%d %H:%M:%S")

# 假设数据加载部分保持不变
with open("./Abstract_Thought/predictions/pre/helpfulness/scores.json", "r") as f:
    data = json.load(f)
    
scores = []
for d in data:
    r = d['responses']
    for item in r:
        scores.append(item['score'])
scores = np.array(scores).reshape(-1, 1)
        
discretizer = KBinsDiscretizer(
            n_bins=5,
            encode='ordinal',
            strategy='uniform'
        )
discretizer.fit(scores)
        
# 提取并处理bins
bins = discretizer.bin_edges_[0]
bins[0] = float('-inf')
bins[-1] = float('inf')

print(f"自动计算的bins: {bins}")
category = np.digitize(scores, bins) - 1
category = category.reshape(-1)

fig, ax = plt.subplots(figsize=(10,6))
ax.hist(category, bins=5, color='skyblue', alpha=0.7, density=True)
ax.set_title("Scores on Helpfulness", fontsize=30)

# 设置横坐标为[0,1,2,3,4]
ax.set_xticks(range(5))  # 设置刻度位置
ax.set_xticklabels(range(5), fontsize=25)  # 设置刻度标签及字体大小

# 设置横纵坐标标签及字体大小
ax.set_xlabel("Score", fontsize=25)
ax.set_ylabel("Density", fontsize=25)

# 设置纵坐标刻度字体大小
ax.tick_params(axis='y', labelsize=25)

plt.savefig("test.pdf", format="pdf")
plt.close()
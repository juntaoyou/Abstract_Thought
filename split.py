import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import json
import datetime
import matplotlib.pyplot as plt
nowtime = datetime.datetime.now()
date_string = nowtime.strftime("%Y-%m-%d %H:%M:%S")

with open("/NAS/yjt/Abstract_Thought/predictions/pre/helpfulness/scores_2025-09-04 17:41:39.json", "r") as f:
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
# 确保区间覆盖可能的极端值
bins[0] = float('-inf')
bins[-1] = float('inf')

print(f"自动计算的bins: {bins}")
category = np.digitize(scores, bins) - 1
category = category.reshape(-1)

fig, ax = plt.subplots(figsize=(10,6))
# ax.set_xlim(0,4.5)
# ax.set_ylim(0,1)
# ax.set_xticks([0.5,1.5,2.5,3.5,4.5], labels = map(str, list(range(0, 5))), fontsize=25)
# ax.set_yticklabels(map(str, list(range(0, 5))), fontsize=25)
ax.hist(category, bins=5,color='skyblue', 
            alpha=0.7,density=True)
ax.set_title("Scores on Helpfulness", fontsize=30)

ax.set_xlabel("Score", fontsize=30)
ax.set_ylabel("Density", fontsize=30)
plt.savefig("test.pdf",format="pdf")
# 确保分类结果在0-4范围内
# category = max(0, min(4, category))

# i = 0
# for d in data:
#     r = d['responses']
#     for item in r:
#         item['score2'] = category[i].astype(np.float64)
#         i += 1
        
# with open(f"/NAS/yjt/Abstract_Thought/predictions/pre/helpfulness/scores_{date_string}.json", "w") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)
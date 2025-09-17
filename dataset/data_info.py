# 

import matplotlib.pyplot as plt
import torch
import numpy as np

# data = torch.load("/NAS/yjt/Abstract_Thought/activation_mask/Qwen3-8B.single")
# # stats1 = data['over_zero'] / data['n']
# # print(len(data))
# preference_set = ["Expertise", "Informativeness", "Style"]
# keys_set = ['A', 'B']
# plt.figure(figsize=(14,10))  
# nums = [[0 for _ in range(36)] for _ in range(6)]
# for idx, d in enumerate(data):
#     for i in range(36):
#         nums[idx][i] = len(d[i])

# nums = np.array(nums)
# for i in range(6):
#     plt.plot(list(range(36)), nums[i], label = f"{preference_set[i // 2]}_{keys_set[i % 2]}", linewidth=3)
# plt.xlabel("Layer Num", fontsize=30)
# plt.ylabel("Num of Preference-Specific neurons", fontsize=30)
# plt.legend(fontsize=25)
# plt.tick_params(axis='both', labelsize=20, pad=12)
# plt.savefig("test.pdf")
# num_layers, inter_size = stats1.shape


# data = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A")
# stats2 = data['over_zero'] / data['n']

# data = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A")
# stats3 = data['over_zero'] / data['n']

# data = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness.AA")
# stats4 = data['over_zero'] / data['n']

# data1 = (stats2 - stats1) / stats1
# data1[torch.isnan(data1)] = 0

# data2 = (stats3 - stats1) / stats1
# data2[torch.isnan(data2)] = 0


# data3 = (stats3 - stats1) / stats1
# data3[torch.isnan(data3)] = 0

# k_range = [0.01] + list(map(lambda x: x * 0.05, list(range(1, 11))))
# res = []
# for k in k_range:
#     map1 = torch.zeros_like(stats1)
#     map2 = torch.zeros_like(stats1)
#     map3 = torch.zeros_like(stats1)
#     d = int(k * inter_size * num_layers)
#     values1, indices1 = torch.topk(data1.flatten(), d, dim = -1)
#     for idx in indices1:
#         map1[idx // inter_size][idx % inter_size] = 1
        
#     values2, indices2 = torch.topk(data2.flatten(), d, dim = -1)
#     for idx in indices2:
#         map2[idx // inter_size][idx % inter_size] = 1
        
#     values3, indices3 = torch.topk(data3.flatten(), d, dim = -1)
#     for idx in indices3:
#         map3[idx // inter_size][idx % inter_size] = 1
        
#     map12 = torch.logical_or(map1, map2)
#     intersection = torch.logical_and(map12, map3).sum().item()
#     union = torch.logical_or(map12, map3).sum().item()
#     res.append(intersection / union)
# plt.figure(figsize=(12,10))
# plt.plot(k_range, res, linewidth=3)
# plt.grid(True)
# plt.xlabel("Ratio of Top Value", fontsize=30)
# plt.ylabel("IoU", fontsize=30)
# plt.tick_params(axis='both', labelsize=20, pad=12)
# plt.savefig("test.pdf")
        
    

# print(y)
# x = torch.quantile(y, 0.25).item()
# print(y)


# y1 = stats2 > stats1
# y2 = stats3 > stats1
# y12 = torch.logical_or(y1, y2)
# y3 = (stats4 > stats1)
# intersection = torch.logical_and(y12, y3).sum().item()
# union = torch.logical_or(y12, y3).sum().item()
# print(intersection / union)

# p1 = y1.sum(dim = -1).cpu().tolist()
# p2 = y2.sum(dim = -1).cpu().tolist()
# p12 = y12.sum(dim = -1).cpu().tolist()
# p3 = y3.sum(dim = -1).cpu().tolist()

# plt.plot(list(range(num_layers)), p1, label="Expertise_A")
# plt.plot(list(range(num_layers)), p2, label="Informativeness_A")
# plt.plot(list(range(num_layers)), p12, label="Expertise_A or Informativeness_A")
# plt.plot(list(range(num_layers)), p3, label="Expertise_Informativeness_AA")
# # y = (stats3 - stats1) / stats1
# # y[torch.isnan(y)] = 0
# # print(y)
# # y = ((y > 0).sum(dim = -1) / inter_size).cpu().tolist()
# # print(y)
# # plt.plot(list(range(num_layers)), y)
# plt.legend()
# plt.savefig("test3.pdf")


# qs = [0.25, 0.5, 0.75]
# i = 0
# for q in qs:
#     delta = torch.quantile(stats1, q).item()
#     restrictions = stats1 <= delta
#     z1 = torch.logical_and(y1, restrictions)
#     z2 = torch.logical_and(y2, restrictions)
#     z12 = torch.logical_or(z1, z2)
#     z3 = torch.logical_and(y3, restrictions)
#     intersection = torch.logical_and(z12, z3).sum().item()
#     union = torch.logical_or(z12, z3).sum().item()
#     print(intersection / union)

#     p1 = z1.sum(dim = -1).cpu().tolist()
#     p2 = z2.sum(dim = -1).cpu().tolist()
#     p12 = z12.sum(dim = -1).cpu().tolist()
#     p3 = z3.sum(dim = -1).cpu().tolist()
    
#     fig, ax = plt.subplots(figsize=(10,10))
#     ax.plot(list(range(num_layers)), p1, label="Expertise_A")
#     ax.plot(list(range(num_layers)), p2, label="Informativeness_A")
#     ax.plot(list(range(num_layers)), p12, label="Expertise_A or Informativeness_A")
#     ax.plot(list(range(num_layers)), p3, label="Expertise_Informativeness_AA")
# # y = (stats3 - stats1) / stats1
# # y[torch.isnan(y)] = 0
# # print(y)
# # y = ((y > 0).sum(dim = -1) / inter_size).cpu().tolist()
# # print(y)
# # plt.plot(list(range(num_layers)), y)
#     ax.legend()
#     plt.savefig(f"quantile_{i + 1}.pdf")
#     i += 1
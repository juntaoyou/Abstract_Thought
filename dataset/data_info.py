# 

import matplotlib.pyplot as plt
import torch

data = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base")
stats1 = data['over_zero'] / data['n']
num_layers, inter_size = stats1.shape


data = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A")
stats2 = data['over_zero'] / data['n']

data = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A")
stats3 = data['over_zero'] / data['n']

data = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness.AA")
stats4 = data['over_zero'] / data['n']

y = (stats2 - stats1) / stats1
y[torch.isnan(y)] = 0
# print(y)
x = torch.quantile(y, 0.25).item()
plot_data = ((y > x).sum(dim = -1) / inter_size).cpu().tolist()
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
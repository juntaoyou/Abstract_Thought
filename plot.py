import matplotlib.pyplot as plt
import torch
from utils.plot import plot0, plot1, plot2, plot3,plot4, load, calc, plot7, plot8, plot1_quantile, plot2_quantile, plot8_quantile, plot5
from matplotlib import cm
import matplotlib
import numpy as np
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['patch.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# plot5()

file_paths = [
    # "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base",
    "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
    "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
    "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
    "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B",
    "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
    "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B"    
]
# data = torch.load(base_file_path)
# print(data['activated_value'] / data['n'])

base_file_path = "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base"
data = torch.load(base_file_path)

stats = data['over_zero'] / data['n']
y = torch.quantile(stats, 0.2).item()
print(y)
# positions1 = (stats > 0).float()
num_layers, inter_size = stats.shape
total_num = num_layers * inter_size
count = []
for i, path in enumerate(file_paths):
    # file_path = "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A"
    data = torch.load(path)
    stats1 = data['over_zero'] / data['n']
    # positions2 = (stats1 > 0).float()
    data_p = (stats1 - stats) / (stats + 1e-10)
    data_p = torch.where(stats >= y, data_p ,0)
    x = torch.quantile(data_p, 0.99).item()
    positions1 = (data_p >= x).float()
    if i == 0:
        positions = positions1
    else:
        positions = torch.logical_and(positions1, positions).float()
    count.append(positions1)
    # intersection = torch.logical_and(positions1, positions2).float()
    # target_neurons = positions2 - intersection
    # count.append(target_neurons.sum(-1))
fig, ax = plt.subplots(figsize=(20,15))
preference_set = ["Expertise","Informativeness","Style"]
keys_set = ['A', 'B']

indices = [[[] for _ in range(num_layers)] for _ in range(6)]
for i in range(len(count)):
    count[i] = count[i] - positions
    plt.plot(list(range(num_layers)), count[i].sum(-1) / inter_size, label=f"{preference_set[i // 2]}_{keys_set[i % 2]}",linewidth=3)
    rows_idx, cols_idx =  torch.where(count[i] > 0)
    for r, c in zip(rows_idx, cols_idx):
        indices[i][r].append(c)
    for j in range(num_layers):
        indices[i][j] = torch.tensor(indices[i][j]).long()
    # print(count[i].sum())
torch.save(indices, "./activation_mask/Qwen3-8B.All")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(list(range(num_layers)), positions.sum(dim=-1) / inter_size, label=f"Overlapping",linewidth=3)
ax.set_xlabel("Layer Num", fontsize=30)
ax.set_ylabel("Num of Preference-specific neurons", fontsize=30)
ax.legend(fontsize=25,bbox_to_anchor=(0.5, 1.1), loc="upper center",ncol=4)
# plt.tight_layout()
ax.tick_params(axis='both',labelsize=25, pad=12)
plt.savefig("ALL.pdf")

# file_paths = [
#     # "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base",
#     # "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#     # "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
#     "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
#     "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B",
#     # "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
#     # "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B"    
# ]
# # data = torch.load(base_file_path)
# # print(data['activated_value'] / data['n'])

# base_file_path = "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base"
# data = torch.load(base_file_path)

# stats = data['over_zero'] / data['n']
# # positions1 = (stats > 0).float()
# num_layers, inter_size = stats.shape
# total_num = num_layers * inter_size
# count = []
# for i, path in enumerate(file_paths):
#     # file_path = "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A"
#     data = torch.load(path)
#     stats1 = data['over_zero'] / data['n']
#     # positions2 = (stats1 > 0).float()
#     data_p = (stats1 - stats) / (stats + 1e-10)
#     # data_p[torch.isnan(data_p)] = 0
#     data_p = torch.where(stats != 0, data_p ,0)
#     x = torch.quantile(data_p, 0.99).item()
#     positions1 = (data_p >= x).float()
#     if i == 0:
#         positions = positions1
#     else:
#         positions = torch.logical_and(positions1, positions).float()
#     count.append(positions1)
#     # intersection = torch.logical_and(positions1, positions2).float()
#     # target_neurons = positions2 - intersection
#     # count.append(target_neurons.sum(-1))
# fig, ax = plt.subplots(figsize=(20,15))
# preference_set = ["Informativeness"]
# keys_set = ['A', 'B']

# indices = [[[] for _ in range(num_layers)] for _ in range(2)]
# for i in range(len(count)):
#     count[i] = count[i] - positions
#     plt.plot(list(range(num_layers)), count[i].sum(-1) / inter_size, label=f"{preference_set[i // 2]}_{keys_set[i % 2]}",linewidth=3)
#     rows_idx, cols_idx =  torch.where(count[i] > 0)
#     for r, c in zip(rows_idx, cols_idx):
#         indices[i][r].append(c)
#     for j in range(num_layers):
#         indices[i][j] = torch.tensor(indices[i][j]).long()
#     # print(count[i].sum())
# # print(indices)
# torch.save(indices, "./activation_mask/Qwen3-8B.Informativeness")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.plot(list(range(num_layers)), positions.sum(dim=-1) / inter_size, label=f"Overlapping",linewidth=3)
# ax.set_xlabel("Layer Num", fontsize=30)
# ax.set_ylabel("Num of Preference-specific neurons", fontsize=30)
# ax.legend(fontsize=25,bbox_to_anchor=(0.5, 1.1), loc="upper center",ncol=4)
# # plt.tight_layout()
# ax.tick_params(axis='both',labelsize=25, pad=12)
# plt.savefig("Informativeness.pdf")


# file_paths = [
#     # "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base",
#     # "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#     # "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
#     # "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
#     # "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B",
#     "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
#     "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B"    
# ]
# # data = torch.load(base_file_path)
# # print(data['activated_value'] / data['n'])

# base_file_path = "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base"
# data = torch.load(base_file_path)

# stats = data['over_zero'] / data['n']
# # positions1 = (stats > 0).float()
# num_layers, inter_size = stats.shape
# total_num = num_layers * inter_size
# count = []
# for i, path in enumerate(file_paths):
#     # file_path = "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A"
#     data = torch.load(path)
#     stats1 = data['over_zero'] / data['n']
#     # positions2 = (stats1 > 0).float()
#     data_p = (stats1 - stats) / (stats + 1e-10)
#     # data_p[torch.isnan(data_p)] = 0
#     data_p = torch.where(stats != 0, data_p ,0)
#     x = torch.quantile(data_p, 0.99).item()
#     positions1 = (data_p >= x).float()
#     if i == 0:
#         positions = positions1
#     else:
#         positions = torch.logical_and(positions1, positions).float()
#     count.append(positions1)
#     # intersection = torch.logical_and(positions1, positions2).float()
#     # target_neurons = positions2 - intersection
#     # count.append(target_neurons.sum(-1))
# fig, ax = plt.subplots(figsize=(20,15))
# preference_set = ["Style"]
# keys_set = ['A', 'B']

# indices = [[[] for _ in range(num_layers)] for _ in range(2)]
# for i in range(len(count)):
#     count[i] = count[i] - positions
#     plt.plot(list(range(num_layers)), count[i].sum(-1) / inter_size, label=f"{preference_set[i // 2]}_{keys_set[i % 2]}",linewidth=3)
#     rows_idx, cols_idx =  torch.where(count[i] > 0)
#     for r, c in zip(rows_idx, cols_idx):
#         indices[i][r].append(c)
#     for j in range(num_layers):
#         indices[i][j] = torch.tensor(indices[i][j]).long()
#     # print(count[i].sum())
# torch.save(indices, "./activation_mask/Qwen3-8B.Style")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.plot(list(range(num_layers)), positions.sum(dim=-1) / inter_size, label=f"Overlapping",linewidth=3)
# ax.set_xlabel("Layer Num", fontsize=30)
# ax.set_ylabel("Num of Preference-specific neurons", fontsize=30)
# ax.legend(fontsize=25,bbox_to_anchor=(0.5, 1.1), loc="upper center",ncol=4)
# # plt.tight_layout()
# ax.tick_params(axis='both',labelsize=25, pad=12)
# plt.savefig("Style.pdf")


    



# print(count[0].sum(-1))

# plt.figure(figsize=(14,10))    
# count = np.array(count)
# for i in range(6):
#     plt.plot(list(range(36)), count[i], label = f"{preference_set[i // 2]}_{keys_set[i % 2]}", linewidth=3)
# plt.xlabel("Layer Num", fontsize=30)
# plt.ylabel("Num of Preference-Specific neurons", fontsize=30)
# plt.legend(fontsize=25)
# plt.tick_params(axis='both', labelsize=20, pad=12)
# plt.savefig("test2.pdf")





# print(target_neurons.sum(-1))
# plot4()

# plot8_quantile(base_file_path, file_paths, q = 0.99)
# plot8_quantile(base_file_path, file_paths, q = 0.8)
# plot8_quantile(base_file_path, file_paths, q = 0.85)
# plot8_quantile(base_file_path, file_paths, q = 0.7)
# plot8_quantile(base_file_path, file_paths, q = 0.75)
# plot1(base_file_path, file_paths, title="Ratio of Preference-related Neurons")
# plot1_quantile(base_file_path, file_paths, title="Ratio of Preference-related Neurons", q=0.25)
# plot1_quantile(base_file_path, file_paths, title="Ratio of Preference-related Neurons", q=0.5)
# plot1_quantile(base_file_path, file_paths, title="Ratio of Preference-related Neurons", q=0.75)
# print(data['over_zero'] / data['n'])

# base_file_path = "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A"
# data = torch.load(base_file_path)
# print(data['over_zero'] / data['n'])
# plot7(base_file_path, file_paths)

# data = torch.load(base_file_path)
# stats = data['over_zero'] / data['n']
# # print(data)
# qs = [0.25, 0.5, 0.75]
# maxv = [0, 0, 0]
# fig, ax = plt.subplots(figsize=(15, 10))
# for i, q in enumerate(qs):
#     b = torch.quantile(stats, q, dim=None).item()
#     print(b)
#     positions = (stats <= b).float() 

#     data = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A")
#     stats2 = data['over_zero'] / data['n']
#     num_layers, inter_size = stats.shape
#     positions2 = (stats2 > stats).float()
#     positions3 = torch.logical_and(positions2, positions)
#     print(positions3.sum().item() / num_layers / inter_size)


#     data1 = positions3.sum(dim = -1).cpu().numpy() / inter_size
#     maxv[i] = np.max(data1)
#     x = list(range(num_layers))
#     ax.plot(x, data1, label=f"{q} quantile", linewidth=3)
#     ax.legend(fontsize=20, loc='lower right')
#     ax.set_xlabel("Layer Num", fontsize=30)
#     ax.set_ylabel("Num of Preference-related Neurons", fontsize=30)
#     ax.tick_params(axis='both', labelsize=20, pad=15)
# plt.savefig("test.pdf")
# print(maxv)
# calc(file_paths, base_file_path)
# # stats = data['over_zero'] / data['n']


# # plot0(file_paths, title="Ratio_of_activated_neurons")
# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B"]
# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B"]

# calc(file_paths,base_file_path)
# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A"]

# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B"]
# calc(file_paths,base_file_path)
# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A"]


# calc(file_paths,base_file_path)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B"]


# calc(file_paths,base_file_path)

labels = ["Expertise_A or Informativeness_A",
          "Expertise_Informativeness_AA",
          "Expertise_A",
          "Informativeness_A"]
# plot2(base_file_path, file_paths, title="Expertise_A_Informativeness_A", labels=labels)

file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness.BB"]

labels = ["Expertise_B or Informativeness_B",
          "Expertise_Informativeness_BB",
          "Expertise_B",
          "Informativeness_B"]


base_file_path = "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base"
preference_set = ["Expertise", "Informativeness", "Style"]
# keys_set = ['A', 'B']
# for i, p1 in enumerate(preference_set):
#     for j, p2 in enumerate(preference_set):
#         if i >= j: continue
#         for k1 in keys_set:
#             for k2 in keys_set:
#                 file_paths = [f"/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.{p1}.{k1}",
#                             f"/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.{p2}.{k2}",
#                             f"/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.{p1}_{p2}.{k1}{k2}"]

#                 labels = [f"{p1}_{k1} or {p2}_{k2}",
#                         f"{p1}_{p2}_{k1}{k2}",
#                         f"{p1}_{k1}",
#                         f"{p2}_{k2}"]
#                 plot2(base_file_path, file_paths, title=f"{p1}_{k1}_{p2}_{k2}", labels=labels, tag='over_zero')
                
                
# preference_set = ["Expertise", "Informativeness", "Style"]
# keys_set = ['A', 'B']
# for k1 in keys_set:
#     for k2 in keys_set:
#         for k3 in keys_set:
#             file_paths = [f"/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.{k1}",
#                         f"/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.{k2}",
#                         f"/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.{k2}",
#                         f"/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness_Style.{k1}{k2}{k3}"]

#             labels = [f"Expertise_{k1}",
#                     f"Informativeness_{k2}",
#                     f"Style_{k3}",
#                     f"Expertise_{k1} or Informativeness_{k2} or Style_{k3}",
#                     f"Expertise_Informativeness_Style_{k1}{k2}{k3}"]
#             plot3(base_file_path, file_paths, title=f"Expertise_{k1}_Informativeness_{k2}_Style_{k3}", labels=labels, tag='over_zero')
# qs = [0.1, 0.2, 0.3]
# for q in qs:
#     for i, p1 in enumerate(preference_set):
#         for j, p2 in enumerate(preference_set):
#             if i >= j: continue
#             for k1 in keys_set:
#                 for k2 in keys_set:
#                     file_paths = [f"/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.{p1}.{k1}",
#                                 f"/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.{p2}.{k2}",
#                                 f"/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.{p1}_{p2}.{k1}{k2}"]

#                     labels = [f"{p1}_{k1} or {p2}_{k2}",
#                             f"{p1}_{p2}_{k1}{k2}",
#                             f"{p1}_{k1}",
#                             f"{p2}_{k2}"]
#                     plot2_quantile(base_file_path, file_paths, title=f"{p1}_{k1}_{p2}_{k2}", labels=labels, tag='over_zero', q=q)

file_paths = [
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness_Style.AB"]

labels = ["Informativeness_A or Style_B",
          "Informativeness_Style_AB",
          "Informativeness_A",
          "Style_B",
          ]
# plot2(base_file_path, file_paths, title="Informativeness_A_Style_B", labels=labels, tag='over_zero')


file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Style.BB"]

labels = ["Expertise_B or Style_B",
          "Expertise_Style_BB",
          "Expertise_B",
          "Style_B"]
# plot2(base_file_path, file_paths, title="Expertise_B_Style_B", labels=labels)

file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness_Style.AAA"]

labels = ["Expertise_A",
          "Informativeness_A",
          "Style_A", 
          "Expertise_A or Informativeness_A or Style_A",
          "Expertise_Informativeness_Style_AAA"]

# plot3(base_file_path, file_paths, title="Expertise_A_Informativeness_A_Style_A", labels=labels)

# plot5()
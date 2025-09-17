import matplotlib.pyplot as plt
import torch
import matplotlib
import numpy as np
from matplotlib import cm
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

def plot0(base_file_path, file_paths, title="", label='over_zero'):
    data = torch.load(base_file_path)
    base_probs = data[label] / data['n']
    q = 0.75
    x = torch.quantile(base_probs, q).item()
    fig, ax = plt.subplots(figsize=(20, 16))
    for file_path in file_paths:
        data = torch.load(file_path)
        activation_probs = data[label] / data['n']
        nums_layer, inter_size = activation_probs.shape
        stats = ((activation_probs > x).sum(dim = -1) / inter_size).cpu().tolist()
        
        
        tags = file_path.split('/')[-1].split('.')
        t, model_name, preference = tags[0], tags[2], tags[3]
        if len(tags) == 5:
            index = tags[4]
        else:
            index = ""
        
        ax.plot(list(range(nums_layer)), stats, label=f"{preference}_{index}" if index != "" else "Only Context", linewidth=3)
        ax.set_ylabel("Ratio of activated neurons", fontsize=30)
        ax.set_xlabel("Layer num", fontsize=30)
        ax.legend(fontsize=25)
    # ax.set_title(title, fontsize=30)
    ax.tick_params(axis='both', labelsize=25, pad=15)
    plt.savefig(f"stats.{title}.pdf")

def load(file_path, label='over_zero'):
    data =  torch.load(file_path)
    activation_probs = data[label] / data['n']
    # _ , inter_size = activation_probs.shape
    # stats = (activation_probs > 0).float().cpu()
    return activation_probs


def plot1(base_file_path, file_paths, title=None):
    stats = load(base_file_path)
    num_layers, inter_size = stats.shape
    fig, ax = plt.subplots(figsize=(20,16))
    for path in file_paths:
        tags = path.split('/')[-1].split('.')
        t, model_name, preference = tags[0], tags[2], tags[3]
        if len(tags) == 5:
            index = tags[4]
        else:
            index = ""
        P_stats = load(path)
        positions = (P_stats > stats).float().cpu()
        y = (positions.sum(dim = -1)/inter_size) .tolist()
        ax.plot(list(range(num_layers)), y, label=f"{preference}_{index}" if index != "" else "Only Context", linewidth=5)
    ax.set_xlabel("Layer Num", fontsize=30)
    ax.set_ylabel("Specific Neuron Num", fontsize=30)
    ax.tick_params(axis='both', labelsize=30)
    ax.legend(fontsize=25, loc='lower right')
    plt.savefig(f"{title}.pdf")
    
def plot1_quantile(base_file_path, file_paths, title=None, q = 0.25):
    stats = load(base_file_path)
    num_layers, inter_size = stats.shape
    fig, ax = plt.subplots(figsize=(20,16))
    x = torch.quantile(stats, q).item()
    restrictions = stats <= x
    for path in file_paths:
        tags = path.split('/')[-1].split('.')
        t, model_name, preference = tags[0], tags[2], tags[3]
        if len(tags) == 5:
            index = tags[4]
        else:
            index = ""
        P_stats = load(path)
        positions = (P_stats > stats).float().cpu()
        positions = torch.logical_and(positions, restrictions)
        y = (positions.sum(dim = -1)/inter_size) .tolist()
        ax.plot(list(range(num_layers)), y, label=f"{preference}_{index}" if index != "" else "Only Context", linewidth=5)
    ax.set_xlabel("Layer Num", fontsize=30)
    ax.set_ylabel("Ratio of Preference-Related Neurons", fontsize=30)
    ax.tick_params(axis='both', labelsize=30)
    ax.legend(fontsize=25, loc='lower right')
    plt.savefig(f"{title}_quantile_{q}.pdf")

def plot2(base_file_path, file_paths, title = "Expertise_A_Informativeness_A", labels = None, tag='activated_value'):
    stats = load(base_file_path, label=tag)
    num_layers, inter_size = stats.shape
    if len(file_paths) != 3: raise ValueError
    P1_stats = load(file_paths[0], label=tag)
    P2_stats = load(file_paths[1], label=tag)
    P12_stats = load(file_paths[2], label=tag)

    # x = torch.quantile(stats, 0.25).item()
    # restrictions = (stats <= x).float().cpu()
    positions1 = (P1_stats > stats).float().cpu()
    positions2 = (P2_stats > stats).float().cpu()
    positions12 = torch.logical_or(positions1, positions2)
    positions3 = (P12_stats > stats).float().cpu()

    y1 = (positions12.sum(dim = -1) / inter_size).tolist()
    y2 = (positions3.sum(dim = -1) / inter_size).tolist()
    y3 = (positions1.sum(dim = -1) / inter_size).tolist()
    y4 = (positions2.sum(dim = -1) / inter_size).tolist()
    # map23 = torch.logical_or(positions1, positions2)
    intersection = torch.logical_and(positions12, positions3).sum().item()
    union = positions12.sum().item() + positions3.sum().item() - intersection  # 1的并集数：(4)+(4)-3=5

    p = intersection / union
    
    fig, ax = plt.subplots(figsize=(20,14))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(list(range(num_layers)), y3, linewidth=4, label=labels[2])
    ax.plot(list(range(num_layers)), y4, linewidth=4, label=labels[3])
    ax.plot(list(range(num_layers)), y1, linewidth=4, label=labels[0])
    ax.plot(list(range(num_layers)), y2, linewidth=4, label=labels[1])
    ax.set_xlabel("Layer Num", fontsize=30)
    ax.set_ylabel("Ratio of Preference-Specific Neurons", fontsize=30)
    ax.text(15, 0.8, s = f"Overlapping Ratio: {round(p,3)}", fontsize=25, color='black') # 添加文本
    ax.legend(fontsize=25)
    ax.tick_params(axis='both', labelsize=25, pad=15)

    plt.savefig(f"{title}.pdf")
    
def plot2_quantile(base_file_path, file_paths, title = "Expertise_A_Informativeness_A", labels = None, tag='activated_value', q=0.25):
    stats = load(base_file_path, label=tag)
    num_layers, inter_size = stats.shape
    if len(file_paths) != 3: raise ValueError
    P1_stats = load(file_paths[0], label=tag)
    P2_stats = load(file_paths[1], label=tag)
    P12_stats = load(file_paths[2], label=tag)
    x = torch.quantile(stats, q).item()
    restrictions = stats <= x
    # x = torch.quantile(stats, 0.25).item()
    # restrictions = (stats <= x).float().cpu()
    positions1 = (P1_stats > stats).float().cpu()
    positions1 = torch.logical_and(positions1, restrictions)
    
    positions2 = (P2_stats > stats).float().cpu()
    positions2 = torch.logical_and(positions2, restrictions)
    
    positions12 = torch.logical_or(positions1, positions2)
    positions3 = (P12_stats > stats).float().cpu()
    positions3 = torch.logical_and(positions3, restrictions)
    
    

    y1 = (positions12.sum(dim = -1) / inter_size).tolist()
    y2 = (positions3.sum(dim = -1) / inter_size).tolist()
    y3 = (positions1.sum(dim = -1) / inter_size).tolist()
    y4 = (positions2.sum(dim = -1) / inter_size).tolist()
    # map23 = torch.logical_or(positions1, positions2)
    intersection = torch.logical_and(positions12, positions3).sum().item()
    union = positions12.sum().item() + positions3.sum().item() - intersection  # 1的并集数：(4)+(4)-3=5

    p = intersection / union
    
    fig, ax = plt.subplots(figsize=(20,14))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(list(range(num_layers)), y3, linewidth=4, label=labels[2])
    ax.plot(list(range(num_layers)), y4, linewidth=4, label=labels[3])
    ax.plot(list(range(num_layers)), y1, linewidth=4, label=labels[0])
    ax.plot(list(range(num_layers)), y2, linewidth=4, label=labels[1])
    ax.set_xlabel("Layer Num", fontsize=30)
    ax.set_ylabel("Ratio of Preference-Specific Neurons", fontsize=30)
    ax.text(15, 0.8, s = f"Overlapping Ratio: {round(p,3)}", fontsize=25, color='black') # 添加文本
    ax.legend(fontsize=25)
    ax.tick_params(axis='both', labelsize=25, pad=15)

    plt.savefig(f"{title}_quantile_{q}.pdf")
    
def plot3(base_file_path, file_paths, title = "Expertise_A_Informativeness_A", labels = None, tag='over_zero'):
    stats = load(base_file_path)
    num_layers, inter_size = stats.shape
    if len(file_paths) != 4: raise ValueError
    P1_stats = load(file_paths[0],label=tag)
    P2_stats = load(file_paths[1],label=tag)
    P3_stats = load(file_paths[2],label=tag)
    P123_stats = load(file_paths[3],label=tag)

    positions1 = (P1_stats > stats).float().cpu()
    positions2 = (P2_stats > stats).float().cpu()
    positions3 = (P3_stats > stats).float().cpu()
    positions12 = torch.logical_or(positions1, positions2)
    positions123 = torch.logical_or(positions12, positions3)
    positions4 = (P123_stats > stats).float().cpu()

    y1 = (positions1.sum(dim = -1) / inter_size).tolist()
    y2 = (positions2.sum(dim = -1) / inter_size).tolist()
    y3 = (positions3.sum(dim = -1) / inter_size).tolist()
    y4 = (positions123.sum(dim = -1) / inter_size).tolist()
    y5 = (positions4.sum(dim = -1) / inter_size).tolist()
    intersection = (torch.logical_and(positions123, positions4) == 1).sum().item()
    union = positions123.sum().item() + positions4.sum().item() - intersection
    p = intersection / union
    # p = float(p.cpu().numpy())
    
    fig, ax = plt.subplots(figsize=(20,14))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(list(range(num_layers)), y1, linewidth=4, label=labels[0])
    ax.plot(list(range(num_layers)), y2, linewidth=4, label=labels[1])
    ax.plot(list(range(num_layers)), y3, linewidth=4, label=labels[2])
    ax.plot(list(range(num_layers)), y4, linewidth=4, label=labels[3])
    ax.plot(list(range(num_layers)), y5, linewidth=4, label=labels[4])
    ax.set_xlabel("Layer Num", fontsize=30)
    ax.set_ylabel("Num of Preference-Specific Neurons", fontsize=30)
    ax.text(15, 0.8, s = f"Overlapping Ratio: {round(p,3)}", fontsize=25, color='black') # 添加文本
    ax.legend(fontsize=25, loc='lower right')
    ax.tick_params(axis='both', labelsize=25, pad=15)

    plt.savefig(f"{title}.pdf")
    
def calc(file_paths, base_file_path):
    stats = load(base_file_path)
    intersection, union = torch.ones_like(stats), torch.zeros_like(stats)
    for i, path in enumerate(file_paths):
        stats_cur = load(path)
        # stats_cur = data['over_zero'] / data['n']
        positions_cur = (stats_cur > stats).float()
        intersection = torch.logical_and(intersection, positions_cur)
        union = torch.logical_or(union, positions_cur)
    print(intersection.sum().item() / union.sum().item())

def plot4():
    table = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base")
    table2 = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A")
    table3 = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A")
    table4 = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness.AA")
    data = table['over_zero'] / table['n']
    data2 = table2['over_zero'] / table2['n']
    data3 = table3['over_zero'] / table3['n']
    data4 = table4['over_zero'] / table4['n']
    data2 = (data2 - data) / (data + 1e-10)
    data3 = (data3 - data) / (data + 1e-10)
    data4 = (data4 - data) / (data + 1e-10)
    import numpy as np
    num_layers, inter_size = data.shape
    x, y = list(range(num_layers)), list(range(inter_size))
    topk_ratios = [0.01] + list(map(lambda x: x * 0.05, list(range(1, 21))))
    res = []
    for topk_ratio in topk_ratios:
        topk_num = int(topk_ratio * inter_size)
        values2, indices2 = torch.topk(data2, topk_num, dim=1, largest=True)
        map2 = torch.zeros_like(data)
        for i in range(num_layers):
            map2[i][indices2[i]] = 1

        values3, indices3 = torch.topk(data3, topk_num, dim=1, largest=True)   
        map3 = torch.zeros_like(data)
        for i in range(num_layers):
            map3[i][indices3[i]] = 1

        values4, indices4 = torch.topk(data4, topk_num, dim=1, largest=True)

        map4 = torch.zeros_like(data)
        for i in range(num_layers):
            map4[i][indices4[i]] = 1


        map23 = torch.logical_or(map2, map3)

        intersection = torch.logical_and(map23, map4).sum().item()
        union = torch.logical_or(map23, map4).sum().item() 

        res.append(round(intersection / union, 3))
    plt.figure(figsize=(14,10))
    plt.plot(topk_ratios, res, linewidth=3, color='blue', label="Probability only difference")
    plt.xlabel("Ratio of Top Value", fontsize=30)
    plt.ylabel("IoU", fontsize=30)
    plt.tick_params(axis='both', labelsize=25, pad=15)
    plt.grid(True)
    plt.savefig("ratios2.pdf")
    
def plot5():
    table = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base")
    table2 = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A")
    table3 = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A")
    table4 = torch.load("/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness.AA")
    data = table['over_zero'] / table['n']
    data2 = table2['over_zero'] / table2['n']
    data3 = table3['over_zero'] / table3['n']
    data4 = table4['over_zero'] / table4['n']
    data2 = (data2 - data) / (data + 1e-10)
    data3 = (data3 - data) / (data + 1e-10)
    data4 = (data4 - data) / (data + 1e-10)
    p = 0.8
    q = 0.2
    x1 = torch.quantile(data2, p).item()
    x2 = torch.quantile(data3, p).item()
    x3 = torch.quantile(data3, p).item()
    
    y = torch.quantile(data, q).item()
    
    data2 = torch.where(data >= y, data2, 0)
    data3 = torch.where(data >= y, data3, 0)
    data4 = torch.where(data >= y, data4, 0)
    data2 = torch.where(data2 >= x1, data2, 0)
    data3 = torch.where(data3 >= x2, data3, 0)
    data4 = torch.where(data4 >= x3, data4, 0)
    # data4 = torch.where(data4 >= -1, data4, -1)
    # positions = data3 > data
    # print(data2)
    num_layers, inter_size = data.shape
    x, y = list(range(num_layers)), list(range(inter_size))
    xv, yv = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xv, yv, data2.numpy().transpose(), cmap=cm.coolwarm, 
                        linewidth=0, antialiased=True)

    ax.set_xlabel('Layer Num', fontsize=20,labelpad=25)
    ax.set_ylabel('Hidden Size', fontsize=20,labelpad=25)
    ax.set_zlabel('Probability Difference', fontsize=20, labelpad=25)
    ax.set_zlim(0, 8)
    ax.tick_params(axis='both', labelsize=10)
    ax.ticklabel_format(style='scientific', axis='z')
    ax.grid(visible=False)
    fig.colorbar(surf, shrink=0.3, location='left', pad=0.001)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig("test2.pdf")
    
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xv, yv, data3.numpy().transpose(), cmap=cm.coolwarm, 
                        linewidth=0, antialiased=True)

    ax.set_xlabel('Layer Num', fontsize=20,labelpad=25)
    ax.set_ylabel('Hidden Size', fontsize=20,labelpad=25)
    ax.set_zlabel('Probability Difference', fontsize=20, labelpad=25)
    ax.set_zlim(0, 8)
    ax.tick_params(axis='both', labelsize=10)
    ax.ticklabel_format(style='scientific', axis='z')
    ax.grid(visible=False)
    fig.colorbar(surf, shrink=0.3, location='left', pad=0.001)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig("test3.pdf")
    
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xv, yv, data4.numpy().transpose(), cmap=cm.coolwarm, 
                        linewidth=0, antialiased=True)

    ax.set_xlabel('Layer Num', fontsize=20,labelpad=25)
    ax.set_ylabel('Hidden Size', fontsize=20,labelpad=25)
    ax.set_zlabel('Probability Difference', fontsize=20, labelpad=25)
    ax.set_zlim(0, 8)
    ax.tick_params(axis='both', labelsize=10)
    ax.ticklabel_format(style='scientific', axis='z')
    ax.grid(visible=False)
    fig.colorbar(surf, shrink=0.3, location='left', pad=0.001)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig("test4.pdf")
    
def plot6(base_file_path, file_path):
    data = torch.load(base_file_path)
    stats = data['over_zero'] / data['n']
    # print(data)
    qs = [0.25, 0.5, 0.75]
    maxv = [0, 0, 0]
    fig, ax = plt.subplots(figsize=(15, 10))
    for i, q in enumerate(qs):
        b = torch.quantile(stats, q, dim=None).item()
        print(b)
        positions = (stats <= b).float() 

        data = torch.load(file_path)
        stats2 = data['over_zero'] / data['n']
        num_layers, inter_size = stats.shape
        positions2 = (stats2 > stats).float()
        positions3 = torch.logical_and(positions2, positions)
        print(positions3.sum().item() / num_layers / inter_size)


        data1 = positions3.sum(dim = -1).cpu().numpy() / inter_size
        maxv[i] = np.max(data1)
        x = list(range(num_layers))
        ax.plot(x, data1, label=f"{q} quantile", linewidth=3)
        ax.legend(fontsize=20, loc='lower right')
        ax.set_xlabel("Layer Num", fontsize=30)
        ax.set_ylabel("Num of Preference-related Neurons", fontsize=30)
        ax.tick_params(axis='both', labelsize=20, pad=15)
    plt.savefig("test.pdf")
    print(maxv)
    
def plot7(base_file_path, file_paths):
    data = torch.load(base_file_path)
    stats1 = data['over_zero'] / data['n']
    # stats1[stats1 == 0] = 1e-10
    num_layers, inter_size = stats1.shape
    fig, ax = plt.subplots(figsize=(12,10))
    for path in file_paths:
        tags = path.split('/')[-1].split('.')
        t, model_name, preference = tags[0], tags[2], tags[3]
        if len(tags) == 5:
            index = tags[4]
        else:
            index = ""
        data = torch.load(path)
        stats = data['over_zero'] / data['n']

        y = (stats - stats1) / (stats1 + 1e-10)
        # print(y)
        y[torch.isnan(y)] = 0
        x = torch.quantile(y, 0.5).item()
        ratio = ((y > x).sum(dim = -1) / inter_size).cpu().tolist()
        # print(y)
        ax.plot(list(range(num_layers)), ratio, label=f"{preference}_{index}", linewidth=3)
    ax.set_xlabel("Layer Num", fontsize=30)
    ax.set_ylabel("Num of Preference-related Neurons", fontsize=30)
    ax.legend(fontsize=20, loc="lower right")
    ax.tick_params(axis='both', labelsize=20, pad=12)
    plt.savefig("test_quantile_0.5.pdf")
    
def plot8(base_file_path, file_paths):
    data = torch.load(base_file_path)
    stats1 = data['over_zero'] / data['n']
    # stats1[stats1 == 0] = 1e-10
    num_layers, inter_size = stats1.shape
    fig, ax = plt.subplots(figsize=(12,10))
    for path in file_paths:
        tags = path.split('/')[-1].split('.')
        t, model_name, preference = tags[0], tags[2], tags[3]
        if len(tags) == 5:
            index = tags[4]
        else:
            index = ""
        data = torch.load(path)
        stats = data['over_zero'] / data['n']

        y = (stats - stats1) / (stats1 + 1e-10)
        y[torch.isnan(y)] = 0
        ratio = ((y > 0).sum(dim = -1) / inter_size)
        ax.plot(list(range(num_layers)), ratio, label=f"{preference}_{index}", linewidth=3)
    ax.set_xlabel("Layer Num", fontsize=30)
    ax.set_ylabel("Num of Preference-related Neurons", fontsize=30)
    ax.legend(fontsize=20, loc="lower right")
    ax.tick_params(axis='both', labelsize=20, pad=12)
    plt.savefig("test2.pdf")
    
def plot8_quantile(base_file_path, file_paths, q=0.25):
    data = torch.load(base_file_path)
    stats1 = data['over_zero'] / data['n']
    num_layers, inter_size = stats1.shape
    delta = torch.quantile(stats1, 0.2).item()
    fig, ax = plt.subplots(figsize=(12, 10))
    for path in file_paths:
        tags = path.split('/')[-1].split('.')
        t, model_name, preference = tags[0], tags[2], tags[3]
        if len(tags) == 5:
            index = tags[4]
        else:
            index = ""
        data = torch.load(path)
        stats = data['over_zero'] / data['n']
        y = (stats - stats1) / stats1
        y[torch.isnan(y)] = 0
        y = torch.where(stats1 >= delta, y, 0)
        x = torch.quantile(y, q)
        ratio = ((y >= x).sum(dim = -1) / inter_size).tolist()
        ax.plot(list(range(num_layers)), ratio, label=f"{preference}_{index}", linewidth=3)
    ax.set_xlabel("Layer Num", fontsize=30)
    ax.set_ylabel("Ratio of Preference-related Neurons", fontsize=30)
    ax.legend(fontsize=20, loc="lower right")
    ax.tick_params(axis='both', labelsize=20, pad=12)
    plt.savefig(f"eval_quantile_{q}.pdf")
    
def plot_entropy():
    data = torch.load("/NAS/yjt/Abstract_Thought/activation_mask/Qwen3-8B.single")
    # stats1 = data['over_zero'] / data['n']
    # print(len(data))
    preference_set = ["Expertise", "Informativeness", "Style"]
    keys_set = ['A', 'B']

    nums = [[0 for _ in range(36)] for _ in range(6)]
    for idx, d in enumerate(data):
        for i in range(36):
            nums[idx][i] = len(d[i])
    plt.figure(figsize=(14,10))  
    nums = np.array(nums)
    for i in range(6):
        plt.plot(list(range(36)), nums[i], label = f"{preference_set[i // 2]}_{keys_set[i % 2]}", linewidth=3)
    plt.xlabel("Layer Num", fontsize=30)
    plt.ylabel("Num of Preference-Specific neurons", fontsize=30)
    plt.legend(fontsize=25)
    plt.tick_params(axis='both', labelsize=20, pad=12)
    plt.savefig("test.pdf")     
import matplotlib.pyplot as plt
import torch
def plot0(file_paths, title=""):
    fig, ax = plt.subplots(figsize=(16, 10))
    for file_path in file_paths:
        data = torch.load(file_path)
        activation_probs = data['over_zero'] / data['n']
        nums_layer, inter_size = activation_probs.shape
        stats = ((activation_probs > 0).sum(dim = -1) / inter_size).cpu().tolist()
        
        
        tags = file_path.split('/')[-1].split('.')
        t, model_name, preference = tags[0], tags[2], tags[3]
        if len(tags) == 5:
            index = tags[4]
        else:
            index = ""
        
        ax.plot(list(range(len(stats))), stats, label=f"{preference}_{index}" if index != "" else "Only Context", linewidth=3)
        ax.set_ylabel("Count of activated neurons", fontsize=30)
        ax.set_xlabel("Layer num", fontsize=30)
        ax.legend(fontsize=25)
    ax.set_title(title, fontsize=30)
    plt.savefig(f"stats.{title}.pdf")

def load(file_path):
    data =  torch.load(file_path)
    activation_probs = data['over_zero'] / data['n']
    _ , inter_size = activation_probs.shape
    stats = (activation_probs > 0).float().cpu()
    return stats


def plot1(base_file_path, file_paths, title=None):
    stats = load(base_file_path)
    num_layers = stats.shape[0]
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
        y = positions.sum(dim = -1).tolist()
        ax.plot(list(range(num_layers)), y, label=f"{preference}_{index}" if index != "" else "Only Context", linewidth=5)
    ax.set_xlabel("Layer Num", fontsize=30)
    ax.set_ylabel("Specific Neuron Num", fontsize=30)
    ax.tick_params(axis='both', labelsize=30)
    ax.legend(fontsize=25, loc='lower right')
    plt.savefig(f"{title}.pdf")

def plot2(base_file_path, file_paths, title = "Expertise_A_Informativeness_A", labels = None):
    stats = load(base_file_path)
    num_layers, inter_size = stats.shape
    if len(file_paths) != 3: raise ValueError
    P1_stats = load(file_paths[0])
    P2_stats = load(file_paths[1])
    P12_stats = load(file_paths[2])

    positions1 = (P1_stats > stats).float().cpu()
    positions2 = (P2_stats > stats).float().cpu()
    positions12 = torch.logical_or(positions1, positions2)
    positions3 = (P12_stats > stats).float().cpu()

    y1 = positions12.sum(dim = -1).tolist()
    y2 = positions3.sum(dim = -1).tolist()
    y3 = positions1.sum(dim = -1).tolist()
    y4 = positions2.sum(dim = -1).tolist()
    x1 = (torch.logical_and(positions12, positions3) == 1).sum()
    x2 = (torch.logical_or(positions12, positions3) == 1).sum()
    p = x1 / x2
    p = float(p.cpu().numpy())
    
    fig, ax = plt.subplots(figsize=(20,14))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(list(range(num_layers)), y3, linewidth=4, label=labels[2])
    ax.plot(list(range(num_layers)), y4, linewidth=4, label=labels[3])
    ax.plot(list(range(num_layers)), y1, linewidth=4, label=labels[0])
    ax.plot(list(range(num_layers)), y2, linewidth=4, label=labels[1])
    ax.set_xlabel("Layer Num", fontsize=30)
    ax.set_ylabel("Specific Neuron Num", fontsize=30)
    ax.text(0.05, 2350, s = f"Overlapping Ratio: {round(p,3)}", fontsize=25, color='black') # 添加文本
    ax.legend(fontsize=25)
    ax.tick_params(axis='both', labelsize=25)

    plt.savefig(f"{title}.pdf")
    
def plot3(base_file_path, file_paths, title = "Expertise_A_Informativeness_A", labels = None):
    stats = load(base_file_path)
    num_layers, inter_size = stats.shape
    if len(file_paths) != 4: raise ValueError
    P1_stats = load(file_paths[0])
    P2_stats = load(file_paths[1])
    P3_stats = load(file_paths[2])
    P123_stats = load(file_paths[3])

    positions1 = (P1_stats > stats).float().cpu()
    positions2 = (P2_stats > stats).float().cpu()
    positions3 = (P3_stats > stats).float().cpu()
    positions12 = torch.logical_or(positions1, positions2)
    positions123 = torch.logical_or(positions12, positions3)
    positions4 = (P123_stats > stats).float().cpu()

    y1 = positions1.sum(dim = -1).tolist()
    y2 = positions2.sum(dim = -1).tolist()
    y3 = positions3.sum(dim = -1).tolist()
    y4 = positions123.sum(dim = -1).tolist()
    y5 = positions4.sum(dim = -1).tolist()
    x1 = (torch.logical_and(positions123, positions4) == 1).sum()
    x2 = (torch.logical_or(positions123, positions4) == 1).sum()
    p = x1 / x2
    p = float(p.cpu().numpy())
    
    fig, ax = plt.subplots(figsize=(20,14))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(list(range(num_layers)), y1, linewidth=4, label=labels[0])
    ax.plot(list(range(num_layers)), y2, linewidth=4, label=labels[1])
    ax.plot(list(range(num_layers)), y3, linewidth=4, label=labels[2])
    ax.plot(list(range(num_layers)), y4, linewidth=4, label=labels[3])
    ax.plot(list(range(num_layers)), y5, linewidth=4, label=labels[4])
    ax.set_xlabel("Layer Num", fontsize=30)
    ax.set_ylabel("Specific Neuron Num", fontsize=30)
    ax.text(15, 2500, s = f"Overlapping Ratio: {round(p,3)}", fontsize=25, color='black') # 添加文本
    ax.legend(fontsize=25, loc='lower right')
    ax.tick_params(axis='both', labelsize=25)

    plt.savefig(f"{title}.pdf")
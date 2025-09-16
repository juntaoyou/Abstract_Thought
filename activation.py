import torch
import torch.nn.functional as F
import numpy as np
# torch.set_printoptions(threshold=np.inf)
keys =  ['A','B']
def activation(activation_probs, num_layers, save_type="single"):
    top_rate = 0.01
    filter_rate = 0.95
    activation_bar_ratio = 0.95
    
    normed_activation_probs = F.softmax(activation_probs, dim = -1)
    # activation_probs[torch.isnan(activation_probs)] = 0
    # print("Act:",normed_activation_probs)
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False
    print(entropy)
    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError
    
    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
    print(top_prob_value)
    # dismiss the neruon if no language has an activation value over top 90%
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index] 
    # print(selected_probs.size())
    print(selected_probs.size(0), torch.bincount(selected_probs.argmax(dim=-1)))
    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
    print((selected_probs > activation_bar).sum(dim=1).tolist())
    lang, indice = torch.where(selected_probs > activation_bar)

    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
        final_indice.append(layer_index)
    torch.save(final_indice, f"activation_mask/{model_suffix}.{save_type}")  
    # print(final_indice)
    
n, conditioned_probs, activation_probs = [], [], []
model_suffix = "Qwen3-8B"
base_data = torch.load(f'data/activation.train.{model_suffix}.base')
base_probs = base_data['over_zero'] / base_data["n"]
# print("Base:", base_probs)
for preference in ['Expertise','Informativeness','Style']:
    for index in keys:
        data = torch.load(f'data/activation.train.{model_suffix}.{preference}.{index}')
        n.append(data['n'])
        x = data['over_zero'] / data['n']
        preference_probs = (x - base_probs) / (base_probs + 1e-10)
        preference_probs[torch.isnan(preference_probs)] = 0
        # print(preference_probs.max())
        activation_probs.append(preference_probs)
        
activation_probs = torch.stack(activation_probs, dim=-1)
num_layers, _ , _ = activation_probs.shape
activation(activation_probs, num_layers, save_type="single")
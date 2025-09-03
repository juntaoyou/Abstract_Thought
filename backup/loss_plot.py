import matplotlib.pyplot as plt
import json
with open("./neuron_deactivation/Qwen3-1.7B_detect_news_headline_generation_shared_neurons_0.01.json", 'r') as f:
    data = json.load(f)

def myplot(keys, values):
    fig_num = len(keys)
    fig, axs = plt.subplots(2, fig_num // 2, figsize=(40,20),layout="constrained",sharey=True) 
    axs = axs.reshape(fig_num) 
    fig.suptitle("Activated Neurons", fontsize=45)
    for i, ax in enumerate(axs):
        value = values[i]
        x = list(range(1, len(value.keys()) + 1))
        y = [len(v) for v in value.values()]
        ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
        ax.set_title(f"{keys[i]}", fontsize=35)
        ax.set_xlabel("layer_num", fontsize=35)
        ax.set_ylabel("Count", fontsize=35)
        ax.set_xlim([0,30])
        ax.set_xticklabels([0,5,10,15,20,25,30], fontsize=30)
        ax.set_yticklabels([0,10,20,30,40,50,60],fontsize=30)
    # plt.show()
    plt.savefig("./figures/neurons.pdf", format='pdf')
    plt.savefig("./figures/neurons.jpg", format='jpg')
    
        
keys = list(data.keys())
values = list(data.values())
myplot(keys, values)

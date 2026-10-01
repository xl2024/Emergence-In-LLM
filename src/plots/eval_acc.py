import os
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import torch


model_color_dict = {
    "gpt2": {
        "label": "GPT-2 Small (124M)", 
        "color": "#98fb98",    # light green
    },
    "gpt2-medium": {
        "label": "GPT-2 Medium (335M)", 
        "color": "#32cd32",    # lime green
    },
    "gpt2-large": {
        "label": "GPT-2 Large (774M)",
        "color": "#006400",    # dark green
    },
    "gpt2-xl": {
        "label": "GPT-2 XL (1.5B)",
        "color": "#808000",    # olive
    },
    "gemma-2-2b": {
        "label": "Gemma-2 2B",
        "color": "#87cefa",    # light blue
    },
    "gemma-2-9b": {
        "label": "Gemma-2 9B",
        "color": "#6495ed",    # cornflower blue
    },
    "gemma-2-27b": {
        "label": "Gemma-2 27B",
        "color": "#000080",    # dark blue / navy
    },
    "Qwen2.5-7B": {
        "label": "Qwen2.5 7B",
        "color": "#f08080",    # light coral
    },
    "Qwen2.5-14B": {
        "label": "Qwen2.5 14B",
        "color": "#ff0000",    # red
    },
    "Qwen2.5-32B": {
        "label": "Qwen2.5 32B",
        "color": "#ff8c00",    # dark orange
    },
    "Qwen2.5-72B": {
        "label": "Qwen2.5 72B",
        "color": "#800000",    # maroon
    },
    "Llama-3.1-8B": {
        "label": "LLama-3.1 8B",
        "color": "#ee82ee",    # violet
    },
    "Llama-3.1-70B": {
        "label": "LLama-3.1 70B",
        "color": "#ba55d3",    # medium orchid
    },
}


def plot_accuracy(sum_dict, rule, save_folder, model_color_dict=model_color_dict):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for model_type in model_color_dict.keys():
        model_data = sum_dict[model_type][rule]
        
        x_vals = []
        y_vals = []
        y_lower = []
        y_upper = []
        
        for eg_num in range(1, 11):
            stats = model_data[eg_num]
            correct_num = stats["correct_num"]
            total_num = stats["total_num"]
            avg_acc = np.mean(stats["acc_list"])
            binom_conf = proportion_confint(count=correct_num, nobs=total_num, method='wilson')
            
            x_vals.append(eg_num)
            y_vals.append(avg_acc)
            y_lower.append(binom_conf[0])
            y_upper.append(binom_conf[1])

        label = model_color_dict[model_type]["label"]
        color = model_color_dict[model_type]["color"]
        ax.plot(x_vals, y_vals, marker='.', markersize=7, label=label, color=color, linewidth=1.5)
        ax.fill_between(x_vals, y_lower, y_upper, color=color, alpha=0.15, edgecolor='none')

    ax.set_xlabel("Number of In-context Examples", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(range(1, 11), fontsize=14)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0.0, 1.1, 0.2)], fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(
        loc='lower right', 
        ncol=2, 
        framealpha=0.9, 
        fontsize=10, 
        borderpad=0.5,
        # handlelength=2.0
    )
    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"accuracy_rule_{rule}.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"fig saved in {save_path}")
    plt.close()


if __name__ == "__main__":
    sum_dict = torch.load("outputs/eval/sum_dict.pt")
    save_folder = "outputs/eval"
    plot_accuracy(sum_dict, rule="ABA", save_folder=save_folder)
    plot_accuracy(sum_dict, rule="ABB", save_folder=save_folder)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import logging

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def plot_curve(schemes_data, top_num_list_data, save_path):
    """
    Plots Ablation, Control, and Random curves in a single figure.
    """
    records = []
    for scheme in schemes_data.keys():
        for k, val in zip(top_num_list_data[scheme], schemes_data[scheme]):
            records.append({"head_k": k, "metric": val, "Scheme": scheme})
    df = pd.DataFrame(records)
    palette = {
        "Ablation": "red",
        "Control": "black",
        "Random": "blue"
    }
    plt.figure(figsize=(6, 5), dpi=300)
    ax = sns.lineplot(
        data=df,
        x="head_k",
        y="metric",
        hue="Scheme",    # group by
        palette=palette,
        hue_order=["Ablation", "Control", "Random"],    # in legend
        errorbar="se"
    )
    plt.xlabel("Number of heads ablated", fontsize=15)
    plt.ylabel("P(Correct Answer)", fontsize=15)
    # plt.xlim(0, max(top_num_list))
    plt.ylim(0.0)
    # plt.yticks(np.arange(0.0, 1.1, 0.2), fontsize=13)
    # plt.xticks(fontsize=13)
    sns.despine(top=True, right=True)
    plt.legend(
        title="",
        loc="upper right",
        frameon=False,
        fontsize=14,
        handlelength=1.5
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"fig saved to {save_path}")


if __name__ == "__main__":
    seed = 1000
    ablation_dict = {
        "Llama-3.1-8B": {
            "Ablation": "ctrl_False_randctrl_False_10",
            "Control": "ctrl_True_randctrl_False_10",
            "Random": "ctrl_False_randctrl_True_5"
        }, 
        "Llama-3.1-70B": {
            "Ablation": "ctrl_False_randctrl_False_10",
            "Control": "ctrl_True_randctrl_False_10",
            "Random": "ctrl_False_randctrl_True_5"
        }
    }
    token_pos = {
        "symbol_abstraction_head": [5, 11],
        "symbolic_induction_head": [-1],
        "retrieval_head": [-1]
    }
    for model_type in ["Llama-3.1-8B", "Llama-3.1-70B"]:
        for head_type in ["symbol_abstraction_head", "symbolic_induction_head", "retrieval_head"]:
            for rule in ["ABA", "ABB"]:
                for ONLY_PREFILL in [True]:
                    schemes_data = {}
                    top_num_list_data = {}
                    common_path = f"results\\identity_rules\\ablation\\meta-llama\\{model_type}\\rule_{rule}\\z_seed_{seed}_shuffle_True_ONLY_PREFILL_{ONLY_PREFILL}\\{head_type}\\sample_num_20_gen_acc_0.9"
                    for scheme, ablation_type in ablation_dict[model_type].items():
                        metric_file = f"{common_path}\\{ablation_type}\\token_pos_{token_pos[head_type]}\\real_ans_prob_patched.pt"
                        real_ans_prob_patched = torch.load(metric_file, map_location="cpu", weights_only=True)
                        schemes_data[scheme] = real_ans_prob_patched.mean(dim=0).cpu().numpy()

                        top_num_list_file = f"{common_path}\\{ablation_type}\\token_pos_{token_pos[head_type]}\\top_num_list.npy"
                        top_num_list_data[scheme] = np.load(top_num_list_file)

                    save_path = f"{common_path}\\Ablation_{model_type}_{head_type}_{rule}_{ONLY_PREFILL}.png"
                    
                    plot_curve(schemes_data, top_num_list_data, save_path)
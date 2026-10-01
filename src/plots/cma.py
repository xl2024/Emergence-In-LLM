import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"] 
HF_HOME = os.environ["HF_HOME"]

from src.utils.utils import HEAD_ACTIVATIONS, get_model_id_family
from src.utils.tools import has_prepend_bos
from src.plots.utils import apply_fwer_threshold


def plot(result_array, save_folder, metric_name="Test Accuracy", num_for_one_rule=None, xlabel_name="Head Index", ylabel_name="Layer Index", pos_labels=None, xpos_labels=None, ypos_labels=None, invert_y_axis=False, fig_w=30, fig_h = 22, annot_labels=None, colormap="viridis"):
    annot_font_size = 8    
    colorbar_font_size = 24
    linewidth = 5
    orig_rcParams = plt.rcParams.copy()
    plt.rcParams.update({
    # 'font.size': 12,           # default font size
    'xtick.labelsize': 18,     # xtick label size
    'ytick.labelsize': 18,     # ytick label size
    'axes.labelsize': 40,      # xlabel, ylabel fontsize
    'axes.titlesize': 40,      # title fontsize
    # 'cbar.labelsize': 12       # colorbar fontsize
    })
    fig = plt.figure(figsize=(fig_w, fig_h))
    # if j_ == 0:
    #     sns.heatmap(heatmap_dict[act].cpu().numpy(), annot=True, fmt=".2f", cbar=True, cmap = "RdYlBu_r", vmin=-1.0, vmax=1.0, annot_kws={"size": annot_font_size})
    # else:
    
    if annot_labels is None:
        ax = sns.heatmap(result_array, annot=True, fmt=".2f", cbar=True, cmap = colormap, annot_kws={"size": annot_font_size}, mask=(np.isnan(result_array))) ##, mask=(result_array == -1))
    else:
        ax = sns.heatmap(result_array, annot=annot_labels, fmt="", cbar=True, cmap = colormap, annot_kws={"size": annot_font_size}, mask=(np.isnan(result_array)))
    remark = ""
    if invert_y_axis:
        ax.invert_yaxis()
        remark = "_invert_y_axis"

    if num_for_one_rule is not None:
        assert not invert_y_axis
        plt.axvline(x=num_for_one_rule, color='black', linewidth=linewidth)
        plt.axhline(y=num_for_one_rule, color='black', linewidth=linewidth)

    colorbar = fig.axes[-1] 
    colorbar.tick_params(labelsize=colorbar_font_size)
    os.makedirs(save_folder, exist_ok=True)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)

    xpos_labels = xpos_labels if xpos_labels is not None else pos_labels
    ypos_labels = ypos_labels if ypos_labels is not None else pos_labels

    if xpos_labels is not None:
        # assert not invert_y_axis
        plt.xticks(np.arange(len(xpos_labels)) + 0.5, xpos_labels)
    
    if ypos_labels is not None:
        assert not invert_y_axis
        plt.yticks(np.arange(len(ypos_labels)) + 0.5, ypos_labels)

    plt.title(f"{metric_name}")
    plt.savefig(os.path.join(save_folder, f"{metric_name}{remark}_heatmap.png"))
    plt.close(fig)

    plt.rcParams.update(orig_rcParams)
    return

        
if __name__ == "__main__":
    # update scores_file, sub_save_folder, base_rule, exp_rule, activation_name, model_type accordingly
    scores_file = "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\token_context\\base_rule_ABB_exp_rule_ABB\\attn_out_seed_0_shuffle_True\\logit\\sample_num_20_gen_acc_0.9\\patch_mlp_out_True\\causal_scores.pt"
    
    if os.path.exists(scores_file):
        filtered_array = apply_fwer_threshold(scores_file)
        save_folder = "outputs/cma"
        os.makedirs(save_folder, exist_ok=True)
        sub_save_folder = os.path.join(save_folder, "Llama-3.1-70B\\token_context\\base_rule_ABB_exp_rule_ABB\\attn_out_seed_0_shuffle_True")
        base_rule = "ABB"
        exp_rule = "ABB"
        activation_name = "attn_out"
        model_type = "Llama-3.1-70B"

        model_id, model_family =  get_model_id_family(model_type)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN) 
        prepend_bos = has_prepend_bos(tokenizer)
        pos_label_dict = {
            "ABA": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "A1", "\\n", "A2", "^", "B2", "^", "A2", "\\n", "A3", "^", "B3", "^"],
            "ABB": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "B1", "\\n", "A2", "^", "B2", "^", "B2", "\\n", "A3", "^", "B3", "^"],
            "Both": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "A1/B1", "\\n", "A2", "^", "B2", "^", "A2/B2", "\\n", "A3", "^", "B3", "^"],
        } ## used as axis labels of the plotted figures
        
        if activation_name in HEAD_ACTIVATIONS:
            xpos_labels = None
            xlabel_name = "Head Index"

        elif "resid" in activation_name or "out" in activation_name:
            xpos_labels = pos_label_dict["Both"]
            xlabel_name = "Token Position"

        os.makedirs(sub_save_folder, exist_ok=True)

        plot(filtered_array, sub_save_folder, metric_name=f"Score_Patching_{base_rule}_to_{exp_rule}", xpos_labels=xpos_labels, xlabel_name=xlabel_name, invert_y_axis=True)
        print("Success!")
    else:
        print(f"Could not find file: {scores_file}")
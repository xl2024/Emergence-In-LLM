import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from transformers import AutoTokenizer
from nnsight import LanguageModel

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"] 
HF_HOME = os.environ["HF_HOME"]

from src.utils.utils import LINE_SEP, get_model_id_family
from src.utils.tools import has_prepend_bos
from src.plots.utils import apply_fwer_threshold


def load_prompts(prompts_path, checkpoints_folder):
    with open(prompts_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    raw_prompts = [p.strip() for p in content.split(LINE_SEP) if p.strip()]
    search_pattern = os.path.join(checkpoints_folder, "prompt_*_final.pt")
    pt_filepaths = glob.glob(search_pattern)
    pt_filepaths.sort(key=lambda filepath: int(re.search(r'prompt_(\d+)_final\.pt', filepath).group(1)))
    prompts = []
    for pt_filepath in pt_filepaths:
        target_index = int(re.search(r'prompt_(\d+)_final\.pt', pt_filepath).group(1))
        prompts.append(raw_prompts[target_index])
            
    return prompts


def get_attentions(model, prompts):
    with model.trace(prompts, output_attentions=True, remote=True):
        # shape: [batch, head, seq_len, seq_len] x layer
        saved_attentions = model.output.attentions.save()

    attention_tensor = torch.stack([a for a in saved_attentions], dim=1)
    return attention_tensor

def compute_weighted_attention(attention_tensor, cma_scores):
    weights = torch.nan_to_num(torch.tensor(cma_scores), nan=0.0)
    weights = torch.clamp(weights, min=0.0)
    # New shape: [1, num_layers, num_heads, 1, 1]
    weights_expanded = weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    weighted_sum = (attention_tensor * weights_expanded).sum(dim=(1, 2))
    total_weight = weights.sum()
    if total_weight == 0:
        raise ValueError("Total weight is 0. No significant heads were found in cma_scores.")
    weighted_attn = weighted_sum / total_weight
    final_pattern = weighted_attn.mean(dim=0)
    return final_pattern.cpu().numpy()


def plot_attention_heatmap(matrix, labels, A1_index, save_folder, metric_name, red_lines_y=None):
    matrix = matrix[A1_index:, A1_index:]
    assert matrix.shape == (len(labels), len(labels)), f"shape mismatch: {matrix.shape} to {len(labels)}^2"
    plt.figure(figsize=(10, 8))
    
    ax = sns.heatmap(
        matrix, 
        cmap="viridis", 
        vmin=0.0, 
        # vmax=np.percentile(matrix, 99), # Cap outliers in the color scale
        cbar_kws={'label': 'Attention Score'}
    )
    ax.invert_yaxis()
    plt.xticks(np.arange(len(labels)) + 0.5, labels, fontsize=12)
    plt.yticks(np.arange(len(labels)) + 0.5, labels, fontsize=12, rotation=90)
    plt.xlabel("Keys", fontsize=16)
    plt.ylabel("Queries", fontsize=16)
    
    if red_lines_y:
        for y_idx in red_lines_y:
            ax.axhline(y_idx, color='red', linestyle='--', linewidth=2)
            ax.axhline(y_idx + 1, color='red', linestyle='--', linewidth=2)
            
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, f"{metric_name}.png"), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    rule="ABA"
    model_type = "Llama-3.1-70B"
    model_id, model_family =  get_model_id_family(model_type)
    model = LanguageModel(
        model_id, attn_implementation="eager", token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    common_path = "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\abstract_context\\base_rule_ABA_exp_rule_ABB\\z_seed_0_shuffle_True_pos_4_10"
    prompts_path = f"{common_path}\\base_input_prompts_1000.txt"
    checkpoints_folder = f"{common_path}\\checkpoints"
    prompts = load_prompts(prompts_path, checkpoints_folder)

    scores_file = f"{common_path}\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[5, 11]\\causal_scores.pt"
    cma_scores = apply_fwer_threshold(scores_file)

    attention_tensor = get_attentions(model, prompts)
    final_matrix = compute_weighted_attention(attention_tensor, cma_scores)
    
    save_folder = "outputs/attn"
    os.makedirs(save_folder, exist_ok=True)
    sub_save_folder = os.path.join(save_folder, "Llama-3.1-70B\\abstract_context\\base_rule_ABA_exp_rule_ABB\\z_seed_0_shuffle_True_pos_4_10")
    prepend_bos = has_prepend_bos(tokenizer)
    pos_label_dict = {
        "ABA": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "A1", "\\n", "A2", "^", "B2", "^", "A2", "\\n", "A3", "^", "B3", "^"],
        "ABB": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "B1", "\\n", "A2", "^", "B2", "^", "B2", "\\n", "A3", "^", "B3", "^"],
        "Both": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "A1/B1", "\\n", "A2", "^", "B2", "^", "A2/B2", "\\n", "A3", "^", "B3", "^"],
    }
    # Slice off the [BOS] token (Row 0 and Col 0)
    A1_index = 1 if prepend_bos else 0
    labels = pos_label_dict[rule][A1_index:]
    plot_attention_heatmap(
        matrix=final_matrix,
        labels=labels,
        A1_index=A1_index,
        save_folder=sub_save_folder,
        metric_name=f"Attention_Analysis_{rule}",
        red_lines_y=[4, 10]    # 15 for last token pos
    )
from dotenv import load_dotenv

load_dotenv()

# reuse original implementations
from reference.LLMSymbMech.utils import *

from src.plots.utils import apply_fwer_threshold
from src.plots.attn import load_prompts as attn_load_prompts


head_dict = {
    "Llama-3.1-70B": {
        "ABA": {
            "symbol_abstraction_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\abstract_context\\base_rule_ABA_exp_rule_ABB\\z_seed_0_shuffle_True_pos_4_10\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[5, 11]\\causal_scores.pt",
            "symbolic_induction_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\abstract_context\\base_rule_ABA_exp_rule_ABB\\z_seed_0_shuffle_True\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[-1]\\causal_scores.pt",
            "retrieval_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\token_context\\base_rule_ABA_exp_rule_ABA\\z_seed_0_shuffle_True\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[-1]\\causal_scores.pt"
        }, 
        "ABB": {
            "symbol_abstraction_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\abstract_context\\base_rule_ABB_exp_rule_ABA\\z_seed_0_shuffle_True_pos_4_10\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[5, 11]\\causal_scores.pt",
            "symbolic_induction_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\abstract_context\\base_rule_ABB_exp_rule_ABA\\z_seed_0_shuffle_True\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[-1]\\causal_scores.pt",
            "retrieval_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\token_context\\base_rule_ABB_exp_rule_ABB\\z_seed_0_shuffle_True\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[-1]\\causal_scores.pt"
        }, 
    }, 
    "Llama-3.1-8B": {
        "ABA": {
            "symbol_abstraction_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\abstract_context\\base_rule_ABA_exp_rule_ABB\\z_seed_0_shuffle_True_pos_4_10\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[5, 11]\\causal_scores.pt",
            "symbolic_induction_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\abstract_context\\base_rule_ABA_exp_rule_ABB\\z_seed_0_shuffle_True\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[-1]\\causal_scores.pt",
            "retrieval_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\token_context\\base_rule_ABA_exp_rule_ABA\\z_seed_0_shuffle_True\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[-1]\\causal_scores.pt"
        }, 
        "ABB": {
            "symbol_abstraction_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\abstract_context\\base_rule_ABB_exp_rule_ABA\\z_seed_0_shuffle_True_pos_4_10\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[5, 11]\\causal_scores.pt",
            "symbolic_induction_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\abstract_context\\base_rule_ABB_exp_rule_ABA\\z_seed_0_shuffle_True\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[-1]\\causal_scores.pt",
            "retrieval_head": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\token_context\\base_rule_ABB_exp_rule_ABB\\z_seed_0_shuffle_True\\logit\\sample_num_20_gen_acc_0.9\\group_heads_False\\token_pos_[-1]\\causal_scores.pt"
        }, 
    }
}

avg_head_dict = {
    "Llama-3.1-70B": {
        "symbol_abstraction_head": "datasets/cma_scores/llama31_70B/symbol_abstraction_head/causal_scores_avg.pt",
        "symbolic_induction_head": "datasets/cma_scores/llama31_70B/symbolic_induction_head/causal_scores_avg.pt", 
        "retrieval_head": "datasets/cma_scores/llama31_70B/retrieval_head/causal_scores_avg.pt"
    }, 
    "Llama-3.1-8B": {
        "symbol_abstraction_head": "datasets/cma_scores/llama31_8B/symbol_abstraction_head/causal_scores_avg.pt",
        "symbolic_induction_head": "datasets/cma_scores/llama31_8B/symbolic_induction_head/causal_scores_avg.pt", 
        "retrieval_head": "datasets/cma_scores/llama31_8B/retrieval_head/causal_scores_avg.pt"
    }
}

significant_head_dict = {
    "Llama-3.1-70B": {
        "symbol_abstraction_head": "datasets/cma_scores/llama31_70B/symbol_abstraction_head/significant_heads.pt",
        "symbolic_induction_head": "datasets/cma_scores/llama31_70B/symbolic_induction_head/significant_heads.pt", 
        "retrieval_head": "datasets/cma_scores/llama31_70B/retrieval_head/significant_heads.pt"
    }, 
    "Llama-3.1-8B": {
        "symbol_abstraction_head": "datasets/cma_scores/llama31_8B/symbol_abstraction_head/significant_heads.pt",
        "symbolic_induction_head": "datasets/cma_scores/llama31_8B/symbolic_induction_head/significant_heads.pt", 
        "retrieval_head": "datasets/cma_scores/llama31_8B/retrieval_head/significant_heads.pt"
    }
}

prompts_dict = {
    "Llama-3.1-70B": {
        "symbol_abstraction_head": {
            "ABA": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\abstract_context\\base_rule_ABA_exp_rule_ABB\\z_seed_0_shuffle_True_pos_4_10",
            "ABB": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\abstract_context\\base_rule_ABB_exp_rule_ABA\\z_seed_0_shuffle_True_pos_4_10"
        }, 
        "symbolic_induction_head": {
            "ABA": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\abstract_context\\base_rule_ABA_exp_rule_ABB\\z_seed_0_shuffle_True",
            "ABB": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\abstract_context\\base_rule_ABB_exp_rule_ABA\\z_seed_0_shuffle_True"
        }, 
        "retrieval_head": {
            "ABA": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\token_context\\base_rule_ABA_exp_rule_ABA\\z_seed_0_shuffle_True",
            "ABB": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-70B\\token_context\\base_rule_ABB_exp_rule_ABB\\z_seed_0_shuffle_True"
        }
    }, 
    "Llama-3.1-8B": {
        "symbol_abstraction_head": {
            "ABA": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\abstract_context\\base_rule_ABA_exp_rule_ABB\\z_seed_0_shuffle_True_pos_4_10",
            "ABB": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\abstract_context\\base_rule_ABB_exp_rule_ABA\\z_seed_0_shuffle_True_pos_4_10"
        }, 
        "symbolic_induction_head": {
            "ABA": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\abstract_context\\base_rule_ABA_exp_rule_ABB\\z_seed_0_shuffle_True",
            "ABB": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\abstract_context\\base_rule_ABB_exp_rule_ABA\\z_seed_0_shuffle_True"
        }, 
        "retrieval_head": {
            "ABA": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\token_context\\base_rule_ABA_exp_rule_ABA\\z_seed_0_shuffle_True",
            "ABB": "results\\identity_rules\\cma\\meta-llama\\Llama-3.1-8B\\token_context\\base_rule_ABB_exp_rule_ABB\\z_seed_0_shuffle_True"
        }, 
    }
}

def rsa_get_head_list(model_type, head_type, head_dict=head_dict, avg_head_dict=avg_head_dict, significant_head_dict=significant_head_dict):
    """
    rewrites `get_head_list` to redirect cma score files for both Llama-3.1-70B and Llama-3.1-8B models. 
    """
    avg_save_path = avg_head_dict[model_type][head_type]
    if not os.path.exists(avg_save_path):
        file_name_ABA = head_dict[model_type]["ABA"][head_type]
        score_ABA = torch.load(file_name_ABA, map_location="cpu")
        file_name_ABB = head_dict[model_type]["ABB"][head_type]
        score_ABB = torch.load(file_name_ABB, map_location="cpu")
        score_avg = (score_ABA + score_ABB) / 2.0
        torch.save(score_avg, avg_save_path)

    sig_head_path = significant_head_dict[model_type][head_type]
    if not os.path.exists(sig_head_path):
        score_sig = apply_fwer_threshold(avg_save_path, save_path=sig_head_path)

    return get_head_list(head_type, head_dict=significant_head_dict[model_type])

def get_prompts_for_cma(model_type, head_type, rule):
    common_path = prompts_dict[model_type][head_type][rule]
    prompts_path = f"{common_path}\\base_input_prompts_1000.txt"
    checkpoints_folder = f"{common_path}\\checkpoints"
    prompts = attn_load_prompts(prompts_path, checkpoints_folder)

    export_folder = os.path.join("datasets", "cma_scores", model_type, head_type)
    os.makedirs(export_folder, exist_ok=True)
    export_path = os.path.join(export_folder, f"base_{rule}_prompt_for_cma_20.txt")
    with open(export_path, "w") as f:
        for p in prompts:
            f.write(f"{LINE_SEP}" + p + "\n")
    print(f"[info] Prompts for cma({model_type} - {head_type} - {rule}) saved to {export_path}")
    return export_path
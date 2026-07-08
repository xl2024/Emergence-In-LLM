import argparse
import torch

def _get_args(type_str: str):
    parser = argparse.ArgumentParser()
    # default_device = "cuda" if torch.cuda.is_available() else "cpu"    # "--device"
    
    if type_str == 'rsa':
        parser.add_argument("--seed", type=int, default=0, nargs="?", help="random seed")

        ## parameters for the model
        parser.add_argument("--model_type", type=str, default="Llama-3.1-70B", help="model type")   
        parser.add_argument("--device_map", type=str, default="cpu", help="device map")
        parser.add_argument("--device", type=str, default="cuda", help="device")
        parser.add_argument("--n_devices", type=int, default=1, help="number of devices")
        parser.add_argument("--fold_ln", action="store_true", help="fold layer normalization")

        ## parameters for the prompts
        parser.add_argument("--base_rule", type=str, default="ABA", help="base rule")
        parser.add_argument("--in_context_example_num", type=int, default=2, help="in-context example number")
        parser.add_argument(
            "--token_set_file", type=str, default="datasets/llama31_70B_correct_common_tokens_0.9_1378.txt", #None, 
            help="the file contains token sets that formed sequences of rules ABA and ABB, with the model predicting both correctly, each line is a set of (2*N) tokens separated by space, 'A_1 B_1 ... A_N B_N', e.g., 'la li te to hi ha' (N-1) in-context examples."
        )
        parser.add_argument("--add_swap_1_2_question", action="store_true")
        parser.add_argument("--prompt_num", type=int, default=None, help="prompt number")
        parser.add_argument("--sep_symbol", type=str, default="^", help="separator symbol")
        parser.add_argument("--do_shuffle", action="store_true")


        ## parameters for rsa
        parser.add_argument("--cmp_with_abstract", action="store_true", help="compare with abstract variables")
        parser.add_argument("--cmp_with_token_id", action="store_true", help="compare with token id")
        parser.add_argument("--sel_pos_list", nargs="+", help="selected position list", type=int)
        parser.add_argument("--act_list", nargs="+", default=["z"], help="the list of activations to be recorded")
        parser.add_argument("--use_attn_result", action="store_true", help="whether to split w_o into sub-blocks and apply each subblock on corresponding attention head's output") 
        parser.add_argument("--low_sim", type=float, default=0.0, help="low similarity")
        parser.add_argument("--high_sim", type=float, default=1.0, help="high similarity")

        parser.add_argument("--only_for_significant_heads", action="store_true", help="conduct rsa only for significant heads")
        parser.add_argument("--head_type", type=str, default="symbol_abstraction_head", choices=["symbol_abstraction_head", "symbolic_induction_head", "retrieval_head"], help="the head type to be analyzed")

        parser.add_argument("--start_layer_idx", type=int, default=0, help="start layer index")
        parser.add_argument("--end_layer_idx", type=int, default=None, help="end layer index")
        parser.add_argument("--start_head_idx", type=int, default=0, help="start head index")
        parser.add_argument("--end_head_idx", type=int, default=None, help="end head index")
        parser.add_argument("--transpose", action="store_true", help="transpose the feature matrix before calculating similarity")
        parser.add_argument("--plot_hand_code", action="store_true", help="plot hand-coded")
        parser.add_argument("--plot_similarity", action="store_true", help="plot similarity")
        parser.add_argument("--plot_rsa", action="store_true", help="plot RSA")
        parser.add_argument("--save_similarity", action="store_true", help="save similarity")
        parser.add_argument("--verbose", action="store_true", help="verbose")
        parser.add_argument("--log_dir", type=str, default="results/identity_rules/rsa", help="log directory")
   
    elif type_str == 'cma':
        parser.add_argument("--seed", type=int, default=0, nargs="?", help="random seed")

        ### load the model
        parser.add_argument("--model_type", type=str, default="Llama-3.1-70B", help="model type")
        parser.add_argument("--device_map", type=str, default="cpu", help="device map (cpu or auto)")
        parser.add_argument("--device", type=str, default="cuda", help="device")
        parser.add_argument("--n_devices", type=int, default=2, help="number of devices. please set to 2 for Llama-3.1-70B and Qwen2,5-72B while 1 for others")
        parser.add_argument("--fold_ln", action="store_true", help="fold layer normalization weights into the weights of the preceeding layer")

        #### build the prompt (context) pairs 
        parser.add_argument("--context_type", type=str, default="abstract", help="whether to build abstract context (symbol abstraction heads/ symbolic induction heads) or token context (retrieval heads)", choices=["abstract", "token"])
        parser.add_argument("--base_rule", type=str, default="ABA", help="rule for base prompt in prompt pairs", choices=["ABA", "ABB"])
        parser.add_argument("--prompt_num", type=int, default=None, help="the number of prompt pairs to be generated")
        parser.add_argument("--in_context_example_num", type=int, default=2, help="in-context example number")
        parser.add_argument("--sep_symbol", type=str, default="^", help="separator symbol")
        parser.add_argument("--do_shuffle", action="store_true", help="whether to shuffle the vocabulary list for generating prompts")

        ### parameters for generation
        parser.add_argument("--eos_token", type=str, default=None, help="eos token") # will set as "\n" if not specified
        parser.add_argument("--max_new_tokens", type=int, default=10, help="max new tokens")
        parser.add_argument("--sample_size", type=int, default=4, help="sample size per prompt during generation")
        parser.add_argument("--load_generation_config", action="store_true")
        parser.add_argument("--generation_config_name", type=str, default=None, help="generation config file name, used when loading generation config from file")

        ## parameters for patching the activations
        parser.add_argument("--activation_name", default="z", type=str, help="activation name, 'z' for individual attention head output or 'attn_out' for the whole attention block output (after aggregating all attention heads through w_o) in each layer")
        parser.add_argument("--patch_mlp_out", action="store_true", help="[necessary and only applicable when activation_name == 'attn_out'] whether to patch the MLP output so that we can replace all the information added into the residual stream in each layer, i.e., attention block output and MLP output")
        parser.add_argument("--token_pos_list", nargs="+", default=[-1], type=int, help="token positions where we do patching")
        parser.add_argument("--min_valid_sample_num", default=-1, type=int, help="minimum number of valid prompt pairs on which model made correct predictions, used for causal mediation score calculation")
        parser.add_argument("--eval_metric", default="gen_acc", help="evaluation metric for filtering prompts", choices=["gen_acc", "ans_prob"])
        parser.add_argument("--low_prob_threshold", default=0.9, type=float, help="threshold for filtering low probability/accuracy samples")
        parser.add_argument("--ungroup_grouped_query_attention", action="store_true", help="ungroup the grouped query attention")

        parser.add_argument("--log_dir", type=str, default="results/identity_rules/cma", help="log directory")
        parser.add_argument("--generate", action="store_true", help="whether to measure the causal effects by actually generating the responses")
        parser.add_argument("--group_heads", action="store_true", help="whether to patch the activations of grouped heads which share the keys/values (GQA) at the same time")
        parser.add_argument("--verbose", action="store_true", help="verbose")
    
    elif type_str == 'eval':
        parser.add_argument("--seed", type=int, default=0, help="random seed")

        parser.add_argument("--rule", type=str, default="ABA", help="rule type", choices=["ABA", "ABB"])
        parser.add_argument("--model_type", type=str, default="Llama-3.1-70B", help="model type")
        parser.add_argument("--in_context_example_num", type=int, default=2, help="in-context example number, n-shot")
        parser.add_argument("--prompt_num", type=int, default=None, help="prompt number")

        parser.add_argument("--sep_symbol", type=str, default="^", help="separator symbol")
        parser.add_argument("--eos_token", type=str, default=None, help="add an eos token, will set as '\n' if not specified")
        parser.add_argument("--max_new_tokens", type=int, default=10, help="max new tokens")
        parser.add_argument("--device_map", type=str, default="auto", help="device map")
        parser.add_argument("--prompt_file", type=str, default=None, help="prompt file")
        parser.add_argument("--sample_size", type=int, default=4, help="sample size per prompt, each prompt will be sampled for N times. If the greedy sampling is used, N does not matter")
        parser.add_argument("--acc_threshold", type=float, default=0.9, help="accuracy threshold above which the prompt will be stored as a correct generation prompt")
        parser.add_argument("--save_generation_config", action="store_true", help="save generation config")
        parser.add_argument("--load_generation_config", action="store_true", help="load generation config from file")
        parser.add_argument("--generation_config_name", type=str, default=None, help="generation config file name, used when loading generation config from file")

        parser.add_argument("--log_dir", type=str, default="results/identity_rules/eval", help="log directory")
        parser.add_argument("--verbose", action="store_true", help="verbose")
        parser.add_argument("--sample_remark", type=str, default="", help="additional remark for the sampling process, used for folder naming")

    elif type_str == 'ablation':
        parser.add_argument("--seed", type=int, default=0, nargs="?", help="random seed")
        parser.add_argument("--rule", type=str, default="ABA", help="rule type")
        parser.add_argument("--head_type", type=str, default="symbol_abstraction_head", help="head type")

        ### load the model
        parser.add_argument("--model_type", type=str, default="Llama-3.1-70B", help="model type")
        parser.add_argument("--device_map", type=str, default="cpu", help="device map (cpu or auto)")
        parser.add_argument("--device", type=str, default="cuda", help="device")
        parser.add_argument("--n_devices", type=int, default=2, help="number of devices. please set to 2 for Llama-3.1-70B and Qwen2,5-72B while 1 for others")
        parser.add_argument("--fold_ln", action="store_true", help="fold layer normalization weights into the weights of the preceeding layer")

        #### build the prompt
        parser.add_argument("--prompt_num", type=int, default=None, help="the number of prompts, which will be used to filter out the correct prompts for ablation")
        parser.add_argument("--in_context_example_num", type=int, default=2, help="in-context example number")
        parser.add_argument("--sep_symbol", type=str, default="^", help="separator symbol")
        parser.add_argument("--do_shuffle", action="store_true")
        parser.add_argument(
            "--token_set_file", type=str, default="datasets/llama31_70B_correct_common_tokens_0.9_1378.txt", #None, 
            help="the file of token sets which form sequences of rules ABA and ABB on which the model could make correct predictions, each line is a set of (2*N) tokens separated by space, 'A_1 B_1 ... A_N B_N', e.g., 'la li te to hi ha' (N-1) in-context examples."
        )
        
        ### parameters for generation
        parser.add_argument("--eos_token", type=str, default=None, help="eos token") # will set as "\n" if not specified
        parser.add_argument("--max_new_tokens", type=int, default=10, help="max new tokens")
        parser.add_argument("--sample_size", type=int, default=4, help="sample size per prompt during generation")
        parser.add_argument("--load_generation_config", action="store_true")
        parser.add_argument("--generation_config_name", type=str, default=None, help="generation config file name, used when loading generation config from file")


        ### parameters for ablation
        parser.add_argument("--activation_name", default="z", type=str)
        parser.add_argument("--token_pos_list", nargs="+", default=[-1], type=int)
        parser.add_argument("--control", action="store_true", help="control group, ablate the same number of heads which have the lowest causal mediation scores in each layer")
        parser.add_argument("--random_control", action="store_true", help="random control group, randomly ablate the same number of heads among all the attention heads")
        parser.add_argument("--random_times_per_step", type=int, default=10, help="the number of random sampling trials at each cumulative step, only used for random control")
        parser.add_argument("--min_valid_sample_num", default=-1, type=int, help="minimum number of valid prompt pairs on which model made correct predictions, used for causal mediation score calculation")

        parser.add_argument("--eval_metric", type=str, default="gen_acc", choices=["gen_acc", "ans_prob"], help="evaluation metric for the model performance, gen_acc: generation accuracy, ans_prob: the probability of the correct answer")
        parser.add_argument("--low_prob_threshold", default=0.9, type=float)
        # add ungroup_grouped_query_attention for k/v ablation
        parser.add_argument("--ungroup_grouped_query_attention", action="store_true", help="ungroup the grouped query attention")
        parser.add_argument("--start_head_idx", type=int, default=0, help="the index of the first head to ablate")
        parser.add_argument("--end_head_idx", type=int, default=None, help="the index of the last head to ablate")
        parser.add_argument("--step_size", type=int, default=1, help="step size for ablation")
        parser.add_argument("--prompt_file_for_causal_scores_exp", type=str, default=None, help="prompts which were used for calculating the causal scores, these prompts will not be used for ablation")

        parser.add_argument("--adaptive_step_size", action="store_true", help="adaptive step size for ablation")
        parser.add_argument("--patch_all_token_pos", action="store_true", help="patch all token positions")
        parser.add_argument("--log_dir", type=str, default="results/identity_rules/ablation", help="log directory")
        parser.add_argument("--verbose", action="store_true", help="verbose")
    
    else:
        raise ValueError(f"Unknown type: {type_str}")
    
    args = parser.parse_args()

    return args
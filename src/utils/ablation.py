#####################################
import os 

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"] 
HF_HOME = os.environ["HF_HOME"]
#####################################

import logging

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from src.utils.utils import LINE_SEP, plot, HEAD_ACTIVATIONS, avg_head_dict, set_seed, vocab_dict, get_model_id_family, get_prompts_for_cma
from src.utils.args import _get_args
from src.utils.tools import generate_prompts, _resolve_layer_path, get_num_hidden_layers, _resolve_text_model_dims, generate_response_eval, get_layer_paths, has_prepend_bos, get_eos_token_id, get_token_sets, get_generation_kwargs
from src.utils.ungroup import check_remote_kv_metadata

from transformers import AutoTokenizer, GenerationConfig
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict 
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
from nnsight import LanguageModel, ndif

ndif.register("src.utils.tools")
ndif.register("src.utils.ungroup")

# Model-level head ablation effect: mask during both prefill and decoding
# Prompt-position causal effect: mask only when the full prompt positions exist; skip decode steps (what the paper did)
ONLY_PREFILL = True

def get_head_list(model_type, head_type, score_dict=avg_head_dict):
    ## Load the causal mediation scores of each attention head for a specified head type 
    file = score_dict[model_type][head_type]
    avg_score = torch.load(file, map_location="cpu", weights_only=True)
    avg_score = avg_score.mean(dim=0).cpu().numpy()

    n_layers = avg_score.shape[0]
    n_heads = avg_score.shape[1]

    layer_rank_head_list = np.argsort(avg_score, axis=1) ## rank the heads in each layer based on causal scores, from low to high
    indices = np.argsort(avg_score.reshape(-1))[::-1] ## rank all the heads from high to low
    layer_indices = indices // n_heads
    head_indices = indices % n_heads
    sorted_head_list = list(zip(layer_indices, head_indices)) ## get the (layer_idx, head_idx) for each head

    return sorted_head_list, layer_rank_head_list 

def get_args():
    return _get_args('ablation')

def ablate_head(
        model, 
        ranked_head_list, ## ranked head list from top to bottom
        input_ids, 
        ans, ## the correct answer
        tokenizer,
        layer_paths,
        prompt=None,
        eos_token_id=None,
        token_pos=[-1], 
        layer_ranked_heads=None, ## from bottom to top in each layer
        total_layers=80, total_heads=64, 
        activation_name="z",
        start_head_idx=0, 
        end_head_idx=None,
        step_size=1,
        control=False,
        random_control=False,
        random_times_per_step=5,
        seed = 42, 
        adaptive_step_size=False,
        patch_all_token_pos=False, 
        eval_metric="gen_acc",
        checkpoint_path=None,
        save_interval=100,
        **generation_kwargs
    ):
    
    assert not random_control or not control, "random control and control cannot be both True"
    assert input_ids is not None
    token_len = input_ids.shape[-1]
    assert token_len > 1, "token length should be larger than 1"
    assert ranked_head_list is not None
    if patch_all_token_pos:
        token_pos = range(token_len) 
        logger.info(f"patch all token positions: {token_pos}")

    ans_prob_ablated_all = []
    top_num_list = []
    i_ = start_head_idx

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ans_prob_ablated_all = ckpt["ans_prob_ablated_all"]
            top_num_list = ckpt["top_num_list"]
            i_ = ckpt["current_head_idx"]
            logger.info(f"Resumed ablation from heads top-{i_}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint, starting from {start_head_idx}: {e}")

    random_i_ = 0
    last_saved_i = i_
    end_head_idx = len(ranked_head_list) if end_head_idx is None else end_head_idx
    tqdm_bar = tqdm(total=end_head_idx - start_head_idx + 1)

    random_command = random.Random(seed)
    while i_ <= end_head_idx: 
        print(f"=================== head_idx: {i_}/{end_head_idx}")
        if random_control:
            full_head_set = [(i, j) for i in range(total_layers) for j in range(total_heads)]
            selected_heads = random_command.sample(full_head_set, i_)
            random_i_ += 1
        else:  
            selected_heads = ranked_head_list[:i_]
        selected_heads_dict = defaultdict(list)
        for layer_idx, head_idx in selected_heads:
            # if layer_idx < 32 and head_idx < 32:    # TODO
            selected_heads_dict[layer_idx].append(head_idx)
        
        top_num_list.append(len(selected_heads))

        def intervention_fn(model):
            head_dict = selected_heads_dict    # TODO
            assert head_dict is not None
            head_dict = dict(sorted(head_dict.items()))
            for layer_idx, head_indices in head_dict.items():
                if control:
                    ablate_heads = layer_ranked_heads[layer_idx][:len(head_indices)]
                else:
                    ablate_heads = head_indices

                if len(ablate_heads) == 0:
                    continue
                
                target_module = _resolve_layer_path(model, layer_paths[layer_idx][activation_name])
                
                if activation_name in ["z", "o"]:
                    activation = target_module.input
                else:
                    activation = target_module.output
                    
                s, h_dim = activation.shape[-2:]
                b = 1 if len(activation.shape) == 2 else activation.shape[0]
                
                assert s == token_len or s == 1, f"activation({activation_name}).shape: {activation.shape}, seq_len({s}) != token_len({token_len})"
                if s == 1:
                    if ONLY_PREFILL:
                        continue
                    else:
                        _token_pos = -1
                else:
                    _token_pos = token_pos
                activation = activation.view(b, s, total_heads, -1)
                for head_idx in ablate_heads:
                    activation[:, _token_pos, head_idx, :] = 0.0
                activation = activation.view(b, s, h_dim)
                        
                if activation_name in ["z", "o"]:
                    target_module.input = activation
                else:
                    target_module.output = activation

        # if eval_metric == "ans_prob":
        with torch.no_grad():
            with model.trace(input_ids, remote=True):
                intervention_fn(model)
                logits = model.lm_head.output.save()

        ans_id = tokenizer.encode(ans, add_special_tokens=False)[-1]
        ans_prob_ablated = torch.softmax(logits, dim=-1)[0,-1,ans_id].cpu()
        ans_prob_ablated_all.append(ans_prob_ablated)
        del logits
        # elif eval_metric == "gen_acc":    # [Note: too many no-eos and multi-token responses]
        #     correct_num, total_num, acc = generate_response_eval(
        #         model, input_ids, eos_token_id, tokenizer, ans, prompt,
        #         intervention_fn=intervention_fn,
        #         sample_size=generation_kwargs.pop("sample_size", 4),
        #         max_new_tokens=generation_kwargs.pop("max_new_tokens", 10), 
        #         **generation_kwargs
        #     )
        #     ans_prob_ablated_all.append(acc)

        if adaptive_step_size: ## [TODO] make it more general
            if not control and not random_control: ## only for the non-control group
                if i_ > 3500: ## after ablating more than 3500 heads, the probability of the correct answer will be very low, so we can increase the step size to speed up the ablation process
                    step_size = 8
            else:
                if len(ranked_head_list) - i_ < 500:
                    step_size = 4 #1

        torch.cuda.empty_cache()
        if random_control and random_i_ < random_times_per_step:
            continue
        else:
            i_ += step_size
            tqdm_bar.update(step_size)
            random_i_ = 0
            
            if checkpoint_path is not None and (i_ - last_saved_i >= save_interval):
                torch.save({
                    "ans_prob_ablated_all": ans_prob_ablated_all,
                    "top_num_list": top_num_list,
                    "current_head_idx": i_
                }, checkpoint_path)
                last_saved_i = i_
    
    tqdm_bar.close()
    return (
        torch.tensor(ans_prob_ablated_all),
        top_num_list
    )


def cumulative_ablation(
        model, 
        tokenizer, 
        prompt_list, 
        correct_ans_list, 
        head_type,
        activation_name, 
        save_folder, 
        layer_paths, 
        token_pos=[-1], 
        ranked_head_list=None,
        layer_ranked_heads=None,
        start_head_idx=0,
        end_head_idx=None,
        step_size=1,
        control=False,
        random_control=False,
        random_times_per_step=5,
        prompt_for_causal_scores=None,
        seed = 42,
        adaptive_step_size=False,
        eval_metric="gen_acc",
        min_valid_sample_num=-1, 
        low_prob_threshold=0.9, 
        eos_token_id=None, 
        patch_all_token_pos=False,
        **generation_kwargs
    ):
    """
    Conduct cumulative ablation on the model for a list of prompts.

    Args:
        model (HookedTransformer): The model to be ablated.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        prompt_list (list): List of prompts to be ablated.
        correct_ans_list (list): List of correct answers corresponding to the prompts.
        head_type (str): Type of the head to be ablated.
        activation_name (str): Name of the activation to be ablated.
        save_folder (str): Folder to save the results.
        token_pos (int or list): Position of the token to be ablated. If -1, the last token will be ablated.
        device (str): Device to run the model.
        ranked_head_list (list): List of heads ranked by causal mediation scoresf from top to bottom.
        layer_ranked_heads (dict): Dictionary of heads ranked by causal mediation scores from bottom to top in each layer.
        start_head_idx (int): Index of the first head to ablate
        end_head_idx (int): Index of the last head to ablate. If None, all heads will be ablated.
        step_size (int): Step size for ablation.
        control (bool): Whether to use control group for ablation.
        random_control (bool): Whether to use random control group for ablation.
        random_times_per_step (int): Number of random times per step for random control group.
        prompt_for_causal_scores (list): List of prompts which were used for causal scores calculation
        seed (int): Random seed for reproducibility.
        adaptive_step_size (bool): Whether to use ununiform step size for ablation.
        eval_metric (str): Evaluation metric to use for ablation, default is "gen_acc".
        min_valid_sample_num (int): Minimum number of valid samples required for ablation.
        low_prob_threshold (float): Low probability threshold for filtering prompts.
        eos_token_id (int): End-of-sequence token ID.
        patch_all_token_pos (bool): [Optional] Whether to patch all token positions.
        **generation_kwargs: Additional keyword arguments for generation.

    """
    assert activation_name in HEAD_ACTIVATIONS, f"activation_name {activation_name} not in {HEAD_ACTIVATIONS}"
    total_layers = get_num_hidden_layers(model)
    _, total_heads = _resolve_text_model_dims(model)
    real_ans_prob_patched_list = []
    valid_num = 0
    selected_prompt_list = []
    
    token_pos_tag = f"token_pos_{token_pos}" 
    if patch_all_token_pos:
        token_pos_tag = f"token_pos_all" 
    
    ablation_type = f"ctrl_{control}_randctrl_{random_control}_{random_times_per_step}"    # filename too long

    checkpoint_dir = os.path.join(save_folder, head_type, "checkpoints", ablation_type, token_pos_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for idx_, prompt in enumerate(prompt_list):
        print(f"===================== prompt {idx_+1}/{len(prompt_list)}")
        logger.info(f"Processing {idx_}th sample")
        ans = correct_ans_list[idx_]

        if prompt_for_causal_scores is not None and prompt in prompt_for_causal_scores:
            print(f"Skip the base prompt {prompt} which was used for calculating the causal scores")
            continue

        completed_path = os.path.join(checkpoint_dir, f"prompt_{idx_}_completed.pt")
        if os.path.exists(completed_path):
            logger.info(f"Prompt {idx_} already completed. Loading results from checkpoint.")
            results = torch.load(completed_path, map_location="cpu", weights_only=False)
            real_ans_prob_patched_list.append(results["real_ans_prob_patched_all"].float())
            top_num_list = results["top_num_list"]
            selected_prompt_list.append(prompt)
            valid_num += 1
            if min_valid_sample_num != -1 and valid_num >= min_valid_sample_num: 
                break
            continue

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        
        if eval_metric == "gen_acc":
            # evaluate the model performance through generation accuracy
            correct_num, total_num, acc = generate_response_eval(
                model, input_ids, eos_token_id, tokenizer, ans, prompt,
                sample_size=generation_kwargs.pop("sample_size", 4), 
                max_new_tokens=generation_kwargs.pop("max_new_tokens", 10), 
                **generation_kwargs
            )
            if acc < low_prob_threshold:
                logger.warning(f"low acc {acc} < {low_prob_threshold}")
                continue

        elif eval_metric == "ans_prob":
            with torch.no_grad():
                with model.trace(input_ids, remote=True):
                    logits = model.lm_head.output.save()
            ans_id = tokenizer.encode(ans, add_special_tokens=False)[-1]
            ans_real_prob = torch.softmax(logits, dim=-1)[0,-1,ans_id]

            if ans_real_prob < low_prob_threshold:
                logger.info(f"true ans {ans_real_prob} lower than {low_prob_threshold}")
                continue
        
        selected_prompt_list.append(prompt)
        checkpoint_path = os.path.join(checkpoint_dir, f"prompt_{idx_}_checkpoint.pt")
        results = ablate_head(
            model, 
            ranked_head_list,
            input_ids, 
            ans,
            tokenizer,
            layer_paths,
            prompt,
            eos_token_id,
            token_pos,
            layer_ranked_heads=layer_ranked_heads,
            total_layers=total_layers, 
            total_heads=total_heads,
            activation_name=activation_name,
            start_head_idx=start_head_idx,
            end_head_idx=end_head_idx,
            step_size=step_size,
            control=control,
            random_control=random_control,
            random_times_per_step=random_times_per_step,
            seed=seed,
            adaptive_step_size=adaptive_step_size,
            patch_all_token_pos=patch_all_token_pos,
            eval_metric=eval_metric,
            checkpoint_path=checkpoint_path,
            save_interval=100,
            **generation_kwargs
        )
        real_ans_prob_patched_all, top_num_list = results
        real_ans_prob_patched_list.append(real_ans_prob_patched_all.float())

        torch.save({
            "real_ans_prob_patched_all": real_ans_prob_patched_all,
            "top_num_list": top_num_list
        }, completed_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        valid_num += 1

        if min_valid_sample_num != -1 and valid_num >= min_valid_sample_num: ## run ablation on enough valid samples on which the model could generate the correct answers
            break

    if valid_num == 0:
        logger.warning("No samples to patch")
        return
    
    save_folder = os.path.join(save_folder, head_type, f"sample_num_{valid_num}_{eval_metric}_{low_prob_threshold}")
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, f"prompt_{valid_num}.txt"), "w") as f:
        for prompt in selected_prompt_list:
            f.write(f"{LINE_SEP}" + prompt + "\n")

    sub_save_folder = os.path.join(save_folder, ablation_type, token_pos_tag)
    os.makedirs(sub_save_folder, exist_ok=True)
    mean_real_ans_prob_patched = torch.stack(real_ans_prob_patched_list, dim=0).mean(dim=0)
    torch.save(torch.stack(real_ans_prob_patched_list, dim=0), os.path.join(sub_save_folder, "real_ans_prob_patched.pt")) 
    np.save(os.path.join(sub_save_folder, f"top_num_list.npy"), np.array(top_num_list))
    plot_curve(mean_real_ans_prob_patched.cpu().numpy(), top_num_list, sub_save_folder, metric_name=f"Cor Ans Prob")


def plot_curve(metric_list, top_num_list, save_folder, metric_name):

    df = pd.DataFrame({
        "head_k": top_num_list,
        "metric": metric_list
    })
    plt.figure()

    sns.lineplot(data=df, x="head_k", y="metric", errorbar="se")
    plt.xlabel("Number of Heads")
    plt.ylabel(metric_name)
    plt.title(f"Effect of Top-K Heads on {metric_name}")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{metric_name}_topk.png")
    plt.savefig(save_path)
    logger.info(f"fig saved to {save_path}")

def main(args):
    set_seed(args.seed)

    # get the model id and family
    model_id, model_family =  get_model_id_family(args.model_type) 
    vocab_file = vocab_dict[model_family] ## get the vocabulary file path
    logger.info(f"model type: {args.model_type}, model id: {model_id}, vocab file: {vocab_file}")

    #################################################################
    # 0. Load the model and tokenizer, 
    # and specify the generation config which will be used to filter out the correct prompts for CMA
    #################################################################
    logger.info(f"Loading {model_id} via nnsight...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    torch.set_grad_enabled(False)
    
    kwargs = {
        "sample_size": args.sample_size,
        "max_new_tokens": args.max_new_tokens
    }
    generation_kwargs = get_generation_kwargs(args, **kwargs)

    ## Load the model
    try:
        model = LanguageModel(model_id, device_map=args.device_map, dtype=torch.bfloat16, token=HF_TOKEN)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error("Failed to load the model. Please check the model id")
        return
    model_remark = ""
    if args.ungroup_grouped_query_attention:
        # Ungroup the grouped query attention, **NECESSARY** for CMA on keys/values, which are not done in the paper 
        logger.info("ungroup the keys/values which were grouped in the grouped query attention, this is necessary for CMA on keys/values (not done in the paper)")
        # model.set_ungroup_grouped_query_attention(True)
        logger.info(f"set use_past_kv_cache to False to avoid the error occurring in the generation with ungrouped keys/values")
        # generation_kwargs["use_past_kv_cache"] = False # set to False to avoid the error occurring in the generation with ungrouped keys/values
        generation_kwargs["use_cache"] = False
        model_remark += "_ungroup_gqa"

    eos_token_id = get_eos_token_id(tokenizer, args.eos_token)

    activation_name = args.activation_name
    layer_paths = get_layer_paths(model, [activation_name])
    ## create the result folder
    remark = f"{model_id}{model_remark}/rule_{args.rule}/{activation_name}_seed_{args.seed}_shuffle_{args.do_shuffle}_ONLY_PREFILL_{ONLY_PREFILL}"
    save_folder = os.path.join(args.log_dir, remark)
    os.makedirs(save_folder, exist_ok=True)
    logger.info(f"save_folder: {save_folder}")

    #################################################################
    #### 1. Build the prompt dataset ####
    #################################################################
    prompts_cache_path = os.path.join(save_folder, f"prompts_data_{args.prompt_num}.pt")
    if os.path.exists(prompts_cache_path):
        logger.info(f"Loading cached prompts from {prompts_cache_path}...")
        cached_data = torch.load(prompts_cache_path, map_location="cpu", weights_only=False)
        prompts = cached_data['prompts']
        correct_ans_list = cached_data['correct_ans_list']
    else:
        logger.info(f"generate prompts.... (SEP symbol: {args.sep_symbol})")
        assert args.prompt_num is not None
        token_sets = get_token_sets(args.token_set_file)

        prompts, correct_ans_list = generate_prompts(
            args, tokenizer, vocab_file, sep_symbol = args.sep_symbol, token_sets=token_sets, do_shuffle=args.do_shuffle, 
            base_rule=args.rule, return_format="others"
            )
        torch.save({
            'prompts': prompts,
            'correct_ans_list': correct_ans_list
        }, prompts_cache_path)
    
    with open(os.path.join(save_folder, f"input_prompts_{len(prompts)}.txt"), "w") as f:
        for p in prompts:
            f.write(f"{LINE_SEP}" + p + "\n")
    
    ## load the prompts which were used for calculating the causal scores, these prompts will not be used for ablation
    if args.prompt_file_for_causal_scores_exp is not None:
        prompt_file_for_cma = args.prompt_file_for_causal_scores_exp
    else:
        prompt_file_for_cma = get_prompts_for_cma(args.model_type, args.head_type, args.rule)
    with open(prompt_file_for_cma, "r") as f:
        lines = f.read()
        lines = lines.split(LINE_SEP)   

        assert lines[0] == ""
        lines = lines[1:] 
    prompt_for_causal_scores = [line.strip() for line in lines]

    #################################################################
    #### 2. Get the head list ranked by the causal mediation scores ####
    #################################################################

    ranked_head_list, layer_rank_head_dict  = get_head_list(args.model_type, args.head_type)


    ##################################################################
    #### 3. Conduct cumulative ablation ####
    ##################################################################

    token_pos = args.token_pos_list
    prepend_bos = has_prepend_bos(tokenizer)
    logger.info(f"prepend_bos: {prepend_bos}")
    if prepend_bos:
        token_pos = [pos+1 if pos != -1 else pos for pos in token_pos]
    min_valid_sample_num = args.min_valid_sample_num if args.min_valid_sample_num is not None else -1
    low_prob_threshold = args.low_prob_threshold #0.9

    logger.info(f"Ablate the activations {activation_name} at pos {token_pos} (min_valid_sample_num: {min_valid_sample_num})")
    logger.info(f"control: {args.control}, random_control: {args.random_control}, random_times_per_step: {args.random_times_per_step}")
    cumulative_ablation(
        model, tokenizer, prompts, correct_ans_list, args.head_type,
        activation_name, 
        save_folder, 
        layer_paths=layer_paths,
        token_pos=token_pos, 
        min_valid_sample_num=min_valid_sample_num, 
        ranked_head_list=ranked_head_list, layer_ranked_heads=layer_rank_head_dict,
        start_head_idx=args.start_head_idx,
        end_head_idx=args.end_head_idx,
        step_size=args.step_size,
        control=args.control,
        random_control=args.random_control,
        random_times_per_step=args.random_times_per_step, 
        eval_metric=args.eval_metric,
        eos_token_id=eos_token_id, 
        low_prob_threshold=low_prob_threshold, 
        prompt_for_causal_scores=prompt_for_causal_scores,
        seed=args.seed,
        adaptive_step_size=args.adaptive_step_size,
        patch_all_token_pos=args.patch_all_token_pos,
        **generation_kwargs
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
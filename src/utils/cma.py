#####################################
import os 
import glob

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"] 
HF_HOME = os.environ["HF_HOME"]
#####################################
## implementation of CMA method (Section 3.1 and Alogorithm 1 in https://arxiv.org/pdf/2502.20332?)
## context_type: abstract (symbol abstraction heads/ symbolic induction heads) or token (retrieval heads)
## notations: (patching from base context to exp context (c2 -> c1))
# base context (c2): base_rule/base_prompt/base_ans (y_c2)
# exp context (c1): exp_rule/exp_prompt/exp_ans (y_c1)
# patched exp context (c1*): expected answer after patching activations from c2 to c1, causal_ans (y_c1*)
######################################

import logging

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from src.utils.utils import LINE_SEP, plot, HEAD_ACTIVATIONS, set_seed, vocab_dict, get_model_id_family
from src.utils.args import _get_args
from src.utils.tools import generate_prompts, generate_response_eval, get_num_hidden_layers, _resolve_layer_path, _resolve_text_model_dims, has_prepend_bos, get_layer_paths, get_eos_token_id, get_generation_kwargs
from src.utils.ungroup import needs_kv_head_ungrouping, ungroup_remote_layer_kv_heads, apply_remote_kv_ungrouping, check_remote_kv_metadata, check_remote_layer_kv_metadata


from transformers import AutoTokenizer, GenerationConfig
import torch
import random
import numpy as np
from tqdm import tqdm
from functools import partial
from nnsight import LanguageModel, ndif

ndif.register("src.utils.tools")
ndif.register("src.utils.ungroup")


def get_args():
    return _get_args('cma')

def cal_logit_prob_diff(logits, tokenizer, causal_ans, original_ans):
    """
        calculate the logit differences between the expected answer of the patched context and the correct answer of the original context. 
    """
    causal_ans_id = tokenizer.convert_tokens_to_ids(causal_ans)
    original_ans_id = tokenizer.convert_tokens_to_ids(original_ans)
    logits_diff = logits[0, -1, causal_ans_id] - logits[0, -1, original_ans_id]
    return logits_diff

### [Important Function] ###
def intervention_fn(model, need_ungroup=False, patched_output=None, **hook_kwargs):
    layer_idx = hook_kwargs.pop('layer_idx')
    token_pos = hook_kwargs.pop('token_pos')
    layer_paths = hook_kwargs.pop('layer_paths')
    activation_name = hook_kwargs.pop('activation_name')
    base_cache = hook_kwargs.pop('base_cache')
    patch_mlp_out = hook_kwargs.pop('patch_mlp_out', False)

    target = _resolve_layer_path(model, layer_paths[layer_idx][activation_name])
    if activation_name in ["z", "o"]:
        activation = target.input
    elif activation_name in ("k", "v") and need_ungroup:
        if patched_output is not None:
            activation = patched_output
        else:
            raise AssertionError(
                f"Missing patched ungrouped {activation_name} output for layer {layer_idx}."
            )
    else:
        activation = target.output
    base_tensor = base_cache[activation_name][layer_idx]

    by_head = hook_kwargs.pop('by_head', False)
    
    squeeze_batch = len(activation.shape) == 2
    if squeeze_batch:
        activation = activation.unsqueeze(0)
    squeeze_base = len(base_tensor.shape) == 2
    if squeeze_base:
        base_tensor = base_tensor.unsqueeze(0)

    if by_head:
        head_idx = hook_kwargs.pop('head_idx')
        total_heads = hook_kwargs.pop('total_heads')
        group_size = hook_kwargs.pop('group_size')
        
        b, s, h_dim = activation.shape
        reshaped = activation.view(b, s, total_heads, -1)
        base_tensor = base_tensor.view(b, s, total_heads, -1)
        
        for t_pos in token_pos:
            reshaped[:, t_pos, head_idx * group_size : (head_idx+1) * group_size, :] = \
                base_tensor[:, t_pos, head_idx * group_size : (head_idx+1) * group_size, :]
                
        activation = reshaped.view(b, s, h_dim)
    else:
        activation[:, token_pos, :] = base_tensor[:, token_pos, :]

    if squeeze_batch:
        activation = activation.squeeze(0)
    
    if activation_name in ["z", "o"]:
        target.input = activation
    else:
        target.output = activation
            
    if activation_name == "attn_out" and patch_mlp_out: ## patch the MLP output as well
        assert not by_head, "'patch_mlp_out' does not support patching by head."
        mlp_target = _resolve_layer_path(model, layer_paths[layer_idx]["mlp_out"])
        mlp_activation = mlp_target.output
        mlp_base_tensor = base_cache["mlp_out"][layer_idx]
        squeeze_mlp = len(mlp_activation.shape) == 2
        if squeeze_mlp:
            mlp_activation = mlp_activation.unsqueeze(0)

        if len(mlp_base_tensor.shape) == 2:
            mlp_base_tensor = mlp_base_tensor.unsqueeze(0)

        mlp_activation[:, token_pos, :] = mlp_base_tensor[:, token_pos, :]

        if squeeze_mlp:
            mlp_activation = mlp_activation.squeeze(0)
        mlp_target.output = mlp_activation

### [1] patching the activations in different layers x different heads at certain token positions
### [2] patching the activations in different layers x token positions
def ablate_head_or_layer(
        model, 
        base_cache, ## activation cache
        input_ids, ## input ids
        layer_paths,
        token_pos=[-1], ## the token positions where activations are patched
        exp_logits_diff=None,
        # ans_1=None, ans_2=None,
        patched_exp_ans=None, exp_ans=None,
        total_layers=80, total_heads=64, group_size=1,
        by_head=True,
        activation_name="z", patch_mlp_out=False,
        device="cuda", 
        ## whether to generate the responses while patching the activations
        generate=False, tokenizer=None, correct_ans=None, prompt=None, eos_token_id=None, 
        checkpoint_dir=None, prompt_idx=0,
        ungroup_grouped_query_attention=False,
        **generation_kwargs
        ):
    """
    Patch the activations in different layers and heads at certain token positions. Please refer to CMA section and Algorithm 1 in https://arxiv.org/pdf/2502.20332? for more details.
    
    Args:
        model: CustomHookedTransformer, the model to be patched.
        base_cache: ActivationCache, the activation cache from the base context (c2) in Algorithm 1 https://arxiv.org/pdf/2502.20332?.
        input_ids: Tensor, the input ids for the model.
        token_pos: int or list, the token positions where activations are patched.
        exp_logits_diff: 
            Tensor, 
            in the exp context (c1), the difference between the logits f(.) for the expected answer (y_c1*) in the patched context and the correct answer (y_c1) of the exp context, 
            i.e., \delta(f_c1) = f(c1)[y_c1*] - f(c1)[y_c1].
        patched_exp_ans: 
            str: The expected answer for the patched context c1*: y_c1* 
            After patching the activations, the answer for the patched context would change 
            according to our hypotheses about the representations of the activations (i.e., whether they represent abstract symbols or literal tokens).
        exp_ans: 
            str: The correct answer for the original context c1: y_c1.
        total_layers: int, the total number of layers in the model.
        total_heads: int, the total number of heads in the model.
        group_size: int, the group size for head splitting.
        activation_name: str, the name of the activation to be patched.
        device: str, the device to run the model on.

        ## whether to generate the responses and test the accuracy of y_c1 while patching the activations
        generate: bool, whether to actually generate responses while patching activations.
        tokenizer: PreTrainedTokenizer, the tokenizer to use for decoding.
        correct_ans: str, the correct answer string.
        prompt: str, the prompt string.
        eos_token_id: int, the end of sequence token id.
        **generation_kwargs: additional keyword arguments for generation.

    """
    final_path = os.path.join(checkpoint_dir, f"prompt_{prompt_idx}_final.pt")
    if os.path.exists(final_path):
        logger.info(f"Prompt {prompt_idx} already completely processed. Loading final checkpoint.")
        return torch.load(final_path, map_location=device, weights_only=False)
    
    assert input_ids is not None
    token_len = input_ids.shape[-1]
    assert token_len > 1, "token length should be larger than 1"

    if generate: ## whether to measure the causal effects by actually generating the responses
        correct_num_all = torch.zeros((total_layers, total_heads if by_head else token_len), device=device)
        total_num_all = torch.zeros((total_layers, total_heads if by_head else token_len), device=device)
        acc_all = torch.zeros((total_layers, total_heads if by_head else token_len), device=device)
    else:
        logits_diff_change = torch.zeros((total_layers, total_heads if by_head else token_len), device=device)

    start_layer = 0
    layer_files = glob.glob(os.path.join(checkpoint_dir, f"prompt_{prompt_idx}_layer_*.pt"))
    if layer_files:
        start_layer = max([int(f.split("_layer_")[1].split(".pt")[0]) for f in layer_files])
        logger.info(f"Resuming Prompt {prompt_idx} from layer {start_layer + 1}...")
        state = torch.load(os.path.join(checkpoint_dir, f"prompt_{prompt_idx}_layer_{start_layer}.pt"), map_location=device, weights_only=False)
        if generate:
            correct_num_all, total_num_all, acc_all = state
        else:
            logits_diff_change = state
        start_layer += 1 # Resume at the NEXT layer

    for layer_idx in tqdm(range(start_layer, total_layers)):
        print("================================")
        print(f"================ layer: {layer_idx+1}/{total_layers}")
        for idx in range(total_heads if by_head else token_len):
            if by_head:
                hook_kwargs = {
                    'layer_idx': layer_idx, 
                    'token_pos': token_pos,
                    'layer_paths': layer_paths,
                    'activation_name': activation_name,
                    'base_cache': base_cache,
                    'by_head': True,
                    'head_idx': idx,
                    'total_heads': total_heads,
                    'group_size': group_size
                }
            else:
                hook_kwargs = {
                    'layer_idx': layer_idx, 
                    'token_pos': idx,
                    'layer_paths': layer_paths,
                    'activation_name': activation_name,
                    'base_cache': base_cache,
                    'patch_mlp_out': patch_mlp_out,
                }
            fwd_hooks = partial(
                intervention_fn, 
                **hook_kwargs,
            )
            combined_hooks = partial(
                apply_remote_kv_ungrouping,
                ungroup_grouped_query_attention=ungroup_grouped_query_attention,
                hook_fn=fwd_hooks,
                **hook_kwargs,
            )

            if generate: ## generating the response while implementing activation patching during the run
                correct_num, total_num, acc = generate_response_eval(
                    model, input_ids, eos_token_id, tokenizer, correct_ans, prompt,
                    sample_size=args.sample_size, 
                    max_new_tokens=args.max_new_tokens,
                    intervention_fn=combined_hooks, 
                    **generation_kwargs
                )
                correct_num_all[layer_idx, idx] = correct_num
                total_num_all[layer_idx, idx] = total_num
                acc_all[layer_idx, idx] = acc
            else:
                with torch.no_grad():
                    with model.trace(input_ids, remote=True):
                        combined_hooks(model)
                        patched_logits = model.lm_head.output.save()

                patched_exp_logits_diff =  cal_logit_prob_diff(
                    patched_logits, tokenizer, causal_ans=patched_exp_ans, original_ans=exp_ans)
                logits_diff_change[layer_idx, idx] = patched_exp_logits_diff -  exp_logits_diff ## calculate the change in logits difference after patching the activations
            
        with torch.no_grad():
            with model.trace(input_ids, remote=True):
                check_remote_layer_kv_metadata(model, layer_idx, ungroup_grouped_query_attention)

        current_state = (correct_num_all, total_num_all, acc_all) if generate else logits_diff_change
        current_path = os.path.join(checkpoint_dir, f"prompt_{prompt_idx}_layer_{layer_idx}.pt")
        torch.save(current_state, current_path)
        
        # Space management: Delete previous layer's checkpoint
        if layer_idx > 0:
            prev_path = os.path.join(checkpoint_dir, f"prompt_{prompt_idx}_layer_{layer_idx - 1}.pt")
            if os.path.exists(prev_path):
                os.remove(prev_path)
     
    final_state = (correct_num_all, total_num_all, acc_all) if generate else logits_diff_change
    torch.save(final_state, final_path)
    last_layer_path = os.path.join(checkpoint_dir, f"prompt_{prompt_idx}_layer_{total_layers - 1}.pt")
    if os.path.exists(last_layer_path):
        os.remove(last_layer_path)

    return final_state

def activation_patching(
        model, tokenizer, 
        prompt_pairs, correct_ans_pairs, causal_ans_list, 
        activation_name, 
        base_rule, exp_rule,
        save_folder, pos_label_dict, 
        layer_paths,
        token_pos=[-1], device="cuda", low_prob_threshold=0.9, min_valid_sample_num=-1, 
        generate=False, eos_token_id=None, group_heads=False, 
        patch_mlp_out=False, 
        eval_metric="gen_acc", ## gen_acc or ans_prob
        ungroup_grouped_query_attention=False,
        **generation_kwargs
        ):

    total_layers = get_num_hidden_layers(model)
    hidden_size, total_heads = _resolve_text_model_dims(model)
    num_heads = total_heads
    _, n_key_value_heads = _resolve_text_model_dims(model, kv_heads=True)
    should_ungroup_kv = needs_kv_head_ungrouping(
        num_heads,
        n_key_value_heads,
        ungroup=ungroup_grouped_query_attention,
    )
    
    if ("k" in activation_name or "v" in activation_name) and total_heads != n_key_value_heads: 
        if should_ungroup_kv:
            logger.info(
                "Ungrouping GQA remotely for %s activations: expanding %s KV heads to %s heads.",
                activation_name,
                n_key_value_heads,
                num_heads,
            )
        else:
            logger.warning("Ungrouping GQA on native HF models via nnsight requires modifying the attention computation. Proceeding with native KV heads.")
            ## for the keys/values which are already grouped in the Grouped Query Attention (GQA), total heads should be the number of groups
            total_heads = n_key_value_heads
    group_size = 1
    if group_heads and not should_ungroup_kv: # [Optional] Default is False
        ## whether to patch the activations (any types in HEAD_ACTIVATIONS) of grouped heads which share the keys/values (GQA) at the same time
        group_size = total_heads // n_key_value_heads
        total_heads = total_heads // group_size

    if generate:
        correct_num_list = []   
        total_num_list = []
        acc_list = []
    else:
        logits_diff_ch_list = []
    valid_num = 0
    base_prompt_list = []
    exp_prompt_list = []

    checkpoint_dir = os.path.join(save_folder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    def get_group_intervention_fn(model):
        group_intervention_fn = None
        if should_ungroup_kv:
            group_intervention_fn = partial(
                check_remote_kv_metadata,
                ungroup_grouped_query_attention=ungroup_grouped_query_attention
            )
        return group_intervention_fn
    
    for idx_, pair in enumerate(prompt_pairs):
        logger.info(f"Processing {idx_}th sample")
        base_prompt, exp_prompt = pair
        base_ans, exp_ans = correct_ans_pairs[idx_]
        causal_ans = causal_ans_list[idx_] ## the expected answer of the patched context

        #### base rule
        base_input = tokenizer(base_prompt, return_tensors="pt")
        base_input_ids = base_input.input_ids

        
        if eval_metric == "gen_acc":
            ## evaluate the model performance through generation accuracy
            _, _, base_acc = generate_response_eval(
                model, base_input_ids, eos_token_id, tokenizer, base_ans, base_prompt, 
                sample_size=args.sample_size, 
                max_new_tokens=args.max_new_tokens, 
                intervention_fn=get_group_intervention_fn(model), **generation_kwargs    # for local runs across contexts
            )
            if base_acc < low_prob_threshold:
                logger.warning(f"base context: low acc {base_acc} < {low_prob_threshold}")
                continue
        
        with torch.no_grad():
            with model.trace(base_input_ids, remote=True):
                base_cache = {activation_name: []} ## cache the internal activations after feeding the base prompt to the model
                if patch_mlp_out: 
                    base_cache["mlp_out"] = []
                base_cache.save()
                for layer_idx in range(total_layers):
                    patched_outputs = {}
                    if should_ungroup_kv:
                        for act in ['k', 'v']:
                            patched_outputs[act] = ungroup_remote_layer_kv_heads(
                                model,
                                layer_idx=layer_idx,
                                ungroup=True,
                                activation_name=act,
                            )
                    target = _resolve_layer_path(model, layer_paths[layer_idx][activation_name])
                    if activation_name in ["z", "o"]:
                        base_cache[activation_name].append(target.input)
                    elif activation_name in patched_outputs:
                        base_cache[activation_name].append(patched_outputs[activation_name])
                    else:
                        base_cache[activation_name].append(target.output)
                    
                    if activation_name == "attn_out" and patch_mlp_out:
                        mlp_target = _resolve_layer_path(model, layer_paths[layer_idx]["mlp_out"])
                        base_cache["mlp_out"].append(mlp_target.output)
                base_logits = model.lm_head.output.save()
                    
        base_real_prob = torch.softmax(base_logits, dim=-1)[0, -1, tokenizer.convert_tokens_to_ids(base_ans)].item() 

        #### exp rule
        exp_input = tokenizer(exp_prompt, return_tensors="pt")
        exp_input_ids = exp_input.input_ids
       
        if eval_metric == "gen_acc":
            ## evaluate the model performance through generation accuracy
            _, _, exp_acc = generate_response_eval(
                model, exp_input_ids, eos_token_id, tokenizer, exp_ans, exp_prompt, 
                sample_size=args.sample_size, 
                max_new_tokens=args.max_new_tokens, 
                intervention_fn=get_group_intervention_fn(model), **generation_kwargs
            )
            if exp_acc < low_prob_threshold:
                logger.warning(f"exp context: low acc {exp_acc} < {low_prob_threshold}")
                continue

        with torch.no_grad():
            with model.trace(exp_input_ids, remote=True):
                # if should_ungroup_kv:
                #     # not strictly necessary here
                #     for layer_idx in range(total_layers):
                #         hook_kwargs = {
                #         'layer_idx': layer_idx
                #         }
                #         apply_remote_kv_ungrouping(
                #             model,
                #             ungroup_grouped_query_attention=True,
                #             **hook_kwargs
                #         )
                exp_logits = model.lm_head.output.save()

        exp_logits_diff = cal_logit_prob_diff(
            exp_logits, tokenizer, causal_ans=causal_ans, original_ans=exp_ans) ## causal answer vs exp answer
        exp_real_prob = torch.softmax(exp_logits, dim=-1)[0,-1,tokenizer.convert_tokens_to_ids(exp_ans)].item() 
        if (eval_metric == "ans_prob") and (exp_real_prob < low_prob_threshold or base_real_prob < low_prob_threshold): 
            ## filter out the samples with low probability answers 
            ## this is more strict than the generation accuracy 
            logger.warning(f"base context answer probability {base_real_prob}; exp context answer probability {exp_real_prob} while the threshold is {low_prob_threshold}")
            continue
        
        base_prompt_list.append(base_prompt)
        exp_prompt_list.append(exp_prompt)
        #### patch from base context (c2 in the Algorithm 1) to exp context (c1 in the Algorithm 1)  -> patched exp context (c1* in the Algorithm 1)   
        kwargs = {}
        if generate:
            kwargs.update({
                "eos_token_id": eos_token_id,
                # "tokenizer": tokenizer,
                "prompt": exp_prompt,
                "correct_ans": exp_ans,
            })
            kwargs.update(generation_kwargs)
        
        if activation_name in HEAD_ACTIVATIONS:
            results = ablate_head_or_layer(
                model, 
                base_cache,
                input_ids=exp_input_ids,
                layer_paths=layer_paths,
                token_pos=token_pos,
                exp_logits_diff=exp_logits_diff,
                patched_exp_ans=causal_ans, exp_ans=exp_ans,
                total_layers=total_layers, 
                total_heads=total_heads, 
                group_size=group_size,
                by_head=True,
                activation_name=activation_name, 
                device=device,
                generate=generate,
                tokenizer=tokenizer,
                checkpoint_dir=checkpoint_dir, prompt_idx=idx_,
                ungroup_grouped_query_attention=ungroup_grouped_query_attention,
                **kwargs
            )

        elif "resid" in activation_name or "out" in activation_name: 
            ## operate on the output of the whole attention block (i.e., attn_out) or the residual stream (e.g, resid_pre) in each layer
            results = ablate_head_or_layer(
                model, 
                base_cache,
                input_ids=exp_input_ids,
                layer_paths=layer_paths,
                exp_logits_diff=exp_logits_diff,
                patched_exp_ans=causal_ans, exp_ans=exp_ans,
                total_layers=total_layers, 
                by_head=False,
                activation_name=activation_name, patch_mlp_out=patch_mlp_out,
                device=device,
                generate=generate,
                tokenizer=tokenizer,
                checkpoint_dir=checkpoint_dir, prompt_idx=idx_,
                ungroup_grouped_query_attention=ungroup_grouped_query_attention,
                **kwargs
            )

        if generate:
            correct_num_all, total_num_all, acc_all = results
            correct_num_list.append(correct_num_all)
            total_num_list.append(total_num_all)
            acc_list.append(acc_all)

        else:
            logits_diff_change = results
            logits_diff_ch_list.append(logits_diff_change)
        
        valid_num += 1

        if min_valid_sample_num != -1 and valid_num >= min_valid_sample_num: ## run CMA on enough valid samples on which the model could generate the correct answers
            break

    if valid_num == 0:
        logger.error("No samples to patch")
        return
    
    generate_remark = "generate" if generate else "logit"
    save_folder = os.path.join(save_folder, generate_remark, f"sample_num_{valid_num}_{eval_metric}_{low_prob_threshold}")
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, f"base_prompt_{valid_num}.txt"), "w") as f:
        for prompt in base_prompt_list:
            f.write(f"{LINE_SEP}" + prompt + "\n")
    with open(os.path.join(save_folder, f"exp_prompt_{valid_num}.txt"), "w") as f:
        for prompt in exp_prompt_list:
            f.write(f"{LINE_SEP}" + prompt + "\n")

    if activation_name in HEAD_ACTIVATIONS:
        sub_save_folder = os.path.join(save_folder, f"group_heads_{group_heads}", f"token_pos_{token_pos}")
        xpos_labels = None
        xlabel_name = "Head Index"

    elif "resid" in activation_name or "out" in activation_name:
        if "out" in activation_name:
            sub_save_folder = os.path.join(save_folder, f"patch_mlp_out_{patch_mlp_out}")
        else:
            sub_save_folder = save_folder 

        xpos_labels = pos_label_dict["Both"]
        xlabel_name = "Token Position"

    os.makedirs(sub_save_folder, exist_ok=True)

    logger.info(f"Plot and save results to {sub_save_folder}")
    if generate:
        correct_num_all = torch.stack(correct_num_list, dim=0).sum(dim=0)
        total_num_all = torch.stack(total_num_list, dim=0).sum(dim=0)
        acc_all = torch.stack(acc_list, dim=0).mean(dim=0)
        torch.save(torch.stack(acc_list, dim=0), os.path.join(sub_save_folder, "acc_all.pt"))
        plot(acc_all.cpu().numpy(), sub_save_folder, metric_name=f"Accuracy of Original Answers in {exp_rule}", xpos_labels=xpos_labels, xlabel_name=xlabel_name)
    else:
        mean_logits_diff_ch = torch.stack(logits_diff_ch_list, dim=0).mean(dim=0)
        torch.save(torch.stack(logits_diff_ch_list, dim=0), os.path.join(sub_save_folder, "causal_scores.pt"))
        plot(mean_logits_diff_ch.cpu().numpy(), sub_save_folder, metric_name=f"Score_Patching_{base_rule}_to_{exp_rule}", xpos_labels=xpos_labels, xlabel_name=xlabel_name)    # Filename too long

    
def main(args):
    set_seed(args.seed)

    rule_list = ["ABA", "ABB"]

    # get the model id and family
    model_id, model_family =  get_model_id_family(args.model_type) 
    vocab_file = vocab_dict[model_family] ## get the vocabulary file path
    logger.info(f"model type: {args.model_type}, model id: {model_id}, vocab file: {vocab_file}")

    #################################################################
    # 0. Load the model and tokenizer, 
    # and specify the generation config which will be used to filter out the correct prompts for CMA
    #################################################################

    logger.info(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    torch.set_grad_enabled(False)

    generation_kwargs = get_generation_kwargs(args)

    ## Load the model
    try:
        model = LanguageModel(model_id, device_map=args.device_map, dtype=torch.bfloat16, token=HF_TOKEN)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error("Failed to load the model. Please check the model id")
        return
    model_remark = ""

    if args.ungroup_grouped_query_attention: # Ungroup the grouped query attention, **NECESSARY** for CMA on keys/values, which are not done in the paper 
        logger.info("ungroup the keys/values which were grouped in the grouped query attention, this is necessary for CMA on keys/values (not done in the paper)")
        logger.info("apply remote ungrouping in nnsight trace contexts; prompt-filtering generation remains native")
        generation_kwargs["use_cache"] = False
        model_remark += "_ungroup_gqa"

    eos_token_id = get_eos_token_id(tokenizer, args.eos_token)

    ## create the result folder
    ## for symbol abstraction heads and symbolic induction heads, the base rule and exp rule are different
    ## for retrieval heads, the base rule and exp rule are the same but the correct answers
    exp_rule_idx = 1 - rule_list.index(args.base_rule) if args.context_type == "abstract" else rule_list.index(args.base_rule)
    exp_rule = rule_list[exp_rule_idx]
    activation_name = args.activation_name
    causal_remark = f"{args.context_type}_context"
    remark = f"{model_id}{model_remark}/{causal_remark}/base_rule_{args.base_rule}_exp_rule_{exp_rule}/{activation_name}_seed_{args.seed}_shuffle_{args.do_shuffle}"
    save_folder = os.path.join(args.log_dir, remark)
    logger.info(f"save_folder: {save_folder}")
    os.makedirs(save_folder, exist_ok=True)

    #################################################################
    #### 1. Build the prompt dataset ####
    #################################################################
    prompts_cache_path = os.path.join(save_folder, f"prompts_data_{args.prompt_num}.pt")
    if os.path.exists(prompts_cache_path):
        logger.info(f"Loading cached prompts from {prompts_cache_path}...")
        cached_data = torch.load(prompts_cache_path, map_location="cpu", weights_only=False)
        prompts = cached_data['prompts']
        correct_ans_list = cached_data['correct_ans_list']
        causal_ans_list = cached_data['causal_ans_list']
    else:
        logger.info(f"generate prompts.... (SEP symbol: {args.sep_symbol})")
        assert args.prompt_num is not None

        prompts, correct_ans_list, causal_ans_list = generate_prompts(
            args, tokenizer, vocab_file, context_type=args.context_type, sep_symbol=args.sep_symbol, base_rule=args.base_rule, do_shuffle=args.do_shuffle, return_format="cma"
            )
        torch.save({
            'prompts': prompts,
            'correct_ans_list': correct_ans_list,
            'causal_ans_list': causal_ans_list
        }, prompts_cache_path)

    with open(os.path.join(save_folder, f"base_input_prompts_{args.prompt_num}.txt"), "w") as f:
        for p in prompts:
            f.write(f"{LINE_SEP}" + p[0] + "\n")
    
    with open(os.path.join(save_folder, f"exp_input_prompts_{args.prompt_num}.txt"), "w") as f:
        for p in prompts:
            f.write(f"{LINE_SEP}" + p[1] + "\n")
    
    with open(os.path.join(save_folder, f"ans_{args.prompt_num}.txt"), "w") as f:
        for idx_, ans_pair in enumerate(correct_ans_list):
            f.write(" ".join(ans_pair) + " " + causal_ans_list[idx_] + "\n")

    min_valid_sample_num = args.min_valid_sample_num if args.min_valid_sample_num is not None else -1
    token_pos = args.token_pos_list
    prepend_bos = has_prepend_bos(tokenizer)
    logger.info(f"prepend_bos: {prepend_bos}")
    if prepend_bos:
        token_pos = [pos+1 if pos != -1 else pos for pos in token_pos]
    pos_label_dict = {
        "ABA": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "A1", "'\n'", "A2", "^", "B2", "^", "A2", "'\n'", "A3", "^", "B3", "^"],
        "ABB": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "B1", "'\n'", "A2", "^", "B2", "^", "B2", "'\n'", "A3", "^", "B3", "^"],
        "Both": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "A1/B1", "'\n'", "A2", "^", "B2", "^", "A2/B2", "'\n'", "A3", "^", "B3", "^"],
    } ## used as axis labels of the plotted figures

    ##################################################################
    #### 2. Activation Patching and Measure Causal Effects ####
    ##################################################################
    logger.info(f"Patching the activations {activation_name} at position {token_pos} for more than {min_valid_sample_num} valid prompt pairs where the model could meet the low threshold of {args.low_prob_threshold} for {args.eval_metric}")
    layer_paths = get_layer_paths(model, [args.activation_name, "mlp_out"])
    activation_patching(
        model, tokenizer, prompts, correct_ans_list, causal_ans_list, 
        activation_name, args.base_rule, exp_rule, 
        save_folder, pos_label_dict, layer_paths=layer_paths, token_pos=token_pos, device=args.device,
        low_prob_threshold=args.low_prob_threshold, min_valid_sample_num=min_valid_sample_num, 
        generate=args.generate, eos_token_id=eos_token_id, group_heads=args.group_heads, 
        patch_mlp_out=args.patch_mlp_out, 
        eval_metric=args.eval_metric,
        ungroup_grouped_query_attention=args.ungroup_grouped_query_attention,
        **generation_kwargs
    )


if __name__ == "__main__":
    args = get_args()
    main(args)

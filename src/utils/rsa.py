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

from src.utils.utils import LINE_SEP, plot, set_seed, vocab_dict, get_model_id_family, rsa_get_head_list
from src.utils.args import _get_args
from src.utils.tools import generate_prompts, _resolve_text_model_dims, _resolve_layer_path, get_num_hidden_layers, get_layer_paths, has_prepend_bos, get_token_sets

from transformers import AutoTokenizer
from nnsight import LanguageModel, ndif
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict 
from scipy.stats import pearsonr
import gc

ndif.register("src.utils.tools")
ndif.register("src.utils.ungroup")


def get_args():
    return _get_args('rsa')

def main(args):
    set_seed(args.seed)
    act_list = args.act_list ## ["z", "v", "rot_k", "rot_q", "result", "k", "q"]

    model_id, model_family =  get_model_id_family(args.model_type) 
    vocab_file = vocab_dict[model_family] ## get the vocabulary file path
    logger.info(f"model type: {args.model_type}, model id: {model_id}, vocab file: {vocab_file}")


    #################################################################
    # 0. Load the model and tokenizer, 
    # and specify the generation config which will be used to filter out the correct prompts for CMA
    #################################################################

    logger.info(f"Loading {model_id} via nnsight...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = HF_TOKEN)
    model = LanguageModel(
        model_id, 
        # device_map=args.device_map,    # remove to prevent local downloading
        dtype=torch.bfloat16,    # `torch_dtype` is deprecated!
        token=HF_TOKEN
        )
    
    hidden_size, n_heads = _resolve_text_model_dims(model)
    d_head = hidden_size // n_heads
    _, num_kv_heads = _resolve_text_model_dims(model, kv_heads=True)

    logger.info("ungroup the keys/values which were grouped in the grouped query attention, this is necessary for CMA on keys/values (not done in the paper)")
    # TODO
    # ungroup logics
    # model = ungroup_nnsight_llm(model, hidden_size, n_heads, num_kv_heads)
    n_layers = get_num_hidden_layers(model)
    model_remark = "_ungroup_gqa"

    low_sim = args.low_sim
    high_sim = args.high_sim

    causal_remark = "abstract"
    if args.add_swap_1_2_question:
        causal_remark += "_add_swap_1_2_question"

    # This block seems unused
    # if "result" in args.act_list or args.use_attn_result: 
    #     # whether to split w_o into sub-blocks and apply each sublock on corresponding attention head's output, 
    #     # so the final output of the activation block w_o * [z[0] ... z[63]] is rewritten into the summation of w_o[i] * z[i] over all attention heads
    #     # And conduct rsa on the activations after applying sub-blocks of w_o on each attention head's output, i.e., "result" in act_list
    #     logger.info("Splitting w_o into sub-blocks and applying each subblock on corresponding attention head's output")
    #     model.set_use_attn_result(True)

    folder_remark = f"{model_id}{model_remark}/in_ctx_{args.in_context_example_num}/{causal_remark}_base_rule_{args.base_rule}/prompt_{args.prompt_num}_seed_{args.seed}_shuffle_{args.do_shuffle}" 
    save_folder = os.path.join(args.log_dir, folder_remark)
    logger.info(f"save_folder: {save_folder}")
    os.makedirs(save_folder, exist_ok=True)

    #################################################################
    #### 1. Build the prompt dataset ####
    #################################################################
    logger.info(f"generate prompts.... (SEP symbol: {args.sep_symbol})")
    assert args.prompt_num is not None

    token_sets = get_token_sets(args.token_set_file)

    prompts, correct_ans_list, rule_group_list = generate_prompts(
        args, tokenizer, vocab_file, sep_symbol=args.sep_symbol, token_sets=token_sets,
        add_swap_1_2_question=args.add_swap_1_2_question, base_rule=args.base_rule, 
        do_shuffle=args.do_shuffle, return_format="rsa"
    )

    for idx_ in range(len(prompts[0])):
        with open(os.path.join(save_folder, f"set_{idx_}_input_prompts_{args.prompt_num}.txt"), "w") as f:
            for p in prompts:
                f.write(f"{LINE_SEP}" + p[idx_] + "\n")

    with open(os.path.join(save_folder, f"ans_{args.prompt_num}.txt"), "w") as f:
        for idx_, ans_pair in enumerate(correct_ans_list):
            f.write(" ".join(ans_pair) + "\n")

    with open(os.path.join(save_folder, f"rule_{args.prompt_num}.txt"), "w") as f:
        for idx_, rule_pair in enumerate(rule_group_list):
            f.write(" ".join(rule_pair) + "\n")
    logger.info(f"# total prompts: {len(prompts)}")


    #################################################################
    #### 2. Collect Activations ####
    #################################################################
    all_act_dict = {"ABA": defaultdict(list), "ABB": defaultdict(list)}
    cache_input_ids_dict = defaultdict(list)
    layer_paths = get_layer_paths(model, act_list)
    for i, prompt_pair in enumerate(tqdm(prompts)):
        print("=====================================================")
        print(f"==================== {i+1} / {len(prompts)} ================")
        print("=====================================================")
        for j_ in range(len(prompt_pair)):
            prompt = [prompt_pair[j_]]
            correct_ans = correct_ans_list[i][j_]
            rule_j = rule_group_list[i][j_] ## "ABA" or "ABB"
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids
            # shape and data are unchanged
            # cache_input_ids = input_ids[0].unsqueeze(0)    
            cache_input_ids_cp = input_ids.clone()
            
            correct_ans_id = tokenizer.convert_tokens_to_ids(correct_ans)
            assert correct_ans_id is not None
            cache_input_ids_cp[:, -1] = correct_ans_id
            cache_input_ids_dict[rule_j].append(cache_input_ids_cp)

            with torch.no_grad():
                with model.trace(input_ids, remote=True):
                    cache = {act: [] for act in act_list}.save()
                    for layer_idx in range(n_layers):
                        for act in act_list:
                            target_module = _resolve_layer_path(model, layer_paths[layer_idx][act])
                            
                            # z/o are inputs to o_proj, q/k/v are outputs of their respective projs
                            if act in ["z", "o"]:
                                cache[act].append(target_module.input)
                            else:
                                cache[act].append(target_module.output)
                                if layer_idx == 0:  # Only log for the first layer to avoid spam
                                    # target_tensor shape is (batch, seq_len, total_features)
                                    total_features = target_module.output.shape[-1]
                                    active_heads = total_features // d_head
                                    print(f"DEBUG: Layer {layer_idx} | {act}_proj | features: {total_features} | calculated heads: {active_heads} | n_heads: {n_heads}")

            act_dict = {}
            for act in act_list:
                act_values = [cache[act][layer_idx].view(1, -1, n_heads, d_head).cpu() for layer_idx in range(n_layers)]
                # act == "pattern" or act == "attn_scores" seems unused
                act_dict[act] = torch.stack(act_values, dim=2) # batch_size x seq_len x n_layers x head_index x d_head

            for act in act_dict.keys():
                # (for same prompt, the activations for each layer are the same)
                act_dict[act] = act_dict[act].mean(dim=0) # (seq_len x n_layers x head_index x d_head) or (seq_len x d_model)
                all_act_dict[rule_j][act].append(act_dict[act])


            del cache
            del act_values
            del act_dict 
            gc.collect()
            torch.cuda.empty_cache()

    logger.info("Finish recording activations")
    del model
    gc.collect()
    torch.cuda.empty_cache()


    if len(list(all_act_dict.keys())) == 0:
        logger.warning("No activations are recorded")
        return

    if args.only_for_significant_heads:
        head_list, head_weight_score = rsa_get_head_list(args.model_type, args.head_type)
        weighted_avg_sim = None
        weighted_rsa_corr = 0
        weighted_sum = 0
        logger.info(f"Focusing on significant heads for {args.head_type}: {head_list}")

    cache_input_ids_dict["ABA"] = torch.concat(cache_input_ids_dict["ABA"], dim=0)    
    cache_input_ids_dict["ABB"] = torch.concat(cache_input_ids_dict["ABB"], dim=0)
    
    prepend_bos = has_prepend_bos(tokenizer)
    ## key, query, value, output
    for act in all_act_dict["ABA"].keys():
        all_act_dict["ABA"][act] = torch.stack(all_act_dict["ABA"][act], dim=0) 
        all_act_dict["ABB"][act] = torch.stack(all_act_dict["ABB"][act], dim=0) 
        logger.info(f"Aggregate activations for {act} ABA: {all_act_dict['ABA'][act].shape} ABB: {all_act_dict['ABB'][act].shape}")
        assert args.cmp_with_abstract or args.cmp_with_token_id 
        all_act = torch.concat([all_act_dict["ABA"][act], all_act_dict["ABB"][act]], dim=0).float()
        seq_len = all_act.shape[1]
        sel_pos_list = args.sel_pos_list
        if prepend_bos:
            sel_pos_list = [pos + 1 if pos != -1 else seq_len-1 for pos in sel_pos_list]
        # relative position for each item in the in-context example
        # pos_table = {   
        #     i_: 1 for i_ in range(1, all_act.shape[1], 6)
        # }
        # pos_table.update({   
        #     i_: 2 for i_ in range(3, all_act.shape[1], 6)
        # })
        # pos_table.update({
        #     i_: 3 for i_ in range(5, all_act.shape[1], 6)
        # })
        # pos_table[all_act.shape[1]-1] = 3 

        #################################################################
        #### 3. Getting Expected Similarity Matrix ####
        #################################################################
        logger.info("Getting Expected Similarity Matrix Based on Abstract Variables or Literal Tokens...")
        ### table of abstract variables for each token in the sequences of rules ABA and ABB
        ## 1: A; 2: B
        logger.info(f"all_act.shape: {all_act.shape}, seq_len: {seq_len}")
        abs_table = {   
            i_: 1 for i_ in range(1, seq_len, 6)
        }
        abs_table.update({   
            i_: 2 for i_ in range(3, seq_len, 6)
        })
        ABA_abs_table = abs_table.copy()
        ABA_abs_table.update({
            i_: 1 for i_ in range(5, seq_len, 6)
        })            
        ABA_abs_table[seq_len-1] = 1
        ABB_abs_table = abs_table.copy()
        ABB_abs_table.update({
            i_: 2 for i_ in range(5, seq_len, 6)
        })            
        ABB_abs_table[seq_len-1] = 2

        ### all_act: (2 * prompt_num) x seq_len x n_layers x n_heads x d_head
        total_prompt_num = all_act.shape[0]
        
        if args.cmp_with_token_id:

            sel_pos_list_ABA = sel_pos_list
            sel_pos_list_ABB = sel_pos_list

            ABA_input_ids = cache_input_ids_dict["ABA"][:, sel_pos_list_ABA]
            ABB_input_ids = cache_input_ids_dict["ABB"][:, sel_pos_list_ABB]

            ABA_act = all_act[:total_prompt_num//2, sel_pos_list_ABA, ...]
            ABB_act = all_act[total_prompt_num//2:, sel_pos_list_ABB, ...]

            all_input_ids = torch.concat([ABA_input_ids, ABB_input_ids], dim=0)
            all_act_sel = torch.concat([ABA_act, ABB_act], dim=0)
            if args.transpose:
                all_act_sel = all_act_sel.swapaxes(0,1)
                all_input_ids = all_input_ids.swapaxes(0,1)

            all_input_ids = all_input_ids.reshape(-1)

            hand_code_sim_matrix = all_input_ids.unsqueeze(1) == all_input_ids.unsqueeze(0)
            hand_code_sim_matrix = hand_code_sim_matrix.float()
            hand_code_sim_matrix = hand_code_sim_matrix * high_sim + (1 - hand_code_sim_matrix) * low_sim

            all_act_sel = all_act_sel.reshape(-1, *all_act_sel.shape[2:]) ## (2 * prompt_num * len(sel_pos_list)) x n_layers x n_heads x d_head
            token_sub_save_folder = os.path.join(save_folder, f"{act}_token/token_pos_{'_'.join([str(pos) for pos in sel_pos_list])}_transpose_{args.transpose}")


        elif args.cmp_with_abstract:
            
            ABA_abstract = torch.tensor([ABA_abs_table[pos] for pos in sel_pos_list])
            ABB_abstract = torch.tensor([ABB_abs_table[pos] for pos in sel_pos_list])

            ABA_abstract = ABA_abstract.unsqueeze(0).repeat(total_prompt_num//2, 1)
            ABB_abstract = ABB_abstract.unsqueeze(0).repeat(total_prompt_num//2, 1)

            all_abstract = torch.concat([ABA_abstract, ABB_abstract], dim=0)
            all_act_sel = all_act[:, sel_pos_list, ...]

            if args.transpose:
                all_act_sel = all_act_sel.swapaxes(0,1)
                all_abstract = all_abstract.swapaxes(0,1)

            all_abstract = all_abstract.reshape(-1)
            all_act_sel = all_act_sel.reshape(-1, *all_act_sel.shape[2:]) ## (2 * prompt_num * len(sel_pos_list)) x n_layers x n_heads x d_head

            hand_code_sim_matrix = all_abstract.unsqueeze(1) == all_abstract.unsqueeze(0)
            hand_code_sim_matrix = hand_code_sim_matrix.float()
            hand_code_sim_matrix = hand_code_sim_matrix * high_sim + (1 - hand_code_sim_matrix) * low_sim

            token_sub_save_folder = os.path.join(save_folder, f"{act}_abstract/token_pos_{'_'.join([str(pos) for pos in sel_pos_list])}_transpose_{args.transpose}")

        os.makedirs(token_sub_save_folder, exist_ok=True)
        torch.save(hand_code_sim_matrix.cpu(), os.path.join(token_sub_save_folder, f"hand_code_sim_matrix.pt"))
        if args.plot_hand_code:
            plot(hand_code_sim_matrix.cpu().numpy(), token_sub_save_folder, metric_name=f"Hand_Code_H_{high_sim}_L_{low_sim}", num_for_one_rule=all_act_sel.shape[0]//2, xlabel_name="Index", ylabel_name="Index")

        #################################################################
        #### 3. Calculate the pairwise similarity for activations ####
        #################################################################
        logger.info("Calculating RSA...")
        rsa_correlation_matrix = np.zeros((n_layers, n_heads)) * np.nan
        start_layer_idx = args.start_layer_idx
        end_layer_idx = args.end_layer_idx if args.end_layer_idx is not None else n_layers
        start_head_idx = args.start_head_idx 
        end_head_idx = args.end_head_idx if args.end_head_idx is not None else n_heads

        for layer_idx in tqdm(range(start_layer_idx, end_layer_idx)):
            for head_idx in range(start_head_idx, end_head_idx):

                if args.only_for_significant_heads and (layer_idx, head_idx) not in head_list:
                    continue
                        
                feature = all_act_sel[:, layer_idx, head_idx, :]
                feature_sim = cal_similarity(feature)

                if args.plot_similarity:
                    plot(
                        feature_sim.cpu().numpy(), token_sub_save_folder, metric_name=f"Layer_{layer_idx}_Head_{head_idx}", 
                        num_for_one_rule=feature_sim.shape[0]//2 if not args.transpose else None, xlabel_name="Index", ylabel_name="Index",
                        fig_w=55, fig_h=35
                        )
                
                if args.save_similarity:
                    torch.save(feature_sim.cpu(), os.path.join(token_sub_save_folder, f"similarity_matrix_Layer_{layer_idx}_Head_{head_idx}.pt"))

                rsa_correlation = compare_two_similarity_matrix(feature_sim, hand_code_sim_matrix, include_diagnoal=False) 
                rsa_correlation_matrix[layer_idx, head_idx] = rsa_correlation
                if args.only_for_significant_heads:
                    weight = head_weight_score[layer_idx, head_idx]
                    if weighted_avg_sim is None:
                        weighted_avg_sim = feature_sim.cpu() * weight
                    else:
                        weighted_avg_sim += feature_sim.cpu() * weight
                    weighted_rsa_corr += rsa_correlation * weight.item()
                    weighted_sum += weight.item()

        np.save(os.path.join(token_sub_save_folder, f"corr_mtx_h_{high_sim}_l_{low_sim}.npy"), rsa_correlation_matrix)
        if args.plot_rsa:
            plot(
                rsa_correlation_matrix, token_sub_save_folder, metric_name=f"Corr_Mtx_{act}_H_{high_sim}_L_{low_sim}",
            )
    
        if args.only_for_significant_heads:
            weighted_avg_sim = weighted_avg_sim / weighted_sum
            weighted_rsa_corr /= weighted_sum
            torch.save(weighted_avg_sim.cpu(), os.path.join(token_sub_save_folder, f"{args.head_type}_significant_head_{len(head_list)}_weighted_sim.pt"))
            plot(weighted_avg_sim.cpu().numpy(), token_sub_save_folder, f"{args.head_type} WSim", xlabel_name="", ylabel_name="")
            logger.info(f"<{args.head_type}|{act}> Weighted RSA correlation across all {len(head_list)} significant heads: {weighted_rsa_corr}")
            with open(os.path.join(token_sub_save_folder, f"significant_head_weighted_rsa_corr.txt"), "a") as f:
                f.write(f"{args.head_type} ({len(head_list)}): {weighted_rsa_corr}\n")


def cal_similarity(features):
    norm_features = torch.nn.functional.normalize(features, p=2, dim=-1)
    norm_features = torch.movedim(norm_features, 0, -2)
    sim = torch.matmul(norm_features, norm_features.transpose(-1, -2))
    return sim

def compare_two_similarity_matrix(real_matrix, exp_matrix, include_diagnoal=False):
    ## Calculate the Pearson correlation coefficient between the lower triangular part of two similarity matrices
    tril_indices = torch.tril_indices(exp_matrix.shape[0], exp_matrix.shape[1], offset=-1 if not include_diagnoal else 0)
    exp_matrix_tril_values = exp_matrix[tril_indices[0], tril_indices[1]].cpu().numpy()
    real_matrix_tril_values = real_matrix[tril_indices[0], tril_indices[1]].cpu().numpy()
    psr, _ = pearsonr(exp_matrix_tril_values, real_matrix_tril_values)
    return psr


if __name__ == "__main__":
    args = get_args()
    main(args)
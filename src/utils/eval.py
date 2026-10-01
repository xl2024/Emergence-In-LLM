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

import torch
import random
import numpy as np
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
from transformers import AutoTokenizer, GenerationConfig
from nnsight import LanguageModel

from src.utils.utils import set_seed, LINE_SEP, vocab_dict, get_model_id_family, load_prompts
from src.utils.args import _get_args
from src.utils.tools import generate_prompts, generate_response_eval, get_eos_token_id, get_generation_kwargs

def get_args():
    return _get_args('eval')

def main(args):
    set_seed(args.seed)
    assert args.rule in ["ABA", "ABB"] 

    # get the model id and family
    model_id, model_family =  get_model_id_family(args.model_type) 
    vocab_file = vocab_dict[model_family] ## get the vocabulary file path
    logger.info(f"model type: {args.model_type}, model id: {model_id}, vocab file: {vocab_file}")

    # 0. Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = HF_TOKEN)
    logger.info(f"Loading {model_id} via nnsight...")
    model = LanguageModel(
        model_id, device_map=args.device_map, 
        dtype=torch.bfloat16, token=HF_TOKEN
    ) 
    torch.set_grad_enabled(False)
    model.eval()    # no Dropout, frozen BatchNorm
    # the generation_config of the model initialized by from_pretrained() could be top-k/top-p sampling, please check the generation_config
    # different sampling strategies may bring slightly differences in model accuracy but overall findings remain the same.
    generation_kwargs = get_generation_kwargs(args)
    eos_token_id = get_eos_token_id(tokenizer, args.eos_token)
    
    #### 1. Build the prompt dataset ####
    if args.prompt_file is None:
        logger.info(f"generate prompts.... (SEP symbol: {args.sep_symbol})")
        assert args.prompt_num is not None
        prompts, correct_ans_list = generate_prompts(
            args, tokenizer, vocab_file, sep_symbol=args.sep_symbol, 
            base_rule=args.rule, return_format="others"
        )

    else:
        assert os.path.exists(args.prompt_file)
        prompts, correct_ans_list = load_prompts(args.prompt_file, args.rule, sep_symbol = args.sep_symbol, in_context_num=args.in_context_example_num, sample_num=args.prompt_num)
        logger.info(f"load prompts from file: {args.prompt_file}")

    logger.info(f"# total prompts: {len(prompts)}")
    remark = f"in_context_example_{args.in_context_example_num}/{args.model_type}/rule_{args.rule}/prompts_{len(prompts)}_seed_{args.seed}"
    sup_folder = os.path.join(args.log_dir, remark)
    os.makedirs(sup_folder, exist_ok=True)
    with open(os.path.join(sup_folder, f"input_prompts_{len(prompts)}.txt"), "w") as f:
        for p in prompts:
            f.write(f"{LINE_SEP}" + p + "\n")

    sample_remark = f"sample_size_per_prompt_{args.sample_size}{args.sample_remark}"
    logdir = os.path.join(sup_folder, sample_remark)
    os.makedirs(logdir, exist_ok=True)
    logger.info(f"create logdir: {logdir}")

    acc_list = []
    abnormal_num = 0
    correct_num = 0
    total_num = 0
    correct_generation_prompt = []
    acc_threshold = args.acc_threshold

    for i, prompt in enumerate(tqdm(prompts)):
        # prompt = [prompt] * args.sample_size 
        ## for each prompt, we sample 4 times. If sampling strategy is not greedy, the responses may be different.
        correct_ans = correct_ans_list[i]

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids

        # return_dict_in_generate=True, output_logits=True: unused
        # attention_mask=inputs.attention_mask: default to all 1's
        _correct_num, _total_num, acc = generate_response_eval(
            model, input_ids, eos_token_id, tokenizer, correct_ans, prompt,
            sample_size=args.sample_size, max_new_tokens=args.max_new_tokens, **generation_kwargs
        )
        correct_num += _correct_num
        total_num += _total_num
        if acc >= acc_threshold: ## save the prompts on which the model accuracy is larger than the threshold
            correct_generation_prompt.append(prompt[0])
        acc_list.append(acc)

    avg_acc = np.mean(acc_list)
    binom_conf = proportion_confint(count=correct_num, nobs=total_num, method='wilson')
    print_str = ""
    logger.info("========================================")
    print_str += f"model type: {args.model_type}; rule: {args.rule}\n" 
    print_str += f"# in-context example: {args.in_context_example_num}; # prompt: {args.prompt_num}; # sample per prompt: {args.sample_size}; seed: {args.seed}\n"
    print_str += f"<accuracy> average: {avg_acc}; std: {np.std(acc_list)}; binom_conf: {binom_conf}; set: {set(acc_list)}\n"
    logger.info(print_str)
    logger.info("========================================")
    with open(os.path.join(logdir, f"log_generated.txt"), "w") as f:
        f.write(print_str + "\n")
    
    with open(os.path.join(logdir, f"correct_generated_prompt_{len(correct_generation_prompt)}_threshold_{acc_threshold}.txt"), "w") as f:
        for p in correct_generation_prompt:
            f.write(f"{LINE_SEP}" + p + "\n")

    sum_path = os.path.join("outputs", "eval", "sum_dict.pt")
    if os.path.exists(sum_path):
        sum_dict = torch.load(sum_path, map_location="cpu", weights_only=False)
    else:
        sum_dict = {}
    if args.model_type not in sum_dict:
        sum_dict[args.model_type] = {}
    if args.rule not in sum_dict[args.model_type]:
        sum_dict[args.model_type][args.rule] = {}
    sum_dict[args.model_type][args.rule][args.in_context_example_num] = {
        "acc_list": acc_list,
        "correct_num": correct_num,
        "total_num": total_num
    }
    torch.save(sum_dict, sum_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
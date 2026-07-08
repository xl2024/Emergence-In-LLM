import os

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"] 
HF_HOME = os.environ["HF_HOME"]

import logging
logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import random
import numpy as np
import torch
from nnsight import LanguageModel
from typing import Dict, Any, List, Tuple
from transformers import GenerationConfig

from src.utils.utils import get_model_id_family

class PrintLogger:
    def info(self, *args, **kwargs):
        print("== [logger.info] ==", *args)

    def warning(self, *args, **kwargs):
        print("== [logger.warning] ==", *args)

    def error(self, *args, **kwargs):
        print("== [logger.error] ==", *args)

    def debug(self, *args, **kwargs):
        print("== [logger.debug] ==", *args)

# logger = PrintLogger()

def generate_prompts(
        args, tokenizer, vocab_file, sep_symbol="^", token_sets=None, do_shuffle=False, base_rule="ABA", context_type="abstract", add_swap_1_2_question=False, return_format="others"
    ):
    """
    return_format choices: "others" (Eval/Ablation), "cma" (CMA), "rsa" (RSA)
    """
    prompts, correct_ans_list, rule_group_list, causal_ans_list = [], [], [], []

    assert os.path.exists(vocab_file), f"Vocabulary file {vocab_file} does not exist."
    with open(vocab_file, "r") as f:
        vocab_list = [l.rstrip() for l in f.readlines()]
    
    if do_shuffle: 
        np.random.shuffle(vocab_list)
        if token_sets is not None:
            np.random.shuffle(token_sets)
    assert context_type in ["abstract", "token"], f"Invalid context type: {context_type}"

    while len(prompts) < args.prompt_num:
        ## generate the prompt
        ## 1. get the tokens for each in-context example
        if token_sets is None:
            tokens = random.sample(vocab_list, k =(args.in_context_example_num + 1) * 2)
        else:
            tokens = token_sets.pop(0) # la li te to hi ha 
            
        base_example_list, exp_example_list = [], []
        for idx_ in range(args.in_context_example_num):
            if base_rule == "ABA":
                base_tokens = [tokens[idx_*2], tokens[idx_*2+1], tokens[idx_*2]]  ## la li la
            elif base_rule == "ABB":
                base_tokens = [tokens[idx_*2], tokens[idx_*2+1], tokens[idx_*2+1]] ## la li li

            base_example = f"{sep_symbol}".join(base_tokens) 
            base_example_list.append(base_example)
            
            if return_format != "others":
                if context_type == "token":
                    exp_tokens = base_tokens
                elif context_type == "abstract":
                    if base_rule == "ABA":
                        exp_tokens = [tokens[idx_*2+1], tokens[idx_*2], tokens[idx_*2]] ## li la la
                    elif base_rule == "ABB":
                        exp_tokens = [tokens[idx_*2+1], tokens[idx_*2], tokens[idx_*2+1]] ## li la li
                
                exp_example = f"{sep_symbol}".join(exp_tokens)
                exp_example_list.append(exp_example)
        
        # incomplete query (question) for the last in-context example
        base_question_tokens = [tokens[-2], tokens[-1]] # hi ha
        base_ans_index = 0 if base_rule == "ABA" else 1
        base_ans = base_question_tokens[base_ans_index]

        base_question = f"{sep_symbol}".join(base_question_tokens) + f"{sep_symbol}"
        base_example_list.append(base_question)
        base_prompt = "\n".join(base_example_list)        

        if return_format == "others":
            ### sanity check to make sure that the tokens will not be grouped with separation symbols in the tokenization. ###
            proc_tokens = tokenizer.tokenize(base_prompt)
            token_num = len(proc_tokens)
            if token_num != 3 * 2 * args.in_context_example_num + 4:
                logger.info("====================================================")
                logger.info(f"Invalid prompt: {base_prompt}, token_num: {token_num}, expected: {3 * 2 * args.in_context_example_num + 4}")
                continue
            else:
                if np.sum(~np.isin(proc_tokens[1::2], [sep_symbol, tokenizer.tokenize("\n")[0]])):
                    logger.info("====================================================")
                    logger.info(f"Invalid prompt: {base_prompt}, tokens: {proc_tokens}")
                    continue

            if base_prompt not in prompts:
                prompts.append(base_prompt)
                correct_ans_list.append(base_ans)
            else:
                logger.info("====================================================")
                logger.info(f"Repeated prompt: {base_prompt}")
  
            continue

        exp_question_tokens = [tokens[-1], tokens[-2]] # ha hi # swap the 1/2 item in the last incomplete query (question)
        exp_question = f"{sep_symbol}".join(exp_question_tokens) + f"{sep_symbol}" 
        exp_example_list.append(exp_question)
        exp_prompt = "\n".join(exp_example_list)
        if context_type == "abstract":
            exp_ans_index = 1 - base_ans_index
            causal_ans = exp_question_tokens[base_ans_index]
        else:
            exp_ans_index = base_ans_index
            causal_ans = base_ans 
        
        exp_ans = exp_question_tokens[exp_ans_index]
        
        prompt_group = [base_prompt, exp_prompt]
        ans_group = [base_ans, exp_ans]

        if return_format == "rsa":
            if base_rule == "ABA":
                exp_rule = "ABB"
            elif base_rule == "ABB":
                exp_rule = "ABA"
            rule_group = [base_rule, exp_rule]
    
            if add_swap_1_2_question:
                add_base_question_tokens = base_question_tokens[::-1]
                add_base_question = f"{sep_symbol}".join(add_base_question_tokens) + f"{sep_symbol}"
                add_base_ans = add_base_question_tokens[base_ans_index]

                base_example_list[-1] = add_base_question
                add_base_prompt = "\n".join(base_example_list)
                prompt_group.append(add_base_prompt)
                ans_group.append(add_base_ans)

                add_exp_question_tokens = exp_question_tokens[::-1]
                add_exp_question = f"{sep_symbol}".join(add_exp_question_tokens) + f"{sep_symbol}"
                add_exp_ans = add_exp_question_tokens[exp_ans_index]

                exp_example_list[-1] = add_exp_question
                add_exp_prompt = "\n".join(exp_example_list)
                prompt_group.append(add_exp_prompt)
                ans_group.append(add_exp_ans)

                rule_group = rule_group * 2
            
        ### sanity check ###
        pass_tag = True
        for prompt in prompt_group:
            proc_tokens = tokenizer.tokenize(prompt)
            token_num = len(proc_tokens)
            if token_num != 3 * 2 * args.in_context_example_num + 4:
                if args.verbose:
                    logger.info("====================================================")
                    logger.info(f"Invalid prompt: {prompt}, token_num: {token_num}, expected: {3 * 2 * args.in_context_example_num + 4}")
                pass_tag = False
                break
            else:
                # if np.sum([1 for k_ in proc_tokens[1::2] if k_ not in [sep_symbol, tokenizer.tokenize("\n")[0]]]):
                if np.sum(~np.isin(proc_tokens[1::2], [sep_symbol, tokenizer.tokenize("\n")[0]])):
                    if args.verbose:
                        logger.info("====================================================")
                        logger.info(f"Invalid prompt: {prompt}, tokens: {proc_tokens}")
                    pass_tag = False
                    break

        if not pass_tag:
            continue

        if return_format == "rsa":
            if prompt_group not in prompts:
                prompts.append(prompt_group)
                correct_ans_list.append(ans_group)
                rule_group_list.append(rule_group)
            else:
                if args.verbose:
                    logger.info("====================================================")
                    logger.info(f"Repeated prompt: {prompt}")

        elif return_format == "cma":
            if prompt_group not in prompts:
                prompts.append(prompt_group)
                correct_ans_list.append(ans_group)
                causal_ans_list.append(causal_ans)
            else:
                if args.verbose:
                    logger.info("====================================================")
                    logger.info(f"Repeated prompt: {prompt}")

    if return_format == "others":
        return prompts, correct_ans_list
    
    if return_format == "cma":
        return prompts, correct_ans_list, causal_ans_list
    
    if return_format == "rsa":
        return prompts, correct_ans_list, rule_group_list

# borrowed from the VLM reproduction repo
def _resolve_layer_path(model: LanguageModel, path_string: str):
    """
    Safely traverses the nnsight model architecture to return the exact 
    PyTorch module based on the config's string path.
    
    Example: 
        path_string = "model.language_model.model.layers[8]"
    """
    # We split by '.' and handle list indices like 'layers[8]'
    current_module = model
    parts = path_string.split('.')
    
    for part in parts:
        if '[' in part and ']' in part:
            attr_name, index_part = part.split('[')
            index = int(index_part.replace(']', ''))
            current_module = getattr(current_module, attr_name)[index]
        else:
            current_module = getattr(current_module, part)
            
    return current_module

# modified from the VLM reproduction repo (support gpt-2)
def _resolve_text_model_dims(model: Any, kv_heads: bool = False) -> Tuple[int, int]:
    """
    Resolve (hidden_size, num_attention_heads) across wrapped/unwrapped VLM models.
    Works when `model.config` is missing/None (common with wrappers) and supports legacy models like GPT-2.
    """
    candidate_configs: List[Any] = []

    # Direct config on the visible object
    candidate_configs.append(getattr(model, "config", None))

    # Common nnsight/HF wrapper patterns
    local_model = getattr(model, "local_model", None)
    if local_model is not None:
        candidate_configs.append(getattr(local_model, "config", None))

    nested_model = getattr(model, "model", None)
    if nested_model is not None:
        candidate_configs.append(getattr(nested_model, "config", None))
        language_model = getattr(nested_model, "language_model", None)
        if language_model is not None:
            candidate_configs.append(getattr(language_model, "config", None))

    # Some multimodal models expose text dims under text_config
    expanded_configs: List[Any] = []
    for cfg in candidate_configs:
        if cfg is None:
            continue
        expanded_configs.append(cfg)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            expanded_configs.append(text_cfg)

    for cfg in expanded_configs:
        # Support GPT-2 legacy naming (n_embd, n_head)
        hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
        num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
        if kv_heads:
            # Fall back to standard num_heads if num_key_value_heads is missing (e.g. GPT-2 standard MHA)
            num_heads = getattr(cfg, "num_key_value_heads", num_heads)
        if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0:
            return hidden_size, num_heads

    raise AttributeError(
        "Could not resolve hidden_size/num_attention_heads from model object. "
        "Expected fields on config or text_config."
    )

# modified from the VLM reproduction repo (support standard text models and gpt-2)
def get_num_hidden_layers(model: Any) -> int:
    """
    Resolve decoder layer count across wrapped/unwrapped VLM model objects,
    including standard HF LLMs, legacy GPT-2, and modern Vision-Language wrappers.
    """
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "local_model"):
        config = getattr(model.local_model, "config", None)
    if config is None and hasattr(model, "_model"):
        config = getattr(model._model, "config", None)
    if config is None and hasattr(model, "model"):
        config = getattr(model.model, "config", None)
        
    if config is None:
        config = model 
        
    for attr in ["num_hidden_layers", "n_layer", "n_layers"]:
        if hasattr(config, attr):
            return getattr(config, attr)


    # Typical HF multimodal configs (e.g., LlavaForConditionalGeneration)
    if hasattr(model, "config") and hasattr(model.config, "text_config") and hasattr(model.config.text_config, "num_hidden_layers"):
        return model.config.text_config.num_hidden_layers

    # Some wrappers expose the nested module path directly
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return len(model.model.language_model.layers)

    # IDEFICS2: nested model.text_model
    if (
        hasattr(model, "model")
        and hasattr(model.model, "text_model")
        and hasattr(model.model.text_model, "config")
        and hasattr(model.model.text_model.config, "num_hidden_layers")
    ):
        return model.model.text_model.config.num_hidden_layers

    # Legacy/alternate wrapper pattern
    if (
        hasattr(model, "local_model")
        and hasattr(model.local_model, "config")
        and hasattr(model.local_model.config, "text_config")
        and hasattr(model.local_model.config.text_config, "num_hidden_layers")
    ):
        return model.local_model.config.text_config.num_hidden_layers

    raise AttributeError("Could not infer number of hidden layers from model object.")

def get_layer_and_proj_name(model: Any, activation_name: str = "z") -> Tuple[str, str]:
    """
    Returns the format string for the layer and the exact projection module name 
    based on the target activation (z/o, q, k, v).
    """
    logger = PrintLogger()    # for remote running
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "")
    
    # Map legacy GPT-2 structure
    if model_type == "gpt2":
        layer_fmt = "transformer.h[{}]"
        if activation_name in ["z", "o"]:
            proj_name = "attn.c_proj"
        elif activation_name in ["q", "k", "v"]:
            logger.warning(f"GPT-2 combines QKV into `attn.c_attn`. Nnsight cannot isolate '{activation_name}' directly via module path.")
            proj_name = "attn.c_attn"
        else:
            proj_name = "attn.c_proj"
            
    # Default modern HF text structure (Llama, Qwen, Gemma, etc.)
    else:
        layer_fmt = "model.layers[{}]"
        if activation_name == "q":
            proj_name = "self_attn.q_proj"
        elif activation_name == "k":
            proj_name = "self_attn.k_proj"
        elif activation_name == "v":
            proj_name = "self_attn.v_proj"
        else:
            proj_name = "self_attn.o_proj" # Default to 'z' / 'o'
            
    return layer_fmt, proj_name

def generate_response_eval(model, input_ids, eos_token_id, tokenizer, correct_ans, prompt, sample_size=4, max_new_tokens=10, intervention_fn=None, **generation_kwargs):
    """
        Generate responses and evaluate the generation accuracy. Capable of adding hooks to the model (e.g., patching activation) during generation.
    """
    if eos_token_id is not None:
        if isinstance(eos_token_id, list) and len(eos_token_id) == 1:
            eos_token_id = eos_token_id[0]

    with torch.no_grad():
        with model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            num_return_sequences=sample_size,
            remote=True,
            **generation_kwargs
        ) as tracer:
            if intervention_fn is not None:
                intervention_fn(model)
            generated_ids = tracer.result.save()
        
    response_ids = generated_ids[:, input_ids.shape[1]:] ## the responses generated by the model
    correct_num = 0
    for response_id in response_ids:
        eos_sign = False
        ## find the first eos token
        for ri, r in enumerate(response_id): 
            if r in eos_token_id:
                r_mask = ri
                eos_sign = True
                break
        if not eos_sign: ## response does not contain the eos token
            r_mask = len(response_id)
            logger.warning("--------------------------------------------------")
            logger.warning(f"Response does not contain the eos token: {response_id} (not in {eos_token_id})")

        valid_response_id = response_id[:r_mask] ## remove the eos token
        gen_ans = tokenizer.decode(valid_response_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        correct_num += (gen_ans == correct_ans)
        if r_mask > 1:  # generated answers include multiple tokens
            multipe_tokens = tokenizer.convert_ids_to_tokens(response_id)
            logger.warning("--------------------------------------------------")
            logger.warning(f"Question: {prompt}")
            logger.warning(f"Response contains multiple tokens: {gen_ans}, correct ans: {correct_ans}, multiple tokens: {multipe_tokens}")
    acc = correct_num / sample_size
    return correct_num, sample_size, acc
    
def has_prepend_bos(tokenizer):
    dummy_ids = tokenizer("a").input_ids
    prepend_bos = (tokenizer.bos_token_id is not None and dummy_ids[0] == tokenizer.bos_token_id)
    return prepend_bos

def get_layer_paths(model, activation_names):
    n_layers = get_num_hidden_layers(model)
    layer_paths = {}
    for layer_idx in range(n_layers):
        layer_paths[layer_idx] = {}
        for act in activation_names:
            layer_fmt, proj_name = get_layer_and_proj_name(model, act)
            layer_paths[layer_idx][act] = f"{layer_fmt.format(layer_idx)}.{proj_name}"
    return layer_paths

def get_generation_kwargs(args, **kwargs):
    ## specify the sampling config for generation
    generation_config_name = args.generation_config_name if args.generation_config_name is not None else f"{args.model_type}.json"
    
    if hasattr(args, "save_generation_config") and args.save_generation_config:
        logger.info(f"save generation config to file: ./generation_config/{generation_config_name}")
        model_id, model_family =  get_model_id_family(args.model_type) 
        generation_config = GenerationConfig.from_pretrained(model_id, token=HF_TOKEN)
        generation_config.save_pretrained("./generation_config/", config_file_name=generation_config_name)

    if args.load_generation_config: ## load the generation config for the model (greedy sampling, used in the paper), 
        ### Use the generation config saved in ./generation_config/ folder
        ### OR GET THE GENERATION CONFIG FILE BY setting args.save_generation_config to True when running the tasks/identity_rules/eval.py 
        ### OR
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_id, device_map=args.device_map, 
        #     dtype=torch.bfloat16
        # )
        # model.generation_config.save_pretrained("./generation_config/", config_file_name=f"{args.model_type}.json")
        ###############
        ###############
        logger.info(f"loading generation config from ./generation_config/{generation_config_name}")
        generation_config = GenerationConfig.from_pretrained("./generation_config/", config_file_name=generation_config_name)
        generation_kwargs = {
            "top_k": generation_config.top_k,
            "top_p": generation_config.top_p,
            "do_sample": generation_config.do_sample,
            "temperature": generation_config.temperature,
            "freq_penalty": getattr(generation_config, "repetition_penalty", None)
        }
    else: ## Some default generation settings for multinomial sampling
        logger.warning("Using default multinomial sampling generation config [Not recommended]")
        generation_kwargs = {
            "top_k": 50,
            "top_p": 0.9,
            "do_sample": True,
            "temperature": 0.6
        }
    generation_kwargs.update(kwargs)
    logger.info(f"generation_kwargs: {generation_kwargs}")

    return generation_kwargs

def get_token_sets(token_set_file):
    ## Strongly recommend to use the token set file that contains the tokens that the model can predict correctly
    if token_set_file is not None:
        assert os.path.exists(token_set_file)
        logger.info(f"Using token set file: {token_set_file}, which contains the tokens that the model can predict correctly")
        with open(token_set_file, "r") as f:
            token_sets = f.readlines()
            token_sets = [t.rstrip().split(" ") for t in token_sets]
    else:
        logger.warning("No token set file is provided, random tokens will be used, which model may not predict correctly")
        token_sets = None

    return token_sets

def get_eos_token_id(tokenizer, eos_token):
    # add new eos token based on the prompt format [Useful for Generation]
    # the prompt format is like: "la^li^la\nte^to^te\nha^hi^", where the eos token should be "\n"
    eos_token_id = tokenizer.eos_token_id
    vocab = tokenizer.get_vocab()
    if eos_token is None: # if eos_token is not specified, we set it as "\n"
        eos_token = "\n"
    add_eos_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(eos_token)[-1])
    logger.info(f"eos_token: {eos_token}, add_eos_token: {add_eos_token}")
    add_eos_token_dict = {v:vocab[v] for v in vocab if v.startswith(add_eos_token)} ## find all tokens which start with the add_eos_token (e.g. "\n")
    add_eos_token_id = list(add_eos_token_dict.values())
    if type(eos_token_id) != list:
        logger.info(f"eos_token_id: {eos_token_id}, eos: {tokenizer.convert_ids_to_tokens(eos_token_id)}")
        eos_token_id = [eos_token_id]
    eos_token_id.extend(add_eos_token_id)

    return eos_token_id
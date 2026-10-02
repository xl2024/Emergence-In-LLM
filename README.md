This repo reproduces the results in the main text of the paper _Emergent Symbolic Mechanisms
Support Abstract Reasoning in LLMs_, reusing their official code when possible.

#### Reproduction of Figure 2
<p align="center">
  <img src="outputs/cma/Llama-3.1-70B/abstract_context/base_rule_ABA_exp_rule_ABB/attn_out_seed_0_shuffle_True/Score_Patching_ABA_to_ABB_invert_y_axis_heatmap.png" width="45%" alt="Llama-3.1-70B_Abstract-Causal-Mediation_ABA">
  <img src="outputs/cma/Llama-3.1-70B/token_context/base_rule_ABA_exp_rule_ABA/attn_out_seed_0_shuffle_True/Score_Patching_ABA_to_ABA_invert_y_axis_heatmap.png" width="45%" alt="Llama-3.1-70B_Token-Causal-Mediation_ABA">
</p>
<p align="center">
  <em>The Abstract Causal Mediation and Token Causal Mediation for Llama-3.1-70B with base rule ABA</em>
</p>

<p align="center">
  <img src="outputs/cma/Llama-3.1-70B/abstract_context/base_rule_ABB_exp_rule_ABA/attn_out_seed_0_shuffle_True/Score_Patching_ABB_to_ABA_invert_y_axis_heatmap.png" width="45%" alt="Llama-3.1-70B_Abstract-Causal-Mediation_ABB">
  <img src="outputs/cma/Llama-3.1-70B/token_context/base_rule_ABB_exp_rule_ABB/attn_out_seed_0_shuffle_True/Score_Patching_ABB_to_ABB_invert_y_axis_heatmap.png" width="45%" alt="Llama-3.1-70B_Token-Causal-Mediation_ABB">
</p>
<p align="center">
  <em>The Abstract Causal Mediation and Token Causal Mediation for Llama-3.1-70B with base rule ABB</em>
</p>

<p align="center">
  <img src="outputs/cma/Llama-3.1-8B/abstract_context/base_rule_ABA_exp_rule_ABB/attn_out_seed_0_shuffle_True/Score_Patching_ABA_to_ABB_invert_y_axis_heatmap.png" width="45%" alt="Llama-3.1-8B_Abstract-Causal-Mediation_ABA">
  <img src="outputs/cma/Llama-3.1-8B/token_context/base_rule_ABA_exp_rule_ABA/attn_out_seed_0_shuffle_True/Score_Patching_ABA_to_ABA_invert_y_axis_heatmap.png" width="45%" alt="Llama-3.1-8B_Token-Causal-Mediation_ABA">
</p>
<p align="center">
  <em>The Abstract Causal Mediation and Token Causal Mediation for Llama-3.1-8B with base rule ABA</em>
</p>

<p align="center">
  <img src="outputs/cma/Llama-3.1-8B/abstract_context/base_rule_ABB_exp_rule_ABA/attn_out_seed_0_shuffle_True/Score_Patching_ABB_to_ABA_invert_y_axis_heatmap.png" width="45%" alt="Llama-3.1-8B_Abstract-Causal-Mediation_ABB">
  <img src="outputs/cma/Llama-3.1-8B/token_context/base_rule_ABB_exp_rule_ABB/attn_out_seed_0_shuffle_True/Score_Patching_ABB_to_ABB_invert_y_axis_heatmap.png" width="45%" alt="Llama-3.1-8B_Token-Causal-Mediation_ABB">
</p>
<p align="center">
  <em>The Abstract Causal Mediation and Token Causal Mediation for Llama-3.1-8B with base rule ABB</em>
</p>

<p align="center">
  <img src="outputs/cma/Llama-3.1-70B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_True_pos_4_10/token_pos_%5B5%2C%2011%5D/Score_Patching_ABA_to_ABB_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-70B_Symbol-Abstraction-Heads_ABA">
  <img src="outputs/cma/Llama-3.1-70B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_True/token_pos_%5B-1%5D/Score_Patching_ABA_to_ABB_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-70B_Symbolic-Induction-Heads_ABA">
  <img src="outputs/cma/Llama-3.1-70B/token_context/base_rule_ABA_exp_rule_ABA/z_seed_0_shuffle_True/token_pos_%5B-1%5D/Score_Patching_ABA_to_ABA_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-70B_Retrieval-Heads_ABA">
</p>
<p align="center">
  <em>The Symbol Abstraction Heads, Symbolic Induction Heads and Retrieval Heads for Llama-3.1-70B with base rule ABA</em>
</p>

<p align="center">
  <img src="outputs/cma/Llama-3.1-70B/abstract_context/base_rule_ABB_exp_rule_ABA/z_seed_0_shuffle_True_pos_4_10/token_pos_%5B5%2C%2011%5D/Score_Patching_ABB_to_ABA_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-70B_Symbol-Abstraction-Heads_ABB">
  <img src="outputs/cma/Llama-3.1-70B/abstract_context/base_rule_ABB_exp_rule_ABA/z_seed_0_shuffle_True/token_pos_%5B-1%5D/Score_Patching_ABB_to_ABA_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-70B_Symbolic-Induction-Heads_ABB">
  <img src="outputs/cma/Llama-3.1-70B/token_context/base_rule_ABB_exp_rule_ABB/z_seed_0_shuffle_True/token_pos_%5B-1%5D/Score_Patching_ABB_to_ABB_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-70B_Retrieval-Heads_ABB">
</p>
<p align="center">
  <em>The Symbol Abstraction Heads, Symbolic Induction Heads and Retrieval Heads for Llama-3.1-70B with base rule ABB</em>
</p>

<p align="center">
  <img src="outputs/cma/Llama-3.1-8B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_True_pos_4_10/token_pos_%5B5%2C%2011%5D/Score_Patching_ABA_to_ABB_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-8B_Symbol-Abstraction-Heads_ABA">
  <img src="outputs/cma/Llama-3.1-8B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_True/token_pos_%5B-1%5D/Score_Patching_ABA_to_ABB_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-8B_Symbolic-Induction-Heads_ABA">
  <img src="outputs/cma/Llama-3.1-8B/token_context/base_rule_ABA_exp_rule_ABA/z_seed_0_shuffle_True/token_pos_%5B-1%5D/Score_Patching_ABA_to_ABA_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-8B_Retrieval-Heads_ABA">
</p>
<p align="center">
  <em>The Symbol Abstraction Heads, Symbolic Induction Heads and Retrieval Heads for Llama-3.1-8B with base rule ABA</em>
</p>

<p align="center">
  <img src="outputs/cma/Llama-3.1-8B/abstract_context/base_rule_ABB_exp_rule_ABA/z_seed_0_shuffle_True_pos_4_10/token_pos_%5B5%2C%2011%5D/Score_Patching_ABB_to_ABA_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-8B_Symbol-Abstraction-Heads_ABB">
  <img src="outputs/cma/Llama-3.1-8B/abstract_context/base_rule_ABB_exp_rule_ABA/z_seed_0_shuffle_True/token_pos_%5B-1%5D/Score_Patching_ABB_to_ABA_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-8B_Symbolic-Induction-Heads_ABB">
  <img src="outputs/cma/Llama-3.1-8B/token_context/base_rule_ABB_exp_rule_ABB/z_seed_0_shuffle_True/token_pos_%5B-1%5D/Score_Patching_ABB_to_ABB_invert_y_axis_heatmap.png" width="30%" alt="Llama-3.1-8B_Retrieval-Heads_ABB">
</p>
<p align="center">
  <em>The Symbol Abstraction Heads, Symbolic Induction Heads and Retrieval Heads for Llama-3.1-8B with base rule ABB</em>
</p>

#### Reproduction of Figure 3
<p align="center">
  <img src="outputs/attn/Llama-3.1-70B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_True_pos_4_10/Attention_Analysis_ABA.png" width="30%" alt="Llama-3.1-70B_Attention-Analysis_Symbol-Abstraction-Heads_ABA">
  <img src="outputs/attn/Llama-3.1-70B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_True/Attention_Analysis_ABA.png" width="30%" alt="Llama-3.1-70B_Attention-Analysis_Symbolic-Induction-Heads_ABA">
  <img src="outputs/attn/Llama-3.1-70B/token_context/base_rule_ABA_exp_rule_ABA/z_seed_0_shuffle_True/Attention_Analysis_ABA.png" width="30%" alt="Llama-3.1-70B_Attention-Analysis_Retrieval-Heads_ABA">
</p>
<p align="center">
  <em>The Attention Analysis for Symbol Abstraction Heads, Symbolic Induction Heads and Retrieval Heads for Llama-3.1-70B with base rule ABA</em>
</p>

<p align="center">
  <img src="outputs/attn/Llama-3.1-70B/abstract_context/base_rule_ABB_exp_rule_ABA/z_seed_0_shuffle_True_pos_4_10/Attention_Analysis_ABB.png" width="30%" alt="Llama-3.1-70B_Attention-Analysis_Symbol-Abstraction-Heads_ABB">
  <img src="outputs/attn/Llama-3.1-70B/abstract_context/base_rule_ABB_exp_rule_ABA/z_seed_0_shuffle_True/Attention_Analysis_ABB.png" width="30%" alt="Llama-3.1-70B_Attention-Analysis_Symbolic-Induction-Heads_ABB">
  <img src="outputs/attn/Llama-3.1-70B/token_context/base_rule_ABB_exp_rule_ABB/z_seed_0_shuffle_True/Attention_Analysis_ABB.png" width="30%" alt="Llama-3.1-70B_Attention-Analysis_Retrieval-Heads_ABB">
</p>
<p align="center">
  <em>The Attention Analysis for Symbol Abstraction Heads, Symbolic Induction Heads and Retrieval Heads for Llama-3.1-70B with base rule ABB</em>
</p>

<p align="center">
  <img src="outputs/attn/Llama-3.1-8B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_True_pos_4_10/Attention_Analysis_ABA.png" width="30%" alt="Llama-3.1-8B_Attention-Analysis_Symbol-Abstraction-Heads_ABA">
  <img src="outputs/attn/Llama-3.1-8B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_True/Attention_Analysis_ABA.png" width="30%" alt="Llama-3.1-8B_Attention-Analysis_Symbolic-Induction-Heads_ABA">
  <img src="outputs/attn/Llama-3.1-8B/token_context/base_rule_ABA_exp_rule_ABA/z_seed_0_shuffle_True/Attention_Analysis_ABA.png" width="30%" alt="Llama-3.1-8B_Attention-Analysis_Retrieval-Heads_ABA">
</p>
<p align="center">
  <em>The Attention Analysis for Symbol Abstraction Heads, Symbolic Induction Heads and Retrieval Heads for Llama-3.1-8B with base rule ABA</em>
</p>

<p align="center">
  <img src="outputs/attn/Llama-3.1-8B/abstract_context/base_rule_ABB_exp_rule_ABA/z_seed_0_shuffle_True_pos_4_10/Attention_Analysis_ABB.png" width="30%" alt="Llama-3.1-8B_Attention-Analysis_Symbol-Abstraction-Heads_ABB">
  <img src="outputs/attn/Llama-3.1-8B/abstract_context/base_rule_ABB_exp_rule_ABA/z_seed_0_shuffle_True/Attention_Analysis_ABB.png" width="30%" alt="Llama-3.1-8B_Attention-Analysis_Symbolic-Induction-Heads_ABB">
  <img src="outputs/attn/Llama-3.1-8B/token_context/base_rule_ABB_exp_rule_ABB/z_seed_0_shuffle_True/Attention_Analysis_ABB.png" width="30%" alt="Llama-3.1-8B_Attention-Analysis_Retrieval-Heads_ABB">
</p>
<p align="center">
  <em>The Attention Analysis for Symbol Abstraction Heads, Symbolic Induction Heads and Retrieval Heads for Llama-3.1-8B with base rule ABB</em>
</p>

#### Reproduction of Figure 4
<p align="center">
  <img src="results/identity_rules/rsa/meta-llama/Llama-3.1-70B_ungroup_gqa/in_ctx_2/abstract_add_swap_1_2_question_base_rule_ABA/prompt_40_seed_1234_shuffle_True/z_abstract/token_pos_11_transpose_True/Hand_Code_H_1.0_L_0.0_heatmap.png" width="45%" alt="Abstract-Similarity-Matrix">
  <img src="results/identity_rules/rsa/meta-llama/Llama-3.1-70B_ungroup_gqa/in_ctx_2/abstract_add_swap_1_2_question_base_rule_ABA/prompt_40_seed_1234_shuffle_True/z_token/token_pos_11_transpose_True/Hand_Code_H_1.0_L_0.0_heatmap.png" width="45%" alt="Token-Similarity-Matrix">
</p>
<p align="center">
  <em>The Predicted Abstract and Token Similarity Matrix</em>
</p>

<p align="center">
  <img src="results/identity_rules/rsa/meta-llama/Llama-3.1-70B_ungroup_gqa/in_ctx_2/abstract_add_swap_1_2_question_base_rule_ABA/prompt_40_seed_1234_shuffle_True/z_token/token_pos_5_transpose_True/symbol_abstraction_head%20WSim_heatmap.png" width="30%" alt="Llama-3.1-70B_RSA_Symbol-Abstraction-Heads_pos-5_ABA">
  <img src="results/identity_rules/rsa/meta-llama/Llama-3.1-70B_ungroup_gqa/in_ctx_2/abstract_add_swap_1_2_question_base_rule_ABA/prompt_40_seed_4567_shuffle_True/z_abstract/token_pos_16_transpose_True/symbolic_induction_head%20WSim_heatmap.png" width="30%" alt="Llama-3.1-70B_RSA_Symbolic-Induction-Heads_ABA">
  <img src="results/identity_rules/rsa/meta-llama/Llama-3.1-70B_ungroup_gqa/in_ctx_2/abstract_add_swap_1_2_question_base_rule_ABA/prompt_40_seed_7890_shuffle_True/z_token/token_pos_16_transpose_True/retrieval_head%20WSim_heatmap.png" width="30%" alt="Llama-3.1-70B_RSA_Retrieval-Heads_ABA">
</p>
<p align="center">
  <em>The RSA for Symbol Abstraction Heads(at the 5th position), Symbolic Induction Heads and Retrieval Heads for Llama-3.1-70B with base rule ABA</em>
</p>

<p align="center">
  <img src="results/identity_rules/rsa/meta-llama/Llama-3.1-8B_ungroup_gqa/in_ctx_2/abstract_add_swap_1_2_question_base_rule_ABA/prompt_40_seed_1234_shuffle_True/z_token/token_pos_5_transpose_True/symbol_abstraction_head%20WSim_heatmap.png" width="30%" alt="Llama-3.1-8B_RSA_Symbol-Abstraction-Heads_pos-5_ABA">
  <img src="results/identity_rules/rsa/meta-llama/Llama-3.1-8B_ungroup_gqa/in_ctx_2/abstract_add_swap_1_2_question_base_rule_ABA/prompt_40_seed_4567_shuffle_True/z_abstract/token_pos_16_transpose_True/symbolic_induction_head%20WSim_heatmap.png" width="30%" alt="Llama-3.1-8B_RSA_Symbolic-Induction-Heads_ABA">
  <img src="results/identity_rules/rsa/meta-llama/Llama-3.1-8B_ungroup_gqa/in_ctx_2/abstract_add_swap_1_2_question_base_rule_ABA/prompt_40_seed_7890_shuffle_True/z_token/token_pos_16_transpose_True/retrieval_head%20WSim_heatmap.png" width="30%" alt="Llama-3.1-8B_RSA_Retrieval-Heads_ABA">
</p>
<p align="center">
  <em>The RSA for Symbol Abstraction Heads(at the 5th position), Symbolic Induction Heads and Retrieval Heads for Llama-3.1-8B with base rule ABA</em>
</p>

#### Reproduction of Figure 5
<p align="center">
  <img src="results/identity_rules/ablation/meta-llama/Llama-3.1-8B/rule_ABA/z_seed_1000_shuffle_True_ONLY_PREFILL_True/symbol_abstraction_head/sample_num_20_gen_acc_0.9/Ablation_Llama-3.1-8B_symbol_abstraction_head_ABA_True.png" width="30%" alt="Llama-3.1-8B_Ablation-Analysis_Symbol-Abstraction-Heads_ABA">
  <img src="results/identity_rules/ablation/meta-llama/Llama-3.1-8B/rule_ABA/z_seed_1000_shuffle_True_ONLY_PREFILL_True/symbolic_induction_head/sample_num_20_gen_acc_0.9/Ablation_Llama-3.1-8B_symbolic_induction_head_ABA_True.png" width="30%" alt="Llama-3.1-8B_Ablation-Analysis_Symbolic-Induction-Heads_ABA">
  <img src="results/identity_rules/ablation/meta-llama/Llama-3.1-8B/rule_ABA/z_seed_1000_shuffle_True_ONLY_PREFILL_True/retrieval_head/sample_num_20_gen_acc_0.9/Ablation_Llama-3.1-8B_retrieval_head_ABA_True.png" width="30%" alt="Llama-3.1-8B_Ablation-Analysis_Retrieval-Heads_ABA">
</p>
<p align="center">
  <em>The Ablation Analysis for Symbol Abstraction Heads, Symbolic Induction Heads and Retrieval Heads for Llama-3.1-8B with base rule ABA</em>
</p>

<p align="center">
  <img src="results/identity_rules/ablation/meta-llama/Llama-3.1-8B/rule_ABB/z_seed_1000_shuffle_True_ONLY_PREFILL_True/symbol_abstraction_head/sample_num_20_gen_acc_0.9/Ablation_Llama-3.1-8B_symbol_abstraction_head_ABB_True.png" width="30%" alt="Llama-3.1-8B_Ablation-Analysis_Symbol-Abstraction-Heads_ABB">
  <img src="results/identity_rules/ablation/meta-llama/Llama-3.1-8B/rule_ABB/z_seed_1000_shuffle_True_ONLY_PREFILL_True/symbolic_induction_head/sample_num_20_gen_acc_0.9/Ablation_Llama-3.1-8B_symbolic_induction_head_ABB_True.png" width="30%" alt="Llama-3.1-8B_Ablation-Analysis_Symbolic-Induction-Heads_ABB">
  <img src="results/identity_rules/ablation/meta-llama/Llama-3.1-8B/rule_ABB/z_seed_1000_shuffle_True_ONLY_PREFILL_True/retrieval_head/sample_num_20_gen_acc_0.9/Ablation_Llama-3.1-8B_retrieval_head_ABB_True.png" width="30%" alt="Llama-3.1-8B_Ablation-Analysis_Retrieval-Heads_ABB">
</p>
<p align="center">
  <em>The Ablation Analysis for Symbol Abstraction Heads, Symbolic Induction Heads and Retrieval Heads for Llama-3.1-8B with base rule ABB</em>
</p>

#### Reproduction of Figure 6
<p align="center">
  <img src="outputs/eval/accuracy_rule_ABA.png" width="45%" alt="Generation-Performance_13-Models_rule-ABA">
  <img src="outputs/eval/accuracy_rule_ABB.png" width="45%" alt="Generation-Performance_13-Models_rule-ABB">
</p>
<p align="center">
  <em>The Generation Performance for 13 models from 4 families for rule ABA and ABB</em>
</p>

_The Figure 6(b) for Number of Significant Heads requires massive CMA experiments with 13 models, and will be shown once they are completed._

## Acknowledgements
This project is built on top of the [official repo](https://github.com/yukang123/LLMSymbMech) and supported by NDIF.

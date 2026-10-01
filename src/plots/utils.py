import torch
import numpy as np
import os


def apply_fwer_threshold(scores_file, num_permutations=5000, p_value=0.05, save_path=None):
    """
    Loads raw prompt scores, runs a Max-Statistic Permutation Test, 
    and returns a masked array for the plotting function.
    """
    print(f"Loading raw scores from: {scores_file}")
    
    # Shape: [num_prompts, num_layers, num_heads]
    scores = torch.load(scores_file, map_location="cpu")
    num_prompts = scores.shape[0]
    
    # Calculate the actual, real average across all prompts
    actual_mean = scores.mean(dim=0)
    
    print(f"Running {num_permutations} permutations on {num_prompts} prompts...")
    max_random_scores = []
    
    for _ in range(num_permutations):
        # Generate a random coin flip (-1 or +1) for each prompt
        # Shape: [num_prompts, 1, 1] so it broadcasts across layers and heads
        coin_flips = torch.randint(0, 2, (num_prompts, 1, 1), dtype=scores.dtype) * 2 - 1
        permuted_mean = (scores * coin_flips).mean(dim=0)
        max_score_in_network = permuted_mean.max().item()
        max_random_scores.append(max_score_in_network)
        
    # Identify the FWER threshold epsilon (e.g., the 95th percentile)
    target_percentile = 100 * (1.0 - p_value)
    epsilon = np.percentile(max_random_scores, target_percentile)
    
    print(f"Calculated FWER Threshold (p < {p_value}): {epsilon:.4f}")
    
    masked_mean = torch.where(
        actual_mean > epsilon, 
        actual_mean, 
        torch.tensor(0.)
        # torch.tensor(np.nan)
    )

    if save_path is not None:
        torch.save(masked_mean, save_path)
        print(f"masked_mean saved to {save_path}")
    
    return masked_mean.numpy()

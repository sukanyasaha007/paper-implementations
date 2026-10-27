"""Held-out evaluation: mean positive-sentiment reward on IMDb test-split prompts."""
import torch

from model import get_reward
from ppo import generate_responses


@torch.no_grad()
def evaluate(model, query_tensors, query_texts, tokenizer, sentiment_model, response_length=8, seed=0):
    # Same seed for every model so baseline and trained policies are compared on equal sampling noise
    torch.manual_seed(seed)
    response_tensors = generate_responses(model, query_tensors, tokenizer, response_length)
    texts = [q + tokenizer.decode(r) for q, r in zip(query_texts, response_tensors)]
    rewards = get_reward(texts, sentiment_model, response_tensors[0].device)
    return rewards, response_tensors


def summarize(rewards):
    # Mean and standard error over prompts
    n = rewards.numel()
    return {
        'mean_reward': rewards.mean().item(),
        'std_error': (rewards.std() / n ** 0.5).item(),
        'n_prompts': n,
    }

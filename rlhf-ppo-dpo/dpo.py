"""DPO: train directly on (prompt, chosen, rejected) pairs, with no reward model or critic in the loop."""
import torch
import torch.nn.functional as F

from model import get_reward
from ppo import generate_responses, get_response_log_probs


@torch.no_grad()
def build_synthetic_preference_pairs(policy, query_tensors, query_texts, tokenizer, sentiment_model, response_length=8):
    """Sample two responses per prompt and let the sentiment model pick the winner.

    These stand in for human preference labels: chosen = higher positive-sentiment score.
    """
    pairs = []

    for query, query_text in zip(query_tensors, query_texts):
        candidates = generate_responses(policy, [query, query], tokenizer, response_length)  # 2 responses: winner and loser
        response_texts = [tokenizer.decode(response, skip_special_tokens=True) for response in candidates]
        full_texts = [query_text + response_text for response_text in response_texts]
        rewards = get_reward(full_texts, sentiment_model, candidates[0].device)

        chosen_index = torch.argmax(rewards).item()
        rejected_index = 1 - chosen_index  # the other index, since there are only 2 candidates
        pairs.append({
            'query': query,
            'chosen': candidates[chosen_index],
            'rejected': candidates[rejected_index],
            'chosen_reward': rewards[chosen_index].item(),
            'rejected_reward': rewards[rejected_index].item(),
        })
    return pairs


def get_sequence_log_probs(model, query_tensors, response_tensors):
    token_log_probs = get_response_log_probs(model, response_tensors, query_tensors)
    return token_log_probs.sum(dim=-1)  # (batch,)


def calculate_dpo_loss(policy_chosen_log_probs, policy_rejected_log_probs, ref_chosen_log_probs, ref_rejected_log_probs, beta=0.1):
    """-log sigmoid(beta * [(log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x))])"""
    chosen_log_ratio = policy_chosen_log_probs - ref_chosen_log_probs
    rejected_log_ratio = policy_rejected_log_probs - ref_rejected_log_probs
    preference_logits = beta * (chosen_log_ratio - rejected_log_ratio)
    loss = -F.logsigmoid(preference_logits).mean()
    return loss, preference_logits


def dpo_update(dpo_policy, optimizer, queries, chosen_responses, rejected_responses, ref_chosen_log_probs, ref_rejected_log_probs, beta=0.1):
    dpo_policy.train()

    policy_chosen_log_probs = get_sequence_log_probs(dpo_policy, queries, chosen_responses)
    policy_rejected_log_probs = get_sequence_log_probs(dpo_policy, queries, rejected_responses)

    loss, preference_logits = calculate_dpo_loss(
        policy_chosen_log_probs,
        policy_rejected_log_probs,
        ref_chosen_log_probs,
        ref_rejected_log_probs,
        beta,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(dpo_policy.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'loss': loss.item(),
        'mean_preference_probability': torch.sigmoid(preference_logits.detach()).mean().item(),
        'gradient_norm': gradient_norm.item(),
    }


def train_dpo_minibatches(dpo_policy, optimizer, train_queries, train_chosen_responses, train_rejected_responses,
                          train_ref_chosen_log_probs, train_ref_rejected_log_probs, epochs=3, batch_size=4, beta=0.1):
    history = []

    for epoch in range(epochs):
        for start in range(0, len(train_queries), batch_size):
            end = start + batch_size
            stats = dpo_update(
                dpo_policy,
                optimizer,
                train_queries[start:end],
                train_chosen_responses[start:end],
                train_rejected_responses[start:end],
                train_ref_chosen_log_probs[start:end],
                train_ref_rejected_log_probs[start:end],
                beta,
            )
            stats['epoch'] = epoch + 1
            history.append(stats)

        print(f"Epoch {epoch + 1}: loss={stats['loss']:.4f}, "
              f"chosen_probability={stats['mean_preference_probability']:.4f}")

    return history

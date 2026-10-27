"""PPO for RLHF: rollouts, KL-shaped token rewards, GAE, and the clipped policy/value/entropy losses."""
import torch
import torch.nn.functional as F

from model import get_reward


def model_device(model):
    return next(model.parameters()).device


def same_length(tensors):
    return len({t.shape[0] for t in tensors}) == 1


def response_batches(query_tensors, response_tensors, device):
    """Yield (input_ids, query_length, responses) chunks for scoring prompt + response sequences.

    Prompts are truncated to a fixed length and responses are fixed-length, so normally the whole batch is a
    single (batch, query_len + response_len) forward pass. Mixed lengths fall back to one sequence at a time.
    """
    if same_length(query_tensors) and same_length(response_tensors):
        queries = torch.stack(list(query_tensors)).to(device)
        responses = torch.stack(list(response_tensors)).to(device)
        yield torch.cat([queries, responses], dim=1), queries.shape[1], responses
    else:
        for query, response in zip(query_tensors, response_tensors):
            query = query.to(device)
            response = response.to(device)
            yield torch.cat([query, response]).unsqueeze(0), query.shape[0], response.unsqueeze(0)


@torch.no_grad()
def generate_responses(policy, query_tensors, tokenizer, max_new_tokens=8):
    policy.eval()
    device = model_device(policy)
    if same_length(query_tensors):
        query_batches = [torch.stack(list(query_tensors))]
    else:
        query_batches = [query.unsqueeze(0) for query in query_tensors]

    responses = []
    for queries in query_batches:
        queries = queries.to(device)
        # Pure sampling (no top-k / top-p) so the sampled tokens come from the same distribution
        # whose log-probabilities PPO uses
        generated = policy.generate(
            input_ids=queries,
            attention_mask=torch.ones_like(queries),
            top_k=0,
            top_p=1.0,
            min_new_tokens=max_new_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        responses.extend(generated[:, queries.shape[1]:])

    return responses


def get_response_log_probs(policy, response_tensors, query_tensors):
    """Log-probability of each generated token: (batch, response_len).

    The logit at position t predicts token t + 1, so the response tokens are predicted by the logits
    from the last prompt position up to the second-to-last position: logits[:, query_length - 1:-1].
    """
    batch_log_probs = []
    device = model_device(policy)

    for input_ids, query_length, responses in response_batches(query_tensors, response_tensors, device):
        logits = policy(input_ids).logits
        response_logits = logits[:, query_length - 1:-1, :]  # (batch, response_len, vocab_size)
        log_probs = F.log_softmax(response_logits, dim=-1)

        # Pick out only the log-prob of the token that was actually generated, along the vocab dimension
        response_log_probs = log_probs.gather(dim=-1, index=responses.unsqueeze(-1))  # (batch, response_len, 1)
        batch_log_probs.append(response_log_probs.squeeze(-1))

    return torch.cat(batch_log_probs)  # (batch, response_len)


def compute_token_rewards(old_log_probs, ref_log_probs, sentiment_reward, kl_beta=0.1):
    # Per-token KL penalty keeps the policy close to the reference model
    kl = old_log_probs - ref_log_probs
    token_rewards = -kl_beta * kl
    # The sentiment score judges the whole text, so it is added only at the last token
    token_rewards[:, -1] = token_rewards[:, -1] + sentiment_reward
    return token_rewards


def get_response_values(policy, value_head, query_tensors, response_tensors):
    """Critic estimate of the future return before each response token is chosen: (batch, response_len)."""
    batch_values = []
    device = model_device(policy)

    for input_ids, query_length, _ in response_batches(query_tensors, response_tensors, device):
        outputs = policy(input_ids, output_hidden_states=True)
        # Same alignment as the log-probs: (batch, query + response len, hidden) -> (batch, response len, hidden)
        response_hidden_states = outputs.hidden_states[-1][:, query_length - 1:-1, :]
        batch_values.append(value_head(response_hidden_states))  # (batch, response_len)

    return torch.cat(batch_values)


def compute_gae(values, rewards, gamma=1.0, gae_lambda=0.95):
    # Store one advantage for each response-token action.
    advantages = torch.zeros_like(rewards)

    # The response ends after its last token, so future value and advantage start at zero.
    next_advantage = torch.zeros(rewards.shape[0], device=rewards.device)
    next_value = torch.zeros(rewards.shape[0], device=rewards.device)

    # Work backward so each token receives information from later rewards.
    for t in reversed(range(rewards.shape[1])):
        # One-step TD error: observed reward plus next value minus current prediction.
        delta = rewards[:, t] + gamma * next_value - values[:, t]

        # Add discounted future TD errors to the current advantage.
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[:, t] = next_advantage

        # Current V_t becomes V_{t+1} when the loop moves to the previous token.
        next_value = values[:, t]

    # Return target for training the value head: G_t = A_t + V_t.
    returns = advantages + values
    return advantages, returns


def calculate_ppo_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2):
    importance_ratio = torch.exp(new_log_probs - old_log_probs)
    clip_ratio = torch.clamp(importance_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    clipped_obj = torch.min(importance_ratio * advantages, clip_ratio * advantages)
    policy_loss = -clipped_obj.mean()
    return policy_loss, importance_ratio


def calculate_value_loss(new_values, returns):
    return torch.mean((new_values - returns) ** 2) / 2


def calculate_entropy(policy, query_tensors, response_tensors):
    """Mean entropy of the policy's next-token distribution over the response positions."""
    batch_entropies = []
    device = model_device(policy)

    for input_ids, q_len, _ in response_batches(query_tensors, response_tensors, device):
        logits = policy(input_ids).logits
        response_logits = logits[:, q_len - 1:-1, :]  # (batch, response_len, vocab_size)
        log_probs = F.log_softmax(response_logits, dim=-1)
        probs = log_probs.exp()
        H = -(probs * log_probs).sum(dim=-1)
        batch_entropies.append(H)
    return torch.cat(batch_entropies).mean()


@torch.no_grad()
def collect_rollout_batch(policy, ref_policy, value_head, sentiment_model, batch, tokenizer,
                          kl_beta=0.1, gamma=1.0, gae_lambda=0.95, response_length=8):
    query_tensors = batch['input_ids']
    response_tensors = generate_responses(policy, query_tensors, tokenizer, response_length)
    old_log_probs = get_response_log_probs(policy, response_tensors, query_tensors)
    ref_log_probs = get_response_log_probs(ref_policy, response_tensors, query_tensors)

    texts = [q + tokenizer.decode(r) for q, r in zip(batch['query'], response_tensors)]
    sentiment_rewards = get_reward(texts, sentiment_model, old_log_probs.device)

    token_rewards = compute_token_rewards(old_log_probs, ref_log_probs, sentiment_rewards, kl_beta)
    old_values = get_response_values(policy, value_head, query_tensors, response_tensors)
    advantages, returns = compute_gae(old_values, token_rewards, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return {
        'query_tensors': query_tensors,
        'response_tensors': response_tensors,
        'old_log_probs': old_log_probs,
        'old_values': old_values,
        'advantages': advantages,
        'returns': returns,
        'sentiment_rewards': sentiment_rewards,
        'kl': (old_log_probs - ref_log_probs).sum(dim=-1).mean(),
    }


def ppo_update(policy, value_head, rollout, optimizer, epochs, clip_ep, coeff_value, coeff_entropy, mini_batch_size=None):
    # Keep dropout OFF while optimizing. The rollout's old log-probs were computed with dropout off
    # (generate_responses calls policy.eval()); with dropout on here, the importance ratio would move away
    # from 1 because of random dropout masks rather than because the policy changed.
    # eval() only changes dropout; gradients still flow.
    policy.eval()
    value_head.train()
    batch_size = len(rollout['query_tensors'])
    mini_batch_size = mini_batch_size or batch_size

    for epoch in range(epochs):
        # As in the PPO paper: several epochs of minibatch updates over the same rollout, reshuffled each epoch
        order = torch.randperm(batch_size).tolist()
        for start in range(0, batch_size, mini_batch_size):
            idx = order[start:start + mini_batch_size]
            queries = [rollout['query_tensors'][i] for i in idx]
            responses = [rollout['response_tensors'][i] for i in idx]

            new_log_probs = get_response_log_probs(policy, responses, queries)
            new_values = get_response_values(policy, value_head, queries, responses)

            policy_loss, imp_ratio = calculate_ppo_loss(rollout['old_log_probs'][idx], new_log_probs, rollout['advantages'][idx], clip_ep)
            value_loss = calculate_value_loss(new_values, rollout['returns'][idx])
            # With no entropy bonus, entropy is only logged, so skip building its graph
            with torch.set_grad_enabled(coeff_entropy > 0):
                entropy = calculate_entropy(policy, queries, responses)
            total_loss = policy_loss + coeff_value * value_loss - coeff_entropy * entropy
            optimizer.zero_grad()
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_head.parameters()), max_norm=1)
            optimizer.step()

            if epoch == 0 and start == 0:
                # Before the first update the policy equals the rollout policy, so this should be exactly 1
                first_ratio = imp_ratio.mean().item()

    return {
        'total_loss': total_loss.item(),
        'value_loss': value_loss.item(),
        'policy_loss': policy_loss.item(),
        'entropy': entropy.item(),
        'first_ratio': first_ratio,
        'mean_ratio': imp_ratio.mean().item(),
        'gradient_norm': gradient_norm.item(),
    }


def train_ppo(policy, ref_policy, value_head, sentiment_model, dataloader, tokenizer, optimizer, max_batch,
              kl_beta, epochs, clip_ep, coeff_value, coeff_entropy, gamma=1.0, gae_lambda=0.95,
              response_length=8, mini_batch_size=None, log_every=1):
    history = []
    for step, batch in enumerate(dataloader):
        if step >= max_batch:
            break
        # 1. collect rollout
        rollout = collect_rollout_batch(policy, ref_policy, value_head, sentiment_model, batch, tokenizer,
                                        kl_beta, gamma, gae_lambda, response_length)
        # 2. optimize rollout with ppo update
        stats = ppo_update(policy, value_head, rollout, optimizer, epochs, clip_ep, coeff_value, coeff_entropy,
                           mini_batch_size)
        stats['mean_reward'] = rollout['sentiment_rewards'].mean().item()
        stats['kl'] = rollout['kl'].item()
        history.append(stats)

        if (step + 1) % log_every == 0:
            print(f"Step {step+1}: Reward:{stats['mean_reward']:.3f}, KL:{stats['kl']:.3f}, "
                  f"Loss:{stats['total_loss']:.3f}, Ratio: {stats['mean_ratio']:.3f}")
    return history

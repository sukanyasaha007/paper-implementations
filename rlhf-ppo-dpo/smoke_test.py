"""Fast checks on a tiny random GPT-2 and a fake reward model; no downloads needed.

Usage: python smoke_test.py
"""
import math

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from dpo import build_synthetic_preference_pairs, calculate_dpo_loss, dpo_update, get_sequence_log_probs
from model import ValueHead, freeze_copy
from ppo import (calculate_ppo_loss, collect_rollout_batch, compute_gae, generate_responses,
                 get_response_log_probs, get_response_values, ppo_update, train_ppo)

VOCAB = 50


class FakeTokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        return ' ' + ' '.join(str(i) for i in ids.tolist())


def fake_sentiment_model(texts, top_k=None):
    # Same output format as the transformers sentiment pipeline with top_k=None
    outputs = []
    for text in texts:
        p = (sum(map(ord, text)) % 100) / 100
        outputs.append([{'label': 'POSITIVE', 'score': p}, {'label': 'NEGATIVE', 'score': 1 - p}])
    return outputs


def tiny_policy():
    # Dropout on, so the test below catches it leaking into the PPO ratio
    config = GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=32, n_layer=2, n_head=2,
                        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, bos_token_id=0, eos_token_id=0)
    torch.manual_seed(0)
    policy = GPT2LMHeadModel(config)
    policy.config.pad_token_id = 0
    return policy, freeze_copy(policy)


def fake_batch(batch_size=3, prompt_length=5):
    queries = [torch.randint(1, VOCAB, (prompt_length,)) for _ in range(batch_size)]
    return {'input_ids': queries, 'query': [f'prompt {i}' for i in range(batch_size)]}


def smoke_log_probs():
    policy, _ = tiny_policy()
    policy.eval()
    batch = fake_batch()
    responses = generate_responses(policy, batch['input_ids'], FakeTokenizer(), max_new_tokens=4)
    assert all(r.shape == (4,) for r in responses)

    log_probs = get_response_log_probs(policy, responses, batch['input_ids'])
    assert log_probs.shape == (3, 4)

    # Reference: cross-entropy of the response tokens given everything before them
    query, response = batch['input_ids'][0], responses[0]
    logits = policy(torch.cat([query, response]).unsqueeze(0)).logits[0]
    expected = -F.cross_entropy(logits[len(query) - 1:-1], response, reduction='none')
    assert torch.allclose(log_probs[0], expected, atol=1e-5)
    print('response log-probs match cross-entropy: ok')


def smoke_batched_matches_loop():
    # The batched fast path must give the same numbers as scoring one sequence at a time
    policy, _ = tiny_policy()
    policy.eval()
    value_head = ValueHead(policy.config.n_embd)
    batch = fake_batch(batch_size=4)
    responses = generate_responses(policy, batch['input_ids'], FakeTokenizer(), max_new_tokens=4)

    batched = get_response_log_probs(policy, responses, batch['input_ids'])
    looped = torch.cat([get_response_log_probs(policy, [r], [q]) for q, r in zip(batch['input_ids'], responses)])
    assert torch.allclose(batched, looped, atol=1e-5)

    with torch.no_grad():
        batched_values = get_response_values(policy, value_head, batch['input_ids'], responses)
        looped_values = torch.cat([get_response_values(policy, value_head, [q], [r])
                                   for q, r in zip(batch['input_ids'], responses)])
    assert torch.allclose(batched_values, looped_values, atol=1e-5)

    # Prompts of different lengths take the one-at-a-time path
    mixed = [torch.randint(1, VOCAB, (n,)) for n in (3, 5, 7)]
    mixed_responses = generate_responses(policy, mixed, FakeTokenizer(), max_new_tokens=4)
    assert get_response_log_probs(policy, mixed_responses, mixed).shape == (3, 4)
    print('batched scoring matches one-at-a-time: ok')


def smoke_gae():
    # gamma = lambda = 1: advantage = (sum of future rewards) - V_t
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.5, 0.5, 0.5]])
    advantages, returns = compute_gae(values, rewards, gamma=1.0, gae_lambda=1.0)
    assert torch.allclose(advantages, torch.tensor([[0.5, 0.5, 0.5]]))
    assert torch.allclose(returns, torch.tensor([[1.0, 1.0, 1.0]]))

    # lambda = 0: advantage is the one-step TD error r_t + V_{t+1} - V_t
    advantages, _ = compute_gae(values, rewards, gamma=1.0, gae_lambda=0.0)
    assert torch.allclose(advantages, torch.tensor([[0.0, 0.0, 0.5]]))
    print('gae: ok')


def smoke_ppo_loss():
    old = torch.zeros(1, 2)
    adv = torch.tensor([[1.0, -1.0]])
    loss, ratio = calculate_ppo_loss(old, old, adv)
    assert torch.allclose(ratio, torch.ones(1, 2)) and abs(loss.item()) < 1e-6

    # Ratio 2 with a positive advantage is clipped to 1 + epsilon
    loss, _ = calculate_ppo_loss(old, torch.full((1, 2), math.log(2.0)), torch.tensor([[1.0, 1.0]]), clip_epsilon=0.2)
    assert abs(loss.item() + 1.2) < 1e-6
    print('ppo clipping: ok')


def smoke_ppo_step():
    policy, ref_policy = tiny_policy()
    value_head = ValueHead(policy.config.n_embd)
    optimizer = torch.optim.AdamW(list(policy.parameters()) + list(value_head.parameters()), lr=1e-3)
    tokenizer = FakeTokenizer()

    rollout = collect_rollout_batch(policy, ref_policy, value_head, fake_sentiment_model, fake_batch(), tokenizer,
                                    response_length=4)
    assert rollout['old_log_probs'].shape == rollout['advantages'].shape == (3, 4)

    before = next(policy.parameters()).detach().clone()
    stats = ppo_update(policy, value_head, rollout, optimizer, epochs=2, clip_ep=0.2, coeff_value=0.5, coeff_entropy=0.01,
                       mini_batch_size=2)
    # The policy has not changed before the first update, so the ratio must be exactly 1 even with dropout configured
    assert abs(stats['first_ratio'] - 1.0) < 1e-5, stats['first_ratio']
    assert not torch.equal(before, next(policy.parameters()))

    history = train_ppo(policy, ref_policy, value_head, fake_sentiment_model, [fake_batch(), fake_batch()], tokenizer,
                        optimizer, max_batch=2, kl_beta=0.1, epochs=1, clip_ep=0.2, coeff_value=0.5,
                        coeff_entropy=0.01, response_length=4)
    assert len(history) == 2 and all(math.isfinite(h['total_loss']) for h in history)
    print('ppo update and training loop: ok')


def smoke_dpo():
    policy, ref_policy = tiny_policy()
    batch = fake_batch()
    pairs = build_synthetic_preference_pairs(ref_policy, batch['input_ids'], batch['query'], FakeTokenizer(),
                                             fake_sentiment_model, response_length=4)
    assert all(p['chosen_reward'] >= p['rejected_reward'] for p in pairs)

    queries = [p['query'] for p in pairs]
    chosen = [p['chosen'] for p in pairs]
    rejected = [p['rejected'] for p in pairs]
    with torch.no_grad():
        ref_chosen = get_sequence_log_probs(ref_policy, queries, chosen)
        ref_rejected = get_sequence_log_probs(ref_policy, queries, rejected)

    # Policy identical to the reference: preference logit 0, loss log 2
    policy.eval()
    loss, logits = calculate_dpo_loss(get_sequence_log_probs(policy, queries, chosen),
                                      get_sequence_log_probs(policy, queries, rejected), ref_chosen, ref_rejected)
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-5)
    assert abs(loss.item() - math.log(2)) < 1e-5

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    stats = dpo_update(policy, optimizer, queries, chosen, rejected, ref_chosen, ref_rejected)
    assert math.isfinite(stats['loss'])
    print('dpo loss and update: ok')


if __name__ == '__main__':
    smoke_log_probs()
    smoke_batched_matches_loop()
    smoke_gae()
    smoke_ppo_loss()
    smoke_ppo_step()
    smoke_dpo()

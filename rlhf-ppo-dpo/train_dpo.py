"""Train GPT-2 with DPO on synthetic preference pairs ranked by the sentiment model.

Usage: python train_dpo.py
Writes results/dpo_history.json and results/dpo_eval.json.
"""
import json
from pathlib import Path

import torch
from torch.optim import Adam

from config import get_config
from data import build_prompt_dataset, build_tokenizer
from dpo import build_synthetic_preference_pairs, get_sequence_log_probs, train_dpo_minibatches
from evaluate import evaluate, summarize
from model import freeze_copy, load_policy_and_reference, load_sentiment_model


def main():
    config = get_config()
    torch.manual_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(config['results_dir'])
    results_dir.mkdir(exist_ok=True)

    tokenizer = build_tokenizer(config['policy_model'])
    train_prompts = build_prompt_dataset(config['dataset'], 'train', tokenizer, config['prompt_length'],
                                         config['num_dpo_pairs'], config['seed'])
    test_prompts = build_prompt_dataset(config['dataset'], 'test', tokenizer, config['prompt_length'],
                                        config['num_eval_prompts'], config['seed'])

    _, ref_policy = load_policy_and_reference(config['policy_model'], device)
    sentiment_model = load_sentiment_model(config['reward_model'], device)

    # 1. Preference pairs sampled from the reference (SFT) policy
    train_queries = [train_prompts[i]['input_ids'] for i in range(len(train_prompts))]
    train_query_texts = [train_prompts[i]['query'] for i in range(len(train_prompts))]
    pairs = build_synthetic_preference_pairs(ref_policy, train_queries, train_query_texts, tokenizer,
                                             sentiment_model, config['response_length'])
    reward_gaps = [p['chosen_reward'] - p['rejected_reward'] for p in pairs]
    print(f'Pairs: {len(pairs)}, mean reward gap: {sum(reward_gaps) / len(reward_gaps):.3f}')

    queries = [p['query'] for p in pairs]
    chosen = [p['chosen'] for p in pairs]
    rejected = [p['rejected'] for p in pairs]

    # 2. Reference scores are fixed, so compute them once (in chunks, to bound the vocab-sized logits in memory)
    with torch.no_grad():
        chunks = range(0, len(queries), 32)
        ref_chosen_log_probs = torch.cat([get_sequence_log_probs(ref_policy, queries[i:i + 32], chosen[i:i + 32])
                                          for i in chunks])
        ref_rejected_log_probs = torch.cat([get_sequence_log_probs(ref_policy, queries[i:i + 32], rejected[i:i + 32])
                                            for i in chunks])

    # 3. The DPO policy starts as a trainable copy of the reference
    dpo_policy = freeze_copy(ref_policy)
    for param in dpo_policy.parameters():
        param.requires_grad = True
    optimizer = Adam(dpo_policy.parameters(), lr=config['dpo_lr'])

    history = train_dpo_minibatches(
        dpo_policy, optimizer, queries, chosen, rejected, ref_chosen_log_probs, ref_rejected_log_probs,
        epochs=config['dpo_epochs'], batch_size=config['dpo_batch_size'], beta=config['dpo_beta'],
    )

    # 4. Held-out sentiment reward
    test_queries = [test_prompts[i]['input_ids'] for i in range(len(test_prompts))]
    test_query_texts = [test_prompts[i]['query'] for i in range(len(test_prompts))]
    baseline_rewards, baseline_responses = evaluate(ref_policy, test_queries, test_query_texts, tokenizer,
                                                    sentiment_model, config['response_length'], config['seed'])
    trained_rewards, trained_responses = evaluate(dpo_policy, test_queries, test_query_texts, tokenizer,
                                                  sentiment_model, config['response_length'], config['seed'])

    baseline = summarize(baseline_rewards)
    trained = summarize(trained_rewards)
    print(f"Reference policy: {baseline['mean_reward']:.3f} ± {baseline['std_error']:.3f}")
    print(f"DPO policy:       {trained['mean_reward']:.3f} ± {trained['std_error']:.3f}")

    samples = [
        {
            'prompt': test_query_texts[i],
            'reference': tokenizer.decode(baseline_responses[i]),
            'dpo': tokenizer.decode(trained_responses[i]),
        }
        for i in range(10)
    ]

    with open(results_dir / 'dpo_history.json', 'w') as f:
        json.dump(history, f, indent=1)
    with open(results_dir / 'dpo_eval.json', 'w') as f:
        json.dump({'reference': baseline, 'dpo': trained, 'mean_reward_gap': sum(reward_gaps) / len(reward_gaps),
                   'config': config, 'samples': samples}, f, indent=1)


if __name__ == '__main__':
    main()

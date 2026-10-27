"""Train GPT-2 with PPO to continue IMDb review prompts with positive sentiment.

Usage: python train_ppo.py
Writes results/ppo_history.json and results/ppo_eval.json.
"""
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import get_config
from data import build_prompt_dataset, build_tokenizer, collator
from evaluate import evaluate, summarize
from model import ValueHead, get_reward, load_policy_and_reference, load_sentiment_model
from ppo import train_ppo


def main():
    config = get_config()
    torch.manual_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(config['results_dir'])
    results_dir.mkdir(exist_ok=True)

    tokenizer = build_tokenizer(config['policy_model'])
    train_prompts = build_prompt_dataset(config['dataset'], 'train', tokenizer, config['prompt_length'],
                                         config['num_train_prompts'], config['seed'])
    test_prompts = build_prompt_dataset(config['dataset'], 'test', tokenizer, config['prompt_length'],
                                        config['num_eval_prompts'], config['seed'])
    dataloader = DataLoader(train_prompts, batch_size=config['ppo_batch_size'], shuffle=True, collate_fn=collator)

    policy, ref_policy = load_policy_and_reference(config['policy_model'], device)
    value_head = ValueHead(policy.config.n_embd).to(device)
    sentiment_model = load_sentiment_model(config['reward_model'], device)

    parameters = list(policy.parameters()) + list(value_head.parameters())
    optimizer = AdamW(parameters, lr=config['ppo_lr'])

    start = time.time()
    history = train_ppo(
        policy, ref_policy, value_head, sentiment_model, dataloader, tokenizer, optimizer,
        max_batch=config['ppo_steps'],
        kl_beta=config['kl_beta'],
        epochs=config['ppo_epochs'],
        clip_ep=config['clip_epsilon'],
        coeff_value=config['value_coeff'],
        coeff_entropy=config['entropy_coeff'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        response_length=config['response_length'],
        mini_batch_size=config['ppo_mini_batch_size'],
        log_every=10,
    )
    minutes = (time.time() - start) / 60
    print(f'Training took {minutes:.1f} min')

    test_queries = [test_prompts[i]['input_ids'] for i in range(len(test_prompts))]
    test_query_texts = [test_prompts[i]['query'] for i in range(len(test_prompts))]
    baseline_rewards, baseline_responses = evaluate(ref_policy, test_queries, test_query_texts, tokenizer,
                                                    sentiment_model, config['response_length'], config['seed'])
    trained_rewards, trained_responses = evaluate(policy, test_queries, test_query_texts, tokenizer,
                                                  sentiment_model, config['response_length'], config['seed'])

    baseline = summarize(baseline_rewards)
    trained = summarize(trained_rewards)
    print(f"Reference policy: {baseline['mean_reward']:.3f} ± {baseline['std_error']:.3f}")
    print(f"PPO policy:       {trained['mean_reward']:.3f} ± {trained['std_error']:.3f}")

    samples = [
        {
            'prompt': test_query_texts[i],
            'reference': tokenizer.decode(baseline_responses[i]),
            'ppo': tokenizer.decode(trained_responses[i]),
        }
        for i in range(10)
    ]
    for sample in samples[:3]:
        print(sample)

    with open(results_dir / 'ppo_history.json', 'w') as f:
        json.dump(history, f, indent=1)
    with open(results_dir / 'ppo_eval.json', 'w') as f:
        json.dump({'reference': baseline, 'ppo': trained, 'train_minutes': minutes,
                   'config': config, 'samples': samples}, f, indent=1)


if __name__ == '__main__':
    main()

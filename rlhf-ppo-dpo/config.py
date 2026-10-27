def get_config():
    return {
        # Models: GPT-2 already fine-tuned on IMDb reviews (the SFT stage), and a DistilBERT sentiment classifier
        # used as a fixed reward model
        'policy_model': 'lvwerra/gpt2-imdb',
        'reward_model': 'lvwerra/distilbert-imdb',
        'dataset': 'stanfordnlp/imdb',

        # Prompts are the first few tokens of a review; the policy continues them
        'prompt_length': 8,
        'response_length': 8,
        'num_train_prompts': 12800,
        'num_eval_prompts': 200,
        'seed': 0,

        # PPO
        'ppo_batch_size': 64,
        'ppo_steps': 200,
        'ppo_epochs': 4,
        'ppo_mini_batch_size': 16,
        'ppo_lr': 1e-5,
        'kl_beta': 0.1,
        'clip_epsilon': 0.2,
        'gamma': 1.0,
        'gae_lambda': 0.95,
        'value_coeff': 0.1,
        # An entropy bonus pushed GPT-2 toward random tokens (KL up, reward flat), so it is off;
        # the KL penalty already keeps the policy from collapsing
        'entropy_coeff': 0.0,

        # DPO
        'num_dpo_pairs': 256,
        'dpo_epochs': 3,
        'dpo_batch_size': 8,
        'dpo_lr': 1e-5,
        'dpo_beta': 0.1,

        'results_dir': 'results',
    }

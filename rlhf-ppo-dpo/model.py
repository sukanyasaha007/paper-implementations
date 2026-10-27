"""Policy, frozen reference policy, value head, and the sentiment reward model."""
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, pipeline


def load_policy_and_reference(model_name, device):
    policy = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    policy.config.pad_token_id = policy.config.eos_token_id

    ref_policy = freeze_copy(policy)
    return policy, ref_policy


def freeze_copy(model):
    # The reference policy anchors the KL penalty; it never receives gradients
    ref_model = deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model


class ValueHead(nn.Module):
    """Predicts the expected future reward from each token's last hidden state.

    Shares the policy's GPT-2 backbone and adds one linear layer: (batch, seq_len, hidden) -> (batch, seq_len).
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, hidden_states):
        x = self.linear(hidden_states)
        return x.squeeze(-1)


def load_sentiment_model(model_name, device):
    return pipeline(
        task='sentiment-analysis',
        model=model_name,
        device=0 if device.type == 'cuda' else -1,
    )


def get_reward(texts, sentiment_model, device):
    # The classifier returns both POSITIVE and NEGATIVE probabilities; the reward is the POSITIVE one
    outputs = sentiment_model(texts, top_k=None)
    rewards = []
    for output in outputs:
        for result in output:
            if result['label'] == 'POSITIVE':
                rewards.append(result['score'])

    return torch.tensor(rewards, device=device)

import torch
import torch.nn as nn

from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embed, n_token):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        # Learned position embeddings, one per token slot
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embed))

    def forward(self, tokens):
        # tokens - (B, seq_len)
        x = self.token_embedding(tokens)
        x = x + self.position_embedding  # (B, seq_len, dim)
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_embed):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x):
        # x - (B, seq_len, dim)
        residue = x
        x = self.attention(self.layernorm_1(x), causal_mask=True)
        x = x + residue

        residue = x
        x = self.linear_1(self.layernorm_2(x))
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU, the activation CLIP was trained with
        x = self.linear_2(x)
        return x + residue


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)  # (vocab, dim, max_tokens)
        self.layers = nn.ModuleList(CLIPLayer(12, 768) for _ in range(12))
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens):
        # tokens (B, 77) -> (B, 77, 768)
        state = self.embedding(tokens.long())
        for layer in self.layers:
            state = layer(state)
        return self.layernorm(state)

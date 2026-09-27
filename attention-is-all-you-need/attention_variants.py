"""Attention variants used at inference time: multi-head attention with a KV cache, and grouped-query attention."""
import math

import torch
import torch.nn as nn


class MultiHeadAttentionKVCache(nn.Module):
    def __init__(self, d_model, h, dropout = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by heads"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        # q, k, v wights
        self.w_q, self.w_k, self.w_v, self.w_o = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, mask=None):
        # attention values = softmax(Q@K.T/sqrt(d_k)) * V
        # (seq_len, d_model) - (seq_len, seq_len)
        attention_score = q @ k.transpose(-2, -1)/ math.sqrt(self.d_k)

        if mask is not None:
            # (seq_len, seq_len)
            attention_score.masked_fill_(mask == 0, float('-inf'))

        attention_score = attention_score.softmax(dim=-1)
        attention_score = self.dropout(attention_score)
        values = attention_score @ v

        return values

    def forward(self, q, k, v, mask=None, cache=None, enable_cache=False):
        query, key, value = self.w_q(q), self.w_k(k), self.w_v(v)
        # (batch, seq_len, d_model) - (batch, seq_len, h, d_k) - (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        if enable_cache and cache is not None:
            cached_key, cached_value = cache
            # (batch, h, cached_len, d_k)
            key = torch.cat([cached_key, key], dim = 2)
            value = torch.cat([cached_value, value], dim = 2)

        new_cache = (key, value)

        values = self.attention(query, key, value, mask)
        #(batch, h, seq_len, d_k) - (batch, seq_len, h, d_k) - (batch, seq_len, d_model)
        values = values.transpose(1, 2).contiguous()
        values = values.view(values.shape[0], values.shape[1], self.h * self.d_k)

        out = self.w_o(values)

        if enable_cache:
            return out, new_cache
        return out


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, heads, kv_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.kv_heads = kv_heads
        self.dropout = nn.Dropout(dropout)
        assert d_model % heads == 0, "Can not creat heads"
        assert heads % kv_heads == 0, "Can not kv heads"

        self.d_k = d_model // heads
        self.group_size = heads // kv_heads
        self.w_q = nn.Linear(d_model, d_model)
        # (seq_len, d_model) -> 8 q heads - (seq_len, d_k) -> 2 kv_heads - (512, 64 * 2) - (512, 128)
        # q1, q2 - kv1, q3, q4 - kv2, q5, q6 - kv3, q7, q8 - kv4
        self.w_k = nn.Linear(d_model, self.d_k * self.kv_heads)
        self.w_v = nn.Linear(d_model, self.d_k * self.kv_heads)
        self.w_o = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, mask=None):
        # q-(batc, s, d_k) - k.t - (batch, d_k, s) -> (batch, s, s) -> (batch, s, s) @ (batch, s, d_k) -> (batch, s, d_k)
        scores = (q@k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            assert mask.dtype == torch.bool, "mask must be boolean"
            scores = scores.masked_fill(mask==0, float('-inf'))
        softmax_scores = scores.softmax(dim=-1)
        softmax_scores = self.dropout(softmax_scores)

        values = softmax_scores @ v
        return values

    def forward(self, q, k, v, mask=None, cache=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        # (b, s, d_model) - (b, s, h, d_k) - (b, h, s, d_k)
        q = q.view(q.shape[0], q.shape[1], self.heads, self.d_k).transpose(1,2)
        k = k.view(k.shape[0], k.shape[1], self.kv_heads, self.d_k).transpose(1,2)
        v = v.view(v.shape[0], v.shape[1], self.kv_heads, self.d_k).transpose(1,2)

        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)

        new_cache = (k, v)
        # (batch, kv_heads, seq_len, d_k) -> (batch, kv_heads * group_size, seq_len, d_k)
        extended_k = k.repeat_interleave(self.group_size, dim=1)
        # heads = 8, kv_heads = 2, group_size = 4, [k0, k0, k0, k0, k1, k1, k1, k1] - [q0, q1...]
        extended_v = v.repeat_interleave(self.group_size, dim=1)

        values = self.attention(q, extended_k, extended_v, mask)
        # (batch, heads, s, dk) - (batch, s, h, dk) - (batch, s, d_model)
        values = values.transpose(1, 2).contiguous()
        values = values.view(values.shape[0], values.shape[1], self.d_model)

        return self.w_o(values), new_cache

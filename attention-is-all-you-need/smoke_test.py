"""Quick shape and one-step training checks on random data; no dataset download needed.

Usage: python smoke_test.py
"""
import torch
import torch.nn as nn

from attention_variants import GroupQueryAttention, MultiHeadAttentionKVCache
from dataset import causal_mask
from model import build_transformer


def smoke_transformer():
    # One-batch smoke test before launching full training
    batch_size = 2
    seq_len = 6
    src_vocab_size = 20
    target_vocab_size = 24

    smoke_model = build_transformer(
        src_vocab=src_vocab_size,
        target_vocab=target_vocab_size,
        src_seq_len=seq_len,
        target_seq_len=seq_len,
        d_model=16,
        N=2,
        h=4,
        dropout=0.1,
        d_ff=32,
    )

    encoder_input = torch.randint(0, src_vocab_size, (batch_size, seq_len))
    decoder_input = torch.randint(0, target_vocab_size, (batch_size, seq_len))
    label = torch.randint(0, target_vocab_size, (batch_size, seq_len))
    encoder_mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.int)
    decoder_mask = causal_mask(seq_len).unsqueeze(1).expand(batch_size, -1, -1, -1)

    optimizer = torch.optim.Adam(smoke_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    encoder_output = smoke_model.encode(encoder_input, encoder_mask)
    decoder_output = smoke_model.decode(
        decoder_input,
        encoder_output,
        encoder_mask,
        decoder_mask,
    )
    projection_output = smoke_model.project(decoder_output)
    loss = loss_fn(projection_output.view(-1, target_vocab_size), label.view(-1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print('encoder_output:', tuple(encoder_output.shape))
    print('decoder_output:', tuple(decoder_output.shape))
    print('projection_output:', tuple(projection_output.shape))
    print('loss:', loss.item())


def smoke_kv_cache():
    layer = MultiHeadAttentionKVCache(512, h= 8, dropout=0.1)
    cache = None

    token1 = torch.randn([2, 1, 512])
    out1, cache = layer(token1, token1, token1, cache = cache, enable_cache=True)
    assert out1.shape == (2, 1, 512)
    assert cache[0].shape == (2, 8, 1, 64)
    assert cache[1].shape == (2, 8, 1, 64)

    token2 = torch.randn([2, 1, 512])
    out2, cache = layer(token2, token2, token2, cache = cache, enable_cache = True)
    assert out2.shape == (2, 1, 512)
    assert cache[0].shape == (2, 8, 2, 64)
    assert cache[1].shape == (2, 8, 2, 64)
    print('kv cache: ok')


def smoke_gqa():
    gqa = GroupQueryAttention(64, 8, 2, 0.01)
    t = torch.rand(1, 64, 64)
    out, cache = gqa(t, t, t)
    assert out.shape == (1, 64, 64)
    assert cache[0].shape == (1, 2, 64, 8) # only kv_heads are cached
    print('gqa:', tuple(out.shape))


if __name__ == '__main__':
    smoke_transformer()
    smoke_kv_cache()
    smoke_gqa()

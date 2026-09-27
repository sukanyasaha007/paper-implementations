# Attention Is All You Need: Transformer from scratch

PyTorch implementation of the encoder-decoder Transformer from [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), trained for English → Italian translation on [opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books).

Followed along with Umar Jamil's videos:
- [Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://www.youtube.com/watch?v=bCz4OMemCcA)
- [Coding a Transformer from scratch on PyTorch, with full explanation, training and inference](https://www.youtube.com/watch?v=ISNdQcPhsts)

## Files

| File | What it contains |
|---|---|
| `model.py` | Input embeddings, sinusoidal positional encoding, layer norm, feed-forward, multi-head attention, residual connection, encoder/decoder blocks, projection layer, `build_transformer` |
| `dataset.py` | `BilingualDataset` (adds `[SOS]`/`[EOS]`/`[PAD]`, builds encoder and causal decoder masks), `causal_mask` |
| `config.py` | Hyperparameters and checkpoint paths |
| `train.py` | Word-level tokenizers, data loaders, greedy decoding, validation, training loop with checkpoint resume |
| `validate.py` | Load a checkpoint and print greedy translations |
| `visualize.py` | Encoder, decoder and cross-attention heatmaps (Altair, saved as HTML) |
| `attention_variants.py` | Multi-head attention with a KV cache, and grouped-query attention (GQA) |
| `smoke_test.py` | Random-data shape checks and one training step; no dataset download |

## Run

```bash
pip install -r requirements.txt
python smoke_test.py        # quick sanity check
python train.py             # trains, writes weights/tmodel_XX.pt and runs/ for TensorBoard
python validate.py 19       # translations from epoch 19's checkpoint
python visualize.py 19      # attention_*.html
```

Set `preload` in `config.py` to an epoch string (e.g. `'05'`) to resume training.

## Notes

**Positional encoding.** The frequency term $1/10000^{2i/d_{model}}$ is computed in log space:

$$
\frac{1}{10000^{2i/d_{model}}} = e^{(2i)\left(-\frac{\ln 10000}{d_{model}}\right)}
$$

which maps to `torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))`. Even dimensions get `sin(position * div_term)`, odd dimensions get `cos(...)`. Early dimension pairs vary quickly with position and later pairs slowly, so both short- and long-range positions can be represented.

**Mask shapes.** The encoder mask `(batch, 1, 1, seq_len)` broadcasts across heads and query positions against attention scores of shape `(batch, heads, seq_len, seq_len)`. The decoder mask `(batch, 1, seq_len, seq_len)` combines padding with the causal mask.

**Inference.** Greedy decoding runs the encoder once (prefill) and then appends one predicted token at a time to the decoder input (decode) until `[EOS]` or `seq_len`.

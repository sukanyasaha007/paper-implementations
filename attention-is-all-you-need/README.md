# Attention Is All You Need: Transformer from scratch

PyTorch implementation of the encoder-decoder Transformer, trained for English → Italian translation on [opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books).

## Paper

**Attention Is All You Need**
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
*Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*

- arXiv: https://arxiv.org/abs/1706.03762
- PDF: https://arxiv.org/pdf/1706.03762
- Proceedings: https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}
```

## Related papers

Used for [attention_variants.py](attention_variants.py), which goes beyond the original paper:

| Paper | Authors | Venue | Link | Used for |
|---|---|---|---|---|
| Fast Transformer Decoding: One Write-Head is All You Need | Noam Shazeer | arXiv 2019 | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) | Multi-query attention, and caching keys/values during incremental decoding |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai | EMNLP 2023 | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | Grouped-query attention |

## Dataset

[opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books), from the OPUS collection: Jörg Tiedemann, *Parallel Data, Tools and Interfaces in OPUS*, LREC 2012.

## Learning resources

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

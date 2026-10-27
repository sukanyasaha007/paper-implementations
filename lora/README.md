# LoRA: Low-Rank Adaptation from scratch

PyTorch implementation of LoRA as a weight parametrization: an MLP is pretrained on MNIST, then fine-tuned on a single
digit by training only two small matrices per layer while the original weights stay frozen.

## Paper

**LoRA: Low-Rank Adaptation of Large Language Models**
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
*International Conference on Learning Representations (ICLR 2022)*

- arXiv: https://arxiv.org/abs/2106.09685
- PDF: https://arxiv.org/pdf/2106.09685

```bibtex
@inproceedings{hu2022lora,
  title     = {{LoRA}: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}
```

## Learning resources

Followed along with Umar Jamil's video:
[LoRA: Low-Rank Adaptation of Large Language Models - Explained visually + PyTorch code from scratch](https://www.youtube.com/watch?v=PXWYUTMt-AU)

## Files

| File | What it contains |
|---|---|
| `model.py` | `RichBoyNet`: a deliberately oversized 3-layer MLP for MNIST (784 → 1000 → 3000 → 10) |
| `lora.py` | `LoRAParameterization`, registering it on linear layers, enabling/disabling it, freezing, parameter counts |
| `train.py` | Pretrain on MNIST, add rank-1 LoRA, fine-tune on digit 9 only, compare with LoRA on and off |
| `smoke_test.py` | Random-data checks of initialization, freezing, parameter counts, and enable/disable; no download |

## Run

```bash
pip install -r requirements.txt
python smoke_test.py        # quick checks, no download
python train.py             # downloads MNIST to ./data, writes results/lora_results.json
```

## Results

Pretrained for one epoch on all of MNIST, then rank-1 LoRA trained for 100 steps on images of the digit 9 only.
Test set: 10,000 MNIST images.

| Model | Accuracy | Errors on digit 9 |
|---|---|---|
| Pretrained MLP (3,818,010 parameters) | 87.6% | 315 |
| + LoRA, fine-tuned on 9s (8,794 trainable parameters, 0.23%) | 37.7% | 1 |
| LoRA switched off | 87.6% | 315 |

The fine-tuned model has only ever been shown 9s, so it now predicts 9 almost everywhere: errors on 9 drop from 315
to 1 while the other digits get worse. The point of the experiment is the last row: the frozen weights are untouched,
so switching the adapter off recovers the original model exactly, and adapters for different tasks can be swapped
on the same base weights.

## Notes

**The update.** For a frozen weight $W_0 \in \mathbb{R}^{d \times k}$, LoRA learns $B \in \mathbb{R}^{d \times r}$
and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$:

$$
W' = W_0 + \frac{\alpha}{r} B A
$$

That is $r(d + k)$ trainable parameters per layer instead of $dk$; here rank 1 adds 8,794 parameters to 3.8 million.

**Initialization matters.** $B$ starts at zero and $A$ is random Gaussian, so $BA = 0$ and the adapted model starts
identical to the original. Both cannot be zero: the gradients are $\nabla_B = \nabla_{W'} A^\top$ and
$\nabla_A = B^\top \nabla_{W'}$, so with $A = B = 0$ both stay zero and nothing ever trains.
`smoke_test.py` checks that $B$ receives a gradient on the first step.

**Parametrization.** `torch.nn.utils.parametrize.register_parametrization` makes `layer.weight` a computed
property: every access returns `LoRAParameterization(layer.parametrizations.weight.original)`. The original weight
is kept untouched in `parametrizations.weight.original`, and the forward pass needs no changes.

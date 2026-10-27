# paper-implementations

My learning repo for ML research papers. I implement each paper from scratch in PyTorch to understand how it actually works, not just how to call a library.

The code puts readability and comments ahead of speed. Comments record what I worked out along the way: tensor shapes, why a step is done a certain way, and the math behind it.

## Papers

| Paper | Authors | Venue | Links | Implementation |
|---|---|---|---|---|
| Attention Is All You Need | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin | NeurIPS 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) · [PDF](https://arxiv.org/pdf/1706.03762) · [Proceedings](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | [attention-is-all-you-need/](attention-is-all-you-need/): encoder-decoder Transformer, English → Italian translation, greedy decoding, attention heatmaps, KV cache, grouped-query attention |
| Proximal Policy Optimization Algorithms; Direct Preference Optimization | Schulman, Wolski, Dhariwal, Radford, Klimov; Rafailov, Sharma, Mitchell, Ermon, Manning, Finn | arXiv 2017; NeurIPS 2023 | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) · [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) | [rlhf-ppo-dpo/](rlhf-ppo-dpo/): RLHF on GPT-2 with no RL library. PPO (value head, GAE, KL-shaped reward) raises held-out sentiment reward from 0.61 to 0.80; DPO on 256 preference pairs reaches 0.73 |
| LoRA: Low-Rank Adaptation of Large Language Models | Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen | ICLR 2022 | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) · [PDF](https://arxiv.org/pdf/2106.09685) | [lora/](lora/): LoRA as a weight parametrization; rank-1 adapters (0.23% extra parameters) fine-tune an MNIST MLP, and switching them off recovers the original model exactly |
| High-Resolution Image Synthesis with Latent Diffusion Models | Rombach, Blattmann, Lorenz, Esser, Ommer | CVPR 2022 | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) · [PDF](https://arxiv.org/pdf/2112.10752) | [stable-diffusion/](stable-diffusion/): Stable Diffusion v1.5 architecture (VAE, CLIP text encoder, U-Net, DDPM sampler) with parameter counts matching v1.5 exactly; text-to-image pipeline in progress |

Full citations (BibTeX) are in each folder's README.

## What's next

- [ROADMAP.md](ROADMAP.md): the papers I plan to implement next, in three tracks (LLMs up to DeepSeek-V4.1-Flash,
  vision and multimodal models, and RL for LLMs), with time estimates.
- [core_concepts/](core_concepts/): short notebooks for the ideas each paper depends on, each with a reference
  implementation, a small experiment, and drills to write from memory.

## How each folder is organized

Every paper has its own folder with:

- `README.md`: the paper, the resources I learned from, how to run the code, and notes on the tricky parts
- `model.py` (or one module per component for larger models): the architecture from the paper
- training, evaluation and visualization scripts where they apply
- `smoke_test.py`: fast checks on random data that need no dataset download
- `requirements.txt`: that folder's dependencies

## Getting started

```bash
git clone git@github.com:sukanyasaha007/paper-implementations.git
cd paper-implementations/attention-is-all-you-need
pip install -r requirements.txt
python smoke_test.py
```

See each folder's README for training and inference commands.

## Credits

Implementations follow the original papers, with help from the tutorials listed in each folder's README.

# Core concepts

Short notebooks for the ideas each paper in the [roadmap](../ROADMAP.md) depends on. Work through a paper's notebooks
before implementing it.

Every notebook has the same shape:

1. A short explanation, with the math where it helps.
2. A reference implementation, checked against PyTorch or a slow loop version.
3. A small experiment that shows the idea working (or failing) with real numbers.
4. "Things to be able to explain": questions to answer out loud without notes.
5. **Drill:** empty functions to write cold, without scrolling up. Each prints `TODO` until written and `PASS` when it
   matches the reference; a wrong answer raises the test's assertion error.

## Running them

Only PyTorch is required (2.5 or newer). Notebook 05 also needs Triton and an NVIDIA or AMD GPU; its cells skip
cleanly without one. Everything else runs on CPU in under a minute; the training demos are faster on a GPU.

On Colab, upload a notebook or open it from GitHub, then choose an L4 runtime (T4 lacks bf16, which notebook 02 and
the Triton benchmarks use).

## LLM track

| # | Notebook | Covers | For papers |
|---|---|---|---|
| 01 | [BPE tokenizer](01_bpe_tokenizer.ipynb) | byte-level BPE training, encoding, decoding, regex pre-tokenization | L1 |
| 02 | [Optimization and mixed precision](02_optim_schedules_mixed_precision.ipynb) | AdamW from scratch, warmup + cosine, global-norm clipping, gradient accumulation, bf16, weight tying | L1 |
| 03 | [RMSNorm, RoPE, SwiGLU, GQA](03_rope_rmsnorm_swiglu_gqa.ipynb) | both RoPE layouts, relative-position property, KV cache sizing | L2, L6, V6 |
| 04 | [Online softmax and FlashAttention](04_online_softmax_flash_attention.ipynb) | running max/sum, merging partial states, tiled attention in PyTorch, roofline | L3 |
| 05 | [Triton kernels](05_triton_kernels.ipynb) | vector add, fused softmax, tiled matmul, benchmarking with `do_bench` | L3, L4, L9 |
| 06 | [Quantization formats](06_quantization_formats.ipynb) | int8, int4 groups, NF4, FP8 blockwise, MXFP4 / NVFP4 | L4, L7, L10 |
| 07 | [MoE routing](07_moe_routing.ipynb) | top-k gates, dispatch/combine, capacity, aux loss, aux-loss-free bias, shared experts | L5, L7 |
| 08 | [MLA](08_mla_low_rank_kv.ipynb) | latent KV cache, decoupled RoPE, absorbed decoding | L6 |
| 09 | [Multi-token prediction](09_multi_token_prediction.ipynb) | DeepSeek-V3 MTP modules, index alignment, causality test | L7 |
| 10 | [Sinkhorn and hyper-connections](10_sinkhorn_hyper_connections.ipynb) | Sinkhorn-Knopp, n-stream residuals, why mHC's constraint keeps signals bounded | L8 |
| 11 | [Sparse attention](11_sparse_attention.ipynb) | sliding window, NSA compression + block selection, DSA lightning indexer | L9 |
| 12 | [KV cache engineering](12_kv_cache_engineering.ipynb) | prefill vs decode, paged cache, cross-layer sharing, ring buffer, FP4 cache | L10 |

## Vision and multimodal track

| # | Notebook | Covers | For papers |
|---|---|---|---|
| 13 | [ViT patch embeddings](13_vit_patch_embeddings.ipynb) | patchify vs conv, 2-D sin-cos, resizing position tables, a tiny ViT | V1, V4, V6 |
| 14 | [CLIP and SigLIP](14_contrastive_clip_siglip.ipynb) | symmetric InfoNCE, sigmoid loss, retrieval on synthetic pairs | V2 |
| 15 | [VLM connectors and M-RoPE](15_vlm_connectors_mrope.ipynb) | projector, image-token splicing, label masking, pixel shuffle, resampler, M-RoPE | V5, V6, V8 |
| 16 | [DDPM and DDIM](16_diffusion_ddpm_ddim.ipynb) | noise schedule, noise-prediction training, ancestral and DDIM sampling | V3 |
| 17 | [DiT, flow matching, CFG](17_dit_flow_matching_cfg.ipynb) | adaLN-Zero, rectified flow, Euler sampling, classifier-free guidance | V4 |
| 18 | [Vector quantization](18_vector_quantization.ipynb) | VQ layer, straight-through estimator, dead codes, FSQ | V7, V8 |

## RL track

| # | Notebook | Covers | For papers |
|---|---|---|---|
| 19 | [Policy gradients and GAE](19_policy_gradients_gae.ipynb) | returns with episode boundaries, GAE, what a baseline buys | R1 |
| 20 | [PPO and KL estimators](20_ppo_and_kl_estimators.ipynb) | clipped loss and where its gradient vanishes, k1/k2/k3, full PPO | R2, R3 |
| 21 | [Reward models and DPO](21_reward_models_dpo.ipynb) | sequence log-probs, Bradley-Terry, DPO recovering the analytic optimum, IPO / SimPO | R3, R4 |
| 22 | [GRPO, DAPO, verifiable rewards](22_grpo_dapo_verifiable_rewards.ipynb) | group advantages, loss aggregation, clip-higher, dynamic sampling, pass@k, entropy collapse | R5, R6, R7 |
| 23 | [Batched LLM rollouts](23_llm_rollouts_batching.ipynb) | left padding, position ids, KV cache with padding, response masks, log-prob recompute | R3–R7 |

# Roadmap

The papers I plan to implement next, in three tracks. Each paper lists the [core concept notebooks](core_concepts/) to
work through first. Times assume part-time work; line counts are for the implementation folder, not the notebooks.

Most of the track can run on a laptop GPU for debugging and a Colab L4 for real training runs. Only the last step of
each track wants an A100.

## Suggested order across tracks

Tracks share prerequisites, so interleave them rather than finishing one before starting the next:

1. GPT-2 → LLaMA (L1, L2): the base model everything else builds on
2. ViT → CLIP/SigLIP (V1, V2): short, and independent of the LLM track
3. REINFORCE/GAE → PPO (R1, R2)
4. FlashAttention-2 → quantization (L3, L4)
5. DPO → GRPO → R1-Zero-style run (R4, R5, R6) on the LLaMA from step 1
6. LLaVA (V5) on the LLaMA from step 1 plus the SigLIP from step 2
7. DeepSeekMoE → MLA → DeepSeek-V3 → mHC → sparse attention → V4.1-Flash (L5–L10)
8. DDPM → DiT → VQ-VAE → Janus (V3, V4, V7, V8)

## LLM track

| # | Paper | What to build | Core concepts | Lines | Time |
|---|---|---|---|---|---|
| L1 | GPT-2 (Radford et al., 2019) | Decoder-only model, BPE tokenizer, pretraining on TinyStories / FineWeb-Edu | [01](core_concepts/01_bpe_tokenizer.ipynb), [02](core_concepts/02_optim_schedules_mixed_precision.ipynb) | 700–900 | 1–2 wk |
| L2 | LLaMA (Touvron et al., 2023) | Upgrade L1 with RMSNorm, RoPE, SwiGLU, GQA, KV cache; load TinyLlama weights to check | [03](core_concepts/03_rope_rmsnorm_swiglu_gqa.ipynb) | 300–500 | 1 wk |
| L3 | FlashAttention-2 (Dao, 2023) | Triton forward kernel (backward as a stretch), benchmarked against SDPA | [04](core_concepts/04_online_softmax_flash_attention.ipynb), [05](core_concepts/05_triton_kernels.ipynb) | 400–600 | 2 wk |
| L4 | GPTQ / AWQ, QLoRA's NF4 | 4-bit weights for the L2 model, accuracy loss, a Triton int4 matmul | [06](core_concepts/06_quantization_formats.ipynb) | 500–700 | 2 wk |
| L5 | DeepSeekMoE (Dai et al., 2024) | Fine-grained + shared experts, top-k routing, load balancing | [07](core_concepts/07_moe_routing.ipynb) | 400–600 | 1–2 wk |
| L6 | DeepSeek-V2: MLA | Latent KV cache, decoupled RoPE, absorbed decoding; measure cache bytes per token | [08](core_concepts/08_mla_low_rank_kv.ipynb) | 400–600 | 2 wk |
| L7 | DeepSeek-V3 | Aux-loss-free balancing, multi-token prediction, FP8 blockwise training (simulated if needed) | [06](core_concepts/06_quantization_formats.ipynb), [07](core_concepts/07_moe_routing.ipynb), [09](core_concepts/09_multi_token_prediction.ipynb) | 600–900 | 3 wk |
| L8 | Hyper-Connections, then mHC | n residual streams; Sinkhorn-constrained mixing | [10](core_concepts/10_sinkhorn_hyper_connections.ipynb) | 300–500 | 1–2 wk |
| L9 | Native Sparse Attention / DeepSeek Sparse Attention (V3.2) | Compressed + selected + sliding-window branches; lightning indexer; block-sparse Triton kernel | [11](core_concepts/11_sparse_attention.ipynb), [05](core_concepts/05_triton_kernels.ipynb) | 700–1,000 | 3–4 wk |
| L10 | DeepSeek-V4.1-Flash (2026) | Combine L5–L9, then CSA2, the causal encoder-decoder, FP4 KV cache, bounded sliding-window cache | [12](core_concepts/12_kv_cache_engineering.ipynb) | 1,500–2,500 | 4–6 wk |

Read alongside: Switch Transformer (before L5), Cross-Layer Attention and YOCO (before L10), and DeepSeek's
open-source inference code for V4 (the paper leaves details out).

## Vision and multimodal track

| # | Paper | What to build | Core concepts | Lines | Time |
|---|---|---|---|---|---|
| V1 | ViT (Dosovitskiy et al., 2020) | Patch embeddings, encoder, train on CIFAR-10 | [13](core_concepts/13_vit_patch_embeddings.ipynb) | 300–400 | 1 wk |
| V2 | CLIP (Radford et al., 2021) and SigLIP (Zhai et al., 2023) | Dual encoder on a small captioned set (Flickr8k/30k), zero-shot eval; both losses | [14](core_concepts/14_contrastive_clip_siglip.ipynb) | 400–600 | 1–2 wk |
| V3 | DDPM (Ho et al., 2020) | Small U-Net, noise-prediction training, DDPM + DDIM sampling on MNIST / CIFAR-10 | [16](core_concepts/16_diffusion_ddpm_ddim.ipynb) | 400–600 | 1–2 wk |
| V4 | DiT (Peebles & Xie, 2023) + flow matching | adaLN-Zero transformer on latents, rectified-flow loss, classifier-free guidance | [17](core_concepts/17_dit_flow_matching_cfg.ipynb), [13](core_concepts/13_vit_patch_embeddings.ipynb) | 400–600 | 2 wk |
| V5 | LLaVA (Liu et al., 2023) | Pretrained SigLIP + the L2 LLaMA (or a small open LLM), MLP projector, two-stage training | [15](core_concepts/15_vlm_connectors_mrope.ipynb) | 500–800 | 2–3 wk |
| V6 | Qwen2-VL (2024) | Native dynamic resolution, 2-D RoPE in the ViT, M-RoPE in the LLM | [15](core_concepts/15_vlm_connectors_mrope.ipynb), [13](core_concepts/13_vit_patch_embeddings.ipynb), [03](core_concepts/03_rope_rmsnorm_swiglu_gqa.ipynb) | 400–700 | 2 wk |
| V7 | VQ-VAE (van den Oord et al., 2017) | Image tokenizer with codebook, straight-through estimator, dead-code handling; FSQ as a variant | [18](core_concepts/18_vector_quantization.ipynb) | 300–500 | 1–2 wk |
| V8 | Janus / Janus-Pro (DeepSeek, 2024–25) | One LLM that understands images (SigLIP path) and generates them (VQ tokens) | [15](core_concepts/15_vlm_connectors_mrope.ipynb), [18](core_concepts/18_vector_quantization.ipynb) | 800–1,200 | 3–4 wk |

## RL track

| # | Paper | What to build | Core concepts | Lines | Time |
|---|---|---|---|---|---|
| R1 | REINFORCE, GAE (Schulman et al., 2015) | Policy gradient with baselines and GAE on `gymnasium` CartPole | [19](core_concepts/19_policy_gradients_gae.ipynb) | 200–300 | 1 wk |
| R2 | PPO (Schulman et al., 2017) | Clipped PPO with vectorized envs, on CartPole then LunarLander | [20](core_concepts/20_ppo_and_kl_estimators.ipynb) | 300–500 | 1–2 wk |
| R3 | InstructGPT (Ouyang et al., 2022) | Reward model + PPO-RLHF on a small LLM, with a per-token KL penalty | [21](core_concepts/21_reward_models_dpo.ipynb), [23](core_concepts/23_llm_rollouts_batching.ipynb) | 600–900 | 3 wk |
| R4 | DPO (Rafailov et al., 2023) | DPO on a small LLM with a preference dataset; compare with IPO / SimPO | [21](core_concepts/21_reward_models_dpo.ipynb) | 300–500 | 1 wk |
| R5 | DeepSeekMath: GRPO (Shao et al., 2024) | GRPO with verifiable math rewards on GSM8K | [22](core_concepts/22_grpo_dapo_verifiable_rewards.ipynb), [23](core_concepts/23_llm_rollouts_batching.ipynb) | 500–800 | 2 wk |
| R6 | DeepSeek-R1 (2025) | R1-Zero-style RL from a base model (Qwen2.5-0.5B/1.5B) with format + answer rewards; watch response length grow | [22](core_concepts/22_grpo_dapo_verifiable_rewards.ipynb) | 400–700 | 2–3 wk |
| R7 | DAPO (Yu et al., 2025) and Dr. GRPO (Liu et al., 2025) | Clip-higher, dynamic sampling, token-level loss, overlong shaping; ablate each | [22](core_concepts/22_grpo_dapo_verifiable_rewards.ipynb) | 200–400 | 1–2 wk |

R3–R7 need rollouts from a real LLM: use vLLM for generation on the Colab A100 for the final runs, and debug the
whole loop on a tiny model on the laptop first.

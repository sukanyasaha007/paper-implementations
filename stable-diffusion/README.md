# Stable Diffusion from scratch

PyTorch implementation of the Stable Diffusion v1.5 architecture: the VAE that maps images to latents, the CLIP text
encoder, the text-conditioned U-Net that predicts noise, and the DDPM sampler.

**Status:** the architecture is complete and checked against the v1.5 reference models (below). Still to do: the
text-to-image pipeline (classifier-free guidance sampling loop) and loading the v1.5 checkpoint to generate images.

## Paper

**High-Resolution Image Synthesis with Latent Diffusion Models**
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
*IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022)*

- arXiv: https://arxiv.org/abs/2112.10752
- PDF: https://arxiv.org/pdf/2112.10752

```bibtex
@inproceedings{rombach2022high,
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {10684--10695},
  year      = {2022}
}
```

## Related papers

| Paper | Authors | Venue | Link | Used for |
|---|---|---|---|---|
| Denoising Diffusion Probabilistic Models | Ho, Jain, Abbeel | NeurIPS 2020 | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) | Noise schedule, noise prediction, the sampler (`ddpm.py`) |
| Learning Transferable Visual Models From Natural Language Supervision (CLIP) | Radford et al. | ICML 2021 | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | Text encoder (`clip.py`) |
| U-Net: Convolutional Networks for Biomedical Image Segmentation | Ronneberger, Fischer, Brox | MICCAI 2015 | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) | Encoder-decoder with skip connections (`diffusion.py`) |
| Auto-Encoding Variational Bayes | Kingma, Welling | ICLR 2014 | [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) | VAE and the reparameterization trick (`encoder.py`) |
| GLU Variants Improve Transformer | Shazeer | arXiv 2020 | [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) | GeGLU feed-forward in the U-Net attention blocks |

## Learning resources

Followed along with Umar Jamil's video
[Coding Stable Diffusion from scratch in PyTorch](https://www.youtube.com/watch?v=ZBKpAp_6TGI) and his reference code,
[hkproj/pytorch-stable-diffusion](https://github.com/hkproj/pytorch-stable-diffusion) (MIT). Module and weight names
follow his code so the v1.5 checkpoint can be loaded with his converter.

## Files

| File | What it contains |
|---|---|
| `attention.py` | Multi-head self-attention (with optional causal mask) and cross-attention to the text context |
| `encoder.py` | VAE encoder: image (3, H, W) → latent (4, H/8, W/8), with the reparameterization trick |
| `decoder.py` | VAE residual and attention blocks, and the decoder back to (3, H, W) |
| `clip.py` | CLIP text encoder: 77 tokens → (77, 768) context |
| `diffusion.py` | Time embedding, U-Net residual and attention blocks, the U-Net, and the noise-prediction head |
| `ddpm.py` | DDPM sampler: noise schedule, forward noising, reverse step, image-to-image strength |
| `smoke_test.py` | Architecture checks with random weights; no checkpoint download |

## Run

```bash
pip install -r requirements.txt
python smoke_test.py
```

## Checks

`smoke_test.py` runs in a few seconds on CPU:

- **Parameter counts match Stable Diffusion v1.5 exactly**: U-Net 859,520,964, VAE 83,653,863, CLIP text encoder
  123,060,480 (the counts of diffusers' `UNet2DConditionModel` / `AutoencoderKL` and transformers' `CLIPTextModel`
  built from the v1.5 configs). The full-size models are built on PyTorch's meta device, so no memory is allocated.
- **Full-size shapes**: a 512×512 image encodes to a 4×64×64 latent and decodes back; the U-Net maps a 64×64 latent,
  77-token context and timestep to a 64×64 noise prediction.
- **Attention is correct**: self-attention (with and without the causal mask) and cross-attention match
  `torch.nn.functional.scaled_dot_product_attention` given the same projections.
- **Sampler math**: if the model predicts the exact noise, the final step recovers the clean latent.

## Notes

**Why latents.** Diffusing in the VAE's 4×64×64 latent space instead of 3×512×512 pixels cuts the work per U-Net
step by about 48×; the VAE handles perceptual detail once, at the end.

**Merging attention heads.** After `scores @ v` the tensor is `(batch, heads, seq, d_head)`. It has to be transposed
to `(batch, seq, heads, d_head)` before `view(batch, seq, dim)`. Viewing without the transpose has the right number
of elements, so it raises no error, but it interleaves values from different heads and positions. The SDPA check
in `smoke_test.py` catches this.

**Time conditioning.** The timestep becomes a 320-dim sinusoidal vector, an MLP expands it to 1280, and every U-Net
residual block projects it to its channel count and adds it to every pixel.

**Text conditioning.** In each U-Net attention block, cross-attention takes queries from the latent pixels and keys
and values from CLIP's 77 × 768 output, so `k_proj` and `v_proj` map 768 → the block's channel count (320, 640 or
1280).

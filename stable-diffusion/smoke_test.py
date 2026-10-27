"""Architecture checks with random weights; no checkpoint download needed.

Usage: python smoke_test.py
"""
import torch
import torch.nn.functional as F

from attention import CrossAttention, SelfAttention
from clip import CLIP
from ddpm import DDPMSampler
from decoder import VAEDecoder
from diffusion import Diffusion, get_time_embedding
from encoder import VAEEncoder

# Parameter counts of the Stable Diffusion v1.5 reference implementations (diffusers UNet2DConditionModel and
# AutoencoderKL, transformers CLIPTextModel) built from the v1.5 configs
SD15_PARAMS = {'diffusion': 859_520_964, 'vae': 83_653_863, 'clip': 123_060_480}


def count(model):
    return sum(p.numel() for p in model.parameters())


def smoke_parameter_counts():
    # The meta device builds full-size models without allocating memory
    with torch.device('meta'):
        assert count(Diffusion()) == SD15_PARAMS['diffusion']
        assert count(VAEEncoder()) + count(VAEDecoder()) == SD15_PARAMS['vae']
        assert count(CLIP()) == SD15_PARAMS['clip']
    print('parameter counts match SD v1.5: ok')


def smoke_full_size_shapes():
    # Shape propagation through the full-size models, still on the meta device
    with torch.device('meta'):
        image = torch.randn(1, 3, 512, 512)
        noise = torch.randn(1, 4, 64, 64)
        latent = VAEEncoder()(image, noise)
        assert latent.shape == (1, 4, 64, 64)
        assert VAEDecoder()(latent).shape == (1, 3, 512, 512)

        context = CLIP()(torch.zeros(2, 77, dtype=torch.long))
        assert context.shape == (2, 77, 768)

        time = torch.randn(1, 320)
        assert Diffusion()(torch.randn(2, 4, 64, 64), context, time).shape == (2, 4, 64, 64)
    print('full-size shapes (512x512 image, 64x64 latent, 77 tokens): ok')


def smoke_attention():
    torch.manual_seed(0)
    attn = SelfAttention(n_heads=4, d_embed=32).eval()
    x = torch.randn(2, 10, 32)

    # Reference: PyTorch's fused attention with the same projections
    q, k, v = attn.in_proj(x).chunk(3, dim=-1)
    split = lambda t: t.view(2, 10, 4, 8).transpose(1, 2)
    for causal in (False, True):
        expected = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=causal)
        expected = attn.out_proj(expected.transpose(1, 2).reshape(2, 10, 32))
        assert torch.allclose(attn(x, causal_mask=causal), expected, atol=1e-5)

    cross = CrossAttention(n_heads=4, d_embed=32, d_cross=24).eval()
    y = torch.randn(2, 7, 24)
    qc = cross.q_proj(x).view(2, 10, 4, 8).transpose(1, 2)
    kc = cross.k_proj(y).view(2, 7, 4, 8).transpose(1, 2)
    vc = cross.v_proj(y).view(2, 7, 4, 8).transpose(1, 2)
    expected = cross.out_proj(F.scaled_dot_product_attention(qc, kc, vc).transpose(1, 2).reshape(2, 10, 32))
    assert torch.allclose(cross(x, y), expected, atol=1e-5)
    print('self- and cross-attention match scaled_dot_product_attention: ok')


def smoke_time_embedding():
    emb = get_time_embedding(0)
    assert emb.shape == (1, 320)
    # t = 0: every cosine is 1, every sine is 0
    assert torch.allclose(emb[0, :160], torch.ones(160)) and torch.allclose(emb[0, 160:], torch.zeros(160))
    print('time embedding: ok')


def smoke_sampler():
    sampler = DDPMSampler(torch.Generator().manual_seed(0))
    sampler.set_inference_timesteps(50)
    assert sampler.timesteps[:3].tolist() == [980, 960, 940] and sampler.timesteps[-1].item() == 0

    # If the model predicts the exact noise, the final step (t = 0) returns the clean sample
    x0 = torch.randn(1, 4, 8, 8)
    eps = torch.randn(1, 4, 8, 8)
    a0 = sampler.alphas_cumprod[0]
    x_t = a0 ** 0.5 * x0 + (1 - a0) ** 0.5 * eps
    assert torch.allclose(sampler.step(0, x_t, eps), x0, atol=1e-5)

    # Forward process: noise level rises with t
    noisy = sampler.add_noise(torch.zeros(4096), torch.tensor(999))
    assert abs(noisy.std().item() - (1 - sampler.alphas_cumprod[999]).sqrt().item()) < 0.05
    assert sampler.step(980, torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)).shape == (1, 4, 8, 8)

    sampler.set_strength(0.8)
    assert len(sampler.timesteps) == 40
    print('ddpm sampler: ok')


if __name__ == '__main__':
    smoke_parameter_counts()
    smoke_full_size_shapes()
    smoke_attention()
    smoke_time_embedding()
    smoke_sampler()

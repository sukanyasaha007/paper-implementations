import torch


class DDPMSampler:
    def __init__(self, generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.0120):
        # Stable Diffusion's "scaled linear" schedule: linear in sqrt(beta)
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # alpha_bar_t: how much image survives after t steps
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.arange(num_training_steps - 1, -1, -1)

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps  # 1000 // 50 = 20
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1) * step_ratio  # 980, 960, ..., 0

    def _get_previous_timestep(self, timestep):
        return timestep - self.num_train_timesteps // self.num_inference_steps  # 980 -> 960

    def _get_variance(self, timestep):
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        # Posterior variance (DDPM eq. 7): beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        return torch.clamp(variance, min=1e-20)

    def set_strength(self, strength=1):
        # Image-to-image: skip the noisiest steps so the input image is only partly destroyed
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep, latents, model_output):
        t = timestep
        prev_t = self._get_previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 1. guess the clean latent from the predicted noise (DDPM eq. 15)
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # 2. posterior mean (DDPM eq. 7): blend the guess and the current latent
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 3. add a little fresh noise (except at the last step)
        if t > 0:
            noise = torch.randn(model_output.shape, generator=self.generator,
                                device=model_output.device, dtype=model_output.dtype)
            pred_prev_sample = pred_prev_sample + self._get_variance(t) ** 0.5 * noise
        return pred_prev_sample

    def add_noise(self, original_samples, timesteps):
        # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps    (one jump, no loop)
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5                  # signal strength
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5  # noise strength
        while sqrt_alpha_prod.dim() < original_samples.dim():               # (B,) -> (B, 1, 1, 1)
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noise = torch.randn(original_samples.shape, generator=self.generator,
                            device=original_samples.device, dtype=original_samples.dtype)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

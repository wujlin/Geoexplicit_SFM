"""
DDPM 噪声调度器：
- 管理噪声添加（前向扩散）
- 管理采样（反向去噪）
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Models (DDPM) 调度器
    
    前向过程：q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    反向过程：p(x_{t-1} | x_t) 由模型预测噪声后计算
    """
    
    def __init__(
        self,
        num_diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        
        # 计算 beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_diffusion_steps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 前向扩散所需系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 反向去噪所需系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 后验分布方差 (用于采样时加噪)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        
        # 后验均值系数
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in 'Improved DDPM'"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0, 0.999)
    
    def _get_coefficient(self, coef: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """获取时间步 t 对应的系数，并 broadcast 到目标 shape"""
        device = t.device
        coef = coef.to(device)
        out = coef.gather(-1, t.long())
        # Reshape to (B, 1, 1, ...) for broadcasting
        while len(out.shape) < len(shape):
            out = out.unsqueeze(-1)
        return out
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向扩散：添加噪声
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            original_samples: (B, ...) 原始样本 x_0
            noise: (B, ...) 标准高斯噪声
            timesteps: (B,) 时间步
        
        Returns:
            noisy_samples: (B, ...) 加噪后的样本 x_t
        """
        sqrt_alpha_prod = self._get_coefficient(
            self.sqrt_alphas_cumprod, timesteps, original_samples.shape
        )
        sqrt_one_minus_alpha_prod = self._get_coefficient(
            self.sqrt_one_minus_alphas_cumprod, timesteps, original_samples.shape
        )
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        反向去噪一步：从 x_t 到 x_{t-1}
        
        Args:
            model_output: 模型预测的噪声 epsilon
            timestep: (B,) 当前时间步
            sample: (B, ...) 当前样本 x_t
            generator: 随机数生成器
        
        Returns:
            prev_sample: (B, ...) 去噪后的样本 x_{t-1}
        """
        # 预测 x_0
        sqrt_recip_alphas_cumprod = self._get_coefficient(
            self.sqrt_recip_alphas_cumprod, timestep, sample.shape
        )
        sqrt_recipm1_alphas_cumprod = self._get_coefficient(
            self.sqrt_recipm1_alphas_cumprod, timestep, sample.shape
        )
        
        # x_0 = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
        pred_original_sample = (
            sqrt_recip_alphas_cumprod * sample - sqrt_recipm1_alphas_cumprod * model_output
        )
        
        # Clip x_0 prediction
        if self.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample, 
                -self.clip_sample_range, 
                self.clip_sample_range
            )
        
        # 计算后验均值
        posterior_mean_coef1 = self._get_coefficient(
            self.posterior_mean_coef1, timestep, sample.shape
        )
        posterior_mean_coef2 = self._get_coefficient(
            self.posterior_mean_coef2, timestep, sample.shape
        )
        
        posterior_mean = (
            posterior_mean_coef1 * pred_original_sample + posterior_mean_coef2 * sample
        )
        
        # 添加噪声（除了 t=0）
        posterior_variance = self._get_coefficient(
            self.posterior_variance, timestep, sample.shape
        )
        
        # t=0 时不加噪声
        is_not_zero = (timestep > 0).float()
        while len(is_not_zero.shape) < len(sample.shape):
            is_not_zero = is_not_zero.unsqueeze(-1)
        
        noise = torch.randn(
            sample.shape, 
            device=sample.device, 
            dtype=sample.dtype,
            generator=generator
        )
        
        prev_sample = posterior_mean + is_not_zero * torch.sqrt(posterior_variance) * noise
        
        return prev_sample
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """随机采样时间步"""
        return torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        condition: torch.Tensor,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        完整的采样过程：从纯噪声生成样本
        
        Args:
            model: 去噪模型，输入 (x_t, t, condition)，输出预测噪声
            shape: 目标样本形状 (B, T, D)
            condition: 条件输入 (B, obs_dim)
            device: 设备
            generator: 随机数生成器
        
        Returns:
            samples: (B, T, D) 生成的样本
        """
        batch_size = shape[0]
        
        # 从纯噪声开始
        sample = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        
        # 逐步去噪
        for t in reversed(range(self.num_diffusion_steps)):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 模型预测噪声 (注意 UNet 期望输入是 (B, C, T))
            sample_input = sample.permute(0, 2, 1)  # (B, T, D) -> (B, D, T)
            model_output = model(sample_input, timesteps, condition)
            model_output = model_output.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
            
            # 去噪一步
            sample = self.step(model_output, timesteps, sample, generator=generator)
        
        return sample


class DDIMScheduler(DDPMScheduler):
    """
    Denoising Diffusion Implicit Models (DDIM) 调度器
    可以使用更少的步数进行采样
    
    支持 Classifier-Free Guidance (CFG):
    - 训练时随机丢弃 condition (在 train.py 中实现)
    - 推理时使用 guidance_scale 增强 condition 影响
    """
    
    def __init__(
        self,
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        eta: float = 0.0,  # eta=0 时是确定性采样
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        super().__init__(
            num_diffusion_steps=num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        
        # 计算推理时使用的时间步子集
        self.set_timesteps(num_inference_steps)
    
    def set_timesteps(self, num_inference_steps: int):
        """设置推理时使用的时间步"""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_diffusion_steps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        self.timesteps = timesteps.flip(0)  # 从大到小
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        prev_timestep: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """DDIM 去噪一步"""
        # 获取当前和上一个时间步的 alpha_bar
        alpha_prod_t = self._get_coefficient(self.alphas_cumprod, timestep, sample.shape)
        
        if prev_timestep is None:
            prev_t = timestep - self.num_diffusion_steps // self.num_inference_steps
            prev_t = torch.clamp(prev_t, min=0)
        else:
            prev_t = prev_timestep
        
        alpha_prod_t_prev = self._get_coefficient(self.alphas_cumprod, prev_t, sample.shape)
        
        # 预测 x_0
        sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
        
        pred_x0 = (sample - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t
        
        if self.clip_sample:
            pred_x0 = torch.clamp(pred_x0, -self.clip_sample_range, self.clip_sample_range)
        
        # 计算 x_{t-1}
        sqrt_alpha_prod_t_prev = torch.sqrt(alpha_prod_t_prev)
        sqrt_one_minus_alpha_prod_t_prev = torch.sqrt(1 - alpha_prod_t_prev)
        
        # DDIM 公式
        sigma = self.eta * torch.sqrt(
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        )
        
        pred_direction = sqrt_one_minus_alpha_prod_t_prev * model_output
        
        # 是否加噪
        if self.eta > 0:
            noise = torch.randn(sample.shape, device=sample.device, generator=generator)
        else:
            noise = torch.zeros_like(sample)
        
        prev_sample = sqrt_alpha_prod_t_prev * pred_x0 + pred_direction + sigma * noise
        
        return prev_sample
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        condition: torch.Tensor,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """DDIM 采样（更快）"""
        batch_size = shape[0]
        
        sample = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        
        timesteps = self.timesteps.to(device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 获取上一个时间步
            if i + 1 < len(timesteps):
                prev_t = timesteps[i + 1]
            else:
                prev_t = torch.tensor(0)
            prev_t_batch = torch.full((batch_size,), prev_t, device=device, dtype=torch.long)
            
            # 模型预测
            sample_input = sample.permute(0, 2, 1)
            model_output = model(sample_input, t_batch, condition)
            model_output = model_output.permute(0, 2, 1)
            
            # 去噪
            sample = self.step(
                model_output, t_batch, sample, 
                prev_timestep=prev_t_batch, 
                generator=generator
            )
        
        return sample

    @torch.no_grad()
    def sample_cfg(
        self,
        model: nn.Module,
        shape: tuple,
        condition: torch.Tensor,
        device: torch.device,
        guidance_scale: float = 3.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Classifier-Free Guidance (CFG) 采样
        
        CFG 公式:
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                   = (1 - guidance_scale) * eps_uncond + guidance_scale * eps_cond
        
        Args:
            model: 去噪模型
            shape: 目标形状 (B, T, D)
            condition: 条件输入 (B, cond_dim)
            device: 设备
            guidance_scale: CFG 强度 (1.0=无增强, 3.0=中等, 7.5=强)
            generator: 随机数生成器
        
        Returns:
            samples: (B, T, D) 生成的样本
        """
        batch_size = shape[0]
        
        # 初始噪声
        sample = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        
        # 无条件 condition (全零)
        uncond = torch.zeros_like(condition)
        
        timesteps = self.timesteps.to(device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 获取上一个时间步
            if i + 1 < len(timesteps):
                prev_t = timesteps[i + 1]
            else:
                prev_t = torch.tensor(0)
            prev_t_batch = torch.full((batch_size,), prev_t, device=device, dtype=torch.long)
            
            # 准备输入
            sample_input = sample.permute(0, 2, 1)  # (B, T, D) -> (B, D, T)
            
            # 无条件预测
            eps_uncond = model(sample_input, t_batch, uncond)
            # 有条件预测
            eps_cond = model(sample_input, t_batch, condition)
            
            # CFG 公式
            model_output = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            model_output = model_output.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
            
            # 去噪
            sample = self.step(
                model_output, t_batch, sample,
                prev_timestep=prev_t_batch,
                generator=generator
            )
        
        return sample

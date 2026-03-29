import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# MLP BASELINE VERSIONS
# =========================================================================

class MLPBaseline_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(52, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 100),
        )

    def forward(self, x):
        y = self.net(x)
        return y.view(-1, 1, 100)


# =========================================================================
# DDPM / U-NET VERSIONS
# =========================================================================

def timestep_embedding(t, dim):
    """Sinusoidal timestep embedding (standard DDPM helper)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, time_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond_emb, time_emb):
        h = self.conv1(x)
        h = h + self.cond_proj(cond_emb).unsqueeze(-1) + self.time_proj(time_emb).unsqueeze(-1)
        h = self.act(self.norm1(h))
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.act(self.norm2(h))
        return h + self.skip(x)


# -------------------------------------------------------------------------
# Self-Attention for 1D sequences (channel-last for MultiheadAttention)
# -------------------------------------------------------------------------

class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        # x: [B, C, L] -> MHA expects [B, L, C]
        x_perm = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x_perm, x_perm, x_perm)
        attn_out = attn_out.permute(0, 2, 1)
        x = x + attn_out
        x = self.norm(x)
        return x


class ConditionalUNet1D_v1(nn.Module):
    def __init__(self, cond_dim=52, time_dim=64, base_ch=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )

        # Encoder
        self.enc1 = ResidualBlock1D(1, base_ch, cond_dim, time_dim)
        self.down1 = nn.Conv1d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.enc2 = ResidualBlock1D(base_ch, base_ch * 2, cond_dim, time_dim)
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.mid = ResidualBlock1D(base_ch * 2, base_ch * 2, cond_dim, time_dim)

        # Decoder
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock1D(base_ch * 4, base_ch, cond_dim, time_dim)
        self.up2 = nn.ConvTranspose1d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock1D(base_ch * 2, base_ch, cond_dim, time_dim)

        self.out = nn.Conv1d(base_ch, 1, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        # Embed time and condition once per batch.
        t_emb = self.time_embed(timestep_embedding(t, self.time_embed[0].in_features))
        cond_emb = self.cond_embed(cond)

        # -------------------------------------------------------------
        # PADDING FIX: Feature vector padding for compatibility with pooling
        # Pad the spatial dimension from 100 to 128 to make it nicely divisible.
        # -------------------------------------------------------------
        orig_len = x.shape[-1]  # 100
        pad_left = (128 - orig_len) // 2
        pad_right = 128 - orig_len - pad_left
        x_padded = F.pad(x, (pad_left, pad_right), mode='constant', value=0)

        # Encoder
        e1 = self.enc1(x_padded, cond_emb, t_emb)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, cond_emb, t_emb)
        d2 = self.down2(e2)

        # Bottleneck
        m = self.mid(d2, cond_emb, t_emb)

        # Decoder with skip connections
        u1 = self.up1(m)
        u1 = torch.cat([u1, e2], dim=1)
        u1 = self.dec1(u1, cond_emb, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, e1], dim=1)
        u2 = self.dec2(u2, cond_emb, t_emb)

        out_padded = self.out(u2)

        # Undo padding: Crop back to 100 (original length)
        out = out_padded[..., pad_left:128-pad_right]

        return out


class ConditionalUNet1D_v2(nn.Module):
    def __init__(self, cond_dim=52, time_dim=64, base_ch=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )

        # Encoder
        self.enc1 = ResidualBlock1D(1, base_ch, cond_dim, time_dim)
        self.down1 = nn.Conv1d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.enc2 = ResidualBlock1D(base_ch, base_ch * 2, cond_dim, time_dim)
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.mid = ResidualBlock1D(base_ch * 2, base_ch * 2, cond_dim, time_dim)

        # Decoder
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock1D(base_ch * 4, base_ch, cond_dim, time_dim)
        self.up2 = nn.ConvTranspose1d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock1D(base_ch * 2, base_ch, cond_dim, time_dim)

        self.out = nn.Conv1d(base_ch, 1, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        t_emb = self.time_embed(timestep_embedding(t, self.time_embed[0].in_features))
        cond_emb = self.cond_embed(cond)

        # -------------------------------------------------------------
        # PADDING FIX: Replicate pad spatial dimension to 128 (100 + 14 + 14 = 128)
        # -------------------------------------------------------------
        x = F.pad(x, (14, 14), mode='replicate')

        # Encoder
        e1 = self.enc1(x, cond_emb, t_emb)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, cond_emb, t_emb)
        d2 = self.down2(e2)

        # Bottleneck
        m = self.mid(d2, cond_emb, t_emb)

        # Decoder with skip connections
        u1 = self.up1(m)
        u1 = torch.cat([u1, e2], dim=1)
        u1 = self.dec1(u1, cond_emb, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, e1], dim=1)
        u2 = self.dec2(u2, cond_emb, t_emb)

        out = self.out(u2)

        # Undo padding: Crop the 14 padded pixels off both ends
        out = out[..., 14:-14]

        return out


class ConditionalUNet1D_v3(nn.Module):
    def __init__(self, cond_dim=52, time_dim=64, base_ch=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )

        # Encoder
        self.enc1 = ResidualBlock1D(1, base_ch, cond_dim, time_dim)
        self.down1 = nn.Conv1d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.enc2 = ResidualBlock1D(base_ch, base_ch * 2, cond_dim, time_dim)
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)

        # Bottleneck with attention and higher capacity
        self.mid1 = ResidualBlock1D(base_ch * 2, base_ch * 4, cond_dim, time_dim)
        self.mid_attn = SelfAttention1D(base_ch * 4)
        self.mid2 = ResidualBlock1D(base_ch * 4, base_ch * 2, cond_dim, time_dim)

        # Decoder
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock1D(base_ch * 4, base_ch, cond_dim, time_dim)
        self.up2 = nn.ConvTranspose1d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock1D(base_ch * 2, base_ch, cond_dim, time_dim)

        self.out = nn.Conv1d(base_ch, 1, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        t_emb = self.time_embed(timestep_embedding(t, self.time_embed[0].in_features))
        cond_emb = self.cond_embed(cond)

        # Replicate pad to reach length 128, same as v2
        x = F.pad(x, (14, 14), mode='replicate')

        # Encoder
        e1 = self.enc1(x, cond_emb, t_emb)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, cond_emb, t_emb)
        d2 = self.down2(e2)

        # Bottleneck with self-attention
        m = self.mid1(d2, cond_emb, t_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, cond_emb, t_emb)

        # Decoder with skip connections
        u1 = self.up1(m)
        u1 = torch.cat([u1, e2], dim=1)
        u1 = self.dec1(u1, cond_emb, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, e1], dim=1)
        u2 = self.dec2(u2, cond_emb, t_emb)

        out = self.out(u2)

        # Undo padding: crop 14 pixels off each end
        out = out[..., 14:-14]

        return out


class DDPM_v1(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(self, x0, cond):
        b = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.timesteps, (b,), device=device)
        noise = torch.randn_like(x0)
        alphas_bar = self.alphas_cumprod[t].view(b, 1, 1)
        noisy = torch.sqrt(alphas_bar) * x0 + torch.sqrt(1 - alphas_bar) * noise
        pred_noise = self.model(noisy, t, cond)
        return (pred_noise - noise).pow(2).mean()


class DDPM_v2(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(self, x0, cond):
        b = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.timesteps, (b,), device=device)
        noise = torch.randn_like(x0)
        alphas_bar = self.alphas_cumprod[t].view(b, 1, 1)
        noisy = torch.sqrt(alphas_bar) * x0 + torch.sqrt(1 - alphas_bar) * noise
        pred_noise = self.model(noisy, t, cond)
        return (pred_noise - noise).pow(2).mean()

    @torch.no_grad()
    def sample(self, cond, shape):
        device = cond.device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = self.model(x, t_batch, cond)
            beta_t = self.betas[t]
            alpha_t = 1.0 - beta_t
            alpha_bar_t = self.alphas_cumprod[t]

            sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

            model_mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_alpha_bar * eps)

            if t > 0:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(beta_t) * noise
            else:
                x = model_mean
            
            # Safety Clamping to prevent exploding values
            x = torch.clamp(x, min=-1.0, max=1.0)
            
        return x


class DDPM_v3(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(self, x0, cond):
        b = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.timesteps, (b,), device=device)
        noise = torch.randn_like(x0)
        alphas_bar = self.alphas_cumprod[t].view(b, 1, 1)
        noisy = torch.sqrt(alphas_bar) * x0 + torch.sqrt(1 - alphas_bar) * noise
        pred_noise = self.model(noisy, t, cond)
        return (pred_noise - noise).pow(2).mean()

    @torch.no_grad()
    def sample(self, cond, shape, guidance_scale=3.0):
        device = cond.device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            if guidance_scale > 1.0:
                noise_cond = self.model(x, t_batch, cond)
                noise_uncond = self.model(x, t_batch, torch.zeros_like(cond))
                pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                pred_noise = self.model(x, t_batch, cond)

            beta_t = self.betas[t]
            alpha_t = 1.0 - beta_t
            alpha_bar_t = self.alphas_cumprod[t]

            sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

            model_mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_alpha_bar * pred_noise)

            if t > 0:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(beta_t) * noise
            else:
                x = model_mean

            x = torch.clamp(x, min=-1.0, max=1.0)

        return x

# =========================================================================
# MODEL REGISTRIES
# =========================================================================

# Dictionaries for dynamic model loading by string name
MLP_MODELS = {
    "v1": MLPBaseline_v1
}

UNET_MODELS = {
    "v1": ConditionalUNet1D_v1,
    "v2": ConditionalUNet1D_v2,
    "v3": ConditionalUNet1D_v3
}

DDPM_MODELS = {
    "v1": DDPM_v1,
    "v2": DDPM_v2,
    "v3": DDPM_v3
}

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ======================================================
# Utils
# ======================================================
def normalize_map(x, eps=1e-6):
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def make_group_norm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def repeat_to_three(x):
    if x.size(1) == 3:
        return x
    if x.size(1) == 1:
        return x.repeat(1, 3, 1, 1)
    if x.size(1) == 2:
        return torch.cat([x, x[:, :1]], dim=1)
    return x[:, :3]


def build_texture_map(
    x,
    window_size=5,
    high_freq_ratio=0.5,
    row_chunk=16,
    eps=1e-8,
):
    """
    Local FFT high-frequency energy-ratio map.

    Relative to the original model, only the texture prior itself is
    replaced. The following SADA mask/attention/strength structure is
    unchanged.
    """
    if x.ndim != 4:
        raise ValueError(
            f"Expected x with shape (B, C, H, W), got {tuple(x.shape)}."
        )
    if window_size <= 1 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer greater than 1.")
    if not 0.0 < high_freq_ratio < 1.0:
        raise ValueError("high_freq_ratio must lie strictly between 0 and 1.")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive.")

    _, _, height, width = x.shape
    padding = window_size // 2
    pad_mode = (
        "reflect"
        if height > padding and width > padding
        else "replicate"
    )
    x_padded = F.pad(
        x,
        (padding, padding, padding, padding),
        mode=pad_mode,
    )
    patches = (
        x_padded
        .unfold(2, window_size, 1)
        .unfold(3, window_size, 1)
    )

    freq = torch.fft.fftfreq(window_size, d=1.0, device=x.device)
    freq_y, freq_x = torch.meshgrid(freq, freq, indexing="ij")
    radius = torch.sqrt(freq_x.square() + freq_y.square())
    normalized_radius = radius / radius.max().clamp_min(eps)
    high_freq_mask = (
        normalized_radius >= high_freq_ratio
    ).to(dtype=torch.float32)
    high_freq_mask = high_freq_mask.view(
        1, 1, 1, 1, window_size, window_size
    )

    texture_rows = []
    for row_start in range(0, height, row_chunk):
        row_end = min(row_start + row_chunk, height)
        patch_chunk = patches[:, :, row_start:row_end].float()
        spectrum = torch.fft.fft2(
            patch_chunk,
            dim=(-2, -1),
            norm="ortho",
        )
        energy = spectrum.real.square() + spectrum.imag.square()
        high_energy = (
            energy * high_freq_mask
        ).sum(dim=(1, 4, 5))
        total_energy = energy.sum(dim=(1, 4, 5))
        ratio = high_energy / (total_energy + eps)
        texture_rows.append(ratio.unsqueeze(1))

    return torch.cat(texture_rows, dim=2).to(dtype=x.dtype)


# ======================================================
# New MoE routing cues
# Only T and M are replaced:
#   old router: [X, T, M, capacity, modality]
#   new router: [X, gradient, covariance, context, capacity, modality]
# ======================================================
class GradientTextureCue(nn.Module):
    """Extract local texture-detail cues with fixed Sobel gradients."""

    def __init__(self, dim, grad_dim, cue_dim, eps=1e-6):
        super().__init__()
        if grad_dim < 1:
            raise ValueError("grad_dim must be positive.")
        self.eps = eps
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, grad_dim, 1, bias=False),
            make_group_norm(grad_dim, max_groups=4),
            nn.GELU(),
        )

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ) / 8.0
        sobel_y = sobel_x.transpose(0, 1).contiguous()
        self.register_buffer(
            "sobel_x",
            sobel_x.view(1, 1, 3, 3).repeat(grad_dim, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            sobel_y.view(1, 1, 3, 3).repeat(grad_dim, 1, 1, 1),
            persistent=False,
        )

        self.project = nn.Sequential(
            nn.Conv2d(grad_dim * 3, cue_dim, 1),
            make_group_norm(cue_dim, max_groups=4),
            nn.GELU(),
            nn.Conv2d(cue_dim, cue_dim, 1),
        )

    def forward(self, x, height, width):
        feat_2d = rearrange(
            x,
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )
        z = self.reduce(feat_2d)

        # Keep the fixed derivative calculation in FP32 for stable AMP use.
        z_float = z.float()
        grad_x = F.conv2d(
            z_float,
            self.sobel_x,
            padding=1,
            groups=self.sobel_x.size(0),
        )
        grad_y = F.conv2d(
            z_float,
            self.sobel_y,
            padding=1,
            groups=self.sobel_y.size(0),
        )
        magnitude = torch.sqrt(
            grad_x.square() + grad_y.square() + self.eps
        )
        descriptor = torch.cat(
            [
                torch.log1p(magnitude),
                torch.log1p(grad_x.abs()),
                torch.log1p(grad_y.abs()),
            ],
            dim=1,
        ).to(dtype=z.dtype)
        cue = self.project(descriptor)
        return rearrange(cue, "b c h w -> b (h w) c")


class LocalChannelCovarianceCue(nn.Module):
    """Describe signed local co-variation between reduced feature channels."""

    def __init__(
        self,
        dim,
        stat_dim,
        cue_dim,
        window_size=3,
        eps=1e-6,
    ):
        super().__init__()
        if stat_dim < 2:
            raise ValueError("stat_dim must be at least 2.")
        if window_size <= 1 or window_size % 2 == 0:
            raise ValueError(
                "window_size must be an odd integer greater than 1."
            )
        self.stat_dim = stat_dim
        self.window_size = window_size
        self.eps = eps
        self.reduce = nn.Conv2d(dim, stat_dim, 1, bias=False)

        triangle = torch.triu_indices(stat_dim, stat_dim, offset=1)
        self.register_buffer("tri_row", triangle[0], persistent=False)
        self.register_buffer("tri_col", triangle[1], persistent=False)

        num_pairs = stat_dim * (stat_dim - 1) // 2
        self.project = nn.Sequential(
            nn.LayerNorm(num_pairs),
            nn.Linear(num_pairs, cue_dim),
            nn.GELU(),
            nn.Linear(cue_dim, cue_dim),
        )

    def forward(self, x, height, width):
        feat_2d = rearrange(
            x,
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )
        z = self.reduce(feat_2d).float()

        # Image-wise standardisation makes covariance magnitudes comparable
        # without turning the local statistic into a correlation matrix.
        global_mean = z.mean(dim=(2, 3), keepdim=True)
        global_var = z.var(
            dim=(2, 3),
            keepdim=True,
            unbiased=False,
        )
        z = (z - global_mean) / torch.sqrt(global_var + self.eps)

        padding = self.window_size // 2
        z_pad = F.pad(
            z,
            (padding, padding, padding, padding),
            mode="replicate",
        )
        patches = F.unfold(
            z_pad,
            kernel_size=self.window_size,
            stride=1,
        )
        batch_size = z.size(0)
        samples = self.window_size * self.window_size
        tokens = height * width
        patches = (
            patches.view(batch_size, self.stat_dim, samples, tokens)
            .permute(0, 3, 1, 2)
        )
        centered = patches - patches.mean(dim=-1, keepdim=True)
        covariance = torch.matmul(
            centered,
            centered.transpose(-1, -2),
        ) / float(max(samples - 1, 1))

        # Only off-diagonal terms are retained: they explicitly encode
        # inter-channel relations rather than single-channel variation.
        covariance_pairs = covariance[
            ..., self.tri_row, self.tri_col
        ]
        covariance_pairs = (
            covariance_pairs.sign()
            * torch.log1p(covariance_pairs.abs())
        ).to(dtype=x.dtype)
        return self.project(covariance_pairs)


class MultiScaleContextPredictionRelation(nn.Module):
    """
    MCPR: estimate each centre token from centre-excluded neighbours at
    several spatial lags, then encode prediction, residual, agreement, and
    scale preference as the context-dependency cue.
    """

    def __init__(
        self,
        dim,
        context_dim,
        cue_dim,
        dilations=(1, 2, 4),
        eps=1e-6,
    ):
        super().__init__()
        if context_dim < 1:
            raise ValueError("context_dim must be positive.")
        if len(dilations) < 1:
            raise ValueError("dilations must not be empty.")
        if any(int(item) < 1 for item in dilations):
            raise ValueError("all context dilations must be positive.")

        self.context_dim = context_dim
        self.dilations = tuple(int(item) for item in dilations)
        self.eps = eps
        self.query = nn.Conv2d(dim, context_dim, 1, bias=False)
        self.key = nn.Conv2d(dim, context_dim, 1, bias=False)
        self.value = nn.Conv2d(dim, context_dim, 1, bias=False)
        self.relative_bias = nn.Parameter(
            torch.zeros(len(self.dilations), 8)
        )

        relation_dim = context_dim * 4 + 1
        self.relation_project = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, cue_dim),
            nn.GELU(),
            nn.Linear(cue_dim, cue_dim),
        )
        self.scale_score = nn.Sequential(
            nn.LayerNorm(cue_dim),
            nn.Linear(cue_dim, max(cue_dim // 2, 1)),
            nn.GELU(),
            nn.Linear(max(cue_dim // 2, 1), 1),
        )

        self.last_scale_weights = None
        self.last_context_agreement = None

    @staticmethod
    def _centre_excluded_unfold(z, dilation):
        pad_mode = (
            "reflect"
            if z.size(-2) > dilation and z.size(-1) > dilation
            else "replicate"
        )
        z_pad = F.pad(
            z,
            (dilation, dilation, dilation, dilation),
            mode=pad_mode,
        )
        patches = F.unfold(
            z_pad,
            kernel_size=3,
            dilation=dilation,
            stride=1,
        )
        batch_size, channels, height, width = z.shape
        patches = patches.view(
            batch_size,
            channels,
            9,
            height * width,
        )
        return torch.cat(
            [patches[:, :, :4], patches[:, :, 5:]],
            dim=2,
        )

    def forward(self, x, height, width):
        feat_2d = rearrange(
            x,
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )
        query = self.query(feat_2d)
        key = self.key(feat_2d)
        value = self.value(feat_2d)

        query_tokens = rearrange(
            query,
            "b c h w -> b (h w) c",
        ).float()
        centre_value = rearrange(
            value,
            "b c h w -> b (h w) c",
        ).float()

        relation_features = []
        agreements = []
        scale = float(self.context_dim) ** -0.5
        for scale_index, dilation in enumerate(self.dilations):
            key_neighbours = self._centre_excluded_unfold(
                key, dilation
            ).float()
            value_neighbours = self._centre_excluded_unfold(
                value, dilation
            ).float()

            # (B, N, 8): the centre acts only as a query. It is never one
            # of the values aggregated to form the context estimate.
            scores = torch.einsum(
                "bnc,bcqn->bnq",
                query_tokens,
                key_neighbours,
            ) * scale
            scores = scores + self.relative_bias[
                scale_index
            ].view(1, 1, 8)
            attention = F.softmax(scores, dim=-1)
            context = torch.einsum(
                "bnq,bcqn->bnc",
                attention,
                value_neighbours,
            )

            residual = centre_value - context
            agreement = F.cosine_similarity(
                centre_value,
                context,
                dim=-1,
                eps=self.eps,
            ).unsqueeze(-1)
            relation_descriptor = torch.cat(
                [
                    centre_value,
                    context,
                    residual,
                    centre_value * context,
                    agreement,
                ],
                dim=-1,
            ).to(dtype=x.dtype)
            relation_features.append(
                self.relation_project(relation_descriptor)
            )
            agreements.append(agreement)

        relations = torch.stack(relation_features, dim=2)
        scale_logits = self.scale_score(relations).squeeze(-1)
        scale_weights = F.softmax(scale_logits, dim=-1)
        context_cue = torch.sum(
            relations * scale_weights.unsqueeze(-1),
            dim=2,
        )

        self.last_scale_weights = scale_weights.detach()
        self.last_context_agreement = torch.cat(
            agreements,
            dim=-1,
        ).detach()
        return context_cue

# ======================================================
# Multi-secret SADA with old-version strengths:
# - per-secret attention
# - per-secret multi-scale interact pyramid
# - per-secret mask / strength
# ======================================================
class MultiSecretPyramidSADA(nn.Module):
    def __init__(self, dim, secret_channels):
        super().__init__()
        self.num_secrets = len(secret_channels)
        self.dim = dim

        self.c_proj = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, 1, 1),
            nn.GroupNorm(4, dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
            nn.GELU()
        )

        self.secret_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, dim // 2, 3, 1, 1),
                nn.GroupNorm(4, dim // 2),
                nn.GELU(),
                nn.Conv2d(dim // 2, dim, 3, 1, 1),
                nn.GroupNorm(8, dim),
                nn.GELU()
            )
            for in_ch in secret_channels
        ])

        self.secret_attn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1),
                nn.Sigmoid()
            )
            for _ in secret_channels
        ])

        # old-version multi-scale interaction pyramid per secret
        self.secret_interacts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim * 2, dim, 3, 1, 1),
                    nn.GroupNorm(8, dim),
                    nn.GELU()
                ) for _ in range(3)
            ])
            for _ in secret_channels
        ])

        self.secret_mask = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 3 + 2, dim * 2, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim, 1, 3, 1, 1),
                nn.Sigmoid()
            )
            for _ in secret_channels
        ])

        self.secret_strength = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim * 2, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            for _ in secret_channels
        ])

        self.texture_gate = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, cover, secrets, valid_mask=None, return_vis=False):
        B = cover.size(0)
        if valid_mask is None:
            valid_mask = torch.ones(B, self.num_secrets, device=cover.device)

        c_feat = self.c_proj(cover)
        cover_texture = build_texture_map(cover)
        cover_texture_norm = normalize_map(cover_texture)
        texture_gate = self.texture_gate(cover_texture_norm)

        fused = c_feat
        total_mask = 0.0
        active_count = valid_mask.sum(dim=1, keepdim=True).view(B, 1, 1, 1).clamp(min=1.0)

        mask_list = []
        strength_list = []
        attn_list = []
        secret_texture_list = []

        for i, secret in enumerate(secrets):
            s_feat = self.secret_projs[i](secret)

            attn_map = self.secret_attn[i](torch.cat([c_feat, s_feat], dim=1))
            c_feat_attn = c_feat * (1.0 - attn_map)
            s_feat_attn = s_feat * attn_map

            feat_now = torch.cat([c_feat_attn, s_feat_attn], dim=1)
            pyramid_feats = []
            for j, layer in enumerate(self.secret_interacts[i]):
                if j > 0:
                    feat_now = F.avg_pool2d(feat_now, 2, 2)
                inter_feat = layer(feat_now)
                pyramid_feats.append(
                    F.interpolate(inter_feat, size=c_feat.shape[2:], mode='bilinear', align_corners=False)
                )

            pyramid_cat = torch.cat(pyramid_feats, dim=1)

            secret_texture = build_texture_map(secret)
            secret_texture_norm = normalize_map(secret_texture)

            mask_in = torch.cat([pyramid_cat, cover_texture_norm, secret_texture_norm], dim=1)
            mask_learned = self.secret_mask[i](mask_in)
            mask_raw = mask_learned * texture_gate
            mask_final = 0.2 + 0.8 * mask_raw

            learned_strength = self.secret_strength[i](torch.cat([c_feat, s_feat], dim=1)).view(B, 1, 1, 1)
            strength = 0.6 + 0.7 * learned_strength

            active = valid_mask[:, i].view(B, 1, 1, 1)
            fused = fused + active * (mask_final * strength) * s_feat
            total_mask = total_mask + active * mask_final

            mask_list.append(mask_final)
            strength_list.append(learned_strength)
            attn_list.append(attn_map.mean(dim=1, keepdim=True))
            secret_texture_list.append(secret_texture)

        total_mask = torch.clamp(total_mask / active_count, 0.0, 1.0)

        if return_vis:
            vis = {
                "cover_texture": cover_texture.detach(),
                "cover_texture_norm": cover_texture_norm.detach(),
                "texture_gate": texture_gate.detach(),
                "mask_total": total_mask.detach(),
                "mask_list": [m.detach() for m in mask_list],
                "strength_list": [s.detach() for s in strength_list],
                "attn_list": [a.detach() for a in attn_list],
                "secret_texture_list": [t.detach() for t in secret_texture_list],
            }
            return fused, cover_texture, total_mask, vis

        return fused, cover_texture, total_mask


# ======================================================
# SCPE
# ======================================================
class EnhancedSCPE(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dwc = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GroupNorm(max(1, dim // 4), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

        self.freq_gate = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 2),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, m_freq, H, W):
        feat_2d = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        pos_feat = self.dwc(feat_2d)
        m_freq_resized = F.interpolate(m_freq, size=(H, W), mode='bilinear', align_corners=False)
        gate = self.freq_gate(1.0 - normalize_map(m_freq_resized))
        out_2d = feat_2d + pos_feat * gate
        return rearrange(out_2d, 'b c h w -> b (h w) c')


# ======================================================
# Experts
# ======================================================
class StandardExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return self.net(x)


class GatedExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.gate = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        g = torch.sigmoid(self.gate(x))
        return self.fc2(h * g)


class LocalExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.dwconv = nn.Conv1d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        # x: (B, N, C)
        x = F.gelu(self.fc1(x))
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = F.gelu(x)
        return self.fc2(x)


class LightExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        return self.net(x)


# 新增专家1：更强调边缘/轮廓/结构跃迁
class StructureExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pre = nn.Linear(dim, dim)
        self.dwconv3 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.dwconv5 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.mix = nn.Conv1d(dim * 2, dim, 1)
        self.post = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, N, C)
        h = F.gelu(self.pre(x))
        ht = h.transpose(1, 2)
        edge3 = self.dwconv3(ht)
        edge5 = self.dwconv5(ht)
        feat = torch.cat([edge3, edge5], dim=1)
        feat = self.mix(feat).transpose(1, 2)
        feat = F.gelu(feat)
        return self.post(feat)


# 新增专家2：更强调低频/平滑趋势
class SmoothExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pre = nn.Linear(dim, dim)
        self.smooth = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.post = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, N, C)
        h = F.gelu(self.pre(x))
        ht = h.transpose(1, 2)
        low = self.smooth(ht).transpose(1, 2)
        g = self.gate(h)
        out = low * g + h * (1.0 - g)
        return self.post(out)


# ======================================================
# Capacity/modality-aware MoE
# 完整版：保留 topk/router/load/importance 的缓存
# ======================================================
class UnifiedFreqMoELayer(nn.Module):
    def __init__(self, dim, cue_dim, num_experts, top_k=1, router_noise_std=0.01):
        super().__init__()

        self.top_k = top_k
        self.num_experts = num_experts
        self.router_noise_std = router_noise_std

        # Keep X, capacity and modality exactly as in the old router.
        # Only old T/M are replaced by the three new cue branches.
        self.gradient_norm = nn.LayerNorm(cue_dim)
        self.covariance_norm = nn.LayerNorm(cue_dim)
        self.context_norm = nn.LayerNorm(cue_dim)

        self.router = nn.Sequential(
            nn.Linear(dim + cue_dim * 3 + 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_experts)
        )

        expert_pool = [
            StandardExpert(dim),
            GatedExpert(dim),
            LocalExpert(dim),
            LightExpert(dim),
            StructureExpert(dim),
            SmoothExpert(dim),
        ]

        if num_experts <= len(expert_pool):
            self.experts = nn.ModuleList(expert_pool[:num_experts])
        else:
            extras = [StandardExpert(dim) for _ in range(num_experts - len(expert_pool))]
            self.experts = nn.ModuleList(expert_pool + extras)

        self.last_topk_indices = None
        self.last_topk_weights = None
        self.last_router_weights = None
        self.last_load = None
        self.last_importance = None

    @staticmethod
    def cv_squared(x, eps=1e-6):
        mean = x.mean()
        var = x.var(unbiased=False)
        return var / (mean * mean + eps)

    def forward(
        self,
        x,
        gradient_cue,
        covariance_cue,
        context_cue,
        capacity_token,
        modality_token
    ):
        B, N, C = x.shape

        gradient_cue = self.gradient_norm(gradient_cue)
        covariance_cue = self.covariance_norm(covariance_cue)
        context_cue = self.context_norm(context_cue)

        router_in = torch.cat([
            x,
            gradient_cue,
            covariance_cue,
            context_cue,
            capacity_token,
            modality_token
        ], dim=-1)
        router_logits = self.router(router_in)

        if self.training and self.router_noise_std > 0:
            router_logits = (
                router_logits
                + torch.randn_like(router_logits) * self.router_noise_std
            )

        # Old adaptive temperature was driven by T, which is no longer a
        # router input. Use the midpoint of the old [0.7, 1.0] range so that
        # no extra routing condition is introduced.
        temp = 0.85
        weights = F.softmax(router_logits / temp, dim=-1)

        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        output = torch.zeros_like(x)
        importance = weights.mean(dim=(0, 1))
        expert_usage = torch.zeros(self.num_experts, device=x.device)

        for k in range(self.top_k):
            onehot = F.one_hot(topk_indices[..., k], self.num_experts).float()
            expert_usage += onehot.sum(dim=(0, 1))

            for i in range(self.num_experts):
                idx = onehot[..., i].bool()
                if idx.any():
                    expert_in = x[idx]  # (num_selected, C)

                    # 原始 expert 是按 (B, N, C) 写的，这里包装成 batch=1
                    if expert_in.dim() == 2:
                        expert_in_pack = expert_in.unsqueeze(0)    # (1, Ns, C)
                        expert_out = self.experts[i](expert_in_pack).squeeze(0)
                    else:
                        expert_out = self.experts[i](expert_in)

                    w = topk_weights[..., k][idx].unsqueeze(-1)
                    output[idx] += expert_out * w

        load = expert_usage / (B * N * self.top_k + 1e-6)
        aux_importance = self.cv_squared(importance)
        aux_load = self.cv_squared(load)
        aux_loss = 0.01 * (aux_importance + aux_load)

        self.last_topk_indices = topk_indices.detach()
        self.last_topk_weights = topk_weights.detach()
        self.last_router_weights = weights.detach()
        self.last_load = load.detach()
        self.last_importance = importance.detach()

        return output, aux_loss, load.detach(), importance.detach()


class UnifiedFreqMoEBlock(nn.Module):
    def __init__(self, dim, cue_dim, num_experts):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.moe = UnifiedFreqMoELayer(
            dim=dim,
            cue_dim=cue_dim,
            num_experts=num_experts,
            top_k=1,
            router_noise_std=0.01
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.freq_conv = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, 1)
        )

        self.last_topk_indices = None
        self.last_topk_weights = None
        self.last_router_weights = None

    def forward(
        self,
        x,
        gradient_cue,
        covariance_cue,
        context_cue,
        capacity_token,
        modality_token
    ):
        residual = x
        x = self.norm1(x)

        moe_out, aux_loss, load, importance = self.moe(
            x,
            gradient_cue,
            covariance_cue,
            context_cue,
            capacity_token,
            modality_token
        )

        freq_feat = x.transpose(1, 2)
        freq_feat = self.freq_conv(freq_feat)
        freq_feat = freq_feat.transpose(1, 2)

        x = residual + moe_out + 0.3 * freq_feat

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        self.last_topk_indices = self.moe.last_topk_indices
        self.last_topk_weights = self.moe.last_topk_weights
        self.last_router_weights = self.moe.last_router_weights

        return x, aux_loss, load, importance


# ======================================================
# Main Model
# same model file for single and multi
# ======================================================
class EnhancedHighQualityStegNetUnified(nn.Module):
    def __init__(self, config, mode='encoder'):
        super().__init__()

        dim = config.embed_dim
        self.mode = mode
        self.dim = dim
        self.num_experts = config.num_experts
        self.num_secrets = config.num_secrets
        self.secret_channels = list(config.secret_channels)

        self.in_proj = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
            nn.GELU()
        )

        if mode == 'encoder':
            self.sada = MultiSecretPyramidSADA(dim, self.secret_channels)

        self.down1 = nn.Conv2d(dim, dim * 2, 4, 2, 1)
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 4, 2, 1)

        self.scpe = EnhancedSCPE(dim * 4)

        bottleneck_dim = dim * 4
        cue_dim = getattr(config, "moe_cue_dim", 32)
        grad_dim = getattr(
            config,
            "gradient_stat_dim",
            max(bottleneck_dim // 8, 4)
        )
        cov_stat_dim = getattr(config, "cov_stat_dim", 8)
        cov_window = getattr(config, "cov_window", 3)
        context_dim = getattr(
            config,
            "context_stat_dim",
            max(bottleneck_dim // 6, 8)
        )
        context_dilations = tuple(
            getattr(config, "context_dilations", (1, 2, 4))
        )

        self.gradient_cue = GradientTextureCue(
            dim=bottleneck_dim,
            grad_dim=grad_dim,
            cue_dim=cue_dim
        )
        self.covariance_cue = LocalChannelCovarianceCue(
            dim=bottleneck_dim,
            stat_dim=cov_stat_dim,
            cue_dim=cue_dim,
            window_size=cov_window
        )
        self.context_cue = MultiScaleContextPredictionRelation(
            dim=bottleneck_dim,
            context_dim=context_dim,
            cue_dim=cue_dim,
            dilations=context_dilations
        )

        self.bottleneck = nn.ModuleList([
            UnifiedFreqMoEBlock(
                bottleneck_dim,
                cue_dim,
                config.num_experts
            )
            for _ in range(8)
        ])

        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(dim * 2, dim, 4, 2, 1)

        if mode == 'encoder':
            self.head = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, 3, 3, 1, 1),
                nn.Tanh()
            )
        else:
            self.secret_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim, dim * 4, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(dim * 4, ch, 3, 1, 1),
                    nn.Tanh()
                )
                for ch in self.secret_channels
            ])

        self.residual_scale = nn.Parameter(torch.tensor(0.35))

        self.last_expert_loads = []
        self.last_expert_importance = []
        self.last_routing_maps = []
        self.last_topk_weights = []
        self.last_router_weights = []
        self.last_sada_vis = None

    def _make_condition_tokens(
        self,
        valid_mask,
        secret_type_ids,
        B,
        H,
        W,
        device,
        dtype
    ):
        # Keep the quantity/modality conditions consistent with the
        # original unified model in BOTH encoder and decoder branches.
        # These are task-level configuration signals, not secret content.
        if valid_mask is None:
            valid_mask = torch.ones(
                B,
                self.num_secrets,
                device=device,
                dtype=dtype
            )
        else:
            valid_mask = valid_mask.to(device=device, dtype=dtype)

        capacity_value = (
            valid_mask.sum(dim=1, keepdim=True)
            / float(self.num_secrets)
        )

        if secret_type_ids is None:
            stype = torch.zeros(
                self.num_secrets,
                device=device,
                dtype=dtype
            )
        else:
            stype = secret_type_ids.to(device=device, dtype=dtype)

        stype = stype.view(1, -1).repeat(B, 1)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        modality_value = (
            (stype * valid_mask).sum(dim=1, keepdim=True)
            / denom
        )

        capacity_token = (
            capacity_value
            .reshape(B, 1, 1)
            .repeat(1, H * W, 1)
        )
        modality_token = (
            modality_value
            .reshape(B, 1, 1)
            .repeat(1, H * W, 1)
        )

        return capacity_token, modality_token

    def forward(self, x, secrets=None, valid_mask=None, secret_type_ids=None, return_vis=False):
        cover = x
        sada_vis = None

        if self.mode == 'encoder':
            if secrets is None:
                raise ValueError("encoder mode requires secrets list")
            if valid_mask is None:
                valid_mask = torch.ones(x.size(0), self.num_secrets, device=x.device)
            if secret_type_ids is None:
                secret_type_ids = torch.zeros(self.num_secrets, device=x.device)

            if return_vis:
                x, m_freq, mask, sada_vis = self.sada(x, secrets, valid_mask=valid_mask, return_vis=True)
            else:
                x, m_freq, mask = self.sada(x, secrets, valid_mask=valid_mask, return_vis=False)
        else:
            # Decoder receives the stego image plus the same task-level
            # quantity/modality configuration used by the encoder. Secret
            # image content and encoder-side SADA information are not used.
            if secrets is not None:
                raise ValueError(
                    "Decoder leakage guard: secrets are not allowed."
                )

            x = self.in_proj(x)
            # cover is the received stego tensor in decoder mode.
            m_freq = build_texture_map(cover)
            mask = None

        x1 = self.down1(x)
        x2 = self.down2(x1)

        B, C, H, W = x2.shape
        xf = rearrange(x2, 'b c h w -> b (h w) c')
        xf = self.scpe(xf, m_freq, H, W)

        # New routing cues are computed from the current branch feature.
        # Decoder-side cues are therefore functions of stego only.
        gradient_cue = self.gradient_cue(xf, H, W)
        covariance_cue = self.covariance_cue(xf, H, W)
        context_cue = self.context_cue(xf, H, W)

        capacity_token, modality_token = self._make_condition_tokens(
            valid_mask,
            secret_type_ids,
            B,
            H,
            W,
            xf.device,
            xf.dtype
        )

        aux_loss_total = 0.0
        self.last_expert_loads = []
        self.last_expert_importance = []
        self.last_routing_maps = []
        self.last_topk_weights = []
        self.last_router_weights = []
        self.last_sada_vis = sada_vis if sada_vis is not None else None

        for layer in self.bottleneck:
            xf, aux_loss, load, importance = layer(
                xf,
                gradient_cue,
                covariance_cue,
                context_cue,
                capacity_token,
                modality_token
            )
            aux_loss_total = aux_loss_total + aux_loss
            self.last_expert_loads.append(load)
            self.last_expert_importance.append(importance)

            routing_map = None
            if layer.last_topk_indices is not None:
                routing_map = layer.last_topk_indices[..., 0].reshape(B, H, W).detach()
            self.last_routing_maps.append(routing_map)
            self.last_topk_weights.append(layer.last_topk_weights.detach() if layer.last_topk_weights is not None else None)
            self.last_router_weights.append(layer.last_router_weights.detach() if layer.last_router_weights is not None else None)

        x2_out = rearrange(xf, 'b (h w) c -> b c h w', h=H, w=W)

        y = self.up2(x2_out)
        y = self.up1(y)

        if self.mode == 'encoder':
            residual = self.head(y)
            scale = torch.clamp(self.residual_scale, 0.2, 1.0)
            stego = torch.clamp(cover + residual * scale, 0, 1)

            if return_vis:
                return stego, residual, aux_loss_total, mask, {
                    "sada": self.last_sada_vis,
                    "moe_loads": self.last_expert_loads,
                    "moe_importance": self.last_expert_importance,
                    "routing_maps": self.last_routing_maps,
                    "topk_weights": self.last_topk_weights,
                    "router_weights": self.last_router_weights,
                }

            return stego, residual, aux_loss_total, mask
        else:
            outputs = [torch.clamp((head(y) + 1.0) * 0.5, 0, 1) for head in self.secret_heads]
            return outputs, aux_loss_total
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SGA(nn.Module):
    def __init__(self, L, d_model=64, num_heads=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.group_size = L // 3
        self.dropout = nn.Dropout(dropout)

        self.conv_q = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding='same'),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        self.conv_k = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding='same'),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        self.conv_v = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding='same'),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        self.fc = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, L = x.shape
        group_size = self.group_size

        x1 = x[:, :group_size].unsqueeze(1)  # [B,1,S1]
        x2 = x[:, group_size:2 * group_size].unsqueeze(1)  # [B,1,S2]
        x3 = x[:, 2 * group_size:3 * group_size].unsqueeze(1)  # [B,1,S3]

        q = self.conv_q(x1).permute(0, 2, 1)  # [B,S,d_model]
        k = self.conv_k(x2).permute(0, 2, 1)
        v = self.conv_v(x3).permute(0, 2, 1)

        attn_out, _ = self.attn(q, k, v)  # [B,S,d_model]
        out = attn_out.mean(dim=1)  # [B,d_model]
        out = self.fc(out)

        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)

        return out


class SDSM(nn.Module):

    def __init__(self, spa_channels, d_model=64, d_state_h=8, d_state_w=8, d_conv_h=3, d_conv_w=3,
                 fuse_mode='concat'):
        super().__init__()
        self.fuse_mode = fuse_mode

        self.proj_in = nn.Sequential(
            nn.Conv2d(spa_channels, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        self.mamba_h = Mamba(d_model=d_model, d_state=d_state_h, d_conv=d_conv_h)  # 横向
        self.mamba_w = Mamba(d_model=d_model, d_state=d_state_w, d_conv=d_conv_w)  # 纵向

        if fuse_mode == 'concat':
            self.fuse = nn.Sequential(
                nn.Conv2d(d_model * 2, d_model, kernel_size=1),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True)
            )
        elif fuse_mode == 'gated':
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(d_model * 2, d_model, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.fuse = self.gate = None

    def forward(self, x):

        B, C, H, W = x.shape
        x0 = self.proj_in(x)  # [B, d_model, H, W]

        x_h = x0.permute(0, 2, 3, 1).contiguous().view(B * H, W, x0.size(1))  # [B*H, W, d_model]
        y_h = self.mamba_h(x_h)
        y_h = y_h.view(B, H, W, x0.size(1)).permute(0, 3, 1, 2)  # [B, d_model, H, W]

        x_w = x0.permute(0, 3, 2, 1).contiguous().view(B * W, H, x0.size(1))  # [B*W, H, d_model]
        y_w = self.mamba_w(x_w)
        y_w = y_w.view(B, W, H, x0.size(1)).permute(0, 3, 2, 1)  # [B, d_model, H, W]

        hw = torch.cat([y_h, y_w], dim=1)

        if self.fuse_mode == 'concat':
            y = self.fuse(hw)
        elif self.fuse_mode == 'gated':
            g = self.gate(hw)
            y = g * y_h + (1 - g) * y_w
        else:
            y = 0.5 * (y_h + y_w)
        return y


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # [B, C, H, W] -> [B, C, 1, 1]
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(x)  # [B, C, 1, 1]
        x = x.squeeze(-1)  # [B, C, 1]
        x = self.conv(x)  # [B, C, 1]
        return self.act(x)  # [B, C, 1]


class SpectralAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(-1)  # [B, C] -> [B, C, 1]
        x = self.conv(x)  # [B, C, 1]
        x = self.act(x)  # [B, C, 1]
        x = x.squeeze(-1)  # [B, C, 1] -> [B, C]
        return x


class DBSSM(nn.Module):
    def __init__(self, spe_channels, spa_channels, num_classes, d_model=64):
        super(DBSSM, self).__init__()
        self.d_model = d_model

        self.spectral_branch = SGA(spe_channels, d_model, num_heads=1, dropout=0.1)

        self.spatial_branch = SDSM(spa_channels, d_model)

        self.spectral_attention = SpectralAttention(d_model)

        self.spatial_attention = SpatialAttention(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, patch, spectrum, return_feature=False):
        B, C, H, W = patch.shape

        spectral_feat = self.spectral_branch(spectrum)
        # print(spectral_feat.shape)

        spatial_feat = self.spatial_branch(patch)
        # print(spatial_feat.shape)

        spectral_att = self.spectral_attention(spectral_feat)
        spatial_att = self.spatial_attention(spatial_feat).squeeze(-1)
        spatial_weighted = spatial_feat * spectral_att.view(B, self.d_model, 1, 1)
        spectral_weighted = spectral_feat * spatial_att

        spatial_pooled = F.adaptive_avg_pool2d(spatial_weighted, 1).view(B, self.d_model)
        fused = torch.cat([spatial_pooled, spectral_weighted], dim=1)
        fused = F.layer_norm(fused, fused.shape[-1:])
        # print(fused.shape)

        logits = self.classifier(fused)
        if return_feature:
            return logits, fused
        else:
            return logits

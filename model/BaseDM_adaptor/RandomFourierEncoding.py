import os
import torch
import torch.nn as nn



class FourierEncoding3D(nn.Module):
    def __init__(self,embed_dim,num_frequencies=10,scale=1.0):
        """

        Args:
            embed_dim: 输入的通道数
            num_frequencies: 随机频率的个数
            scale: 频率缩放因子
        """
        super(FourierEncoding3D,self).__init__()
        self.num_frequencies = num_frequencies
        # self.embed_dim = embed_dim
        self.scale = scale
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_frequencies, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # self.frequencies = nn.Parameter(torch.randn(num_frequencies), requires_grad=False)
        self.register_buffer('frequencies', torch.randn(num_frequencies))

    def encode_dim(self,x):
        dim = torch.arange(x, device=x.device).unsqueeze(-1)  # (dim_size, 1)
        features = dim * self.frequencies.to(x.device)  # (dim_size, num_frequencies)
        sin_features = torch.sin(features)
        cos_features = torch.cos(features)
        combined_features = torch.cat([sin_features, cos_features], dim=-1)  # (dim_size, 2*num_frequencies)
        return self.mlp(combined_features)  # (dim_size, embed_dim)

    def forward(self, x):
        """

        Args:
            x: 输入张量 形状为B C t H W

        Returns:

        """
        b,c,t,h,w = x.shape
        device = x.device
        time_enc = self.encode_dim(t).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, T, embed_dim, 1, 1)
        height_enc = self.encode_dim(h).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, H, embed_dim, 1)
        width_enc = self.encode_dim(w).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, W, embed_dim)

        time_enc = time_enc.permute(0, 2, 1, 3, 4).expand(b, -1, -1, h, w)  # (B, embed_dim, T, H, W)
        height_enc = height_enc.permute(0, 2, 3, 1, 4).expand(b, -1, t, -1, w)  # (B, embed_dim, T, H, W)
        width_enc = width_enc.permute(0, 2, 3, 4, 1).expand(b, -1, t, h, -1)  # (B, embed_dim, T, H, W)

        # ===== 合并时间和空间编码 =====
        combined_enc = time_enc + height_enc + width_enc

        # ===== 拼接到输入特征 =====
        return torch.cat([x, combined_enc], dim=1)  # (B, C + embed_dim, T, H, W)


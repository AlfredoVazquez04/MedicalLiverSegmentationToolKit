import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ==============================================================================
# 1. Bloques Constructivos (Atención y Mix-FFN) - SIN CAMBIOS
# ==============================================================================

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

# ==============================================================================
# 2. Encoder (MiT) - SIN CAMBIOS
# ==============================================================================

class MiT(nn.Module):
    def __init__(self, in_chans=1, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], 
                 mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]):
        super().__init__()

        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.block1 = nn.ModuleList([Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], sr_ratio=sr_ratios[0]) for _ in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.block2 = nn.ModuleList([Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], sr_ratio=sr_ratios[1]) for _ in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.block3 = nn.ModuleList([Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], sr_ratio=sr_ratios[2]) for _ in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.block4 = nn.ModuleList([Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], sr_ratio=sr_ratios[3]) for _ in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        B = x.shape[0]
        outs = []

        x, H, W = self.patch_embed1(x)
        for blk in self.block1: x = blk(x, H, W)
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x1)

        x, H, W = self.patch_embed2(x1)
        for blk in self.block2: x = blk(x, H, W)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x2)

        x, H, W = self.patch_embed3(x2)
        for blk in self.block3: x = blk(x, H, W)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x3)

        x, H, W = self.patch_embed4(x3)
        for blk in self.block4: x = blk(x, H, W)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x4)

        return outs

# ==============================================================================
# 3. Decoder & Wrapper Principal (LÓGICA BLINDADA EN FORWARD)
# ==============================================================================

class SegFormerHead(nn.Module):
    def __init__(self, embedding_dim=256, in_channels=[64, 128, 320, 512], num_classes=16):
        super().__init__()
        self.linear_c4 = nn.Conv2d(in_channels[3], embedding_dim, 1)
        self.linear_c3 = nn.Conv2d(in_channels[2], embedding_dim, 1)
        self.linear_c2 = nn.Conv2d(in_channels[1], embedding_dim, 1)
        self.linear_c1 = nn.Conv2d(in_channels[0], embedding_dim, 1)
        
        self.dropout = nn.Dropout2d(0.1)
        self.linear_fuse = nn.Conv2d(embedding_dim*4, embedding_dim, 1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c1.shape
        
        _c4 = F.interpolate(self.linear_c4(c4), size=(h,w), mode='bilinear', align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=(h,w), mode='bilinear', align_corners=False)
        _c2 = F.interpolate(self.linear_c2(c2), size=(h,w), mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x

class SegFormer(nn.Module):
    def __init__(self, in_channels=1, num_classes=16):
        super().__init__()
        
        
        # Configuración B1 (embed_dim=64)
        embed_dims = [64, 128, 320, 512]
        depths = [2, 2, 2, 2]
        num_heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]
        mlp_ratios = [4, 4, 4, 4]
        
        self.encoder = MiT(
            in_chans=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            sr_ratios=sr_ratios,
            mlp_ratios=mlp_ratios
        )
        
        self.decoder = SegFormerHead(
            embedding_dim=256, 
            in_channels=embed_dims,
            num_classes=num_classes
        )

    def forward(self, x):
        # ------------------------------------------------------------------
        # ADAPTADOR DE FORMATO AUTOMÁTICO
        # ------------------------------------------------------------------
        
        # 1. Si es Numpy (matriz 512x512 cruda), convertir a Tensor y mover a GPU
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        # Asegurar float32 (vital para modelos médicos)
        if x.dtype != torch.float32:
            x = x.float()

        # Mover al dispositivo correcto (GPU/CPU) donde esté el modelo
        device = self.encoder.patch_embed1.proj.weight.device
        if x.device != device:
            x = x.to(device)

        # 2. Corregir Dimensiones (El paso clave)
        # Si llega (512, 512) -> Falta Batch y Canal -> Convertir a (1, 1, 512, 512)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            
        # Si llega (Batch, 512, 512) -> Falta Canal (típico en dataloaders grayscale) -> (Batch, 1, 512, 512)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
            
        # ------------------------------------------------------------------
        
        # Guardar tamaño original para el upsampling final
        input_shape = x.shape[2:] 
        
        features = self.encoder(x)
        logits = self.decoder(features)
        
        # Upsampling bilineal para devolver el tamaño exacto de entrada
        logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)
        
        return logits
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import math


class mlp_head(nn.Sequential):
    def __init__(self, dim):
        super().__init__(nn.LayerNorm(dim),
                         nn.Linear(dim, 2)
                         )


class SpaceCenter_Concentrate_Attention(torch.nn.Module):
    def __init__(self, args, weight=0.5):
        super(SpaceCenter_Concentrate_Attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.patches = args.patches
        self.batch_size = args.batch_size
        self.linear = nn.Linear(self.patches*self.patches, 1)
        self.fc = nn.Sequential(
            nn.Linear(self.patches*self.patches, self.patches*self.patches // 2, False),
            nn.ReLU(),
            nn.Linear(self.patches*self.patches // 2, 1, False),
            # nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.Softplus = nn.Softplus()
        self.mask = torch.zeros([self.patches, self.patches]).cuda()
        self.cal_mask()
        self.weight = weight

    def cal_mask(self):
        for x in range(0, self.patches):
            for y in range(0, self.patches):
                len = math.sqrt((x + 0.5 - self.patches / 2) ** 2 + (y + 0.5 - self.patches / 2) ** 2)
                self.mask[x][y] = len

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        feature_map = torch.cat([avgout, maxout], dim=1)
        feature_map = self.conv(feature_map)
        feature_map = self.sigmoid(feature_map)
        b, c, h, w = feature_map.size()

        para_b = feature_map.view(b, -1)[:, (self.patches*self.patches)//2]
        para_k = self.fc(feature_map.view(b, -1))
        para_k = self.Softplus(para_k)*(-1)  # k<0
        attention_map = torch.zeros([b, 1, self.patches, self.patches]).cuda()

        for i in range(0, self.patches):
            for j in range(0, self.patches):
                attention_map[:, 0, i, j] = self.mask[i][j] * para_k.squeeze(dim=-1) + para_b
        attention_map = self.sigmoid(attention_map)

        mask = torch.zeros([b, 1, self.patches, self.patches]).cuda()
        mask += para_b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cut_map = abs(feature_map - mask)
        cut_map = 1-cut_map

        attention_map = self.weight * attention_map + (1 - self.weight) * cut_map

        return attention_map * x, attention_map



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, x3, **kwargs):
        return self.fn(x, x2, x3, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, x3, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), self.norm(x3), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, y, mask=None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(y)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, sub_pha, lent, softmax=True ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.sub_pha = sub_pha
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, softmax=softmax))),
                # Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
        self.linear = nn.Linear(self.sub_pha, lent)

    def forward(self, x, m, n, mask=None):
        """target(query), memory"""
        x = rearrange(x, 'b c l -> b l c')
        x = self.linear(x)
        x = rearrange(x, 'b c l -> b l c')
        for attn, ff in self.layers:
            x = attn(x, m, n, mask=mask)
            x = ff(x)
            # m = attn(m, mask=mask)
            # m = ff(m)
        return x


class TransformerDecoderSEQ(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, sub_pha, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.sub_pha = sub_pha
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, softmax=softmax))),
                # Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
        self.linear = nn.Linear(self.sub_pha, dim)

    def forward(self, x, m, n, mask=None):
        """target(query), memory"""
        x = self.linear(x)
        for attn, ff in self.layers:
            x = attn(x, m, n, mask=mask)
            x = ff(x)
            # m = attn(m, mask=mask)
            # m = ff(m)
        return x

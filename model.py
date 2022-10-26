from einops import rearrange
from torch import nn
import torch
import torch.nn.functional
import collections
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import reduce
from operator import mul
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
NEG_INF = -1000000

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """ Group based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted group.
    Args:
        dim (int): Number of input channels.
        group_size (tuple[int]): The temporal length, height and width of the group.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, group_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.group_size = group_size  # Gd, Gh, Gw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            nG = mask.shape[0]
            attn = attn.view(B_ // nG, nG, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # (B, nG, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DFormerBlock3D(nn.Module):
    """ D-Former Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        group_size (tuple[int]): Group size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, group_size=(2, 7, 7), interval=8, gsm=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.gsm = gsm
        self.interval = interval


        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, group_size=self.group_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x):
        B, D, H, W, C = x.shape

        x = self.norm1(x)

        if H < self.group_size[1]:
            # if group size is larger than input resolution, we don't partition group
            self.gsm = 0
            self.group_size = (D, H, W)
        # pad feature maps to multiples of group size
        size_div = self.interval if self.gsm == 1 else self.group_size
        if isinstance(size_div, int): size_div = to_3tuple(size_div)
        pad_l = pad_t = pad_d0 = 0
        pad_d = (size_div[0] - D % size_div[0]) % size_div[0]
        pad_b = (size_div[1] - H % size_div[1]) % size_div[1]
        pad_r = (size_div[2] - W % size_div[2]) % size_div[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d))
        _, Dp, Hp, Wp, _ = x.shape

        mask = torch.zeros((1, Dp, Hp, Wp, 1), device=x.device)
        if pad_d > 0:
            mask[:, -pad_d:, :, :, :] = -1
        if pad_b > 0:
            mask[:, :, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, :, -pad_r:, :] = -1

        # group embeddings and generate attn_mask
        if self.gsm == 0: # LS-MSA
            Gd = size_div[0]
            Gh = size_div[1]
            Gw = size_div[2]
            B, D2, H2, W2, C = x.shape

            x = x.view(B, D2 // Gd, Gd, H2 // Gh, Gh, W2 // Gw, Gw, C).permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
            x = x.reshape(-1, reduce(mul, size_div), C)

            nG = (Dp * Hp * Wp) // (Gd * Gh * Gw)  # group_num

            if pad_r > 0 or pad_b > 0 or pad_d > 0:
                mask = mask.reshape(1, Dp // Gd, Gd, Hp // Gh, Gh, Wp // Gw, Gw, 1).permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()

                mask = mask.reshape(nG, 1, Gd * Gh * Gw)
                attn_mask = torch.zeros((nG, Gd * Gh * Gw, Gd * Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

        else: # GS-MSA
            B, D2, H2, W2, C = x.shape
            interval_d = Dp // self.group_size[0]
            interval_h = Hp // self.group_size[1]
            interval_w = Wp // self.group_size[2]

            Id, Ih, Iw = interval_d, interval_h, interval_w
            Gd, Gh, Gw = Dp // interval_d, Hp // interval_h, Wp // interval_w
            x = x.reshape(B, Gd, Id, Gh, Ih, Gw, Iw, C).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            x = x.reshape(B * Id * Ih * Iw, Gd * Gh * Gw, C)

            nG = interval_d * interval_h * interval_w  # group_num

            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Gd, Id, Gh, Ih, Gw, Iw, 1).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
                mask = mask.reshape(nG, 1, Gd * Gh * Gw)
                attn_mask = torch.zeros((nG, Gd * Gh * Gw, Gd * Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

        # multi-head self-attention
        x = self.attn(x, mask=attn_mask)

        # ungroup embeddings
        if self.gsm == 0:
            x = x.reshape(B, D2 // size_div[0], H2 // size_div[1], W2 // size_div[2], size_div[0], size_div[1],
                       size_div[2], C).permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous() # B, Hp//G, G, Wp//G, G, C
            x = x.view(B, D2, H2, W2, -1)
        else:
            x = x.reshape(B, interval_d, interval_h, interval_w,
                          D2 // interval_d, H2 // interval_h, W2 // interval_w, C)\
                .permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous() # B, Gh, I, Gw, I, C

            x = x.view(B, D2, H2, W2, -1)

        # remove padding
        if pad_d > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x)
        else:
            x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, 0::2, 0::2, 1::2, :]  # B D H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B D H/2 W/2 C
        x3 = x[:, 0::2, 1::2, 1::2, :]  # B D H/2 W/2 C
        x4 = x[:, 1::2, 0::2, 0::2, :]  # B D H/2 W/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B D H/2 W/2 C
        x6 = x[:, 1::2, 1::2, 0::2, :]  # B D H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D H/2 W/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpand3D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 4 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        x: B, D*H*W, C
        """
        x = self.expand(x)
        B, D, H, W, C = x.shape

        x = rearrange(x, 'b d h w (p0 p1 p2 c)-> b (d p0) (h p1) (w p2) c', p0=2, p1=2, p2=2, c=C//8)
        x = self.norm(x)
        return x


# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv3d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        feat_token = x
        cnn_feat = rearrange(feat_token, 'b d h w c -> b c d h w')
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = rearrange(x, 'b c d h w -> b d h w c')

        return x


class BasicLayer(nn.Module):
    """ D-Former blocks for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        group_size (tuple[int]): Local group size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 interval,
                 depth,
                 num_heads,
                 group_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 i_layer=None):
        super().__init__()
        self.group_size = group_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            DFormerBlock3D(
                dim=dim,
                num_heads=num_heads,
                group_size=group_size,
                interval=interval,
                gsm=0 if (i % 2 == 0) else 1,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

        self.pos_block = PosCNN(in_chans=dim, embed_dim=dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.pos_block(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class BasicLayer_up(nn.Module):
    """ D-Former blocks for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        group_size (int): Local group size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, group_size, interval,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.group_size = group_size
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            DFormerBlock3D(dim=dim,
                                   num_heads=num_heads,
                                   group_size=group_size,
                                   interval=interval,
                                   gsm=0 if (i % 2 == 0) else 1,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer)
            for i in range(depth)])

        self.pos_block = PosCNN(in_chans=dim, embed_dim=dim)
        # patch merging layer
        self.upsample = upsample
        if upsample is not None:
            self.upsample = PatchExpand3D(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = self.pos_block(x)

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed3D(nn.Module):
    """
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input CT image channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x

class DFormer3D(nn.Module):
    """ D-Former backbone.

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        group_size (int): group size. Default: (2,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 interval_list=[8, 4, 2, 1],
                 patch_size=(2, 4, 4),
                 in_chans=1,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 group_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.group_size = group_size
        self.patch_size = patch_size
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                group_size=group_size,
                interval=interval_list[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                i_layer=i_layer
            )
            self.layers.append(layer)


        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()

            if i_layer == 0:
                layer_up = PatchExpand3D(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         interval=interval_list[(self.num_layers - 1 - i_layer)],
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         group_size=group_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand3D if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        self._freeze_stages()
        self.init_weights()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        """Inflate the 2d parameters to 3d.
        The differences between 3d and 2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of 2d models should be inflated to fit in the shapes of
        the 3d counterpart.
        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        pos_keys = [k for k in state_dict.keys() if "pos" in k]
        for k in pos_keys:
            del state_dict[k]

        # delete relative position biases
        biases_keys = [k for k in state_dict.keys() if "biases" in k]
        for k in biases_keys:
            del state_dict[k]
        attn_index_keys = [k for k in state_dict.keys() if "attn" in k]
        weight = collections.OrderedDict()
        for k in attn_index_keys:
            weight[k] = state_dict[k]

        mlp_keys = [k for k in state_dict.keys() if "mlp" in k]
        for k in mlp_keys:
            weight[k] = state_dict[k]

        up_attn_keys = [k for k in self.state_dict().keys() if ("attn" or "layers_up.") in k]
        up_attn_keys = up_attn_keys[len(attn_index_keys):]
        match_attn_keys = attn_index_keys[:len(up_attn_keys)]
        s1_up = up_attn_keys[0:24]
        s1_match = match_attn_keys[16:]
        s2_up = up_attn_keys[24:32]
        s2_match = match_attn_keys[8:16]
        s3_up = up_attn_keys[32:]
        s3_match = match_attn_keys[:8]

        for i in range(len(s1_up)):
            weight[s1_up[i]] = state_dict[s1_match[i]]

        for i in range(len(s2_up)):
            weight[s2_up[i]] = state_dict[s2_match[i]]

        for i in range(len(s3_up)):
            weight[s3_up[i]] = state_dict[s3_match[i]]


        self.load_state_dict(weight, strict=False)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.pretrained != None:
            if isinstance(self.pretrained, str):
                self.apply(_init_weights)
                if self.pretrained2d:
                    # Inflate 2D model into 3D model.
                    self.inflate_weights()
            else:
                raise TypeError('pretrained must be a str or None')


    def forward_features(self, x):
        """Encoder and Bottleneck."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(rearrange(x, 'n c d h w -> n d h w c'))
            x = layer(x.contiguous())
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        """Dencoder and Skip connection"""
        x_upsample = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[(self.num_layers-1) - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            if inx < 2:
                x_upsample.append(x)
        x = self.norm_up(x)  # B L C
        x_upsample.append(x)
        return x_upsample


    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x_upsample = self.forward_up_features(x, x_downsample)
        return x_upsample


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, patch_size=(2, 4, 4), norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.up_scale = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        self.expand = nn.Linear(dim, self.up_scale*dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, D*H*W, C
        """
        x = self.expand(x)
        B, D, H, W, C = x.shape

        x = x.view(B, D, H, W, C)

        x = rearrange(x, 'b d h w (p0 p1 p2 c)-> b (d p0) (h p1) (w p2) c', p0=self.patch_size[0], p1=self.patch_size[1], p2=self.patch_size[2], c=C//self.up_scale)
        x = self.norm(x)
        x = x.view(B, self.patch_size[0] * D, self.patch_size[1] * H, self.patch_size[2] * W, -1)
        x = x.permute(0, 4, 1, 2, 3)  # B,C,D,H,W

        return x

class SegNetwork(nn.Module):

    def __init__(self, num_classes,
                 in_chan=1,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 patch_size=(2, 4, 4),
                 group_size=(2, 8, 8),
                 deep_supervision=True,
                 pretrain=None):
        super(SegNetwork, self).__init__()

        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes

        self.upscale_logits_ops = []

        self.upscale_logits_ops.append(lambda x: x)
        self.model_down = DFormer3D(pretrained=pretrain,
                                        in_chans=in_chan,
                                        group_size=group_size,
                                        patch_size=patch_size,
                                        depths=depths,
                                        num_heads=num_heads)

        self.final = []
        for i in range(len(depths) - 1):
            self.final.append(nn.Sequential(FinalPatchExpand_X4(embed_dim * 2 ** i, patch_size=patch_size),
                                            nn.Conv3d(in_channels=embed_dim * 2 ** i, out_channels=14, kernel_size=1, bias=False)))
        self.final = nn.ModuleList(self.final)

    def forward(self, x):

        seg_outputs = []
        out = self.model_down(x)

        for i in range(len(out)):
            seg_outputs.append(self.final[-(i + 1)](out[i]))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

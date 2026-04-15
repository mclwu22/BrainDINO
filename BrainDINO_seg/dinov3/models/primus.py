import torch
from einops import rearrange
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from typing import Tuple
import torch
from torch import nn

from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from einops import rearrange
import numpy as np

import math

class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
        self,
        patch_size: int, 
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()
        assert patch_size > 0
        n = int(math.log2(patch_size))

        assert 2 ** n == patch_size and n >= 1

        ch = [embed_dim]
        for _ in range(n):
            ch.append(ch[-1]//2)
        ch.append(out_channels)

        stages = []
        for i in range(n):
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch[i], ch[i + 1], kernel_size=2, stride=2),
                    norm(ch[i + 1]),
                    activation(),
                )
            )
        stages.append(nn.Conv2d(ch[-2], ch[-1], kernel_size=1))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = False

class Primus(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        x = x.repeat(1,3,1,1)
        x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        dec_out = self.up_projection(x)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")

class Primus_conv(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        num_modalities=4,
    ):
        """
        Enhanced version of Primus with learnable modality fusion through convolution.

        Similar to Primus but supports multiple input modalities with a learnable 1x1
        convolution layer to fuse them into 3 channels for the DINO encoder.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))

        # Register a Conv layer with max modalities support
        self.max_modalities = 4
        self.to3 = nn.Conv2d(self.max_modalities, 3, kernel_size=1, bias=False)

        # Initialize with average fusion
        with torch.no_grad():
            w = torch.full((3, self.max_modalities, 1, 1), 1.0 / self.max_modalities)
            self.to3.weight.copy_(w)

    def forward(self, x, ret_mask=False):
        """
        x: shape [B, C, H, W], where C is the number of modalities
        Dynamically crops self.to3 weights to only use the first C channels.
        """
        in_chans = x.shape[1]

        # Extract the first in_chans weights from the registered to3 convolution
        w = self.to3.weight[:, :in_chans, :, :]

        # Manual convolution using F.conv2d (since weight is cropped)
        import torch.nn.functional as F
        x = F.conv2d(x, w, bias=None)

        # Forward through dino encoder
        x = self.dino_encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        dec_out = self.up_projection(x)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")

class Primus_Multiscale_conv(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        num_modalities=4,
        interaction_indices =[1,2,3,4]
    ):
        """
        We follow a similar design as ViT-adapter, using intermediate layers and concat along channel dimension.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim * len(interaction_indices), num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices

        in_chans = num_modalities
        # 注册一个最大通道数的Conv层（例如支持最多8个模态）
        self.max_modalities = 4
        self.to3 = nn.Conv2d(self.max_modalities, 3, kernel_size=1, bias=False)

        # 初始化为平均融合
        with torch.no_grad():
            w = torch.full((3, self.max_modalities, 1, 1), 1.0 / self.max_modalities)
            self.to3.weight.copy_(w)


    # def forward(self, x, ret_mask=False):
    #     to3 = nn.Conv2d(x.shape[1], 3, kernel_size=1, bias=False).to(x.device)
    #     with torch.no_grad():
    #         w = torch.full((3, in_chans, 1, 1), 1.0 / max(1, in_chans))
    #         self.to3.weight.copy_(w)
    #     hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
    #     hier = torch.cat(hier, dim=1)
    #     dec_out = self.up_projection(hier)
    #     return dec_out
    def forward(self, x, ret_mask=False):
        """
        x: shape [B, C, H, W], C为模态数
        动态裁剪 self.to3 的权重，只取前 C 个通道参与卷积。
        """
        in_chans = x.shape[1]

        # 从已注册的 to3 卷积中取前 in_chans 个权重
        w = self.to3.weight[:, :in_chans, :, :]

        # 用 F.conv2d 手动卷积（因为 weight 已被裁剪）
        import torch.nn.functional as F
        x = F.conv2d(x, w, bias=None)

        # 后续交给 dino encoder
        hier = self.dino_encoder.get_intermediate_layers(
            x, n=self.interaction_indices, reshape=True
        )
        hier = torch.cat(hier, dim=1)
        dec_out = self.up_projection(hier)
        return dec_out
    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


class Primus_Multiscale(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        interaction_indices =[1,2,3,4]
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim * len(interaction_indices), num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        x = x.repeat(1,3,1,1)
        hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        dec_out = self.up_projection(hier)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
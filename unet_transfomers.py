from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
#
# from utils import config
# from models.EncoderDecoder import ConvBlock, EncoderLayer, DecoderLayer



#### CONFIG ####
# import torch

DTYPE = torch.float32
USE_GPU = True

ACCURACY = 'accuracy'
DICE_SCORE = 'dice_score'
JACCARD_INDEX = 'jaccard_index'
PRECISION = 'precision'
RECALL = 'recall'
SPECIFICITY = 'specificity'
F1_SCORE = 'f1_score'
AUROC_ = 'auroc'
AUPRC = 'auprc'


# def get_device() -> torch.device:
#     device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
#     return device
# #####
#
# DEVICE = get_device()
#### Encoder Decoder ####



class ConvBlock(nn.Module):
    def __init__(self, options, in_channels: int, out_channels: int, is_residual: bool = False,
                 bias=False) -> None:
        super(ConvBlock, self).__init__()
        print("Input channels:", in_channels)
        print("Output channels:", out_channels)
        print("Residual:", is_residual)
        print("Bias:", bias)
        print("Kernel size:", options['conv_kernel_size'])
        print("Stride rate:", options['conv_stride_rate'])
        print("Padding:", options['conv_padding'])
        print("Padding style:", options['conv_padding_style'])
        print("")


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=options['conv_kernel_size'],
                      stride=options['conv_stride_rate'],
                      padding=options['conv_padding'],
                      padding_mode=options['conv_padding_style'],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,kernel_size=options['conv_kernel_size'],
                      stride=options['conv_stride_rate'],
                      padding=options['conv_padding'],
                      padding_mode=options['conv_padding_style'],
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if is_residual:
            self.conv_skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=bias),
                nn.BatchNorm2d(out_channels)
            )

        self.is_residual = is_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shortcut = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.is_residual:
            x_shortcut = self.conv_skip(x_shortcut)
            x = torch.add(x, x_shortcut)

        x = F.relu(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, options, in_channels: int, out_channels: int, is_residual: bool = False,
                 bias=False) -> None:
        super(EncoderLayer, self).__init__()
        # print("Input channels:", in_channels)
        # print("Output channels:", out_channels)
        self.conv = ConvBlock(options, in_channels, out_channels, is_residual, bias)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x = self.conv(x)
        pool = self.pool(x)
        return x, pool


class DecoderLayer(nn.Module):
    def __init__(self, options, in_channels: int, out_channels: int, is_residual: bool = False,
                 bias=False) -> None:
        super(DecoderLayer, self).__init__()

        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=bias)
        self.conv = ConvBlock(options, in_channels, out_channels, is_residual, bias)

    def forward(self, skip_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.transpose(x)
        x = torch.cat((skip_x, x), dim=1)
        return self.conv(x)
    
############################################################
    
# class ExpandingBlock(torch.nn.Module):
#     """Class to perform upward layer in the U-Net."""

#     def __init__(self, options, input_n, output_n):
#         super(ExpandingBlock, self).__init__()

#         self.padding_style = options['conv_padding_style']
#         self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.double_conv = ConvBlock(options, input_n=input_n + output_n, output_n=output_n)

#     def forward(self, x, x_skip):
#         """Pass x through the upward layer and concatenate with opposite layer."""
#         x = self.upsample(x)

#         # Insure that x and skip H and W dimensions match.
#         x = expand_padding(x, x_skip, padding_style=self.padding_style)
#         x = torch.cat([x, x_skip], dim=1)

#         return self.double_conv(x)
    
    
# def expand_padding(x, x_contract, padding_style: str = 'constant'):
#     """
#     Insure that x and x_skip H and W dimensions match.
#     Parameters
#     ----------
#     x :
#         Image tensor of shape (batch size, channels, height, width). Expanding path.
#     x_contract :
#         Image tensor of shape (batch size, channels, height, width) Contracting path.
#         or torch.Size. Contracting path.
#     padding_style : str
#         Type of padding.

#     Returns
#     -------
#     x : ndtensor
#         Padded expanding path.
#     """
#     # Check whether x_contract is tensor or shape.
#     if type(x_contract) == type(x):
#         x_contract = x_contract.size()

#     # Calculate necessary padding to retain patch size.
#     pad_y = x_contract[2] - x.size()[2]
#     pad_x = x_contract[3] - x.size()[3]

#     if padding_style == 'zeros':
#         padding_style = 'constant'

#     x = torch.nn.functional.pad(x, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2], mode=padding_style)

#     return x

############################################################

class FeatureMap(torch.nn.Module):
    """Class to perform final 1D convolution before calculating cross entropy or using softmax."""

    def __init__(self, input_n, output_n):
        super(FeatureMap, self).__init__()

        self.feature_out = torch.nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        """Pass x through final layer."""
        return self.feature_out(x)


class TransformerUNet(nn.Module):
    def __init__(self, options) -> None:
        super(TransformerUNet, self).__init__()
        """
        channels -> unet_conv_filters
        num_heads -> num_heads
        is_residual -> is_residual
        """
        channels = options['unet_conv_filters']
        num_heads = options['num_heads']
        is_residual = options['is_residual']
        bias = options['bias']

        ####
        self.channels = channels
        self.pos_encoding = PositionalEncoding(options=options)
        self.encode = nn.ModuleList(
            [EncoderLayer(options, channels[i], channels[i + 1], is_residual, bias) for i in
             range(len(channels) - 2)])
        self.bottle_neck = ConvBlock(options, channels[-2], channels[-1], is_residual, bias)
        self.mhsa = MultiHeadSelfAttention(channels[-1], num_heads, bias)
        self.mhca = nn.ModuleList(
            [MultiHeadCrossAttention(channels[i], num_heads, channels[i], channels[i + 1], bias) for
             i in reversed(range(1, len(channels) - 1))])
        self.decode = nn.ModuleList(
            [DecoderLayer(options, channels[i + 1], channels[i], is_residual, bias) for i in
             reversed(range(1, len(channels) - 1))])
        # TODO: Change here to AI4EO output function
        # self.output = nn.Conv2d(channels[1], 1, 1) # Classifies only one class

        # Changed unet_conv_filter index from 0 to 1
        self.sic_feature_map = FeatureMap(input_n=options['unet_conv_filters'][1],
                                          output_n=options['n_classes']['SIC'])
        self.sod_feature_map = FeatureMap(input_n=options['unet_conv_filters'][1],
                                          output_n=options['n_classes']['SOD'])
        self.floe_feature_map = FeatureMap(input_n=options['unet_conv_filters'][1],
                                           output_n=options['n_classes']['FLOE'])

        self.init_weights()

    # Change output from torch.Tensor to dictionary
    def forward(self, x: torch.Tensor) -> dict:
        skip_x_list: List[torch.Tensor] = []
        for i in range(len(self.channels) - 2):
            skip_x, x = self.encode[i](x)
            skip_x_list.append(skip_x)

        x = self.bottle_neck(x)
        x = self.pos_encoding(x)
        x = self.mhsa(x)

        for i, skip_x in enumerate(reversed(skip_x_list)):
            x = self.pos_encoding(x)
            skip_x = self.pos_encoding(skip_x)
            skip_x = self.mhca[i](skip_x, x)
            x = self.decode[i](skip_x, x)

        # Change return to three classes
        # return self.output(x)

        return {'SIC': self.sic_feature_map(x),
                'SOD': self.sod_feature_map(x),
                'FLOE': self.floe_feature_map(x)}

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias=False) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=True, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            b, c, h, w = x.size()
            x = x.permute(0, 2, 3, 1).view((b, h * w, c))
            # print(self.training)
            x, _ = self.mha(x, x, x, need_weights=False)
            return x.view((b, h, w, c)).permute(0, 3, 1, 2)
        else:
            # print("Size:", x.shape)
            b, c, h, w = x.size()
            x = x.permute(0, 2, 3, 1).view((b, h * w, c)).detach()#.cpu()
            # print(self.training)
            x, _ = self.mha(x, x, x, need_weights=False)
            return x.view((b, h, w, c)).permute(0, 3, 1, 2)
            # print("Size eval:", x.size)
            # # b, h, c = x.size()
            # x = x.permute(0, 2, 1)
            # x, _ = self.mha(x, x, x, need_weights=False)
            # return x.permute(0, 2, 1)


# class MultiHeadSelfAttentionEval(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, bias=False) -> None:
#         super(MultiHeadSelfAttention, self).__init__()

#         self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, h, c = x.size()
#         x = x.permute(0, 2, 1)
#         x, _ = self.mha(x, x, x, need_weights=False)
#         return x.permute(0, 2, 1)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, channel_S: int, channel_Y: int,
                 bias=False) -> None:
        super(MultiHeadCrossAttention, self).__init__()

        self.conv_S = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channel_S, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU()
        )

        self.conv_Y = nn.Sequential(
            nn.Conv2d(channel_Y, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU()
        )

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

        self.upsample = nn.Sequential(
            nn.Conv2d(channel_S, channel_S, 1, bias=bias).apply(
                lambda m: nn.init.xavier_uniform_(m.weight.data)),
            nn.BatchNorm2d(channel_S),
            nn.Sigmoid(),
            nn.ConvTranspose2d(channel_S, channel_S, 2, 2, bias=bias)
        )

    def forward(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s_enc = s
        s = self.conv_S(s)
        y = self.conv_Y(y)

        b, c, h, w = s.size()
        s = s.permute(0, 2, 3, 1).view((b, h * w, c))

        b, c, h, w = y.size()
        y = y.permute(0, 2, 3, 1).view((b, h * w, c))

        y, _ = self.mha(y, y, s, need_weights=False)
        y = y.view((b, h, w, c)).permute(0, 3, 1, 2)

        y = self.upsample(y)

        return torch.mul(y, s_enc)


class PositionalEncoding(nn.Module):
    def __init__(self, options) -> None:
        super(PositionalEncoding, self).__init__()
        self.dtype = options['dtype']
        self.device = options['device']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        pos_encoding = self.positional_encoding(h * w, c)
        pos_encoding = pos_encoding.permute(1, 0).unsqueeze(0).repeat(b, 1, 1)
        x = x.view((b, c, h * w)) + pos_encoding
        return x.view((b, c, h, w))

    def positional_encoding(self, length: int, depth: int) -> torch.Tensor:
        depth = depth / 2

        positions = torch.arange(length, dtype=self.dtype, device=self.device)
        depths = torch.arange(depth, dtype=self.dtype, device=self.device) / depth

        angle_rates = 1 / (10000 ** depths)
        angle_rads = torch.einsum('i,j->ij', positions, angle_rates)

        pos_encoding = torch.cat((torch.sin(angle_rads), torch.cos(angle_rads)), dim=-1)

        return pos_encoding

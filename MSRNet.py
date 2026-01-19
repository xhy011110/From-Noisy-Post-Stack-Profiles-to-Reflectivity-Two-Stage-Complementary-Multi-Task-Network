"""
MSRNet
Author:Haiyang Xu
Description: This file contains the implementation of MSRNet, featuring
             Dual-Decoder architecture with Multi-scale Dense Connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==============================================================================
#                               Basic Blocks
# ==============================================================================

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out


class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


# ==============================================================================
#                         Advanced Feature Blocks
# ==============================================================================

class MakeDense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class RDB(nn.Module):
    """ Residual Dense Block """

    def __init__(self, nChannels, nDenselayer, growthRate, scale=1.0):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        self.scale = scale
        modules = []
        for i in range(nDenselayer):
            modules.append(MakeDense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out) * self.scale
        out = out + x
        return out


# ==============================================================================
#                       Multi-scale Dense Connections (MDC)
# ==============================================================================

class Decoder_MDCBlock1(nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None,
                 mode='iter1'):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock(num_filter * (2 ** i), num_filter * (2 ** (i + 1)), kernel_size, stride, padding, bias,
                          activation, norm=None)
            )
            self.up_convs.append(
                DeconvBlock(num_filter * (2 ** (i + 1)), num_filter * (2 ** i), kernel_size, stride, padding, bias,
                            activation, norm=None)
            )

    def forward(self, ft_h, ft_l_list):
        ft_fusion = ft_h
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft - len(ft_l_list) + i](ft_h)
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft - i - 1](ft_fusion - ft_l_list[i]) + ft_h_list[
                    len(ft_l_list) - i - 1]

        elif self.mode == 'iter2':
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        elif self.mode == 'iter3':
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i + 1):
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        elif self.mode == 'iter4':
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


class Encoder_MDCBlock1(nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None,
                 mode='iter1'):
        super(Encoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock(num_filter // (2 ** i), num_filter // (2 ** (i + 1)), kernel_size, stride, padding, bias,
                            activation, norm=None)
            )
            self.down_convs.append(
                ConvBlock(num_filter // (2 ** (i + 1)), num_filter // (2 ** i), kernel_size, stride, padding, bias,
                          activation, norm=None)
            )

    def forward(self, ft_l, ft_h_list):
        ft_fusion = ft_l
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_l_list = []
            for i in range(len(ft_h_list)):
                ft_l_list.append(ft_l)
                ft_l = self.up_convs[self.num_ft - len(ft_h_list) + i](ft_l)
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft_fusion = self.down_convs[self.num_ft - i - 1](ft_fusion - ft_h_list[i]) + ft_l_list[
                    len(ft_h_list) - i - 1]

        elif self.mode == 'iter2':
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        elif self.mode == 'iter3':
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[len(ft_h_list) - i - 1]
                for j in range(i + 1):
                    ft = self.down_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        elif self.mode == 'iter4':
            for i in range(len(ft_h_list)):
                ft = ft_l
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


# ==============================================================================
#                               Main Model: MSRNet
# ==============================================================================

class MSRNet(nn.Module):
    """
    MSRNet: Multi-task Super-Resolution Network
    Architecture includes a Shared Encoder and Dual Decoders.
    """

    def __init__(self, n_classes=2, in_channels=1, is_deconv=True, is_batchnorm=True, res_blocks=18):
        super(MSRNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        # Input Stage
        self.conv_input = ConvLayer(in_channels, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            ResidualBlock(16), ResidualBlock(16), ResidualBlock(16)
        )

        # Encoder Stage 1
        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.conv1 = RDB(16, 4, 16)
        self.fusion1 = Encoder_MDCBlock1(16, 2, mode='iter2')
        self.dense1 = nn.Sequential(
            ResidualBlock(32), ResidualBlock(32), ResidualBlock(32)
        )

        # Encoder Stage 2
        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv2 = RDB(32, 4, 32)
        self.fusion2 = Encoder_MDCBlock1(32, 3, mode='iter2')
        self.dense2 = nn.Sequential(
            ResidualBlock(64), ResidualBlock(64), ResidualBlock(64)
        )

        # Encoder Stage 3
        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv3 = RDB(64, 4, 64)
        self.fusion3 = Encoder_MDCBlock1(64, 4, mode='iter2')
        self.dense3 = nn.Sequential(
            ResidualBlock(128), ResidualBlock(128), ResidualBlock(128)
        )

        # Encoder Stage 4
        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.conv4 = RDB(128, 4, 128)
        self.fusion4 = Encoder_MDCBlock1(128, 5, mode='iter2')

        # Dehaze / Bottleneck
        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))

        # ------------------------ Decoder 1 ------------------------
        self.convd16x_1 = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4_1 = nn.Sequential(ResidualBlock(128), ResidualBlock(128), ResidualBlock(128))
        self.conv_4_1 = RDB(64, 4, 64)
        self.fusion_4_1 = Decoder_MDCBlock1(64, 2, mode='iter2')

        self.convd8x_1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3_1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64), ResidualBlock(64))
        self.conv_3_1 = RDB(32, 4, 32)
        self.fusion_3_1 = Decoder_MDCBlock1(32, 3, mode='iter2')

        self.convd4x_1 = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2_1 = nn.Sequential(ResidualBlock(32), ResidualBlock(32), ResidualBlock(32))
        self.conv_2_1 = RDB(16, 4, 16)
        self.fusion_2_1 = Decoder_MDCBlock1(16, 4, mode='iter2')

        self.convd2x_1 = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1_1 = nn.Sequential(ResidualBlock(16), ResidualBlock(16), ResidualBlock(16))
        self.conv_1_1 = RDB(8, 4, 8)
        self.fusion_1_1 = Decoder_MDCBlock1(8, 5, mode='iter2')

        self.convdx_1 = UpsampleConvLayer(16, 8, kernel_size=3, stride=2)
        self.dense_0_1 = nn.Sequential(ResidualBlock(8), ResidualBlock(8), ResidualBlock(8))
        self.conv_0_1 = RDB(4, 4, 4)
        self.fusion_0_1 = Decoder_MDCBlock1(4, 6, mode='iter2')

        self.conv_output1 = ConvLayer(8, 1, kernel_size=3, stride=1)

        # ------------------------ Decoder 2 ------------------------
        self.convd16x_2 = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4_2 = nn.Sequential(ResidualBlock(128), ResidualBlock(128), ResidualBlock(128))
        self.conv_4_2 = RDB(64, 4, 64)
        self.fusion_4_2 = Decoder_MDCBlock1(64, 2, mode='iter2')

        self.convd8x_2 = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3_2 = nn.Sequential(ResidualBlock(64), ResidualBlock(64), ResidualBlock(64))
        self.conv_3_2 = RDB(32, 4, 32)
        self.fusion_3_2 = Decoder_MDCBlock1(32, 3, mode='iter2')

        self.convd4x_2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2_2 = nn.Sequential(ResidualBlock(32), ResidualBlock(32), ResidualBlock(32))
        self.conv_2_2 = RDB(16, 4, 16)
        self.fusion_2_2 = Decoder_MDCBlock1(16, 4, mode='iter2')

        self.convd2x_2 = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1_2 = nn.Sequential(ResidualBlock(16), ResidualBlock(16), ResidualBlock(16))
        self.conv_1_2 = RDB(8, 4, 8)
        self.fusion_1_2 = Decoder_MDCBlock1(8, 5, mode='iter2')

        self.convdx_2 = UpsampleConvLayer(16, 8, kernel_size=3, stride=2)
        self.dense_0_2 = nn.Sequential(ResidualBlock(8), ResidualBlock(8), ResidualBlock(8))
        self.conv_0_2 = RDB(4, 4, 4)
        self.fusion_0_2 = Decoder_MDCBlock1(4, 6, mode='iter2')

        self.conv_output2 = ConvLayer(8, 2, kernel_size=3, stride=1)

    def forward(self, x, label_dsp_dim):
        """
        Args:
            x: Input tensor
            label_dsp_dim: List or tuple [Height, Width] for output cropping
        """
        # --- Encoder Forward ---
        res1x = self.conv_input(x)
        res1x_1, res1x_2 = res1x.split([(res1x.size()[1] // 2), (res1x.size()[1] // 2)], dim=1)
        feature_mem = [res1x_1]
        x_enc = self.dense0(res1x) + res1x

        res2x = self.conv2x(x_enc)
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion1(res2x_1, feature_mem)
        res2x_2 = self.conv1(res2x_2)
        feature_mem.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)
        res2x = self.dense1(res2x) + res2x

        res4x = self.conv4x(res2x)
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion2(res4x_1, feature_mem)
        res4x_2 = self.conv2(res4x_2)
        feature_mem.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion3(res8x_1, feature_mem)
        res8x_2 = self.conv3(res8x_2)
        feature_mem.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        res16x_1 = self.fusion4(res16x_1, feature_mem)
        res16x_2 = self.conv4(res16x_2)
        res16x = torch.cat((res16x_1, res16x_2), dim=1)

        # Bottleneck
        res_dehaze = res16x
        in_ft = res16x * 2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze

        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        feature_mem_up = [res16x_1]
        feature_mem_up1 = feature_mem_up.copy()

        # Save features for skip connections
        res8x3 = res8x
        res4x3 = res4x
        res2x3 = res2x
        x3 = x_enc

        # --- Decoder 1 Forward ---
        res16x_1_dec = self.convd16x_1(res16x)
        # Crop to handle padding differences
        res16x_1_dec = res16x_1_dec[:, :, 0:res8x.size()[2], 0:res8x.size()[3]]
        res8x = torch.add(res16x_1_dec, res8x3)
        res8x = self.dense_4_1(res8x) + res8x - res16x_1_dec
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion_4_1(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4_1(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)

        res8x = self.convd8x_1(res8x)
        res8x = res8x[:, :, 0:res4x.size()[2], 0:res4x.size()[3]]
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3_1(res4x) + res4x - res8x
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion_3_1(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3_1(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)

        res4x = self.convd4x_1(res4x)
        res4x = res4x[:, :, 0:res2x.size()[2], 0:res2x.size()[3]]
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2_1(res2x) + res2x - res4x
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion_2_1(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2_1(res2x_2)
        feature_mem_up.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)

        res2x = self.convd2x_1(res2x)
        res2x = res2x[:, :, 0:x_enc.size()[2], 0:x_enc.size()[3]]
        x_dec1 = torch.add(res2x, x_enc)
        x_dec1 = self.dense_1_1(x_dec1) + x_dec1 - res2x
        x_1, x_2 = x_dec1.split([(x_dec1.size()[1] // 2), (x_dec1.size()[1] // 2)], dim=1)
        x_1 = self.fusion_1_1(x_1, feature_mem_up)
        x_2 = self.conv_1_1(x_2)
        feature_mem_up.append(x_1)
        x_dec1 = torch.cat((x_1, x_2), dim=1)

        x1 = self.convdx_1(x_dec1)
        # Note: Hardcoded crop 256x256 from original code - adjust if input size changes
        x1 = x1[:, :, 0:256, 0:256]
        x_11, x_21 = x1.split([(x1.size()[1] // 2), (x1.size()[1] // 2)], dim=1)
        x_11 = self.fusion_0_1(x_11, feature_mem_up)
        x_21 = self.conv_0_1(x_21)
        x11 = torch.cat((x_11, x_21), dim=1)

        decoder1_output = self.conv_output1(x11[:, :, :label_dsp_dim[0], :label_dsp_dim[1]].contiguous())

        # --- Decoder 2 Forward ---
        res16x_2_dec = self.convd16x_2(res16x)
        res16x_2_dec = res16x_2_dec[:, :, 0:res8x3.size()[2], 0:res8x3.size()[3]]
        res8x3 = torch.add(res16x_2_dec, res8x3)
        res8x3 = self.dense_4_2(res8x3) + res8x3 - res16x_2_dec
        res8x_1, res8x_2 = res8x3.split([(res8x3.size()[1] // 2), (res8x3.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion_4_2(res8x_1, feature_mem_up1)
        res8x_2 = self.conv_4_2(res8x_2)
        feature_mem_up1.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)

        res8x = self.convd8x_2(res8x)
        res8x = res8x[:, :, 0:res4x3.size()[2], 0:res4x3.size()[3]]
        res4x3 = torch.add(res8x, res4x3)
        res4x3 = self.dense_3_2(res4x3) + res4x3 - res8x
        res4x_1, res4x_2 = res4x3.split([(res4x3.size()[1] // 2), (res4x3.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion_3_2(res4x_1, feature_mem_up1)
        res4x_2 = self.conv_3_2(res4x_2)
        feature_mem_up1.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)

        res4x = self.convd4x_2(res4x)
        res4x = res4x[:, :, 0:res2x3.size()[2], 0:res2x3.size()[3]]
        res2x3 = torch.add(res4x, res2x3)
        res2x3 = self.dense_2_2(res2x3) + res2x3 - res4x
        res2x_1, res2x_2 = res2x3.split([(res2x3.size()[1] // 2), (res2x3.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion_2_2(res2x_1, feature_mem_up1)
        res2x_2 = self.conv_2_2(res2x_2)
        feature_mem_up1.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)

        res2x = self.convd2x_2(res2x)
        res2x = res2x[:, :, 0:x3.size()[2], 0:x3.size()[3]]
        x3 = torch.add(res2x, x3)
        x3 = self.dense_1_2(x3) + x3 - res2x
        x_1, x_2 = x3.split([(x3.size()[1] // 2), (x3.size()[1] // 2)], dim=1)
        x_1 = self.fusion_1_2(x_1, feature_mem_up1)
        feature_mem_up1.append(x_1)
        x_2 = self.conv_1_2(x_2)
        x = torch.cat((x_1, x_2), dim=1)

        x12 = self.convdx_2(x)
        x12 = x12[:, :, 0:256, 0:256]  # Hardcoded crop
        x_112, x_212 = x12.split([(x12.size()[1] // 2), (x12.size()[1] // 2)], dim=1)
        x_112 = self.fusion_0_2(x_112, feature_mem_up1)
        x_212 = self.conv_0_2(x_212)
        x112 = torch.cat((x_112, x_212), dim=1)

        decoder2_output = self.conv_output2(x112[:, :, :label_dsp_dim[0], :label_dsp_dim[1]].contiguous())

        return [decoder1_output, decoder2_output]


class LossMSRNet:
    """
    Combined Loss function for MSRNet.
    Includes MSE for reconstruction and CrossEntropy for segmentation/classification.
    """

    def __init__(self, weights=[1, 1], entropy_weight=[1, 1]):
        self.criterion1 = nn.MSELoss()
        if torch.cuda.is_available():
            ew = torch.from_numpy(np.array(entropy_weight).astype(np.float32)).cuda()
        else:
            ew = torch.from_numpy(np.array(entropy_weight).astype(np.float32))

        self.criterion2 = nn.CrossEntropyLoss(weight=ew)
        self.weights = weights

    def __call__(self, outputs1, outputs2, targets1, targets2):
        mse = self.criterion1(outputs1, targets1)
        cross = self.criterion2(outputs2, torch.squeeze(targets2).long())
        loss = (self.weights[0] * mse + self.weights[1] * cross)
        return loss


# ==============================================================================
#                               Usage Example
# ==============================================================================

if __name__ == '__main__':
    print("Testing MSRNet...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example input
    x = torch.randn((2, 1, 256, 256)).to(device)  # Batch size 2

    # Initialize Model
    model = MSRNet(n_classes=1, in_channels=1, is_deconv=True, is_batchnorm=True).to(device)

    # Forward Pass
    try:
        out = model(x, label_dsp_dim=[256, 256])
        print(f"Success! Output shapes: Decoder1 {out[0].size()}, Decoder2 {out[1].size()}")
    except Exception as e:
        print(f"Error during forward pass: {e}")
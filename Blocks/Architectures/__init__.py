# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Blocks.Architectures.MLP import MLP, Dense

from Blocks.Architectures.Vision.CNN import CNN, Conv

from Blocks.Architectures.Residual import Residual

from Blocks.Architectures.Vision.ResNet import MiniResNet, MiniResNet as ResNet, ResNet18, ResNet50
from Blocks.Architectures.Vision.ConvMixer import ConvMixer
from Blocks.Architectures.Vision.ConvNeXt import ConvNeXt, ConvNeXtTiny, ConvNeXtBase

from Blocks.Architectures.MultiHeadAttention import Attention, MHDPA, CrossAttention, SelfAttention, ReLA

from Blocks.Architectures.Transformer import AttentionBlock, CrossAttentionBlock, SelfAttentionBlock, Transformer

from Blocks.Architectures.Vision.ViT import ViT, ViT as VisionTransformer
from Blocks.Architectures.Vision.CoAtNet import CoAtNet, CoAtNet0, CoAtNet1, CoAtNet2, CoAtNet3, CoAtNet4

from Blocks.Architectures.Vision import DCGAN

from Blocks.Architectures.Perceiver import Perceiver

from Blocks.Architectures.RN import RN, RN as RelationNetwork

from Blocks.Architectures.Vision.CNN import AvgPool
from Blocks.Architectures.Vision.ViT import CLSPool

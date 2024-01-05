import re
from collections import OrderedDict
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import Tensor
import numpy as np

from modules.layers import AdaptiveAvgPool2d, AvgPool2d, BatchNorm2d, Cat, Conv2d, Dropout, Linear, MaxPool2d, ReLU, Sequential

__all__ = [
    "DenseNet",
    "densenet121"
]


class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
    
        self.cat = Cat()

        self.norm1 = BatchNorm2d(num_input_features)
        self.relu1 = ReLU(inplace=True)
        self.conv1 = Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = BatchNorm2d(bn_size * growth_rate)
        self.relu2 = ReLU(inplace=True)
        self.conv2 = Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        concated_features = self.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        # if self.drop_rate > 0:
        #     new_features = Dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

    def relprop(self, R, alpha):
        R = self.conv2.relprop(R, alpha)
        R = self.relu2.relprop(R, alpha)
        R = self.norm2.relprop(R, alpha)
        R = self.conv1.relprop(R, alpha)
        R = self.relu1.relprop(R, alpha)
        R = self.norm1.relprop(R, alpha)
        return self.cat.relprop(R, alpha) # a list of tensors

class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)
        
        self.cat = Cat()
        self.num_layers = num_layers

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features) # here is the concatenation of all previous features within the block

        # if you look at the diagram of a DenseNet block, it has one final concatenation of all previous layers
        return self.cat(features, 1)
    
    def relprop(self, R, alpha = 1):
        R = self.cat.relprop(R, alpha)

        # do it in reverse order
        # decompose the relevance value and for the gradient of the same color, add them up for fusion
            # do it yourself first then distribute the rest to its corresponding slots
        # at the end, add up the tensor gradients at each slot

        modules_reversed = reversed(self.items())
        # a list of list, index represents the layer index and the value in the index is a list of accumulating relevance tensors from previous connections
        module_Rs = []

        for i, module in enumerate(modules_reversed):

            # accumulate the relevance values for each layer
            for layer_index, rel in enumerate(R):
                if layer_index not in module_Rs:
                    module_Rs[layer_index] = []
                module_Rs[layer_index].append(rel)
            
            # update the R list for the next iteration
            layer = module[-1]
            R = layer.relprop(sum(module_Rs[-(i+1)]), alpha) # self.modules["denselayer%d" % (i + 1)].relprop(r, alpha)

        return R

class _Transition(Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = BatchNorm2d(num_input_features)
        self.relu = ReLU(inplace=True)
        self.conv = Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000, # TODO: change this depends on the dataset
        memory_efficient: bool = False,
    ) -> None:

        super().__init__()

        self.growth_rate = growth_rate
        self.block_config = block_config
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        assert(self.drop_rate == 0)
        self.num_classes = num_classes
        self.memory_efficient = memory_efficient

        # initial convolution before dense block convolution
        self.features = Sequential(
            OrderedDict(
                [
                    ("conv0", Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", BatchNorm2d(num_init_features)),
                    ("relu0", ReLU(inplace=True)),
                    ("pool0", MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
        self.relu0 = ReLU(inplace=True)
        self.maxpool0 = MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Each denseblock
        num_features = num_init_features

        # break down the above loop into individual layer
        denseblock1, num_features = self._make_layer(num_features, self.block_config[0], 0)
        transition1, num_features = self._make_transition(num_features, self.block_config[0], 0)
        self.features.add_module('denseblock1', denseblock1)
        self.features.add_module("transition1", transition1)

        denseblock2, num_features = self._make_layer(num_features, self.block_config[1], 1)
        transition2, num_features = self._make_transition(num_features, self.block_config[1], 1)
        self.features.add_module('denseblock2', denseblock2)
        self.features.add_module("transition2", transition2)

        denseblock3, num_features = self._make_layer(num_features, self.block_config[2], 2)
        transition3, num_features = self._make_transition(num_features, self.block_config[2], 2)
        self.features.add_module('denseblock3', denseblock3)
        self.features.add_module("transition3", transition3)

        denseblock4, num_features = self._make_layer(num_features, self.block_config[3], 3)
        self.features.add_module('denseblock4', denseblock4)


        # Final batch norm
        # self.norm5 = BatchNorm2d(num_features)
        self.features.add_module("norm5", BatchNorm2d(num_features))

        # Linear layer
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.classifier = Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, num_features, num_layers, i):
        block = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=self.bn_size,
            growth_rate=self.growth_rate,
            drop_rate=self.drop_rate,
            memory_efficient=self.memory_efficient,
        )
        # self.features.add_module("denseblock%d" % (i + 1), block)
        num_features = num_features + num_layers * self.growth_rate
        return block, num_features

    def _make_transition(self, num_features, num_layers, i):
        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        return trans, num_features

    def CLRP(self, x, maxindex = [None]):
        if maxindex == [None]:
            maxindex = torch.argmax(x, dim=1)
        R = torch.ones(x.shape).cuda()
        R /= -self.num_classes
        for i in range(R.size(0)):
            R[i, maxindex[i]] = 1
        return R

    def forward(self, x: Tensor, mode='output', target_class = [None], xMode=False):

        x = self.feature.conv0(x)
        x = self.feature.norm0(x)
        x = self.feature.relu0(x)
        x = self.relu0(x)
        features = self.maxpool0(x)

        # TODO: add transition module here
        denseblock1 = self.features.denseblock1(features)
        denseblock2 = self.features.denseblock2(denseblock1)
        denseblock3 = self.features.denseblock3(denseblock2)
        denseblock4 = self.features.denseblock4(denseblock3)
        
        # activate and downsample
        layer4Norm = self.features.norm5(denseblock4)
        out = ReLU(layer4Norm, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        z = self.classifier(out)

        if mode == 'output':
            # specific to the cheXnet model
            return z
    
        # propagation
        R = self.CLRP(z, target_class)

        R = self.classifier.relprop(R, 1)
        R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.relprop(R, 1)
        R = ReLU.relprop(R)
        R4 = self.features.norm5.relprop(R, 1)

        if mode == 'denseblock4':
            r_weight4 = self._compute_weights(R, denseblock4, xMode)
            r_cam4 = denseblock4 * r_weight4
            r_cam4 = torch.sum(r_cam4, dim=(1), keepdim=True)
            return r_cam4, z
        elif mode == 'denseblock3':
            R3 = self.denseblock4.relprop(R4, 1)
            r_weight3 = self._compute_weights(R3, denseblock3, xMode)
            r_cam3 = denseblock3 * r_weight3
            r_cam3 = torch.sum(r_cam3, dim=(1), keepdim=True)
            return r_cam3, z
        elif mode == 'denseblock2':
            R3 = self.denseblock4.relprop(R4, 1)
            R2 = self.denseblock3.relprop(R3, 1)
            r_weight2 = self._compute_weights(R2, denseblock2, xMode)
            r_cam2 = denseblock2 * r_weight2
            r_cam2 = torch.sum(r_cam2, dim=(1), keepdim=True)
            return r_cam2, z
        elif mode == 'denseblock1':
            R3 = self.denseblock4.relprop(R4, 1)
            R2 = self.denseblock3.relprop(R3, 1)
            R1 = self.denseblock2.relprop(R2, 1)
            r_weight1 = self._compute_weights(R1, denseblock1, xMode)
            r_cam1 = denseblock1 * r_weight1
            r_cam1 = torch.sum(r_cam1, dim=(1), keepdim=True)
            return r_cam1, z

        return out

    def _XRelevanceCAM(self, R, activations): #XRelevanceCAM
        """state of the art among the ones that I tried but visually it is bad
        this works!
        """
        try:
            R = R.cpu().detach().numpy() 
            activations = activations.cpu().detach().numpy()
        except:
            R = R.detach().numpy()
            activations = activations.detach().numpy()
        weights = R / (np.sum(activations, axis=(2, 3), keepdims=True) + 1e-7) # per channel division operation
        
        weights = np.sum(weights, axis=(2, 3), keepdims=True)
        return torch.tensor(weights, device=activations.device)

    def _compute_weights(self, R, activations, xMode):
        # xrelevance 
        if xMode:
            return self._XRelevanceCAM(R, activations)
        
        # relevance
        return torch.mean(R, dim=(2, 3), keepdim=True)

def densenet121(pretrained=False, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

    Args:
        weights (:class:`~torchvision.models.DenseNet121_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.DenseNet121_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.densenet.DenseNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.DenseNet121_Weights
        :members:
    """
    model = DenseNet(32, (6, 12, 24, 16), 64, **kwargs)
    return model
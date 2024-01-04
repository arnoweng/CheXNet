import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.model_zoo as model_zoo
from typing import Optional, Tuple, List, Callable, Any

from modules.layers import *

__all__ = ['GoogLeNet', 'googlenet', "GoogLeNetOutputs", "_GoogLeNetOutputs"]

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


def googlenet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> "GoogLeNet":
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['googlenet']))

        return model

    return GoogLeNet(**kwargs)


class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None
    ) -> None:
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(blocks) == 3


        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(0.2)
        self.fc = Linear(1024, num_classes)

        self.gradients = dict()
        self.activations = dict()

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        def backward_hook(module,input,output):
            self.gradients['value'] = output[0]

        self.inception3b.register_forward_hook(forward_hook)
        self.inception3b.register_backward_hook(backward_hook)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)



        # R = self.CLRP(x)
        #
        # logit = x[:, x.max(1)[-1]].sum()
        # logit.backward()

        # R = self.fc.relprop(R)
        # R = self.dropout.relprop(R)
        # R = R.reshape_as(self.avgpool.Y)
        # R = self.avgpool.relprop(R)
        # R = self.inception5b.relprop(R)
        # R = self.inception5a.relprop(R)
        # R = self.maxpool4.relprop(R)
        # R = self.inception4e.relprop(R)
        # R = self.inception4d.relprop(R)
        # R = self.inception4c.relprop(R)
        # R = self.inception4b.relprop(R)
        # R = self.inception4a.relprop(R)
        # R = self.maxpool3.relprop(R)
        # R = self.inception3b.relprop(R)
        # R = self.inception3a.relprop(R)
        #
        # r_weight = torch.mean(R,dim=(2,3),keepdim=True)
        # r_cam = t*r_weight
        # r_cam = torch.sum(r_cam,dim=(0,1))
        #
        # a = self.activations['value']
        # g = self.gradients['value']
        # g_ = torch.mean(g,dim=(2,3),keepdim=True)
        # grad_cam = a * g_
        # grad_cam = torch.sum(grad_cam,dim=(0,1))
        #
        # g_2 = g ** 2
        # g_3 = g ** 3
        # alpha_numer = g_2
        # alpha_denom = 2 * g_2 + torch.sum(a * g_3, dim=(0, 1), keepdim=True)  # + 1e-2
        #
        # alpha = alpha_numer / alpha_denom
        #
        # w = torch.sum(alpha * torch.clamp(g, min =0), dim=(0, 1), keepdim=True)
        #
        # grad_cam_pp = torch.clamp(w * a, min=0)
        # grad_cam_pp = torch.sum(grad_cam_pp, dim=-1)


        return x

    def CLRP(self,x):
        maxindex = torch.argmax(x)
        R = torch.ones(x.shape).cuda()
        R /= -1000
        R[:, maxindex] = 1

        return R

    def relprop(self,R):
        R = self.fc.relprop(R)
        R = self.dropout.relprop(R)
        R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.relprop(R)
        R = self.inception5b.relprop(R)
        R = self.inception5a.relprop(R)
        R = self.maxpool4.relprop(R)
        R = self.inception4e.relprop(R)
        R = self.inception4d.relprop(R)
        R = self.inception4c.relprop(R)
        R = self.inception4b.relprop(R)
        R = self.inception4a.relprop(R)
        # R = self.maxpool3.relprop(R)
        # R = self.inception3b.relprop(R)
        # R = self.inception3a.relprop(R)
        # R = self.maxpool2.relprop(R)
        # R = self.conv3.relprop(R)
        # R = self.conv2.relprop(R)
        # R = self.maxpool1.relprop(R)
        # R = self.conv1.relprop(R)

        return R


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x

class Inception(nn.Module):

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d


        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
        self.channel1 = ch1x1


        self.branch2 = Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.channel2 = ch3x3


        self.branch3 = Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )
        self.channel3 = ch5x5

        self.branch4 = Sequential(
            MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )
        self.channel4 = pool_proj

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)

        return torch.cat(outputs, 1)

    def relprop(self,R):
        R1 = R[:,:self.channel1]
        R2 = R[:, self.channel1:self.channel1+self.channel2]
        R3 = R[:, self.channel1+self.channel2:self.channel1+self.channel2+self.channel3]
        R4 = R[:, self.channel1+self.channel2+self.channel3:]


        R1 = self.branch1.relprop(R1)
        R2 = self.branch2.relprop(R2)
        R3 = self.branch3.relprop(R3)
        R4 = self.branch4.relprop(R4)

        R = R1+R2+R3+R4

        return R

class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

    def relprop(self,R):
        R = self.bn.relprop(R)
        R = self.conv.relprop(R)
        return R
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from modules.layers import *
import torch
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.clone = Clone()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=True)

        self.add = Add()

        self.register_forward_hook(forward_hook)

    def forward(self, x):
        x1, x2 = self.clone(x, 2)

        out = self.conv1(x1)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x2 = self.downsample(x2)

        out = self.add([out, x2])
        out = self.relu2(out)

        return out

    def relprop(self, R, alpha):
        out = self.relu2.relprop(R, alpha)
        out, x2 = self.add.relprop(out, alpha)

        if self.downsample is not None:
            x2 = self.downsample.relprop(x2, alpha)

        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.relu1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return self.clone.relprop([x1, x2], alpha)

    def m_relprop(self, R, pred, alpha):
        out = self.relu2.m_relprop(R, pred, alpha)
        out, x2 = self.add.m_relprop(out, pred, alpha)

        if self.downsample is not None:
            x2 = self.downsample.m_relprop(x2, pred, alpha)

        out = self.bn2.m_relprop(out, pred, alpha)
        out = self.conv2.m_relprop(out, pred, alpha)

        out = self.relu1.m_relprop(out, pred, alpha)
        out = self.bn1.m_relprop(out, pred, alpha)
        x1 = self.conv1.m_relprop(out, pred, alpha)

        return self.clone.m_relprop([x1, x2], pred, alpha)
    def RAP_relprop(self, R):
        out = self.relu2.RAP_relprop(R)
        out, x2 = self.add.RAP_relprop(out)

        if self.downsample is not None:
            x2 = self.downsample.RAP_relprop(x2)

        out = self.bn2.RAP_relprop(out)
        out = self.conv2.RAP_relprop(out)

        out = self.relu1.RAP_relprop(out)
        out = self.bn1.RAP_relprop(out)
        x1 = self.conv1.RAP_relprop(out)

        return self.clone.RAP_relprop([x1, x2])
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.clone = Clone()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=True)
        self.relu3 = ReLU(inplace=True)

        self.add = Add()

        self.register_forward_hook(forward_hook)

    def forward(self, x):
        # x1, x2 = self.clone(x, 2)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        # out = self.add([out, x2])
        out = self.add([out, x])
        out = self.relu3(out)

        return out

    def relprop(self, R, alpha):
        out = self.relu3.relprop(R, alpha)

        out, x = self.add.relprop(out, alpha)

        if self.downsample is not None:
            x = self.downsample.relprop(x, alpha)

        out = self.bn3.relprop(out, alpha)
        out = self.conv3.relprop(out, alpha)

        out = self.relu2.relprop(out, alpha)
        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.relu1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return x1 + x
        # return self.clone.relprop([x1, x2], alpha)
    def m_relprop(self, R, pred, alpha):
        out = self.relu3.m_relprop(R, pred, alpha)

        out, x = self.add.m_relprop(out, pred, alpha)

        if self.downsample is not None:
            x = self.downsample.m_relprop(x, pred, alpha)

        out = self.bn3.m_relprop(out, pred, alpha)
        out = self.conv3.m_relprop(out, pred, alpha)

        out = self.relu2.m_relprop(out, pred, alpha)
        out = self.bn2.m_relprop(out, pred, alpha)
        out = self.conv2.m_relprop(out, pred, alpha)

        out = self.relu1.m_relprop(out, pred, alpha)
        out = self.bn1.m_relprop(out, pred, alpha)
        x1 = self.conv1.m_relprop(out, pred, alpha)
        if torch.is_tensor(x1) == True:
            return x1 + x
        else:
            for i in range(len(x1)):
                x1[i] = x1[i] + x[i]
            return x1

    def RAP_relprop(self, R):
        out = self.relu3.RAP_relprop(R)

        out, x = self.add.RAP_relprop(out)

        if self.downsample is not None:
            x = self.downsample.RAP_relprop(x)

        out = self.bn3.RAP_relprop(out)
        out = self.conv3.RAP_relprop(out)

        out = self.relu2.RAP_relprop(out)
        out = self.bn2.RAP_relprop(out)
        out = self.conv2.RAP_relprop(out)

        out = self.relu1.RAP_relprop(out)
        out = self.bn1.RAP_relprop(out)
        x1 = self.conv1.RAP_relprop(out)

        return x1 + x

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, long= False, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes)
        self.long = long
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def CLRP(self, x, maxindex = [None]):
        if maxindex == [None]:
            maxindex = torch.argmax(x, dim=1)
        R = torch.ones(x.shape).cuda()
        R /= -self.num_classes
        for i in range(R.size(0)):
            R[i, maxindex[i]] = 1
        return R

    def forward(self, x, mode='output', target_class = [None]):
        
        # this is part of the architecture, where before the block, there is a general feature convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # start blockwise-convolution
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # average pooling it and flattening before linear classifier
        x = self.avgpool(layer4)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        if mode == 'output':
            return z

        # start the CLRP propagagation, start from the end to the start layer by layer (following inverse of above)
        R = self.CLRP(z, target_class)
        R = self.fc.relprop(R, 1)
        R = R.reshape_as(self.avgpool.Y)
        R4 = self.avgpool.relprop(R, 1)

        if mode == 'layer4':
            r_weight4 = torch.mean(R4, dim=(2, 3), keepdim=True)
            r_cam4 = layer4 * r_weight4
            r_cam4 = torch.sum(r_cam4, dim=(1), keepdim=True)
            return r_cam4, z

        elif mode == 'layer3':
            R3 = self.layer4.relprop(R4, 1)
            r_weight3 = torch.mean(R3, dim=(2, 3), keepdim=True)
            r_cam3 = layer3 * r_weight3
            r_cam3 = torch.sum(r_cam3, dim=(1), keepdim=True)
            return r_cam3, z
        elif mode == 'layer2':
            R3 = self.layer4.relprop(R4, 1)
            R2 = self.layer3.relprop(R3, 1)
            r_weight2 = torch.mean(R2, dim=(2, 3), keepdim=True)
            r_cam2 = layer2 * r_weight2
            r_cam2 = torch.sum(r_cam2, dim=(1), keepdim=True)
            return r_cam2, z
        else:
            return z

    def relprop(self, R, alpha, flag = 'inter'):
        # this is the function that perform clrp propagation all the way to the first convolution, might not needed
        if self.long:
            R = self.fc.relprop(R, alpha)
            R = R.reshape_as(self.avgpool.Y)
            R = self.avgpool.relprop(R, alpha)
            R = self.layer4.relprop(R, alpha)
            R = self.layer3.relprop(R, alpha)
            R = self.layer2.relprop(R, alpha)
            R = self.layer1.relprop(R, alpha)
            R = self.maxpool.relprop(R, alpha)
            R = self.relu.relprop(R, alpha)
            R = self.bn1.relprop(R, alpha)
            R = self.conv1.relprop(R, alpha)
        else:
            R = self.fc.relprop(R, alpha)
            R = R.reshape_as(self.avgpool.Y)
            R = self.avgpool.relprop(R, alpha)
            if flag == 'layer4': return R
            R = self.layer4.relprop(R, alpha)
            if flag == 'layer3': return R
            R = self.layer3.relprop(R, alpha)
            if flag == 'layer2': return R
            R = self.layer2.relprop(R, alpha)
            if flag == 'layer1': return R

        return R

    def m_relprop(self, R, pred, alpha):
        R = self.fc.m_relprop(R, pred, alpha)
        if torch.is_tensor(R) == False:
            for i in range(len(R)):
                R[i] = R[i].reshape_as(self.avgpool.Y)
        else:
            R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.m_relprop(R, pred, alpha)

        R = self.layer4.m_relprop(R, pred, alpha)
        R = self.layer3.m_relprop(R, pred, alpha)
        R = self.layer2.m_relprop(R, pred, alpha)
        R = self.layer1.m_relprop(R, pred, alpha)

        R = self.maxpool.m_relprop(R, pred, alpha)
        R = self.relu.m_relprop(R, pred, alpha)
        R = self.bn1.m_relprop(R, pred, alpha)
        R = self.conv1.m_relprop(R, pred, alpha)

        return R

    def RAP_relprop(self, R):
        R = self.fc.RAP_relprop(R)
        R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.RAP_relprop(R)

        R = self.layer4.RAP_relprop(R)
        R = self.layer3.RAP_relprop(R)
        R = self.layer2.RAP_relprop(R)
        R = self.layer1.RAP_relprop(R)

        R = self.maxpool.RAP_relprop(R)
        R = self.relu.RAP_relprop(R)
        R = self.bn1.RAP_relprop(R)
        R = self.conv1.RAP_relprop(R)

        return R

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, long = False,**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], long = long,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



# def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""ResNeXt-50 32x4d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)
#
#
# def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""ResNeXt-101 32x8d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)
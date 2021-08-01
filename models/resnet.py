import torch
import torch.nn as nn
import math
import numpy as np
import pdb
import torch.utils.model_zoo as model_zoo

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
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, low_dim=128, in_channel=3, width=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(self.base * 8 * block.expansion, low_dim)
        self.l2norm = Normalize(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, layer=7):
        if layer <= 0:
            return x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if layer == 1:
            return x
        x = self.layer1(x)
        if layer == 2:
            return x
        x = self.layer2(x)
        if layer == 3:
            return x
        x = self.layer3(x)
        if layer == 4:
            return x
        x = self.layer4(x)
        if layer == 5:
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if layer == 6:
            return x
        x = self.fc(x)
        x = self.l2norm(x)

        return x


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


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
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


class InsResNet50(nn.Module):
    """Encoder for instance discrimination and MoCo"""
    def __init__(self, width=1):
        super(InsResNet50, self).__init__()
        self.encoder = resnet50(width=width)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=7):
        return self.encoder(x, layer)

class MaskBlock(nn.Module):
    def __init__(self, fc_dims, in_dim=2048):
        super(MaskBlock, self).__init__()
        
        self.fcs = nn.ModuleList([]) # Need to do this so fcs are moved to GPU on .to(device)
        fc_dims.insert(0, in_dim)
        for fc_dim_i in range(len(fc_dims) - 1):
            self.fcs.append(nn.Linear(fc_dims[fc_dim_i], fc_dims[fc_dim_i + 1]))
        self.last_fc = nn.Linear(fc_dims[-1], 1)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        out = x.permute(0, 2, 3, 1)
        
        for layer in self.fcs:
            out = layer(out)    
            out = self.relu(out)

        out = self.last_fc(out)
        out = self.sigm(out)

        return out

class MaskBlockConv(nn.Module):
    def __init__(self, filter_dims, use_bn=True, in_dim=2048):
        super(MaskBlockConv, self).__init__()
        
        self.fcs = nn.ModuleList([]) # Need to do this so fcs are moved to GPU on .to(device)
        self.bns = nn.ModuleList([])
        filter_dims.insert(0, in_dim)
        for i in range(len(filter_dims) - 1):
            self.fcs.append(nn.Conv2d(filter_dims[i], filter_dims[i + 1], kernel_size=3, stride=1, padding=1))
            if use_bn:
                self.bns.append(nn.BatchNorm2d(filter_dims[i + 1]))

        self.last_fc = nn.Conv2d(filter_dims[-1], 1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

        self.use_bn = use_bn

    def forward(self, x):
        out = x
        
        if self.use_bn:
            for layer, bnl in zip(self.fcs, self.bns):
                out = layer(out)
                out = bnl(out)
                out = self.relu(out)
        else:
            for layer in self.fcs:
                out = layer(out)    
                out = self.relu(out)

        out = self.last_fc(out)
        out = self.sigm(out)

        out = out.permute(0, 2, 3, 1)

        return out

class MaskedEncoder(nn.Module):
    def __init__(self, encoder, mask_block):
        super(MaskedEncoder, self).__init__()
        self.encoder = encoder
        self.mask = mask_block
        # self.sm = nn.Softmax(dim=1)
    
    def forward(self, x, aux, layer):
        if layer <= 5:
            return self.encoder(x, layer)

        out = self.encoder(x, layer=5) #N x 7 x 7 x 2048
        mask = self.mask(out) #N x 7 x 7 x 1
    
        mask = mask.view(mask.size(0), -1, mask.size(3)) # N x 49 x 1
        # mask = self.sm(mask)
        # pdb.set_trace()
        out = out.view(out.size(0), out.size(1), -1) # N x 2048 x 49

        masked_out = torch.matmul(out, mask).squeeze(-1) # N x 2048
        
        idx, epoch = aux
        if idx == 0: #and (epoch > 8 or epoch == 4 or epoch == 1):
            print(mask[0])
            # pdb.set_trace()
            # see if masked_out is too small
        
        # what happens if this?
        mask_sum = torch.sum(mask, dim=1)
        masked_out /= torch.max(mask_sum, torch.ones_like(mask_sum) * 0.001)

        # masked out if weighted average pool
        if layer == 6:
            return masked_out, mask
        
        masked_out = self.encoder.fc(masked_out)
        masked_out = self.encoder.l2norm(masked_out)

        return masked_out


class MaskInsResNet50(nn.Module):
    """Encoder for mask instance discrimination and MoCo"""
    def __init__(self, width=1):
        super(MaskInsResNet50, self).__init__()
        self.resnet = resnet50(width=width)
        self.mask_block = MaskBlock([1000, 250], 2048)
        
        self.masked_encoder = MaskedEncoder(self.resnet, self.mask_block)
        self.encoder = nn.DataParallel(self.masked_encoder)
        
    def forward(self, x, layer=7):
        return self.encoder(x, layer)


class MaskConvInsResNet50(nn.Module):
    """Encoder for mask instance discrimination and MoCo"""
    def __init__(self, width=1):
        super(MaskConvInsResNet50, self).__init__()
        self.resnet = resnet50(width=width)
        self.mask_block = MaskBlockConv([1000, 250], True, 2048)
        
        self.masked_encoder = MaskedEncoder(self.resnet, self.mask_block)
        self.encoder = nn.DataParallel(self.masked_encoder)
        
    def forward(self, x, aux, layer=7):
        return self.encoder(x, aux, layer)


class ResNetV1(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetV1, self).__init__()
        if name == 'resnet50':
            self.l_to_ab = resnet50(in_channel=1, width=0.5)
            self.ab_to_l = resnet50(in_channel=2, width=0.5)
        elif name == 'resnet18':
            self.l_to_ab = resnet18(in_channel=1, width=0.5)
            self.ab_to_l = resnet18(in_channel=2, width=0.5)
        elif name == 'resnet101':
            self.l_to_ab = resnet101(in_channel=1, width=0.5)
            self.ab_to_l = resnet101(in_channel=2, width=0.5)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab


class ResNetV2(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetV2, self).__init__()
        if name == 'resnet50':
            self.l_to_ab = resnet50(in_channel=1, width=1)
            self.ab_to_l = resnet50(in_channel=2, width=1)
        elif name == 'resnet18':
            self.l_to_ab = resnet18(in_channel=1, width=1)
            self.ab_to_l = resnet18(in_channel=2, width=1)
        elif name == 'resnet101':
            self.l_to_ab = resnet101(in_channel=1, width=1)
            self.ab_to_l = resnet101(in_channel=2, width=1)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab


class ResNetV3(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetV3, self).__init__()
        if name == 'resnet50':
            self.l_to_ab = resnet50(in_channel=1, width=2)
            self.ab_to_l = resnet50(in_channel=2, width=2)
        elif name == 'resnet18':
            self.l_to_ab = resnet18(in_channel=1, width=2)
            self.ab_to_l = resnet18(in_channel=2, width=2)
        elif name == 'resnet101':
            self.l_to_ab = resnet101(in_channel=1, width=2)
            self.ab_to_l = resnet101(in_channel=2, width=2)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab


class MyResNetsCMC(nn.Module):
    def __init__(self, name='resnet50v1'):
        super(MyResNetsCMC, self).__init__()
        if name.endswith('v1'):
            self.encoder = ResNetV1(name[:-2])
        elif name.endswith('v2'):
            self.encoder = ResNetV2(name[:-2])
        elif name.endswith('v3'):
            self.encoder = ResNetV3(name[:-2])
        else:
            raise NotImplementedError('model not support: {}'.format(name))

        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=7):
        return self.encoder(x, layer)

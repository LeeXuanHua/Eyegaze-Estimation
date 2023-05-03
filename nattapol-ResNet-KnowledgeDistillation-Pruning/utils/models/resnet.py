import torch
import torch.nn as nn

# Wide Residual Network
class ConvBlock(nn.Module):
    def __init__(self, input_channel, last = False):
        super(ConvBlock, self).__init__()
        self.Conv_1 = nn.Conv2d(in_channels=input_channel, 
                                out_channels=2*input_channel, 
                                kernel_size=(3,3), 
                                padding='same')
        self.Conv_2 = nn.Conv2d(in_channels=2*input_channel, 
                                out_channels=2*input_channel, 
                                kernel_size=(3,3), 
                                padding='same')
        self.Conv_1l = nn.Conv2d(in_channels=input_channel, 
                                 out_channels=input_channel, 
                                 kernel_size=(3,3), 
                                 padding='same')
        self.Conv_2l = nn.Conv2d(in_channels=input_channel, 
                                 out_channels=input_channel, 
                                 kernel_size=(3,3), 
                                 padding='same')
        self.Conv_skip_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, 
                      out_channels=2*input_channel, 
                      kernel_size=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*input_channel)
        )
        self.Conv_skip_1l = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, 
                      out_channels=input_channel, 
                      kernel_size=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel)
        )
        self.features = nn.Sequential(
            self.Conv_1,
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*input_channel),
            self.Conv_2,
            nn.BatchNorm2d(2*input_channel)
        )
        if last:
            self.features = nn.Sequential(
                self.Conv_1l,
                nn.LeakyReLU(),
                nn.BatchNorm2d(input_channel),
                self.Conv_2l,
                nn.BatchNorm2d(input_channel)
            )
        self.last = last
        self.input_channel = input_channel
        
    def forward(self, input):
        if not self.last:
            input_skip = self.Conv_skip_1(input)
        else:
            input_skip = self.Conv_skip_1l(input)
        input = self.features(input)
        input = input_skip + input
        return nn.ReLU()(input)

class DeConvBlock(nn.Module):
    def __init__(self, input_channel, last = False):
        super(DeConvBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, 
                      out_channels=input_channel//2, 
                      kernel_size=(3,3), 
                      padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel//2),
            nn.Conv2d(in_channels=input_channel//2, 
                      out_channels=input_channel//2, 
                      kernel_size=(3,3), 
                      padding='same'),
            nn.BatchNorm2d(input_channel//2)
        )
        if last: 
            self.features = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, 
                          out_channels=input_channel//4, 
                          kernel_size=(3,3), 
                          padding='same'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(input_channel//4),
                nn.Conv2d(in_channels=input_channel//4, 
                          out_channels=input_channel//4, 
                          kernel_size=(3,3), 
                          padding='same'),
                nn.BatchNorm2d(input_channel//4)
            )
        self.last = last
        self.input_channel = input_channel

    def forward(self, input):
        if not self.last:
            input_skip = nn.Conv2d(in_channels=self.input_channel, 
                                   out_channels=self.input_channel//2, 
                                   kernel_size=(1,1))(input)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = nn.BatchNorm2d(self.input_channel//2)(input_skip)
        else:
            input_skip = nn.Conv2d(in_channels=self.input_channel, 
                                   out_channels=self.input_channel//4, 
                                   kernel_size=(1,1))(input)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = nn.BatchNorm2d(self.input_channel//4)(input_skip)
        input = self.features(input)
        input = input_skip + input
        return nn.ReLU()(input)

class resnet10(nn.Module):
    def __init__(self, channels=64, out_channels=2, input_size=(224,224)) -> None:
        super(resnet10, self).__init__()
        self.ConvBlock_1 = ConvBlock(input_channel=channels, last=True)
        self.ConvBlock_2 = ConvBlock(input_channel=channels)
        self.ConvBlock_3 = ConvBlock(input_channel=2*channels, last=True)
        self.ConvBlock_4 = ConvBlock(input_channel=2*channels)
        self.features_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_1,
        )
        self.features_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_2,
        )
        self.features_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_3,
        )
        self.features_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_4,
        )
        self.decoder = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2,3), stride=(2,3)), # 1/3 1/4 for EyeDiap
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=112*channels, out_features=channels//4),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=channels//4, out_features=out_channels),
        )
        self.channels = channels
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.uniform_(0.0, 1e-3)
            m.bias.data.fill_(0)
            
    def forward(self, input):
        input = self.features_1(input)
        feature_1 = input
        input = self.features_2(input)
        feature_2 = input
        input = self.features_3(input)
        feature_3 = input
        input = self.features_4(input)
        feature_4 = input
        input = self.decoder(input)
        return input , feature_1, feature_2, feature_3, feature_4
    


# Resnet from official pytorch 
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, x1, x2, x3, x4

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2],
                   **kwargs)
    
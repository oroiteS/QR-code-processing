import os

import torch
import torch.nn as nn

from models.mobilenet_v2 import MobileNetV2

'''
这段代码主要实现了一个基于 MobileNetV2 和特征金字塔网络（FPN）的神经网络模型，用于图像相关任务。
整体结构包括 MobileNetV2 作为特征提取器，FPN 模块用于生成多尺度特征图，以及多个分割头部和后续的处理层来生成最终的输出。
'''


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class FPNHead(nn.Module):
    # 一个用于生成多尺度特征图的模块，包含两个卷积层和 ReLU 激活函数
    # 输入是 num_in 个通道的特征图，输出是 num_out 个通道的特征图，num_mid是中间的通道数
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # 在第一个block后面添加激活函数，下同理
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=128, pretrained=True):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super(FPN, self).__init__()
        net = MobileNetV2(n_class=1000)

        if pretrained:
            # Load weights into the project directory
            # add map_location='cpu' if no gpu
            state_dict = torch.load(os.path.join(get_project_root(), 'DeblurGANv2', 'weights', 'mobilenet_v2.pth.tar'))
            net.load_state_dict(state_dict)
        self.features = net.features

        self.enc0 = nn.Sequential(*self.features[0:2])
        self.enc1 = nn.Sequential(*self.features[2:4])
        self.enc2 = nn.Sequential(*self.features[4:7])
        self.enc3 = nn.Sequential(*self.features[7:11])
        self.enc4 = nn.Sequential(*self.features[11:16])

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))

        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(32, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(24, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(16, num_filters // 2, kernel_size=1, bias=False)

        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0)  # 256

        enc2 = self.enc2(enc1)  # 512

        enc3 = self.enc3(enc2)  # 1024

        enc4 = self.enc4(enc3)  # 2048

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.interpolate(map4, scale_factor=2, mode="nearest"))
        map2 = self.td2(lateral2 + nn.functional.interpolate(map3, scale_factor=2, mode="nearest"))
        map1 = self.td3(lateral1 + nn.functional.interpolate(map2, scale_factor=2, mode="nearest"))
        return lateral0, map1, map2, map3, map4


class FPNMobileNet(nn.Module):
    # 一个完整的模型，包括 FPN 和多个分割头部，用于图像去模糊任务
    #  初始化函数，接收归一化层 norm_layer、输出通道数 output_ch、
    #  过滤器数量 num_filters 和 num_filters_fpn 以及是否预训练 pretrained
    def __init__(self, norm_layer, output_ch=3, num_filters=64, num_filters_fpn=128, pretrained=True):
        super().__init__()
        '''
        Feature Pyramid Network (FPN) with four feature maps of resolutions
        1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        '''
        # 定义一个FPN类的实例，用于提取特征并赋给自己的fpn属性
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer, pretrained=pretrained)
        # 创建四个分割头部，分别对应不同尺度的特征图传入输入、中间和输出通道数
        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)
        # 定义一个平滑层序列，用于处理拼接后的特征图，包括卷积、归一化和激活函数
        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
        )
        # 定义另一个平滑层序列，进一步处理特征图，减少通道数并应用激活函数
        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        # 定义最终的卷积层，将处理后的特征图转换为输出通道数
        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        # 使用 FPN 提取不同尺度的特征图
        map0, map1, map2, map3, map4 = self.fpn(x)
        # 对每个分割头部的输出进行上采样,scale_factor代表上采样的倍数，mode代表插值方法，这里使用最近邻插值
        map4 = nn.functional.interpolate(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.interpolate(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.interpolate(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.interpolate(self.head1(map1), scale_factor=1, mode="nearest")
        # 将上采样后的特征图拼接起来，并通过平滑层进行处理
        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.interpolate(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.interpolate(smoothed, scale_factor=2, mode="nearest")
        # 通过最终的卷积层得到输出
        final = self.final(smoothed)
        # 将输出与输入图像相加，并通过 tanh 激活函数进行处理，最后裁剪到 [-1, 1] 范围内返回
        res = torch.tanh(final) + x

        return torch.clamp(res, min=-1, max=1)

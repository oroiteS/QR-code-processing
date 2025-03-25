import functools  # 提供对函数进行操作的工具

import numpy as np
import torch
import torch.nn as nn

from models.fpn_densenet import FPNDense
from models.fpn_inception import FPNInception
from models.fpn_inception_simple import FPNInceptionSimple
from models.fpn_mobilenet import FPNMobileNet
from models.unet_seresnext import UNetSEResNext


###############################################################################
# Functions
###############################################################################


# 此方法用于获取指定类型的归一化层。
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':  # 批量归一化
        '''
        nn.BatchNorm2d是一个用于对输入数据进行批量归一化的层
        使用affine参数设置是否使用学习的缩放和平移参数
        设置 affine=True，这意味着批量归一化层将学会自己的缩放和偏移参数，便于后续的训练
        '''
        # functools.partial的作用是用于修改现有函数的某些参数或行为，而不改变其原始定义
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':  # 实例归一化
        '''
        nn.InstanceNorm2d是一个用于对输入数据进行实例归一化的层
        '''
        # 设置affine=false代表这个实例归一化层将对每个实例的特征进行归一化
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few downsampling/upsampling operations.
# 定义了一个生成器，它在一些下采样 / 上采样操作之间由残差网络（Resnet）模块组成。
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    ##############################################################################
    # 定义了一个生成器，它继承自 PyTorch 的 nn.Module 类
    # 生成器由一系列卷积、归一化和激活函数组成，用于图像去模糊任务
    ##############################################################################
    """
    input_nc 是输入图像的通道数，即输入图像的特征数
    output_nc 是输出图像的通道数，即输出图像的特征数
    ngf是生成器中特征图的初始通道数
    norm_layer是归一化层的类型，默认为nn.BatchNorm2d
    use_dropout是一个布尔值，用于指示是否在生成器中使用dropout层
    n_blocks是生成器中Resnet块的数量
    use_parallel是一个布尔值，用于指示是否使用并行处理
    learn_residual是一个布尔值，用于指示是否学习残差0
    padding_type是填充类型，默认为'reflect'
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 use_parallel=True, learn_residual=True, padding_type='reflect'):
        # assert是一个断言语句，用于检查条件是否为真。如果条件不为真，则会引发AssertionError异常并显示指定的错误消息。
        assert (n_blocks >= 0)
        # super是指是用于调用父类（超类）的方法，也就是调用nn.Module的构造函数
        super(ResnetGenerator, self).__init__()
        # 将各种输入的参数保存为类的参数
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        # 比较normal_layer的类型是否是 functools.partial 类型的
        # 如果是，则根据 norm_layer 的函数类型确定是否使用偏置（use_bias）
        # 否则直接比较它是否是 nn.InstanceNorm2d
        # 在卷积运算中，对于输入数据的每个位置，卷积核与对应位置的数据进行乘法和求和操作
        # 偏置项则是在此基础上加上的一个固定值，它对卷积结果进行了一个额外的调整
        '''
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        modified on 2024.9.25
        '''
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # 构建模型结构
        # nn.ReflectionPad2d(3)是构造一个填充层，用于在输入图像周围填充一定数量的像素
        # nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)是一个二维卷积层，用于对输入图像进行特征提取
        # norm_layer默认是使用 nn.BatchNorm2d 进行归一化
        # nn.relu是激活函数
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]
        '''
        对应于具体的论文当中的一个生成器的结构，在论文当中的生成器的结构是这样的：
        输入层 -> 下采样层 -> 残差块 -> 上采样层 -> 输出层
        其中，下采样层和上采样层都是通过卷积层和激活函数实现的，而残差块则是通过一系列卷积层和激活函数实现的
        '''
        # 进行下采样
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # 添加残差层
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        # 进行上采样
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        # 再添加一个填充层，用于将上采样后的特征图与原始输入特征图进行融合
        model += [nn.ReflectionPad2d(3)]
        # 然后再添加一个卷积层，用于将上采样后的特征图转换为与原始输入特征图相同的通道数
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # nn.tanh是一个激活函数，用于将输出值映射到[-1, 1]之间
        model += [nn.Tanh()]

        # nn.sequential是一个容器，用于将多个模块按照顺序组合在一起
        # 输入的参数*model是指将列表当中的所有元素都添加到model当中
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
        前向传播函数
        这个函数的作用是根据输入进行输出
        其中self.learn_residual是指示是否学习残差
        """
        output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        # padding_type指定的是填充类型，负责填充图像的边缘，以保留图像的边缘信息
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        # 添加一个二维卷积层，输入输出通道数都是dim，卷积核大小为3，并根据前面的if来确定填充方式，并根据use_bias来确定是否使用偏置
        # 同时根据输入的norm_layer和dim来确定是否使用归一化层以及使用什么类型
        # 添加一个ReLU激活函数
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            # 如果使用dropout，则添加一个Dropout层，概率为0.5
            # 这个可以随机的将一些神经元的输出设置为0，以减少过拟合的风险
            conv_block += [nn.Dropout(0.5)]

        # 再次添加一个类似的填充和卷积层，但是不使用激活函数
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class DicsriminatorTail(nn.Module):
    def __init__(self, nf_mult, n_layers, ndf=64, norm_layer=nn.BatchNorm2d, use_parallel=True):
        super(DicsriminatorTail, self).__init__()
        self.use_parallel = use_parallel
        '''
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        '''
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
            use_bias = isinstance(norm_layer.func, nn.InstanceNorm2d)
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d, use_parallel=True):
        super(MultiScaleDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        self.scale_one = nn.Sequential(*sequence)
        self.first_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=3)
        nf_mult_prev = 4
        nf_mult = 8

        self.scale_two = nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        nf_mult_prev = nf_mult
        self.second_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=4)
        self.scale_three = nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        self.third_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=5)

    def forward(self, input):
        x = self.scale_one(input)
        x_1 = self.first_tail(x)
        x = self.scale_two(x)
        x_2 = self.second_tail(x)
        x = self.scale_three(x)
        x = self.third_tail(x)
        return [x_1, x_2, x]


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def get_fullD(model_config):
    model_d = NLayerDiscriminator(n_layers=5,
                                  norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                  use_sigmoid=False)
    return model_d


def get_generator(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'resnet':
        model_g = ResnetGenerator(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                  use_dropout=model_config['dropout'],
                                  n_blocks=model_config['blocks'],
                                  learn_residual=model_config['learn_residual'])
    elif generator_name == 'fpn_mobilenet':
        model_g = FPNMobileNet(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
    elif generator_name == 'fpn_inception':
        model_g = FPNInception(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
    elif generator_name == 'fpn_inception_simple':
        model_g = FPNInceptionSimple(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
    elif generator_name == 'fpn_dense':
        model_g = FPNDense()
    elif generator_name == 'best_fpn':
        model_g = FPNInception(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
    # elif generator_name == 'unet_seresnext':
    # model_g = UNetSEResNext(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
    #                        pretrained=model_config['pretrained'])
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    # nn.DAtaParallel是一个用于并行计算的包装器，它可以将一个模型复制到多个GPU上，并在每个GPU上运行模型的一部分
    return nn.DataParallel(model_g)


def get_discriminator(model_config):
    discriminator_name = model_config['d_name']
    if discriminator_name == 'no_gan':
        model_d = None
    elif discriminator_name == 'patch_gan':
        model_d = NLayerDiscriminator(n_layers=model_config['d_layers'],
                                      norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                      use_sigmoid=False)
        model_d = nn.DataParallel(model_d)
    elif discriminator_name == 'double_gan':
        patch_gan = NLayerDiscriminator(n_layers=model_config['d_layers'],
                                        norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                        use_sigmoid=False)
        patch_gan = nn.DataParallel(patch_gan)
        full_gan = get_fullD(model_config)
        full_gan = nn.DataParallel(full_gan)
        model_d = {'patch': patch_gan,
                   'full': full_gan}
    elif discriminator_name == 'multi_scale':
        model_d = MultiScaleDiscriminator(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
        model_d = nn.DataParallel(model_d)
    else:
        raise ValueError("Discriminator Network [%s] not recognized." % discriminator_name)

    return model_d


def get_nets(model_config):
    return get_generator(model_config), get_discriminator(model_config)

import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator
import pkg_resources


def get_package_path(relative_path):
    """获取包内资源的绝对路径"""
    try:
        # 首先尝试作为已安装的包来获取路径
        return pkg_resources.resource_filename('deblurgan', relative_path)
    except Exception:
        # 如果失败，则使用相对于当前文件的路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, relative_path)


class Predictor:
    # predictor类的构造函数
    # 初始化对象，它会读取配置文件，加载指定模型，并将其设置为训练模式
    def __init__(self, weights_path: str, model_name: str = ''):  # 自身，模型权重路径，模型名称
        # 打开config.yaml文件
        # 使用 get_package_path 获取配置文件路径
        config_path = get_package_path('config/config.yaml')
        with open(config_path, encoding='utf-8') as cfg:
            # 参数Loader=yaml.FullLoader确保加载的是完整的yaml文
            # 如果不设置参数会导致出现错误
            config = yaml.load(cfg, Loader=yaml.FullLoader)

        # 获取一个生成器模型，model_name或者是配置文件当中的model（即配置文件当中指定的模型名称）
        model = get_generator(model_name or config['model'])
        # 使用load方法加载weights_path指定的模型权重
        # torch.load会返回一个包含模型权重的字典，通过索引model获取权重模型
        model.load_state_dict(torch.load(weights_path)['model'])
        # 加载模型到cuda上，且设置自己的参数model为获得的生成器的模型
        self.model = model.cuda()
        # 将模型设置为训练模式。在GAN中，模型在训练模式下可以正确使用归一化层中的统计数据
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        # 获取一个归一化函数，用于数据预处理，确保输入数据的分布适合模型训练或推理
        self.normalize_fn = get_normalize()

    @staticmethod  # 静态方法说明，不需要self变量
    def _array_to_batch(x):  # 这是一个辅助函数，用于将图像数组转换为适合模型输入的格式
        # x代表图像的numpy数组
        # 转置数组，符合模型需要 按顺序调换 从 0 1 2变成 2 0 1
        x = np.transpose(x, (2, 0, 1))
        # 在最左边0增加数组的维度
        x = np.expand_dims(x, 0)
        # 将模型转化成torch所需要的张量并返回
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        # 这个函数对输入的图像和掩码进行预处理，包括归一化、填充以确保图像尺寸符合模型要求
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        # 这个函数对模型的输出进行后处理，将模型输出的张量转换回图像格式
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        # 这是Predictor类的调用函数，它接受一个图像和可选的掩码，然后使用模型进行预测，返回处理后的图像
        # 先对图像和掩码进行预处理
        (img, mask), h, w = self._preprocess(img, mask)
        # torch.no_grad是一个上下文管理器，用于禁用梯度计算。
        # 在推理阶段，我们不需要计算梯度，因此使用torch.no_grad()可以减少内存消耗和计算时间
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                # 如果不忽视mask，则将mask添加到inputs列表中
                inputs += [mask]
            # 调用模型进行预测
            pred = self.model(*inputs)
            # 返回处理后的图像
        return self._postprocess(pred)[:h, :w, :]


def process_video(pairs, predictor, output_dir):
    # 这个函数用于处理视频文件。它读取视频的每一帧，使用Predictor对象进行去模糊处理，然后将处理后的帧写入新的视频文件
    for video_filepath, mask in tqdm(pairs):
        video_filename = os.path.basename(video_filepath)
        output_filepath = os.path.join(output_dir, os.path.splitext(video_filename)[0] + '_deblur.mp4')
        video_in = cv2.VideoCapture(video_filepath)
        fps = video_in.get(cv2.CAP_PROP_FPS)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        # video_out = cv2.VideoWriter(output_filepath.text, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        video_out = cv2.VideoWriter(output_filepath.text, cv2.VideoWriter.fourcc(*'MP4V'), fps, (width, height))
        # 2024-10-27
        tqdm.write(f'process {video_filepath} to {output_filepath}, {fps}fps, resolution: {width}x{height}')
        for frame_num in tqdm(range(total_frame_num), desc=video_filename):
            res, img = video_in.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predictor(img, mask)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            video_out.write(pred)


def main(img_pattern: str,
         mask_pattern: Optional[str] = None,
         weights_path=None,  # 修改默认权重路径
         out_dir='submit/',
         side_by_side: bool = False,
         video: bool = False):
    # 如果未指定权重文件路径，使用默认路径
    if weights_path is None:
        weights_path = get_package_path('weights/fpn_mobilenet.h5')

    # 接受图像路径、权重文件路径等参数，创建Predictor对象，并根据提供的参数处理图像或视频
    def sorted_glob(pattern):
        return sorted(glob(pattern))  # 返回一个已排序的文件名列表，该列表包含与给定模式匹配的所有文件路径
        # glob是一个用于查找文件路径的函数，它接受一个模式作为参数，并返回与该模式匹配的所有文件路径，例如glob('*.txt')

    imgs = sorted_glob(img_pattern)
    # 后面的if语句用于处理可选的mask_pattern参数。如果mask_pattern不为None，则使用sorted_glob函数获取与该模式匹配的文件名列表，否则生成一个全为None的列表
    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    pairs = zip(imgs, masks)  # 用zip函数将imgs和masks列表中的元素一一对应，形成一个元组列表，方便后续处理
    names = sorted([os.path.basename(x) for x in glob(img_pattern)])
    # 根据传入的weights_path来获得一个Predictor对象
    predictor = Predictor(weights_path=weights_path)
    # os.makedirs的作用是创建一个名为out_dir的目录，且如果该目录已经存在，则不会抛出异常
    os.makedirs(out_dir, exist_ok=True)
    if not video:
        # tqdm是一个Python库，用于在循环中显示进度条
        for name, pair in tqdm(zip(names, pairs), total=len(names)):
            f_img, f_mask = pair
            # map是一个内置函数，它接受一个函数和一个可迭代对象作为参数，然后将函数应用于可迭代对象的每个元素，并返回一个迭代器，该迭代器包含应用函数后的结果
            # 在此处map的作用主要是将cv2.imread函数应用于f_img和f_mask，然后将结果转换为RGB格式
            img, mask = map(cv2.imread, (f_img, f_mask))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 返回一个经过去模糊处理的图像pred
            pred = predictor(img, mask)
            if side_by_side:
                # 如果side_by_side为True，则将原始图像和去模糊后的图像水平拼接起来，形成一个新的图像pred
                pred = np.hstack((img, pred))
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, name), pred)
    else:
        process_video(pairs, predictor, out_dir)


# def getfiles():
#     filenames = os.listdir(r'.\dataset1\blur')
#     print(filenames)
def get_files():
    # 用于批量处理信息
    name_list = []
    for filepath, dirnames, filenames in os.walk(r'.\dataset1\blur'):
        for filename in filenames:
            name_list.append(os.path.join(filepath, filename))
    return name_list


if __name__ == '__main__':
    #  Fire(main)
    # 增加批量处理图片：
    # img_path = get_files()
    # for i in img_path:
    #   main(i)
    path = './test_img/49.jpg'
    main(path)

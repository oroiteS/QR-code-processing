import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as SSIM
from util.metrics import PSNR


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)

        # 确保图像至少是 7x7 像素
        min_dim = min(fake.shape[:2])  # 取图像的高度和宽度中的最小值
        win_size = min(11, min_dim)  # 设置 win_size 为 11 或图像的较小边，以较小者为准

        # 确保 win_size 是奇数
        if win_size % 2 == 0:
            win_size -= 1  # 如果 win_size 是偶数，减 1 使其变为奇数

        psnr = PSNR(fake, real)
        ssim = SSIM(fake, real, multichannel=True, win_size=win_size, channel_axis=2)
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img


def get_model(model_config):
    return DeblurModel()

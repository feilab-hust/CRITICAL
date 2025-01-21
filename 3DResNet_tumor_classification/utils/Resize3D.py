import torch.nn.functional as F


# 自定义Transform类，支持 (D, H, W) 和 (C, D, H, W)
class Resize3D:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if len(img.shape) == 4:  # (C, D, H, W) 格式
            img = img.unsqueeze(0)  # 添加一个维度，变为 (1, C, D, H, W)
        else:
            raise ValueError("Unsupported tensor shape, must be (C, D, H, W)")

        # 进行大小调整
        img_resized = F.interpolate(img, size=self.size, mode='trilinear', align_corners=False)

        return img_resized.squeeze(0)  # 返回多通道的 (C, D, H, W)


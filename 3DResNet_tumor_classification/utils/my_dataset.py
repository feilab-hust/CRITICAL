import time

import numpy as np
import tifffile as tiff
import torch
from skimage.color import rgb2gray
from torch.utils.data import Dataset


class MyDataSetTIF(Dataset): #NOTE 支持输入3D或2D的tif图像，包括unit8-RGB和uint8or16-gray。输出(C,D,H,W)或(C,H,W)的tensor图像，C=1or3
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None, convertRGB=False):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.convertRGB = convertRGB
        self.images, self.labels, self.imagesizes = self.load_images_and_labels()  # 加载所有图像和标签数据

    def __len__(self):
        return len(self.images_path)

    def load_images_and_labels(self):
        start_time = time.time()
        images, labels, imagesizes = [], [], []
        for item in range(len(self.images_path)):
            with tiff.TiffFile(self.images_path[item]) as tif:
                img = tif.asarray()  # 读取图像数组
                photometric = tif.pages[0].tags['PhotometricInterpretation'].value  # 读取图像的颜色模式

            imagesize = torch.tensor(img.shape[-3:]).to(torch.float32)
            imagesizes.append(imagesize)

            if photometric == 1:  # 1为灰度图像 uint16-gray (D,H,W)
                if self.convertRGB:
                    img = np.stack([img] * 3, axis=0)  # uint16(D, H, W) -> (3, D, H, W)   因为ToTensor不支持三维RGB数据，因此在这里手动将C通道放到第一个维度上
                else:
                    img = np.expand_dims(img, axis=0)    # uint16(1, D, H, W)
                # uint16
                if img.dtype == np.uint8:
                    img = img.astype(np.float32)
                    img = img.astype(np.float32) / 255.0
                elif img.dtype == np.uint16:
                    img = img.astype(np.float32) / 65535.0
                else:
                    raise ValueError(f"Unsupported dtype format: {img.dtype}")
                img = torch.from_numpy(img)  # 转换为 PyTorch Tensor
            elif photometric == 2:  # 2为RGB图像 uint8-RGB (D,H,W,3)
                if self.convertRGB:
                    if img.ndim == 3:
                        img = np.transpose(img, (2, 0, 1))      # C挪到第一个 uint8(3, H, W)
                    elif img.ndim == 4:
                        img = np.transpose(img, (3, 0, 1, 2))   # C挪到第一个 uint8(3, D, H, W)
                    img = img.astype(np.float32) / 255.0
                else:
                    img = rgb2gray(img).astype(np.float32) # uint8(D, H, W, 3) -> np.float64(D, H, W) -> np.float32
                    img = np.expand_dims(img, axis=0)   # np.float32(1, D, H, W)
                img = torch.from_numpy(img)  # 转换为 PyTorch Tensor
            else:
                raise ValueError(f"Unsupported photometric format: {photometric}")

            if self.transform is not None:
                img = self.transform(img)

            label = self.images_class[item]  # 得到该图像的标签

            images.append(img)
            labels.append(label)

        print(f"加载数据用时: {time.time() - start_time} 秒")
        return images, labels, imagesizes

    def __getitem__(self, idx):
        # 读取TIFF图像，并归一化到0~1，转为torch.float32格式
        image, label, imagesize = self.images[idx], self.labels[idx], self.imagesizes[idx]  # 获取图像数据和标签数据
        return image, label, imagesize

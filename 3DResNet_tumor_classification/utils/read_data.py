import os

import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.padding import padding


class CustomDataset(Dataset):
    def __init__(self, data_dir, image_size, if_transformsdata = False, if_padding = True, if_TH = True):
        self.data_dir = data_dir  # 数据集所在的目录
        self.image_size = image_size  # 每个图像的大小
        self.if_transformsdata = if_transformsdata   # 是否数据增强
        self.if_padding = if_padding   # 是否先padding图像
        self.if_TH = if_TH   # 是否加入阈值
        self.classes = sorted(os.listdir(data_dir))  # 所有类别的名称。在这里将所有类别的名称进行sorted排序，按照排序顺序依次映射为0,1,……等，（默认AAH:0 ADAC:1或AC:0 AD:1）
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}  # 将类别名称映射为类别索引
        self.images, self.labels = self.load_images_and_labels()  # 加载所有图像和标签数据

    def load_images_and_labels(self):
        images, labels = [], []
        for cls_idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(self.data_dir, cls_name)  # 每个类别所在的目录
            image_paths = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.tif')]  # 所有tif格式的图像文件路径
            for path in image_paths:
                # 先判断图像大小，如果过小或过大就剔除训练集
                if self.if_TH:
                    size = os.path.getsize(path)/1024   # 转换为KB
                    if size < 80:
                        continue
                    elif size > 4500:
                        continue

                image = tiff.imread(path)  # 加载图像数据，读取格式为(D,H,W)
                image = np.float32(image)   # 转数据类型，否则上/下采样后会有问题
                image = torch.from_numpy(image)   # 转成tensor
                # 定义数据增强的transform
                if self.if_transformsdata:   # 如果数据增强
                    transform_train = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomRotation(degrees=(0, 90))
                        # 水平、上下翻转，旋转
                    ])
                    image = transform_train(image)
                if self.if_padding:
                    if image.shape[0]<self.image_size[0] or image.shape[1]<self.image_size[1] or image.shape[2]<self.image_size[2]:   # 如果图像任一size小于64，则paddingda到64
                        image = padding(image, self.image_size)
                image = image.unsqueeze(0).unsqueeze(0)   # 将numpy数组转换为PyTorch张量
                # interpolate插值：即将输入图像全部转换为同样的image_size大小
                image = F.interpolate(image, size=self.image_size, mode='trilinear', align_corners=False)   # 三线性插值，用于三维数据，使用目标像素周围的8个像素点的值来计算目标像素的值。
                image = image.squeeze(0).squeeze(0).numpy()   # 将tensor转回numpy数组（三维数据不能ToTensor！）
                if image.shape != self.image_size:  # 如果图像大小不符合要求，则抛出异常
                    raise ValueError(f"Image size of {path} is not {self.image_size}.")
                images.append(image)
                labels.append(cls_idx)
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]  # 获取图像数据和标签数据
        image = torch.from_numpy(image).unsqueeze(0)  # 添加一个通道维度，转换为PyTorch的张量
        return image, label


# data_dir = 'dataset/train'  # 数据集所在的目录
# image_size = (101, 37, 45)  # 每个图像的大小
# batch_size = 4  # 每个批次的大小
# num_workers = 0  # 用于数据加载的线程数。在linux系统中可以使用多个子进程加载数据，而在windows系统中不能，故设置为0。
#
# dataset = CustomDataset(data_dir, image_size)  # 创建数据集对象
# # images={ndarray:(images_num,D,H,W)}（所有图像需要image_size={tuple:DHW}相同）
# # labels={ndarray:(images_num)[0,0,……,0,1,1,……,1]}
# # （AAH:0 ADAC:1）
# dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)  # 创建数据加载器对象
#
# for batch_idx, (images, labels) in enumerate(dataloader):
#     # images={Tensor:(batch,1,D,H,W)}   labels={Tensor:(batch)}
#     # 训练代码
#     pass


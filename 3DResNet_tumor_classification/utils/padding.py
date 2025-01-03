import torch
import tifffile as tiff
import numpy as np
import datetime

def padding(image, image_size):

    image.numpy()   # 先转成np，才能用后面的np.pad
    # 定义需要 padding 到的大小为 (128, 128, 128)
    target_size = image_size   # 输入为(D,H,W)，但下面的pad为(W,H,D)，因此下面的for循环倒着来

    # 计算需要在每个维度上分别进行多少次 padding
    pad_dims = [max(target_size[i] - image.shape[i], 0) for i in range(3)]   # [p_D,p_H,p_W]
    # 将 padding 数量转化为左右两侧各 padding 的数量
    pad_widths = [(pad_dim // 2, pad_dim - pad_dim // 2) for pad_dim in pad_dims]   # [(p_D前,p_D后),(p_H↑,p_H↓),(p_W←,p_W→)]
    # pad_widths = tuple(num for tup in pad_widths for num in tup)  # 转成元组 (p_D前,p_D后,p_H↑,p_H↓,p_W←,p_W→)
    # 在三个维度上进行 padding
    image = np.pad(image, pad_width=pad_widths, mode='minimum')

    image = torch.from_numpy(image)   # 转回tensor

    # 查看 padding 后的图像大小
    return image

if __name__ == '__main__':


    image = tiff.imread('Datasets/dataset5/train/ADAC/lobe2 (4).tif')
    image = np.float32(image)   # 转数据类型，否则上/下采样后会有问题
    image = torch.from_numpy(image)
    print(image.shape)
    t1 = datetime.datetime.now()
    image = padding(image)



    image = image.numpy()   # 测试保存用才转成np
    tiff.imwrite('1.tif',image)
    t2 = datetime.datetime.now()
    h = t2.hour - t1.hour
    m = t2.minute - t1.minute
    s = t2.second - t1.second
    print('总耗时:%dh %dm %ds' % (h, m, s))
    print(image.shape)

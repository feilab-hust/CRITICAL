import glob
import os

import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F

from ResNet_3D_model import ResNet3D18_imgsize


def predict(model, pth_path, data_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")  # cuda报错没有可用的就改成cuda:0123

    # 加载模型和预训练参数
    model.load_state_dict(torch.load(pth_path, map_location='cuda:0'))
    model.to(device)
    model.eval()
    with torch.no_grad():
        AAH = AD = AC = 0
        # 加载数据集
        imglist = glob.glob(os.path.join(data_dir, '*.tif'))
        print(f'{len(imglist)} images found')
        for imgpath in imglist:
            # 读图
            with tiff.TiffFile(imgpath) as tif:
                img = tif.asarray()  # 读取图像数组
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32) / 65535.0
            img = torch.from_numpy(img)  # 转换为 PyTorch Tensor

            imgsize = torch.tensor(img.shape[-3:]).unsqueeze(0).to(torch.float32)

            # Resize3D -> 64
            img = F.interpolate(img.unsqueeze(0), size=(64, 64, 64), mode='trilinear', align_corners=False)

            img, imgsize = img.to(device), imgsize.to(device)

            # 对测试数据进行预测
            predicted_classes = model(img, imgsize).softmax(-1)
            _, predicted = predicted_classes.max(1)
            if predicted == 0:
                AAH += 1
            elif predicted == 1:
                AC += 1
            elif predicted == 2:
                AD += 1

            # print(f'{imgpath}:  {predicted.cpu().numpy()}')   # 输出每个肿瘤的预测结果

    print(f'AAH: {AAH}, AD: {AD}, AC: {AC}')


if __name__ == '__main__':
    pth_path = './save_models/drop20_lr15-2/model_77.pth'
    predict(model = ResNet3D18_imgsize(num_classes=3), pth_path = pth_path, data_dir = 'D:/Data/Sunlab/Dataset_tumor/Test_Dataset')


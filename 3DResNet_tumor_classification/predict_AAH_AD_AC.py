import glob
import os

import numpy as np
import tifffile as tiff
import torch
from scipy import ndimage

from utils.ResNet_3D_model import *
from utils.padding import padding


### 该程序用于预测数据

def img_preprocess(img,if_ADAC,image_size,model,device):
    # 假设你的测试数据是test_img，需要将其转换成统一大小、tensor张量并将其放在正确的设备上
    # 将图像大小转换为image_size
    if if_ADAC:
        img = ndimage.zoom(img, (0.5, 0.5, 0.5), order=1)  # AD、AC在训练时是经过了两倍下采样的
    # 将图像大小转换为image_size
    img = np.float32(img)  # 转数据类型，否则上/下采样后会有问题
    img = torch.from_numpy(img)  # 将numpy数组转换为PyTorch张量
    if if_ADAC:
        if img.shape[0] < image_size or img.shape[1] < image_size or img.shape[2] < image_size:  # 如果图像任一size小于self.image_size，则paddingda到self.image_size
            img = padding(img, (image_size, image_size, image_size))

    img = img.unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, size=image_size, mode='trilinear',
                             align_corners=False)  # 三线性插值，用于三维数据，使用目标像素周围的8个像素点的值来计算目标像素的值。
    # 将数据和模型放入GPU
    img = img.to(device)
    model.to(device)

    # 对测试数据进行预测
    predicted_classes = model(img)
    _, predicted = predicted_classes.max(1)

    return predicted




# 加载训练好的参数
def predict(test_dir, pth_path1, pth_path2, image_size1 = 64, image_size2 = 128, model1=ResNet3D18(2), model2=ResNet3D18_ADAC(2)):

    # pth_path:模型参数路径
    # test_dir:测试图像所在的文件夹路径
    # image_size:训练时图像的大小
    # model:训练时采用的模型

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    TH_AAH_ADAC = True

    AAH = 0
    AD = 0  # 计数
    AC = 0

    # 加载模型参数
    model1.load_state_dict(torch.load(pth_path1, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
    model2.load_state_dict(torch.load(pth_path2, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
    # 设置模型为评估模式
    model1.eval()
    model2.eval()

    # 读取测试数据并进行预测
    with torch.no_grad():
        # 得到所有tif格式的图像文件路径
        image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.tif')]
        for img_path in image_paths:
            # 先判断图像大小，如果过小或过大就直接认定为AAH或ADAC
            if TH_AAH_ADAC:
                size = os.path.getsize(img_path)/1024   # 转换为KB
                if size < 80:   # 直接判断为AAH，进入下一张图像
                    AAH += 1
                    continue
                elif size > 5000:   # 判断为AD或AC，进入第二段网络
                    only_ADAC = True   # 只可能是ADAC
                else:
                    only_ADAC = False   # 既可能是AAH也可能是ADAC

            test_img = tiff.imread(img_path)  # 读图，读取格式为(D,H,W)
            ### ----------------------如果只可能是ADAC，则直接进入第二段网络---------------------------------
            if only_ADAC:
                predicted = img_preprocess(test_img, True, image_size2, model2, device)
                if predicted == 1:   # 0：AC   1:AD
                    AD += 1
                    # print(img_path)
                else:
                    AC += 1
                    # print(img_path)

            ### ----------------------如果AAH和ADAC都有可能，则先进入第一段网络，如果是ADAC，则再进入第二段网络---------------------------------
            else:
                predicted = img_preprocess(test_img, False, image_size1, model1, device)
                if predicted == 1:   # 0：AAH   1:ADAC
                    # -----------------进入第二段网络-----------------
                    predicted = img_preprocess(test_img, True, image_size2, model2, device)
                    if predicted == 1:  # 0：AC   1:AD
                        AD += 1
                        # print(img_path)
                    else:
                        AC += 1
                        # print(img_path)
                else:
                    AAH += 1
                    # print(img_path)

        tumor_num = AAH+AD+AC   # 防止除数为0
        print('该肺内总肿瘤数： %d' % tumor_num)
        print('预测结果：AAH:%d   AD: %d   AC: %d' % (AAH, AD, AC))
        if tumor_num == 0:
            tumor_num = 1
        print('预测比例：AAH: {:.2%}'.format(AAH/tumor_num), 'AD: {:.2%}'.format(AD/tumor_num), 'AC: {:.2%}'.format(AC/tumor_num))
        return AAH,AD,AC

if __name__ == '__main__':
    AAH,AD,AC = predict(r'D:\Data\Sunlab\20250102-肿瘤测试集合集',
                        r'save_models/dataset6_models/ResNet18_dataset6_size64_batch16_TH_lr5/3d_resnet_30.pth',
                        r'save_models/datasetADAC_models/ResNet18_datasetADAC_size128_batch8_padding_lr5/3d_resnet_30.pth')

'''
    basepath = r'D:\Data\Sunlab\20250102-肿瘤测试集'
    dirlist = glob.glob(os.path.join(basepath, '*'))
    AAH = AD = AC = 0
    for dir in dirlist:
        AAH_temp, AD_temp, AC_temp = predict(dir,
                              r'save_models/dataset6_models/ResNet18_dataset6_size64_batch16_TH_lr5/3d_resnet_30.pth',
                              r'save_models/datasetADAC_models/ResNet18_datasetADAC_size128_batch8_padding_lr5/3d_resnet_30.pth')
        AAH += AAH_temp
        AD += AD_temp
        AC += AC_temp

    tumor_num = AAH + AD + AC  # 防止除数为0
    print('该肺内总肿瘤数： %d' % tumor_num)
    print('预测结果：AAH:%d   AD: %d   AC: %d' % (AAH, AD, AC))
    if tumor_num == 0:
        tumor_num = 1
    print('预测比例：AAH: {:.2%}'.format(AAH / tumor_num), 'AD: {:.2%}'.format(AD / tumor_num),
          'AC: {:.2%}'.format(AC / tumor_num))
'''
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 载入其他py文件
from utils.ResNet_3D_model import *
from utils.learning_rate import step_decay_scheduler
from utils.read_data import CustomDataset
from utils.test import test
from utils.train import train

### 该程序用于训练3DResNet模型

def main():
    # 设置参数
    image_size = (128, 128, 128)  # 图像大小。分类AAH和ADAC时默认64，分类AD和AC时默认128。
    epochs = 50
    learning_rate = 1e-3
    num_classes = 2   # 类别数
    # 用来命名的主要参数
    batch_size = 8
    image_pad = True
    enhance = False     # 数据增强
    lr_decrease = True
    TH = False   # 体积阈值：是否筛除掉体积过小和过大的肿瘤，在read_data里调整
    model = ResNet3D18_ADAC(num_classes)   # 选择模型，记得下面的路径同步改名
    # 数据集路径
    data_dir = r'dataset_test/AD_AC'
    train_data_dir = os.path.join(data_dir, 'train')
    test_data_dir = os.path.join(data_dir, 'val')
    # 自动给保存模型参数的文件夹命名（model还需要手动改名）
    pth_dir = 'save_models/datasetADAC_models/ResNet18_datasetADAC_alltrain_size%d_batch%d'%(image_size[0],batch_size)
    if TH:
        pth_dir = pth_dir + '_TH'
    if image_pad:
        pth_dir = pth_dir + '_padding'
    if enhance:
        pth_dir = pth_dir + '_enhance'
    if lr_decrease:
        pth_dir = pth_dir + '_lr5'
    print(pth_dir)   # 记录模型信息
    # 设置GPU或CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")   # cuda报错没有可用的就改成cuda:0123

    num_workers = 0  # 用于数据加载的线程数。在linux系统中可以使用多个子进程加载数据，而在windows系统中不能，故设置为0。
    train_dataset = CustomDataset(train_data_dir, image_size, if_transformsdata = enhance, if_padding = image_pad, if_TH = TH)  # 创建数据集对象
    test_dataset = CustomDataset(test_data_dir, image_size, if_transformsdata = enhance, if_padding = image_pad, if_TH = TH)  # 创建数据集对象
    # dataset:
    # images={ndarray:(images_num,D,H,W)}（所有图像需要image_size={tuple:DHW}相同）
    # labels={ndarray:(images_num)[0,0,……,0,1,1,……,1]}
    # （AAH:0 ADAC:1）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)  # 创建数据加载器对象
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)  # 创建数据加载器对象

    # 初始化模型和优化器
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    print('开始训练')
    # 运行训练和测试循环
    for epoch in range(epochs):

        # lr递减
        if lr_decrease == True:
            learning_rate_epoch = step_decay_scheduler(total_epochs=epochs, initial_lr=learning_rate, decay_epochs=5)
            learning_rate_epoch = learning_rate_epoch[epoch]
        else:
            learning_rate_epoch = learning_rate
        optimizer = optim.Adam(model.parameters(), lr=learning_rate_epoch)

        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

        # 每一轮后保存模型
        if not os.path.exists(pth_dir):
            os.makedirs(pth_dir)
        torch.save(model.state_dict(), os.path.join(pth_dir, "3d_resnet_%d.pth"%(epoch)))   # 记录轮数

        # # 测试集
        # predict(os.path.join(pth_dir, "3d_resnet_%d.pth"%(epoch)), 'Datasets/dataset5/predict/AAH', model=model)
        # predict(os.path.join(pth_dir, "3d_resnet_%d.pth"%(epoch)), 'Datasets/dataset5/predict/ADAC', model=model)
        print('-----------------------------------------------')

if __name__ == '__main__':
    main()



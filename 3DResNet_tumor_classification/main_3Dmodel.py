import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from ResNet_3D_model import ResNet3D18_imgsize
from SWT_utils.Resize3D import Resize3D
from SWT_utils.learning_rate import step_decay_scheduler
from SWT_utils.load_trainval_data import load_trainval_data
from SWT_utils.my_dataset import MyDataSetTIF
from SWT_utils.utils import read_split_data
from test import test
from train import train


def main():
# 设置参数
    pth_savepath = './save_models/newDataset/imgsize_AAH-AD-AC_all-drop20_lr15_train1600'    # 模型保存路径
    # 设置GPU或CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")  # cuda报错没有可用的就改成cuda:0123

    ### 设置训练参数
    batch_size = 16
    num_workers = 0  # 用于数据加载的线程数。在linux系统中可以使用多个子进程加载数据，而在windows系统中不能，故设置为0
    num_classes = 3
    epochs = 80
    learning_rate = 1e-3
    imgsize = 64

# 设置学习率
    learning_rate = step_decay_scheduler(total_epochs=epochs, initial_lr=learning_rate, decay_epochs=15)

    model = ResNet3D18_imgsize(num_classes=num_classes).to(device)

### 设置数据集
    print('加载数据集')
    data_dir = r'D:\Data\Sunlab\Dataset_tumor\AAH-AD-AC-all'  # 训练数据集所在的目录
    # train_images_path, train_images_label, val_images_path, val_images_label = load_trainval_data(data_dir, train_ratio=1.0, random_seed=0)
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_dir, val_rate = 0.461, random_seed = 0)
    data_transform = {
            "train": transforms.Compose([#transforms.ToTensor(),
                                         Resize3D(size=(imgsize, imgsize, imgsize))]),
            "val": transforms.Compose([#transforms.ToTensor(),
                                       Resize3D(size=(imgsize, imgsize, imgsize))])}
    # 实例化数据集
    train_dataset = MyDataSetTIF(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"],
                              convertRGB=False)
    test_dataset = MyDataSetTIF(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"],
                             convertRGB=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=num_workers)

# 设置损失函数
    criterion = nn.CrossEntropyLoss()


    print('开始训练')
    train_acc = []  # 用于记录每轮的准确率
    test_acc = []    # 运行训练和测试循环
    for epoch in range(epochs):

        # 设置优化器
        learning_rate_epoch = learning_rate[epoch]
        optimizer = optim.Adam(model.parameters(), lr=learning_rate_epoch)

        # 训练
        start_time = time.time()
        train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch)
        print(f"训练用时: {time.time() - start_time} 秒")
        start_time = time.time()
        test_accuracy = test(model, device, test_loader, criterion)
        print(f"验证用时: {time.time() - start_time} 秒")

        if not os.path.exists(pth_savepath):
            os.makedirs(pth_savepath)

        if epoch % 1 == 0 or epoch == epochs-1:
            torch.save(model.state_dict(), os.path.join(pth_savepath, "model_%d.pth" % (epoch)))   # 记录轮数
        print('-----------------------------------------------')

        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)

    # function: 计算每5轮的平均准确率，并保存为txt文件
    train_acc_mean5 = [0] * len(train_acc)
    test_acc_mean5 = [0] * len(test_acc)
    for i in range(len(train_acc)):
        # 动态确定分组范围
        start = max(0, i - 4)  # 保证不越界，向前找 4 个数
        end = i + 1  # 当前索引加 1（包含当前数）
        # 计算当前分组的平均值
        train_acc_mean5[i] = np.mean(train_acc[start:end])
        test_acc_mean5[i] = np.mean(test_acc[start:end])
    # 准确率都只保留两位小数
    train_acc = np.round(np.array(train_acc), 2)
    train_acc_mean5 = np.round(np.array(train_acc_mean5), 2)
    test_acc = np.round(np.array(test_acc), 2)
    test_acc_mean5 = np.round(np.array(test_acc_mean5), 2)
    # 将模型准确率保存为 txt 文件
    with open(os.path.join(pth_savepath, "BestAccuracy_{:.2f}%.txt".format(max(test_acc_mean5))), "w") as file:
        file.write("epoch\ttrain\t5Mean\ttest\t5Mean\n")  # 写入标题
        for epoch, train1, train5, test1, test5 in zip(range(1,epochs+1) ,train_acc, train_acc_mean5, test_acc, test_acc_mean5):
            file.write(f"{epoch}\t{train1}\t{train5}\t{test1}\t{test5}\n")  # 使用制表符分隔两列
        file.write(f"Best\t{(max(train_acc))}\t{(max(train_acc_mean5))}\t{(max(test_acc))}\t{(max(test_acc_mean5))}\n")
    print("保存为 BestAccuracy_{:.2f}%.txt 文件成功！".format(max(test_acc_mean5)))

if __name__ == '__main__':
    main()



# 查看模型的输出
# print(model_1)

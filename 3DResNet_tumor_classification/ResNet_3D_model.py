import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageSizeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ImageSizeClassifier, self).__init__()

        # 输入层到隐藏层的全连接层
        self.fc1 = nn.Linear(3, 256)  # 输入3个特征，输出64个神经元
        self.fc2 = nn.Linear(256, 64)  # 第二层，64个神经元到32个神经元
        self.fc3 = nn.Linear(64, num_classes)  # 输出层，3个神经元表示3个分类

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 输入通过第一个全连接层并经过ReLU激活
        x = self.relu(self.fc2(x))  # 第二个全连接层和ReLU激活
        x = self.fc3(x)  # 输出层
        return x  # 输出概率分布


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):   # 每个BasicBlock卷积2次
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet3D, self).__init__()
        self.in_planes = 16   # 第一个卷积层将单通道变成in_planes通道

        self.conv1 = nn.Conv3d(1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(4096, 64)
        self.linear2 = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):   # x:(1,1,64,64,64)
        out = F.relu(self.bn1(self.conv1(x)))   # (1,16,64,64,64)   1conv,1bn,1relu
        out = self.layer1(out)   # (1,16,64,64,64)
        out = self.layer2(out)   # (1,32,32,32,32)
        out = self.layer3(out)   # (1,64,16,16,16)
        out = F.avg_pool3d(out, 4)   # 平均池化，(1,64,4,4,4)
        out = out.view(out.size(0), -1)   # (1,4096)   4096 = 64*4*4*4
        out = self.linear1(out)   # 4096 -> 64
        out = self.linear2(out)   # 64 -> 2
        return out

class ResNet3D_128(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet3D_128, self).__init__()
        self.in_planes = 16   # 第一个卷积层将单通道变成in_planes通道

        self.conv1 = nn.Conv3d(1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(4096, 64)
        self.linear2 = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):   # x:(1,1,128,128,128)
        out = F.relu(self.bn1(self.conv1(x)))   # (1,16,128,128,128)   1conv,1bn,1relu
        out = self.layer1(out)   # (1,16,128,128,128)
        out = self.layer2(out)   # (1,32,64,64,64)
        out = self.layer3(out)   # (1,64,32,32,32)
        out = self.layer4(out)   # (1,64,16,16,16)
        out = F.avg_pool3d(out, 4)   # 平均池化，(1,64,4,4,4)
        out = out.view(out.size(0), -1)   # (1,4096)   4096 = 64*4*4*4
        out = self.linear1(out)   # 4096 -> 64
        out = self.linear2(out)   # 64 -> 2
        return out

class ResNet3D_imgsize(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet3D_imgsize, self).__init__()
        self.in_planes = 16   # 第一个卷积层将单通道变成in_planes通道

        self.conv1 = nn.Conv3d(1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(4160, 64)
        self.linear2 = nn.Linear(64, num_classes)
        self.linear_size = nn.Linear(3, 64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        # self.dropout_size = nn.Dropout(0.2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, size):   # x:(1,1,64,64,64)
        out = F.relu(self.bn1(self.conv1(x)))   # (1,16,64,64,64)   1conv,1bn,1relu
        out = self.layer1(out)   # (1,16,64,64,64)
        out = self.layer2(out)   # (1,32,32,32,32)
        out = self.layer3(out)   # (1,64,16,16,16)
        out = F.avg_pool3d(out, 4)   # 平均池化，(1,64,4,4,4)
        out = out.view(out.size(0), -1)   # (1,4096)   4096 = 64*4*4*4

        size = self.relu(self.linear_size(size))   # (1,3) -> (1,64)
        # size = self.dropout_size(size)
        out = torch.cat((out, size), dim=1) # 4096+64=4160
        out = self.relu(self.linear1(out))   # 4160 -> 64
        out = self.dropout1(out)
        out = self.linear2(out)   # 64 -> 2
        return out

def ResNet3D18(num_classes=2):
    return ResNet3D(BasicBlock, [2, 2, 2], num_classes=num_classes)

def ResNet3D18_imgsize(num_classes=2):
    return ResNet3D_imgsize(BasicBlock, [2, 2, 2], num_classes=num_classes)

def ResNet3D18_128(num_classes=2):
    return ResNet3D_128(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet3D34(num_classes=2):
    return ResNet3D(BasicBlock, [3, 4, 6], num_classes=num_classes)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # cuda报错没有可用的就改成cuda:0123
    x = torch.randn(16, 1, 64, 64, 64).to(device)
    size = torch.randn(16, 3).to(device)
    model = ResNet3D18_imgsize(num_classes=3).to(device)
    start_time = time.time()
    output = model(x, size)
    print(output.size())
    print(f"用时: {time.time() - start_time} 秒")

    # x = torch.randn(16, 3).to(device)
    # model = ImageSizeClassifier().to(device)
    # start_time = time.time()
    # output = model(x)
    # print(output.size())
    # print(f"用时: {time.time() - start_time} 秒")

    # model = ImageSizeClassifier(num_classes=3)
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params_count = sum([p.numel() for p in model_parameters])
    # print(f"模型参数数量: {params_count}")
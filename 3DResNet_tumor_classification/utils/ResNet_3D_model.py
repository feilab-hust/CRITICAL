import torch.nn as nn
import torch.nn.functional as F


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

class ResNet3D_ADAC(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet3D_ADAC, self).__init__()
        self.in_planes = 16   # 第一个卷积层将单通道变成in_planes通道

        self.conv1 = nn.Conv3d(1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(32768, 128)
        self.linear2 = nn.Linear(128, num_classes)

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
        out = F.avg_pool3d(out, 4)   # 平均池化，(1,64,8,8,8)
        out = out.view(out.size(0), -1)   # (1,32768)   32768 = 64*8*8*8
        out = self.linear1(out)   # 32768 -> 128
        out = self.linear2(out)   # 128 -> 2
        return out


def ResNet3D18(num_classes=2):
    return ResNet3D(BasicBlock, [2, 2, 2], num_classes=num_classes)

def ResNet3D18_ADAC(num_classes=2):
    return ResNet3D_ADAC(BasicBlock, [2, 2, 2], num_classes=num_classes)

def ResNet3D34(num_classes=2):
    return ResNet3D(BasicBlock, [3, 4, 6], num_classes=num_classes)

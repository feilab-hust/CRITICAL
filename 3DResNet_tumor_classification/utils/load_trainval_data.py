import os
import glob
import random

def load_trainval_data(data_dir, train_ratio=1.0, test_ratio=1.0, random_seed=0):
    # 设置随机种子以确保结果可复现
    if random_seed is not None:
        random.seed(random_seed)

    # 定义训练集和验证集的路径
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # 初始化用于存储图片路径和标签的列表
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []

    # 加载训练集图片路径和标签
    # 遍历train文件夹中的子文件夹（每个子文件夹代表一个类别）
    for label, class_dir in enumerate(sorted(os.listdir(train_dir))):
        class_path = os.path.join(train_dir, class_dir)  # 获得子文件夹的完整路径
        if os.path.isdir(class_path):  # 确保这是一个文件夹
            # 使用glob获取该类别文件夹下所有图片的路径
            images = glob.glob(os.path.join(class_path, '*.tif'))

            # 随机抽样根据train_ratio保留部分图像
            if train_ratio < 1.0:
                images = random.sample(images, max(1, int(len(images) * train_ratio)))

            train_images_path.extend(images)  # 将图片路径添加到训练集路径列表
            train_images_label.extend([label] * len(images))  # 为每张图片添加对应的标签

    # 加载验证集图片路径和标签（逻辑与训练集相同）
    for label, class_dir in enumerate(sorted(os.listdir(val_dir))):
        class_path = os.path.join(val_dir, class_dir)  # 获取验证集子文件夹的完整路径
        if os.path.isdir(class_path):  # 确保这是一个文件夹
            # 使用glob获取该类别文件夹下所有图片的路径
            images = glob.glob(os.path.join(class_path, '*.tif'))

            # 随机抽样根据train_ratio保留部分图像
            if test_ratio < 1.0:
                images = random.sample(images, max(1, int(len(images) * test_ratio)))

            val_images_path.extend(images)  # 将图片路径添加到验证集路径列表
            val_images_label.extend([label] * len(images))  # 为每张图片添加对应的标签

    # 返回训练集和验证集的图片路径与标签列表
    print("{} images were found in the dataset.".format(len(train_images_path+val_images_path)))
    print("{} train images.".format(len(train_images_path)))
    print("{} val images.".format(len(val_images_path)))

    return train_images_path, train_images_label, val_images_path, val_images_label


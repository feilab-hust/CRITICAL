import numpy as np
import matplotlib.pyplot as plt

# Warmup + Cosine Decay Learning Rate Scheduler

def warmup_cosine_decay_scheduler(total_epochs, warmup_epochs, max_lr, min_lr=0.0001):
    """
    warmup_cosine_decay_scheduler 函数用于实现 Warmup + 余弦衰减的学习率调度策略。

    参数:
    total_epochs (int): 总训练轮数，即模型将要训练的总epoch数。
    warmup_epochs (int): 预热轮数，在此轮数内学习率从 min_lr 线性增加到 max_lr。
    max_lr (float): 最大学习率，预热阶段结束时达到的学习率峰值。
    min_lr (float): 最小学习率，也是学习率的初始值和最终值，余弦衰减时不会低于此值。

    返回:
    lr_schedule (list): 包含每轮次学习率的列表，长度为 total_epochs。
    """

    lr_schedule = []
    for epoch in range(total_epochs):
        if epoch < warmup_epochs:
            # Linearly increase learning rate during warmup phase
            lr = (max_lr - min_lr) * epoch / warmup_epochs + min_lr
        else:
            # Cosine decay after warmup
            decay_epochs = epoch - warmup_epochs
            total_decay_epochs = total_epochs - warmup_epochs
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * decay_epochs / total_decay_epochs))
        lr_schedule.append(lr)
    return lr_schedule


def step_decay_scheduler(total_epochs, initial_lr=0.0001, decay_epochs=10, decay_factor=0.5):
    """
    简单的学习率衰减策略：每经过 decay_epochs 轮，学习率乘以 decay_factor。

    参数:
    total_epochs (int): 总训练轮数
    initial_lr (float): 初始学习率
    decay_epochs (int): 每隔多少轮进行衰减
    decay_factor (float): 学习率衰减系数

    返回:
    lr_schedule (list): 包含每轮次学习率的列表
    """

    lr_schedule = []
    current_lr = initial_lr

    for epoch in range(total_epochs):
        if epoch > 0 and epoch % decay_epochs == 0:
            current_lr *= decay_factor  # 每 decay_epochs 轮，学习率衰减
        lr_schedule.append(current_lr)

    return lr_schedule

if __name__ == '__main__':
    # Define parameters
    total_epochs = 100
    warmup_epochs = 10
    max_lr = 0.01
    min_lr = 0.001

    # Get learning rate schedule
    lr_schedule = warmup_cosine_decay_scheduler(total_epochs, warmup_epochs, max_lr, min_lr)

    # Plot the learning rate schedule
    plt.plot(range(total_epochs), lr_schedule)
    plt.title("Warmup + Cosine Decay Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()

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


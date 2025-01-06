import numpy as np

def step_decay_scheduler(total_epochs, initial_lr=0.0001, decay_epochs=10, decay_factor=0.5):

    lr_schedule = []
    current_lr = initial_lr

    for epoch in range(total_epochs):
        if epoch > 0 and epoch % decay_epochs == 0:
            current_lr *= decay_factor  # 每 decay_epochs 轮，学习率衰减
        lr_schedule.append(current_lr)

    return lr_schedule


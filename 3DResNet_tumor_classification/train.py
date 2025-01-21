def train(model, device, train_loader, optimizer, criterion, epoch):
    # 将模型设置为训练模式
    model.train()

    # 初始化变量用于记录训练损失和准确率
    train_loss = 0
    correct = 0
    total = 0

    # 遍历训练数据集中的所有批次
    for batch_idx, (data, target, imagesize) in enumerate(train_loader):
        # 将数据和目标移动到GPU或CPU上
        data, target, imagesize = data.to(device), target.to(device), imagesize.to(device)
        # 将优化器的梯度清零
        optimizer.zero_grad()
        # 使用模型进行前向传递并计算损失
        output = model(data, imagesize)
        loss = criterion(output, target)
        # 计算反向传递并更新模型参数
        loss.backward()
        optimizer.step()

        # 计算训练损失和准确率
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    # 计算平均训练损失和训练准确率
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / total

    print('Train Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, train_loss, correct, total, train_accuracy))

    return train_accuracy

import torch


def test(model, device, test_loader, criterion):
    # 将模型设置为测试模式
    model.eval()

    # 初始化变量用于记录测试损失和准确率
    test_loss = 0
    correct = 0
    total = 0

    # 禁用梯度计算，因为在测试过程中不需要更新模型参数
    with torch.no_grad():
        # 遍历测试数据集中的所有批次
        for data, target in test_loader:
            # 将数据和目标移动到GPU或CPU上
            data, target = data.to(device), target.to(device)

            # 使用模型进行前向传递并计算损失
            output = model(data)
            loss = criterion(output, target)

            # 计算测试损失和准确率
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    # 计算平均测试损失和测试准确率
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / total
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, total, test_accuracy))

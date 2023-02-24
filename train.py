from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model import SwinTransformer
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# 加载数据
def load_dataset(filePath, batch_sz=128, val_sz=0.4):
    trans = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(root=filePath, 
                          transform=trans)
    data_len = len(dataset)
    val_len = int(data_len * val_sz)
    train_len = data_len - val_len
    train_dataset, val_dataset = random_split(dataset=dataset, 
                                               lengths=[train_len, val_len])
    return (DataLoader(train_dataset, batch_size=batch_sz, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_sz, shuffle=True), 
            dataset.class_to_idx)
train_iter, test_iter, ss = load_dataset('./CK+48')
net = SwinTransformer(in_chans=3,
                            patch_size=2,
                            window_size=5,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=7)
# 每一个训练epoch
def train_epoch(model, train_loader, optimizer, loss_fn, epoch):
    size = len(train_loader.dataset)
    num_batches = len(train_loader)
    model.train() # 将模型设置为训练模式
    train_loss, train_correct = 0, 0
    for batch_idx, (X, y) in enumerate(train_loader):
        pred = model(X)
        y = y.to(pred.device)
        loss = loss_fn(pred, y)
        # 梯度清零， 反向传播，更新网络参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录损失与正确率
        train_loss += loss.item()
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X), size,
                100. * batch_idx / num_batches, loss.item()))
    return train_loss / num_batches, train_correct / size
# 测试epoch
def test_epoch(model, test_loader, loss_fn):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval() # 设为评估模式
    test_loss, test_correct = 0, 0
    # 不记录梯度，节省内存
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            y = y.to(pred.device)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, size, 100. * test_correct / size))
    return test_loss, test_correct / size
# 分类问题使用交叉熵作为损失函数
loss_fn = nn.CrossEntropyLoss()
# 使用随机梯度下降法更新
trainer = torch.optim.AdamW(net.parameters(), lr = 0.0001, weight_decay=5E-2)
# 使用DP模式训练
net = nn.DataParallel(net)
# 获取训练数据集和测试数据集
# 训练轮数
num_epochs = 500
# 记录损失和正确率
train_loss, train_accuracy = [], []
test_loss, test_accuracy = [], []
for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}\n-------------------------------")
    a, b = train_epoch(net, train_iter, trainer, loss_fn, epoch)
    train_loss.append(a)
    train_accuracy.append(b)
    c, d = test_epoch(net, test_iter, loss_fn)
    test_loss.append(c)
    test_accuracy.append(d)
    writer.add_scalar('train/loss',scalar_value= a, global_step=epoch)
    writer.add_scalar('train/Accuracy',  scalar_value=b, global_step=epoch)
    writer.add_scalar('test/loss', scalar_value=c, global_step=epoch)
    writer.add_scalar('test/Accuracy', scalar_value=d,global_step=epoch)
# 保存模型参数
torch.save(net.state_dict(),'./model.pth')
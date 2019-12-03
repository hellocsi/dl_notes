# 最简单的MNIST程序结构

```python
# 常规导入
import torch
import torch.nn as nn
# ...

# 定义网络
class Net(nn.Module):
    # init 函数中定义网络结构
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # ...
    # forward 定义正向传播
    def forward(self, x):
        x = self.conv1(x)
        # ...
        output = F.log_softmax(x, dim=1)
        return output

# 定义训练过程
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # ...

# 定义测试过程
def test(args, model, device, test_loader):
    model.eval()
    # ...

# 定义参数设置及数据处理
def main():
    # 定义参数
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    # ...
    # 数据处理
    train_loader = torch.utils.data.DataLoader('../data')
    # ...
    # 设置optimizer, scheduler 等
    # ...
    # 记得保存模型
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

# main 函数
if __name__ == '__main__':
    main()

```
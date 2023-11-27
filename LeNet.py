import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        self.c2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.avg_pool2d(self.c1(x), kernel_size=(2, 2)))
        x = F.relu(F.avg_pool2d(self.c2(x), kernel_size=(2, 2)))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features

# class LeNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 定义卷定义层卷积层,1个输入通道，6个输出通道，5*5的filter,28+2+2=32
#         # 左右、上下填充padding
#         # MNIST图像大小28，LeNet大小是32
#         self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
#         # 定义第二层卷积层
#         self.conv2 = nn.Conv2d(6, 16, 5)
#
#         # 定义3个全连接层
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     # 前向传播
#     def forward(self, x):
#         # 先卷积，再调用relu激活函数，然后再最大化池化
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
#         # num_flat_features=16*5*5
#         x = x.view(-1, self.num_flat_features(x))
#
#         # 第一个全连接
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]
#         num_features = 1
#         for s in size:
#             num_features = num_features * s
#         return num_features

epochs = 10
BATCH_SIZE = 64
lr = 0.001
model = LeNet()
train_data = datasets.MNIST('../dataset/mnist', train=True, transform=transforms.ToTensor(), download=False)
test_data = datasets.MNIST('../dataset/mnist', train=False, transform=transforms.ToTensor(), download=True)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


for epoch in range(epochs):
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch {}, loss {}'.format(epoch + 1, loss.item()))

torch.save(model, './lenet.pt')
model = torch.load('./lenet.pt')

model.eval()
correct = 0
total = 0

for data in test_dataloader:
    inputs, labels = data
    output = model(inputs)
    total += labels.size(0)
    _, predicted = torch.max(output.data, 1)
    correct = correct + (predicted == labels).sum().item()

print('10000张图像的测试准确率:{}'.format(100 * correct / total))

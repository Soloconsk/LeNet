import os

import cv2
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision import transforms
from PIL import Image
from torchvision.ops import DeformConv2d
from pathlib import Path


class LeNet_WithoutPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        # self.offset_1 = nn.Conv2d(1, out_channels=2 * 5 * 5, kernel_size=5, padding=2)
        # self.offset_2 = nn.Conv2d(6, out_channels=2 * 10 * 10, kernel_size=10, stride=2)
        # self.conv_offset2d_1 = DeformConv2d(1, 6, 5, padding=2)
        # self.conv_offset2d_2 = DeformConv2d(6, 16, 10, stride=2)
        self.c2 = nn.Conv2d(6, 16, 10, 2)
        # self.c2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # offset_1 = self.offset_1(x)
        # x = F.sigmoid((self.conv_offset2d_1(x, offset_1)))
        # offset_2 = self.offset_2(x)
        # x = F.sigmoid((self.conv_offset2d_2(x, offset_2)))
        # x = F.avg_pool2d(F.sigmoid(self.c1(x)), kernel_size=(2, 2))
        # x = F.avg_pool2d(F.sigmoid(self.c2(x)), kernel_size=(2, 2))
        x = F.sigmoid(self.c1(x))
        x = F.sigmoid(self.c2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features


class LeNet_MaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        # self.offset_1 = nn.Conv2d(1, out_channels=2 * 5 * 5, kernel_size=5, padding=2)
        # self.offset_2 = nn.Conv2d(6, out_channels=2 * 10 * 10, kernel_size=10, stride=2)
        # self.conv_offset2d_1 = DeformConv2d(1, 6, 5, padding=2)
        # self.conv_offset2d_2 = DeformConv2d(6, 16, 10, stride=2)
        # self.c2 = nn.Conv2d(6, 16, 10, 2)
        self.c2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # offset_1 = self.offset_1(x)
        # x = F.sigmoid((self.conv_offset2d_1(x, offset_1)))
        # offset_2 = self.offset_2(x)
        # x = F.sigmoid((self.conv_offset2d_2(x, offset_2)))
        # x = F.avg_pool2d(F.sigmoid(self.c1(x)), kernel_size=(2, 2))
        # x = F.avg_pool2d(F.sigmoid(self.c2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.sigmoid(self.c1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.sigmoid(self.c2(x)), kernel_size=(2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features


class LeNet_AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        # self.offset_1 = nn.Conv2d(1, out_channels=2 * 5 * 5, kernel_size=5, padding=2)
        # self.offset_2 = nn.Conv2d(6, out_channels=2 * 10 * 10, kernel_size=10, stride=2)
        # self.conv_offset2d_1 = DeformConv2d(1, 6, 5, padding=2)
        # self.conv_offset2d_2 = DeformConv2d(6, 16, 10, stride=2)
        # self.c2 = nn.Conv2d(6, 16, 10, 2)
        self.c2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # offset_1 = self.offset_1(x)
        # x = F.sigmoid((self.conv_offset2d_1(x, offset_1)))
        # offset_2 = self.offset_2(x)
        # x = F.sigmoid((self.conv_offset2d_2(x, offset_2)))
        x = F.avg_pool2d(F.sigmoid(self.c1(x)), kernel_size=(2, 2))
        x = F.avg_pool2d(F.sigmoid(self.c2(x)), kernel_size=(2, 2))
        # x = F.sigmoid(self.c1(x))
        # x = F.sigmoid(self.c2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    class LeNet_AvgPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 6, 5, padding=2)
            # self.offset_1 = nn.Conv2d(1, out_channels=2 * 5 * 5, kernel_size=5, padding=2)
            # self.offset_2 = nn.Conv2d(6, out_channels=2 * 10 * 10, kernel_size=10, stride=2)
            # self.conv_offset2d_1 = DeformConv2d(1, 6, 5, padding=2)
            # self.conv_offset2d_2 = DeformConv2d(6, 16, 10, stride=2)
            # self.c2 = nn.Conv2d(6, 16, 10, 2)
            self.c2 = nn.Conv2d(6, 16, 5)

            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            # offset_1 = self.offset_1(x)
            # x = F.sigmoid((self.conv_offset2d_1(x, offset_1)))
            # offset_2 = self.offset_2(x)
            # x = F.sigmoid((self.conv_offset2d_2(x, offset_2)))
            x = F.avg_pool2d(F.sigmoid(self.c1(x)), kernel_size=(2, 2))
            x = F.avg_pool2d(F.sigmoid(self.c2(x)), kernel_size=(2, 2))
            # x = F.sigmoid(self.c1(x))
            # x = F.sigmoid(self.c2(x))
            x = x.view(-1, self.num_flat_features(x))
            x = F.sigmoid(self.fc1(x))
            x = F.sigmoid(self.fc2(x))
            x = self.fc3(x)
            return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features


class LeNet_Modified(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        # self.offset_1 = nn.Conv2d(1, out_channels=2 * 5 * 5, kernel_size=5, padding=2)
        # self.offset_2 = nn.Conv2d(6, out_channels=2 * 10 * 10, kernel_size=10, stride=2)
        # self.conv_offset2d_1 = DeformConv2d(1, 6, 5, padding=2)
        # self.conv_offset2d_2 = DeformConv2d(6, 16, 10, stride=2)
        # self.c2 = nn.Conv2d(6, 16, 10, 2)
        self.c2 = nn.Conv2d(6, 12, 2, stride=2)
        self.c3 = nn.Conv2d(12, 24, 2, 2)

        self.fc1 = nn.Linear(24 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # offset_1 = self.offset_1(x)
        # x = F.sigmoid((self.conv_offset2d_1(x, offset_1)))
        # offset_2 = self.offset_2(x)
        # x = F.sigmoid((self.conv_offset2d_2(x, offset_2)))
        x = F.sigmoid(self.c1(x))
        x = F.sigmoid(self.c2(x))
        x = F.sigmoid(self.c3(x))
        # x = F.sigmoid(self.c1(x))
        # x = F.sigmoid(self.c2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features


if __name__ == '__main__':
    epochs = 10
    BATCH_SIZE = 64
    lr = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LeNet_Modified()
    model.to(device)
    validation_data = datasets.ImageFolder(root='test_dataset', transform=transforms.Compose([
        transforms.Grayscale(), transforms.ToTensor()
    ]))
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False)
    train_data = datasets.MNIST('../dataset/mnist', train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST('../dataset/mnist', train=False, transform=transforms.ToTensor(), download=True)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    total = 0
    correct = 0

    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = correct + (predicted == labels).sum().item()
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch {}, loss {}'.format(epoch + 1, loss.item()))
        print('准确率:{}'.format(100 * correct / total))
    torch.save(model, './lenet_modified.pt')
# model = torch.load('./lenet_withoutpool.pt')
#
# model.eval()
# correct = 0
# total = 0
#
# for data in test_dataloader:
#     inputs, labels = data
#     inputs, labels = inputs.to(device), labels.to(device)
#     output = model(inputs)
#     total += labels.size(0)
#     _, predicted = torch.max(output.data, 1)
#     correct = correct + (predicted == labels).sum().item()
#
# print('10000张图像的测试准确率:{}'.format(100 * correct / total))
#
# correct = 0
# total = 0
#
# for data in validation_loader:
#     inputs, labels = data
#     inputs, labels = inputs.to(device), labels.to(device)
#     output = model(inputs)
#     total += labels.size(0)
#     _, predicted = torch.max(output.data, 1)
#     correct = correct + (predicted == labels).sum().item()
#
# print('50张图像的测试准确率:{}'.format(100 * correct / total))

# def test(folder_path):
#     total = 0
#     correct = 0
#     i = 0
#     file = open('test_label.txt', 'r')
#     file_data = file.read().splitlines()
#     for item in os.listdir(folder_path):
#         item_path = os.path.join(folder_path, item)
#         img = cv2.imread(item_path, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (28, 28))
#         img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0).unsqueeze(1)
#         img = img.to(device)
#         output = model(img)
#         total += 1
#         _, predicted = torch.max(output.data, 1)
#         if str(predicted.item()) == file_data[i]:
#             correct += 1
#         i += 1
#     print('50张图像的预测准确率:{}%'.format(100 * correct / total))
#
#
# test('test_dataset')

# def predict_Result(img):
#     """
#     预测结果，返回预测的值
#     img，numpy类型，二值图像
#     """
#     model.eval()
#     img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0).unsqueeze(1)
#     img = img.to(device)
#     out = model(img)
#     print(out)
#     _, predicted = torch.max(out.data, 1)
#     return predicted.item()
#
#
# img = cv2.imread('test_dataset_280/0_4.jpg', cv2.IMREAD_GRAYSCALE)
# # # img2 = img[0:28, 0:28]
# # # img2 = np.array(img2)
# # # print(img2.shape)
# # # print(img2.shape)
# img2 = cv2.resize(img, (28, 28))
# # print(img2.shape)
# # #
# # # img2 = np.array(img2)
# # # img2 = test_data.data[10].numpy()
# # # print(test_data.targets[10])
# # # print(test_data.test_data[25].shape)
# print('预测结果：', predict_Result(img2))
# # # # plt.imshow(img2)
# plt.imshow(img2, cmap='gray')
# plt.show()

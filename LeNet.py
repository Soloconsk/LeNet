import cv2
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
from PIL import Image


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)

        self.c2 = nn.Conv2d(6, 16, 10, 2)
        # self.c2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.sigmoid((self.c1(x)))
        x = F.sigmoid((self.c2(x)))
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


epochs = 10
BATCH_SIZE = 64
lr = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = LeNet()
model.to(device)
train_data = datasets.MNIST('../dataset/mnist', train=True, transform=transforms.ToTensor(), download=False)
new_train_data = datasets.MNIST('../dataset/mnist', train=True,
                                transform=transforms.Compose([transforms.RandomRotation(degrees=(0, 360)),
                                                             transforms.ToTensor()]), download=True)
test_data = datasets.MNIST('../dataset/mnist', train=False, transform=transforms.ToTensor(), download=True)

train_dataloader = DataLoader(new_train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

total = 0
correct = 0

# for epoch in range(epochs):
#     for i, data in enumerate(train_dataloader):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         total += labels.size(0)
#         _, predicted = torch.max(outputs.data, 1)
#         correct = correct + (predicted == labels).sum().item()
#         loss = loss_function(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print('epoch {}, loss {}'.format(epoch + 1, loss.item()))
#     print('准确率:{}'.format(100 * correct / total))
# torch.save(model, './lenet_withoutpool_randomrotate.pt')
model = torch.load('./lenet_withoutpool_randomrotate.pt')

model.eval()
correct = 0
total = 0

for data in test_dataloader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)
    total += labels.size(0)
    _, predicted = torch.max(output.data, 1)
    correct = correct + (predicted == labels).sum().item()

print('10000张图像的测试准确率:{}'.format(100 * correct / total))


def predict_Result(img):
    """
    预测结果，返回预测的值
    img，numpy类型，二值图像
    """
    model.eval()
    img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0).unsqueeze(1)
    img = img.to(device)
    out = model(img)
    print(out)
    _, predicted = torch.max(out.data, 1)
    return predicted.item()


img = cv2.imread('test_dataset/0_1.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = img[0:28, 0:28]
# img2 = np.array(img2)
# print(img2.shape)
# print(img2.shape)
img2 = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
print(img2.shape)
#
# img2 = np.array(img2)
# img2 = test_data.data[10].numpy()
# print(test_data.targets[10])
# print(test_data.test_data[25].shape)
print('预测结果：', predict_Result(img2))
# # plt.imshow(img2)
plt.imshow(img2, cmap='gray')
plt.show()

import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import torch as t
import cv2
from torch import nn
import torch.nn.functional as F


def detect(model, image):
    print("预测开始：")
    model.eval()
    wt = 'wb.pt'
    model.load_state_dict(t.load(wt))
    image = t.from_numpy(image)
    pred_labels = model(image.cuda())
    predicted = t.max(pred_labels, 1)[1].cpu()
    print(type(predicted))
    print(predicted.shape)
    num = predicted.numpy()
    print("num:", num[0])


def load_image(image_path):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.show()
    image = np.array(image)
    image = image[:, :, 0]
    a = image[0][0] - 22
    print(a)
    print(image)
    image = Image.fromarray(image)
    # image=image.convert('L')
    plt.imshow(image)
    plt.show()
    # image.show()
    threshold = a
    table = []
    for i in range(256):
        if i < threshold:
            table.append(1)
        else:
            table.append(0)
    image = image.point(table, "1")
    plt.imshow(image)
    plt.show()
    image = image.convert('L')
    image = image.resize((28, 28), Image.ANTIALIAS)
    plt.imshow(image)
    plt.show()
    image = np.array(image).reshape(1, 1, 28, 28).astype('float32')
    image = image / 255 - 0.5 / 0.5
    print(image)
    return image


def load_image1(file):
    img = cv2.imread(file)
    cv2.imshow("加载完成", img)
    cv2.waitKey(0)
    b, g, r = cv2.split(img)
    cv2.imshow("r", r)
    cv2.waitKey(0)
    threshold = 100
    table = []
    for i in range(256):
        if i < threshold:
            table.append(1)
        else:
            table.append(0)

    # 图片二值化
    img = Image.fromarray(r)
    img = img.point(table, '1')
    plt.imshow(img)
    plt.show()
    print(type(img))
    img = img.convert('L')

    # 预处理
    # 调整图像大小
    plt.imshow(img)
    plt.show()

    img = img.resize((28, 28), Image.ANTIALIAS)

    plt.imshow(img)
    plt.show()
    img = np.array(img).reshape(1, 1, 28, 28).astype('float32')
    # 归一化处理
    img = img / 255 - 0.5 / 0.5
    return img


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)

        # self.c2 = nn.Conv2d(6, 16, 10, 2)
        self.c2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.sigmoid((self.c1(x))), kernel_size=(2, 2))
        x = F.avg_pool2d(F.sigmoid((self.c2(x))), kernel_size=(2, 2))
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
    model = torch.load('lenet.pt')
    image_path = r"../1_320.jpg"
    image = load_image(image_path)
    detect(model=model, image=image)

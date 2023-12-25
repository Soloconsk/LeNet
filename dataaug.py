from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torch
import torchvision

img_path = 'test_dataset'
img_trans = transforms.RandomAffine(0, translate=(0.05, 0.1))

mydata = datasets.ImageFolder('test_dataset_28',
                              transform=transforms.Compose([img_trans, transforms.Grayscale(), transforms.ToTensor()]))

imgLoader = DataLoader(mydata, batch_size=1, shuffle=False)
for i in range(1, 10):
    j = i * 3
    count = 0
    for data in imgLoader:
        inputs, label = data
        # img_nums = inputs.shape[0]
        print(label.shape)
        # print(label.item())
        # for num in range(img_nums):
        inputs = torch.squeeze(inputs[0])
        inputs = transforms.ToPILImage()(inputs)
        inputs.save('test_dataset_28/' + str(label.item()) + '/' + str(label.item()) + '_' + str(j + 1) + '.jpg')
        j += 1
        count += 1
        if count == 3:
            j = i * 3
            count = 0


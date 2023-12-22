from LeNet import LeNet_WithoutPool
from LeNet import LeNet_AvgPool
from LeNet import LeNet_MaxPool
from LeNet import LeNet_Modified
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

model = torch.load('lenet_maxpool.pt')
validation_data = datasets.ImageFolder(root='test_dataset', transform=transforms.Compose([
    transforms.Resize(28), transforms.Grayscale(), transforms.ToTensor()
]))
validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False)

correct = 0
total = 0
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for data in validation_loader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)
    total += labels.size(0)
    _, predicted = torch.max(output.data, 1)
    correct = correct + (predicted == labels).sum().item()

print('50张图像的测试准确率:{}'.format(100 * correct / total))

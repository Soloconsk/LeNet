import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
train_data = datasets.MNIST('../dataset/mnist', train=True, transform=transforms.ToTensor(), download=False)
test_data = datasets.MNIST('../dataset/mnist', train=False, transform=transforms.ToTensor(), download=True)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
for i, data in enumerate(train_dataloader):
    inputs, _ = data
    print(inputs.size())

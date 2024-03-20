import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model.cifar10_model import cifar10net

MEAN = np.array([125.3, 123.0, 113.9]) / 255.0  # = np.array([0.49137255, 0.48235294, 0.44666667])
STD = np.array([63.0, 62.1, 66.7]) / 255.0  # = np.array([0.24705882, 0.24352941, 0.26156863])

normalize = transforms.Normalize(
    mean=MEAN,
    std=STD,
)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

train_data_path = Path.cwd() / 'cifar10/cifar10_64/train'
test_data_path = Path.cwd() / 'cifar10/cifar10_64/test'

train_dataset = ImageFolder(root=train_data_path.__str__(), transform=transform)
test_dataset = ImageFolder(root=test_data_path.__str__(), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

net = cifar10net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


num_epochs = 20

highest_accuracy = 0.0

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', ncols=100)
    for i, data in enumerate(progress_bar):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f'{running_loss / (i + 1):.4f}')

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    current_accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {current_accuracy:.2f}%')

    if current_accuracy > highest_accuracy:
        highest_accuracy = current_accuracy
        torch.save(net.state_dict(), 'best_model.pth')
        print('Saving model with highest accuracy: {:.2f}%'.format(highest_accuracy))

print('Finished Training')

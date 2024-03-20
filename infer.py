from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from model.cifar10_model import cifar10net
from pathlib import Path

net = cifar10net(initialize=False)

model_path = 'best_model.pth'
net.load_state_dict(torch.load(model_path))

net.eval()

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


def predict_image(image_path):
    image = Image.open(image_path)

    image = transform(image)

    image = image.unsqueeze(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    net.to(device)

    with torch.no_grad():
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

image_path = Path.cwd() / 'cifar10/cifar10_64/test/car/img6.png'
predicted_class = predict_image(image_path)
print(f'Predicted class: {classes[predicted_class]}')

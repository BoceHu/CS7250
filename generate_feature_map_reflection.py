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
from model.resnet import ResNet, BasicBlock
import matplotlib.pyplot as plt
from rotate_imgs import rotate_image


def get_feature_map(output, N=4):
    output_sum = torch.sum(output, dim=(1, 2))
    valid_channels = output_sum != 0
    valid_channels = valid_channels.nonzero()
    first_non_zero_channel_idx = valid_channels[0]
    low = first_non_zero_channel_idx // N
    high = low + N
    selected_slice = output[low:high]
    equi_feature_map = torch.mean(selected_slice, dim=0)
    return equi_feature_map

# we want this number to be twice as large is it is in the GFM.py file
# making this 90 gives us reflections of [0, 45, 135, 180]. This is good.
# Note: ref(p) = ref(p + 180), so [0, 45, 135, 180] = [180, 225, 270, 315]
minimum_angle_of_reflection = 45 * 2  
angles = range(0, 360, minimum_angle_of_reflection)

MEAN = np.array([125.3, 123.0, 113.9]) / 255.0
STD = np.array([63.0, 62.1, 66.7]) / 255.0

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
feature_map_path = Path.cwd() / 'feature_map'

train_dataset = ImageFolder(root=train_data_path.__str__(), transform=transform)
test_dataset = ImageFolder(root=test_data_path.__str__(), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

eqnet = cifar10net(N=2, flip=True, initialize=False)
net = ResNet(BasicBlock, [2, 2, 2, 2])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
eqnet.to(device)
net.eval()
eqnet.eval()

eqnet.load_state_dict(torch.load('best_model_D2.pth'))
net.load_state_dict(torch.load('best_model_CNN.pth'))

counter = 0

for data in tqdm(train_loader):

    # gets the reflected images
    inputs, label = data[0].to(device), data[1].to(device)
    # rot(t) + ref(p) = ref(p + t/2)
    reflected_images = [torch.flip(rotate_image(inputs[0], angle), [1]) for angle in angles]
    inputs = torch.stack(reflected_images)

    outputs, resnet_output = net(inputs)
    eq_outputs, eq_resnet_output = eqnet(inputs)

    store_dict = dict()
    store_dict['img'] = inputs[0]
    store_dict['eq'] = dict()
    store_dict['CNN'] = dict()
    for i, angle in enumerate(angles):
        store_dict['eq'][angle // 2] = dict()
        store_dict['CNN'][angle // 2] = dict()
        # terminal eq
        store_dict['eq'][angle // 2]["map"] = get_feature_map(eq_resnet_output[1][i].tensor.squeeze(0), N=4)
        # terminal trad
        store_dict['CNN'][angle // 2]["map"] = get_feature_map(resnet_output[1][i].squeeze(0), N=1)

    torch.save(store_dict, feature_map_path / str(label[0].item()) / f"{str(counter)}.pth")
    counter += 1

for data in tqdm(test_loader):
    
    # gets the reflected images
    inputs, label = data[0].to(device), data[1].to(device)
    # rot(t) + ref(p) = ref(p + t/2)
    reflected_images = [torch.flip(rotate_image(inputs[0], angle), [1]) for angle in angles]
    inputs = torch.stack(reflected_images)

    outputs, resnet_output = net(inputs)
    eq_outputs, eq_resnet_output = eqnet(inputs)

    store_dict = dict()
    store_dict['img'] = inputs[0]
    store_dict['eq'] = dict()
    store_dict['CNN'] = dict()
    for i, angle in enumerate(angles):
        store_dict['eq'][angle // 2] = dict()
        store_dict['CNN'][angle // 2] = dict()
        # terminal eq
        store_dict['eq'][angle // 2]["map"] = get_feature_map(eq_resnet_output[1][i].tensor.squeeze(0), N=4)
        # terminal trad
        store_dict['CNN'][angle // 2]["map"] = get_feature_map(resnet_output[1][i].squeeze(0), N=1)

    torch.save(store_dict, feature_map_path / str(label[0].item()) / f"{str(counter)}.pth")
    counter += 1

print('Finished Generating Feature Maps')

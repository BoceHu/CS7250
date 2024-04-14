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


def get_feature_map(output, N=2):
    output_sum = torch.sum(output, dim=(1, 2))
    valid_channels = output_sum != 0
    valid_channels = valid_channels.nonzero()
    first_non_zero_channel_idx = valid_channels[0]
    low = first_non_zero_channel_idx // N
    high = low + N
    selected_slice = output[low:high]
    equi_feature_map = torch.mean(selected_slice, dim=0)
    return equi_feature_map


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

eqnet = cifar10net(N=4, flip=True, initialize=False)
net = ResNet(BasicBlock, [2, 2, 2, 2])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
eqnet.to(device)
net.eval()
eqnet.eval()

eqnet.load_state_dict(torch.load('best_model_D4.pth'))
net.load_state_dict(torch.load('best_model_CNN.pth'))

counter = 0

for data in tqdm(train_loader):
    inputs, label = data[0].to(device), data[1].to(device)
    C, H, W = inputs.shape[1], inputs.shape[2], inputs.shape[3]
    flip_h = torch.flip(inputs[0], [2]).view(1, C, H, W)
    flip_v = torch.flip(inputs[0], [1]).view(1, C, H, W)
    inputs = torch.vstack([inputs, flip_h, flip_v])

    outputs, resnet_output = net(inputs)
    eq_outputs, eq_resnet_output = eqnet(inputs)

    store_dict = dict()
    store_dict['img'] = inputs[0]
    store_dict['eq'] = dict()
    store_dict['eq']['original'] = dict()
    store_dict['eq']['flip_h'] = dict()
    store_dict['eq']['flip_v'] = dict()

    store_dict['CNN'] = dict()
    store_dict['CNN']['original'] = dict()
    store_dict['CNN']['flip_h'] = dict()
    store_dict['CNN']['flip_v'] = dict()

    flip_types = ['original', 'flip_h', 'flip_v']

    for i in range(inputs.shape[0]):
        flip_type = flip_types[i]

        for num, intermediate_output in enumerate(resnet_output):
            eq_feature_map = get_feature_map(eq_resnet_output[num][i].tensor.squeeze(0), N=8)
            store_dict['eq'][flip_type]["x" + str(num)] = eq_feature_map
            feature_map = get_feature_map(resnet_output[num][i].squeeze(0), N=1)
            store_dict['CNN'][flip_type]["x" + str(num)] = feature_map

    torch.save(store_dict, feature_map_path / str(label[0].item()) / f"{str(counter)}.pth")
    counter += 1

for data in tqdm(test_loader):
    images, label = data[0].to(device), data[1].to(device)
    C, H, W = images.shape[1], images.shape[2], images.shape[3]
    flip_h = torch.flip(images[0], [2]).view(1, C, H, W)
    flip_v = torch.flip(images[0], [1]).view(1, C, H, W)
    inputs = torch.vstack([images, flip_h, flip_v])

    outputs, resnet_output = net(inputs)
    eq_outputs, eq_resnet_output = eqnet(inputs)

    store_dict = dict()
    store_dict['img'] = inputs[0]
    store_dict['eq'] = dict()
    store_dict['eq']['original'] = dict()
    store_dict['eq']['flip_h'] = dict()
    store_dict['eq']['flip_v'] = dict()

    store_dict['CNN'] = dict()
    store_dict['CNN']['original'] = dict()
    store_dict['CNN']['flip_h'] = dict()
    store_dict['CNN']['flip_v'] = dict()

    flip_types = ['original', 'flip_h', 'flip_v']

    for i in range(inputs.shape[0]):
        flip_type = flip_types[i]

        for num, intermediate_output in enumerate(resnet_output):
            eq_feature_map = get_feature_map(eq_resnet_output[num][i].tensor.squeeze(0), N=8)
            store_dict['eq'][flip_type]["x" + str(num)] = eq_feature_map
            feature_map = get_feature_map(resnet_output[num][i].squeeze(0), N=1)
            store_dict['CNN'][flip_type]["x" + str(num)] = feature_map

    torch.save(store_dict, feature_map_path / str(label[0].item()) / f"{str(counter)}.pth")
    counter += 1

print('Finished Generating Feature Maps')

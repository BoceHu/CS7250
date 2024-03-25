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

    # input C x H x W
    output_sum = torch.sum(output, dim=(1, 2))
    # get first non zero channel
    valid_channels = output_sum != 0
    valid_channels = valid_channels.nonzero()
    first_non_zero_channel_idx = valid_channels[0]
    low = first_non_zero_channel_idx // N
    high = low + N
    # pooling to generate equivariant feature map
    selected_slice = output[low:high]
    equi_feature_map = torch.mean(selected_slice, dim=0)
    
    return equi_feature_map


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

# feature maps
feature_map_path = Path.cwd() / 'feature_map'

train_dataset = ImageFolder(root=train_data_path.__str__(), transform=transform)
test_dataset = ImageFolder(root=test_data_path.__str__(), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# equivariant
eqnet = cifar10net(N=2)
# regular CNN
net = ResNet(BasicBlock, [2, 2, 2, 2])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
eqnet.to(device)
net.eval()
eqnet.eval()

# equivariant
eqnet.load_state_dict(torch.load('best_model_C2.pth'))
# regular CNN
net.load_state_dict(torch.load('best_model_CNN.pth'))

counter = 0

for data in tqdm(train_loader):

    inputs, label = data[0].to(device), data[1].to(device)

    C, H, W = inputs.shape[1], inputs.shape[2], inputs.shape[3]
    rot_90 = torch.rot90(inputs[0], k=1, dims=(1, 2)).view(1, C, H, W)
    rot_180 = torch.rot90(inputs[0], k=2, dims=(1, 2)).view(1, C, H, W)
    rot_270 = torch.rot90(inputs[0], k=3, dims=(1, 2)).view(1, C, H, W)
    inputs = torch.vstack([inputs, rot_90, rot_180, rot_270, inputs])
    
    outputs, resnet_output = net(inputs)
    eq_outputs, eq_resnet_output = eqnet(inputs)
    
    # store feature maps
    '''
    dict format:{
        img: tensor
        eq:
            0:
                x0: tensor
                x1: tensor
                ...
                x4: tensor
            90:
                x0: tensor
                ...
                x4: tensor
            180:
                ...
            270:
                ...
            360:
                ...
        CNN:
            0:
                x0: tensor
                x1: tensor
                ...
                x4: tensor
            90:
                x0: tensor
                ...
                x4: tensor
            180:
                ...
            270:
                ...
            360:
                ...
    }
    '''
    store_dict = dict()
    store_dict['img'] = inputs[0]
    store_dict['eq'] = dict()
    store_dict['eq'][0] = dict()
    store_dict['eq'][90] = dict()
    store_dict['eq'][180] = dict()
    store_dict['eq'][270] = dict()
    store_dict['eq'][360] = dict()
    store_dict['CNN'] = dict()
    store_dict['CNN'][0] = dict()
    store_dict['CNN'][90] = dict()
    store_dict['CNN'][180] = dict()
    store_dict['CNN'][270] = dict()
    store_dict['CNN'][360] = dict()

    for i in range(inputs.shape[0]):

        rot = 90 * i

        for num, intermediate_output in enumerate(resnet_output):
            # avg pooling for equi feature map
            eq_feature_map = get_feature_map(eq_resnet_output[num][i].tensor.squeeze(0))
            store_dict['eq'][rot]["x"+str(num)] = eq_feature_map
            # get regular feature map
            feature_map = get_feature_map(resnet_output[num][i].squeeze(0), N=1)
            store_dict['CNN'][rot]["x"+str(num)] = feature_map

    torch.save(store_dict, feature_map_path / str(label[0].item()) / f"{str(counter)}.pth")
            
    counter += 1

for data in tqdm(test_loader):

    images, label = data[0].to(device), data[1].to(device)

    C, H, W = inputs.shape[1], inputs.shape[2], inputs.shape[3]
    rot_90 = torch.rot90(inputs[0], k=1, dims=(1, 2)).view(1, C, H, W)
    rot_180 = torch.rot90(inputs[0], k=2, dims=(1, 2)).view(1, C, H, W)
    rot_270 = torch.rot90(inputs[0], k=3, dims=(1, 2)).view(1, C, H, W)
    inputs = torch.vstack([inputs, rot_90, rot_180, rot_270, inputs])

    outputs, resnet_output = net(images)
    eq_outputs, eq_resnet_output = eqnet(inputs)

    # store feature maps
    store_dict = dict()
    store_dict['img'] = inputs[0]
    store_dict['eq'] = dict()
    store_dict['eq'][0] = dict()
    store_dict['eq'][90] = dict()
    store_dict['eq'][180] = dict()
    store_dict['eq'][360] = dict()
    store_dict['CNN'] = dict()
    store_dict['CNN'][0] = dict()
    store_dict['CNN'][90] = dict()
    store_dict['CNN'][180] = dict()
    store_dict['CNN'][270] = dict()
    store_dict['CNN'][360] = dict()

    for i in range(inputs.shape[0]):

        rot = 90 * i

        for num, intermediate_output in enumerate(resnet_output):
            # avg pooling for equi feature map
            eq_feature_map = get_feature_map(eq_resnet_output[num][i].tensor.squeeze(0))
            store_dict['eq'][rot]["x"+str(num)] = eq_feature_map
            # get regular feature map
            feature_map = get_feature_map(resnet_output[num][i].squeeze(0), N=1)
            store_dict['CNN'][rot]["x"+str(num)] = feature_map

    torch.save(store_dict, feature_map_path / str(label[0].item()) / f"{str(counter)}.pth")

    counter += 1

print('Finished Generating Feature Maps')

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


def get_feature_map(output, N=8):

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

minimum_angle_of_rotation = 45
angles = range(0, 360, minimum_angle_of_rotation)


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

# train_data_path = Path.cwd() / 'cifar10/cifar10_64/train'
test_data_path = Path.cwd() / 'cifar10/cifar10_64/test'

# feature maps
feature_map_path = Path.cwd() / 'feature_map'

train_dataset = ImageFolder(root=train_data_path.__str__(), transform=transform)
test_dataset = ImageFolder(root=test_data_path.__str__(), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# equivariant
eqnet = cifar10net(N=4, flip=True, initialize=False)
# regular CNN
net = ResNet(BasicBlock, [2, 2, 2, 2])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
eqnet.to(device)
net.eval()
eqnet.eval()

# equivariant
eqnet.load_state_dict(torch.load('best_model_D4.pth'))
# regular CNN
net.load_state_dict(torch.load('best_model_CNN.pth'))

counter = 0

for data in tqdm(train_loader):

    # gets the rotated images
    inputs, label = data[0].to(device), data[1].to(device)
    rotated_images = [rotate_image(inputs[0], angle) for angle in angles]
    inputs = torch.stack(rotated_images)
    
    # gets intermediate results for rotated images
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
    store_dict['CNN'] = dict()
    for i, angle in enumerate(angles):
        store_dict['eq'][angle] = dict()
        store_dict['CNN'][angle] = dict()
        # gets terminal GCNN feature map
        store_dict['eq'][angle]["map"] = get_feature_map(eq_resnet_output[1][i].tensor.squeeze(0))
        # gets terminal CNN feature map
        store_dict['CNN'][angle]["map"] = get_feature_map(resnet_output[1][i].squeeze(0), N=1)

    torch.save(store_dict, feature_map_path / str(label[0].item()) / f"{str(counter)}.pth")
            
    counter += 1

for data in tqdm(test_loader):

    # gets the rotated images
    inputs, label = data[0].to(device), data[1].to(device)
    rotated_images = [rotate_image(inputs[0], angle) for angle in angles]
    inputs = torch.stack(rotated_images)
    
    # gets intermediate results for rotated images
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
    store_dict['CNN'] = dict()
    for i, angle in enumerate(angles):
        store_dict['eq'][angle] = dict()
        store_dict['CNN'][angle] = dict()
        # gets terminal GCNN feature map
        store_dict['eq'][angle]["map"] = get_feature_map(eq_resnet_output[1][i].tensor.squeeze(0))
        # gets terminal CNN feature map
        store_dict['CNN'][angle]["map"] = get_feature_map(resnet_output[1][i].squeeze(0), N=1)

    torch.save(store_dict, feature_map_path / str(label[0].item()) / f"{str(counter)}.pth")

    counter += 1

print('Finished Generating Feature Maps')

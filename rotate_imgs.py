import torch
import torchvision.transforms.functional as TF

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

"""
rotates anti-clockwise (degrees) and masks out pixels that lie 
outside the inscribing circle
"""
def rotate_image(image, angle):

    # rotates image using built in torch function
    rotated_image = TF.rotate(image, angle)

    # sets all pixels' values to be 0 that lie outside of the inscribing circle
    # C, H, W = image.shape
    # center_x, center_y = W // 2, H // 2
    # radius = min(center_x, center_y)
    
    # # determines which indicies lie outside circle
    # Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    # distance_from_center = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # # defines our mask
    # mask = distance_from_center <= radius
    # mask = mask.unsqueeze(0).repeat(C, 1, 1)
    
    # # masks out everything outside circle
    # rotated_image[~mask] = 0

    return rotated_image

"""
performs circle mask on single channel feature maps
"""
def circle_mask(image):

    # sets all pixels' values to be 0 that lie outside of the inscribing circle
    H, W = image.shape
    center_x, center_y = W // 2, H // 2
    radius = min(center_x, center_y)
    
    # determines which indicies lie outside circle
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    distance_from_center = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # defines our mask
    mask = distance_from_center < radius
    
    # masks out everything outside circle
    image = image.clone()
    image[~mask] = 0

    return image

# def eval_mask(image):

#     # sets all pixels' values to be 0 that lie outside of the inscribing circle
#     H, W = image.shape
#     center_x, center_y = W // 2, H // 2
#     radius = min(center_x, center_y)
    
#     # determines which indicies lie outside circle
#     Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
#     distance_from_center = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
#     # defines our mask
#     mask = distance_from_center <= radius
    
#     # masks out everything outside circle
#     new_image = image.flatten()[mask.flatten()]

#     return new_image

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

if __name__ == '__main__':

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


    test_data_path = Path.cwd() / 'cifar10/cifar10_64/test'
    test_dataset = ImageFolder(root=test_data_path.__str__(), transform=transform)
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

    for i, data in tqdm(enumerate(test_loader)):

        if i == 700:
            # gets the rotated images
            inputs, label = data[0].to(device), data[1].to(device)
            rotated_images = [rotate_image(inputs[0], angle) for angle in angles]
            inputs = torch.stack(rotated_images)
            
            # gets intermediate results for rotated images
            outputs, resnet_output = net(inputs)
            eq_outputs, eq_resnet_output = eqnet(inputs)

            for i, angle in enumerate(angles):
                eq_map = get_feature_map(eq_resnet_output[1][i].tensor.squeeze(0))

                map = get_feature_map(resnet_output[1][i].squeeze(0), N=1)

                plt.figure(f"input for {angle} degrees" )
                plt.imshow(torch.hstack((inputs[0], inputs[i])).permute(1,2,0).cpu().numpy())
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.suptitle(f"maps for {angle} degres")
                ax1.imshow(eq_map.detach().cpu().numpy())
                ax2.imshow(map.detach().cpu().numpy())
                plt.show()



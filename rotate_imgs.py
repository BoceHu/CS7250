import torch
import torchvision.transforms.functional as TF

"""
rotates anti-clockwise (degrees) and masks out pixels that lie 
outside the inscribing circle
"""
def rotate_image(image, angle):

    # rotates image using built in torch function
    rotated_image = TF.rotate(image, angle)

    return rotated_image

def circle_mask(image):
    
    # sets all pixels' values to be 0 that lie outside of the inscribing circle
    C, H, W = image.shape
    center_x, center_y = W // 2, H // 2
    radius = min(center_x, center_y)
    
    # determines which indicies lie outside circle
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    distance_from_center = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # defines our mask
    mask = distance_from_center <= radius
    mask = mask.unsqueeze(0).repeat(C, 1, 1)
    
    # masks out everything outside circle
    image[~mask] = 0

    return image



if __name__ == '__main__':
    None


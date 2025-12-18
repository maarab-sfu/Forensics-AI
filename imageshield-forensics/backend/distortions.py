import torch
import torchvision
import torch.nn as nn
import itertools
from skimage import exposure
from numpy.random import default_rng
import numpy as np
import config as c
import torch.nn.functional as F
from kornia.filters import MedianBlur, GaussianBlur2d, BoxBlur
import string

import functools
import torch.utils.data as data

from PIL import Image, ImageEnhance
import os
import os.path

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_

from torchvision import models
import torchvision.transforms as T

import math
import cv2

from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms.functional as F_t
from natsort import natsorted
import glob
import random
import skimage

import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



from tqdm import tqdm
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from itertools import chain

from itertools import cycle, compress


from math import exp
from DiffJPEG import DiffJPEG
from torch.nn.functional import conv2d

from inpainting import Generator

# from diff_jpeg import DiffJPEGCoding


IMAGE_COUNTER = 0

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, crop_ratio):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
    This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
    (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
    a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
    that we crop from top/bottom with equal probability.
    The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
    :param image: The image we want to crop
    :param height_ratio_range: The range of remaining height ratio
    :param width_ratio_range:  The range of remaining width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(np.rint(crop_ratio * image_height))
    remaining_width = int(np.rint(crop_ratio * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, noised_and_cover):
        return noised_and_cover

class GaussianNoise(nn.Module):
    def __init__(self, Standard_deviation):
        super(GaussianNoise, self).__init__()
        self.std = Standard_deviation

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noise = torch.randn_like(noised_image) * self.std  # Generate Gaussian noise
        noisy_result = noised_image + noise  # Add noise to the image
        noisy_result = torch.clamp(noisy_result, 0, 1)
        # Find min and max values
        # min_value = noisy_result.min()
        # max_value = noisy_result.max()

        # print(f"Min: {min_value}, Max: {max_value}")
        noised_and_cover[0] = noisy_result
        return noised_and_cover
    #     self.Standard_deviation = Standard_deviation

    # def forward(self, noised_and_cover):
    #     noised_image = noised_and_cover[0]
    #     batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy()
    #     batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
    #     for idx in range(batch_encoded_image.shape[0]):
    #         encoded_image = batch_encoded_image[idx]
    #         noise_image = skimage.util.random_noise(encoded_image, mode= 'gaussian',clip = False, var = (self.Standard_deviation) ** 2 )
    #         noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
    #         if (idx == 0):
    #             batch_noise_image = noise_image.unsqueeze(0)
    #         else:
    #             batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
    #     batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
    #     noised_and_cover[0] = 2*batch_noise_image - 1
    #     return noised_and_cover


class SaltPepper(nn.Module):
    def __init__(self,Amount):
        super(SaltPepper, self).__init__()
        self.Amount = Amount

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        batch_encoded_image = noised_image.cpu().detach().numpy()
        batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(batch_encoded_image.shape[0]):
            encoded_image = batch_encoded_image[idx]
            noise_image = skimage.util.random_noise(encoded_image, mode='s&p', amount = self.Amount)
            noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = batch_noise_image
        return noised_and_cover



class GaussianBlur(nn.Module):
    def __init__(self, sigma = 1.5):
        super(GaussianBlur, self).__init__()
        self.sigma = sigma
        self.gaussian_filters = {
			1: GaussianBlur2d((3,3), (1/4, 1/4)),
		    1.5: GaussianBlur2d((3,3), (1/5, 1/5)),
		    3: GaussianBlur2d((3,3), (1/6, 1/6)),
		}

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        blur_result = self.gaussian_filters[self.sigma](noised_image)
        noised_and_cover[0] = blur_result
        return noised_and_cover


class MedianFilter(nn.Module):
    def __init__(self, kernel = 7):
        super(MedianFilter, self).__init__()
        self.kernel = kernel
        self.median_filters = {
			13: MedianBlur((1, 3)),
		    31: MedianBlur((3, 1)),
		    33: MedianBlur((3, 3)),
		}

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        blur_result = self.median_filters[self.kernel](noised_image)
        noised_and_cover[0] = blur_result
        return noised_and_cover

class BoxFilter(nn.Module):
    def __init__(self, kernel = 7):
        super(BoxFilter, self).__init__()
        self.kernel = kernel
        self.box_filters = {
			13: BoxBlur((1, 3)),
		    31: BoxBlur((3, 1)),
		    33: BoxBlur((3, 3)),
		}

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        blur_result = self.box_filters[self.kernel](noised_image)
        noised_and_cover[0] = blur_result
        return noised_and_cover

class AverageFilter(nn.Module):
    def __init__(self, kernel = 5):
        super(AverageFilter, self).__init__()
        self.kernel = kernel

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(batch_encoded_image.shape[0]):
            encoded_image = batch_encoded_image[idx]
            noise_image = cv2.blur(encoded_image, (self.kernel, self.kernel))
            noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)/255
        return noised_and_cover




class DropOut(nn.Module):
    def __init__(self, prob):
        super(DropOut, self).__init__()
        self.prob = prob

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        
        mask_percent = np.random.uniform(self.prob, 1)
        mask = np.random.choice([0.0, 1.0], noised_image.shape[2:], p=[1 - mask_percent, mask_percent])
        mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)
        mask_tensor = mask_tensor.expand_as(noised_image)
        noised_image = noised_image * mask_tensor + cover_image * (1-mask_tensor)
        noised_and_cover[0] = noised_image
 
        return noised_and_cover



class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, crop_ratio):
        """
        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.crop_ratio = crop_ratio


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)
        crop_mask = torch.zeros_like(noised_image)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, self.crop_ratio)
        crop_mask[:, :, h_start:h_end, w_start:w_end] = 1
        noised_and_cover[0] = noised_image * crop_mask
        return noised_and_cover

class CropOut(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        super(CropOut, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range
        self.i = 0


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]

        cropout_mask = torch.zeros_like(noised_image)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=noised_image,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

        # noised = noised_image * (1-cropout_mask) + cover_image * cropout_mask
        noised_and_cover[0] = noised_image * (1-cropout_mask) + 0 * cropout_mask
        torchvision.utils.save_image(cropout_mask, c.IMAGE_BINMAP_PATH + '%.5d.png' % self.i)
        self.i = self.i + 1
        return  noised_and_cover

def generate_blob_mask(height, width, min_area=0.1, max_area=0.30):
    # read input image
    img = np.zeros((height,width,3), np.uint8)

    # change the pattern
    rng = default_rng()

    # create random noise image
    noise = rng.integers(0, 255, (height,width), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)

    # calculate minimum and maximum areas
    min_area_px = int(min_area * height * width)
    max_area_px = int(max_area * height * width)

    # dynamically adjust the threshold to meet area requirements
    for threshold in range(0, 255):
        thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY)[1]
        area = np.count_nonzero(thresh)
        if min_area_px <= area <= max_area_px:
            break

    # apply morphology open and close to smooth out and make 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask,mask,mask])
    mask = np.transpose(mask, (2,0,1))
    return mask

def random_flip(mask):
    """Randomly flips the mask horizontally or vertically and ensures contiguous memory."""
    if random.random() < 0.5:
        mask = np.flip(mask, axis=2).copy()  # Horizontal Flip (Make Copy)
    if random.random() < 0.5:
        mask = np.flip(mask, axis=1).copy()  # Vertical Flip (Make Copy)
    return mask

def random_rotate(mask):
    """Randomly rotates the mask by 90, 180, or 270 degrees and ensures contiguous memory."""
    rotations = [0, 1, 2, 3]  # 0° (original), 90°, 180°, 270°
    k = random.choice(rotations)
    return np.rot90(mask, k, axes=(1, 2)).copy()  # Ensure contiguous memory

def elastic_transform(mask, alpha=30, sigma=5):
    """Applies elastic deformation to the mask."""
    random_state = np.random.RandomState(None)
    shape = mask.shape[1:]

    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    transformed_mask = np.stack([
        cv2.remap(mask[c], map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        for c in range(mask.shape[0])
    ])
    
    return transformed_mask.copy()  # Ensure contiguous memory

def random_dilate_erode(mask):
    """Randomly applies dilation or erosion to the mask and ensures contiguous memory."""
    kernel = np.ones((3, 3), np.uint8)
    if random.random() < 0.5:
        mask = np.stack([cv2.dilate(mask[c], kernel, iterations=1) for c in range(mask.shape[0])])
    else:
        mask = np.stack([cv2.erode(mask[c], kernel, iterations=1) for c in range(mask.shape[0])])
    return mask.copy()  # Ensure contiguous memory

def load_random_mask(target_height, target_width, mask_folder = "C:/Users/maarab/Forensics/Datasets/masks"):
    """Loads a random mask from a folder and applies random augmentations."""
    
    # Get list of mask files
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not mask_files:
        raise ValueError("No mask images found in the folder!")

    # Select a random mask
    mask_path = os.path.join(mask_folder, random.choice(mask_files))
    
    # Read the mask in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Failed to read mask file: {mask_path}")

    # Resize to match target dimensions
    mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # Convert to binary mask (ensure values are 0 or 255)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Convert to 3-channel format
    mask = np.stack([mask] * 3, axis=0)

    # Apply a random augmentation
    augmentations = [random_flip, random_rotate, elastic_transform, random_dilate_erode]
    mask = random.choice(augmentations)(mask)

    return mask.copy()  # Ensure contiguous memory



class IdentityAttack(nn.Module):
    def __init__(self):
        super(IdentityAttack, self).__init__()
    def forward(self, noised_and_cover):
        noised = noised_and_cover[0]
        # print(noised.shape)
        
        zero_mask = torch.zeros((noised.shape)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # return [noised, zero_mask]
        return noised_and_cover, zero_mask

class Splicing(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, cover, phase, device):
        super(Splicing, self).__init__()
        self.phase = phase
        self.cover = cover
        self.device = device


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0].clone()
        batch_size, channels, height, width = noised_image.shape
        mask = torch.zeros((noised_image.shape))
        # print("height and width: ", height, width)
        for i in range(batch_size):
            image = noised_image[i]
            # print("image shape: ", image.shape)
            
            # Generate a random mask with diverse shapes
            if c.USE_HUMAN_MASKS == 1:
                tampered_mask = torch.from_numpy(load_random_mask(height, width))
            elif c.USE_HUMAN_MASKS == 0:
                tampered_mask = torch.from_numpy(generate_blob_mask(height, width))  # Match mask size to copied region
            else:
                if random.choice([True, False]):
                    tampered_mask = torch.from_numpy(load_random_mask(height, width))
                else:
                    tampered_mask = torch.from_numpy(generate_blob_mask(height, width))

            # print(tampered_mask)
            tampered_image = torch.clone(image)
            tampered_image[tampered_mask > 0] = self.cover[tampered_mask > 0]
                    
            noised_image[i] = tampered_image
            mask[i,:,:,:] = tampered_mask.to(torch.float32).to(self.device)

        noised_and_cover[0] = noised_image
        # print(mask)
        return noised_and_cover, mask

class CopyMove(nn.Module):
    """
    Combines the noised and cover images into a single image using copy-move operation.
    """
    def __init__(self, phase, device):
        super(CopyMove, self).__init__()
        self.phase = phase
        self.device = device

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0].clone()
        batch_size, channels, height, width = noised_image.shape
        mask = torch.zeros((noised_image.shape))
        
        for i in range(batch_size):
            image = noised_image[i]
            # print("image shape: ", image.shape)
            tiled_image = torch.zeros(channels, 2 * height, 2 * width)
            # Tile the original image in the 2x2 grid
            tiled_image[:,0:height, 0:width] = image
            tiled_image[:,0:height, width:2*width] = image
            tiled_image[:,height:2*height, 0:width] = image
            tiled_image[:,height:2*height, width:2*width] = image

            start_x = random.randint(0, width)
            start_y = random.randint(0, height)

            # Extract an image with the size of the original image from the tiled image
            extracted_image = tiled_image[:,start_y:start_y + height, start_x:start_x + width].cuda()

            # Generate a random mask with diverse shapes
            # Generate a random mask with diverse shapes
            if c.USE_HUMAN_MASKS == 1:
                tampered_mask = torch.from_numpy(load_random_mask(height, width))
            elif c.USE_HUMAN_MASKS == 0:
                tampered_mask = torch.from_numpy(generate_blob_mask(height, width))  # Match mask size to copied region
            else:
                if random.choice([True, False]):
                    tampered_mask = torch.from_numpy(load_random_mask(height, width))
                else:
                    tampered_mask = torch.from_numpy(generate_blob_mask(height, width))
            # print(tampered_mask)

            tampered_image = torch.clone(image)
            tampered_image[tampered_mask > 0] = extracted_image[tampered_mask > 0]
  
            noised_image[i] = tampered_image
            mask[i,:,:,:] = tampered_mask.to(torch.float32).to(self.device)

        noised_and_cover[0] = noised_image
        # print(mask)
        return noised_and_cover, mask

def DeepFillV2(generator, image, mask):
    output = generator.infer(image, mask, return_vals=['inpainted'])
    return output

class Inpainting(nn.Module):
    """
    Combines the noised and cover images into a single image using copy-move operation.
    """
    def __init__(self, generator, phase, device):
        super(Inpainting, self).__init__()
        self.generator = generator
        self.phase = phase
        self.device = device

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0].clone()
        mask = torch.zeros((noised_image.shape))
        
        batch_size, channels, height, width = noised_image.shape
        
        for i in range(batch_size):
            image = noised_image[i]
            # print("image shape: ", image.shape)
            tiled_image = torch.zeros(channels, 2 * height, 2 * width)
            # Tile the original image in the 2x2 grid
            tiled_image[:,0:height, 0:width] = image
            tiled_image[:,0:height, width:2*width] = image
            tiled_image[:,height:2*height, 0:width] = image
            tiled_image[:,height:2*height, width:2*width] = image

            start_x = random.randint(0, width)
            start_y = random.randint(0, height)

            # Extract an image with the size of the original image from the tiled image
            extracted_image = tiled_image[:,start_y:start_y + height, start_x:start_x + width].cuda()

            # Generate a random mask with diverse shapes
            # Generate a random mask with diverse shapes
            if c.USE_HUMAN_MASKS == 1:
                tampered_mask = torch.from_numpy(load_random_mask(height, width)).to(self.device)
            elif c.USE_HUMAN_MASKS == 0:
                tampered_mask = torch.from_numpy(generate_blob_mask(height, width)).to(self.device)  # Match mask size to copied region
            else:
                if random.choice([True, False]):
                    tampered_mask = torch.from_numpy(load_random_mask(height, width)).to(self.device)
                else:
                    tampered_mask = torch.from_numpy(generate_blob_mask(height, width)).to(self.device)
            # print(tampered_mask)
            tampered_image = DeepFillV2(self.generator, image, tampered_mask)/255.0

            noised_image[i] = torch.from_numpy(np.transpose(tampered_image, (2,0,1))).to(self.device)
            mask[i,:,:,:] = tampered_mask.to(torch.float32).to(self.device)
    

        noised_and_cover[0] = noised_image

        # print(noised.shape)
        # print(mask)
        return noised_and_cover, mask
        # return noised

class ObjectAddition(nn.Module):
    def __init__(self, device, object_dir):
        super(ObjectAddition, self).__init__()
        self.device = device
        self.object_dir = object_dir
        self.transform = ToTensor()

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0].clone()
        # print(noised_image.shape)
        # noised_image = noised.clone()
        mask_image = torch.zeros((noised_image.shape)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # print(noised_image.shape)
        _, channels, height, width = noised_image.shape

        # Randomly select an object image
        object_files = [f for f in os.listdir(self.object_dir) if f.endswith('.png')]
        if not object_files:
            return noised_image, mask_image  # Return unchanged if no objects are found
        
        obj_path = os.path.join(self.object_dir, random.choice(object_files))
        obj_image = Image.open(obj_path).convert("RGBA")

        # Resize object so that it covers at most 1/4 of the image
        max_size = min(height, width) // 2
        obj_w, obj_h = obj_image.size
        scale_factor = min(max_size / obj_w, max_size / obj_h)
        new_size = (int(obj_w * scale_factor), int(obj_h * scale_factor))
        obj_image = obj_image.resize(new_size, Image.ANTIALIAS)

        # Convert object to tensor
        obj_tensor = self.transform(obj_image)  # Shape: (4, H, W) (RGBA)
        obj_tensor = obj_tensor.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        # Generate a random position for placing the object
        max_x, max_y = width - new_size[0], height - new_size[1]
        start_x, start_y = random.randint(0, max_x), random.randint(0, max_y)

        # Extract object RGB and Alpha channel
        obj_rgb = obj_tensor[:3]  # Shape: (3, H, W)
        obj_alpha = obj_tensor[3].unsqueeze(0)  # Shape: (1, H, W)

        # print(obj_rgb.shape)
        # print(obj_alpha.shape)

        
        
        # Place the object onto the image using alpha blending
        end_x, end_y = start_x + new_size[0], start_y + new_size[1]
        # print(noised_image[:,:, start_y:end_y, start_x:end_x].shape)
        noised_image[:,:, start_y:end_y, start_x:end_x] = (
            noised_image[:,:, start_y:end_y, start_x:end_x] * (1 - obj_alpha) + obj_rgb * obj_alpha
        )

        # Update the mask (mark new edited area with 0s)
        # print("obj_alpha min:", obj_alpha.min().item(), "obj_alpha max:", obj_alpha.max().item())

        mask_image[:,:, start_y:end_y, start_x:end_x] = obj_alpha * 255
        
        noised_and_cover[0] = noised_image

        return noised_and_cover, mask_image




class PixelElimination(nn.Module):
    def __init__(self, pixel_ratio):
        super(PixelElimination, self).__init__()
        self.pixel_ratio = pixel_ratio

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        _,_,H,W = noised_image.shape

        elimination_mask = torch.ones_like(noised_image)

        idx_H = np.random.randint(H, size=(int(self.pixel_ratio*H)))
        idx_W = np.random.randint(W, size=(int(self.pixel_ratio*W)))

        elimination_mask[:, :, :, idx_W] = 0
        elimination_mask[:, :, idx_H, :] = 0

        noised_and_cover[0] = noised_image * elimination_mask
        return noised_and_cover

class AdjustHue(nn.Module):
    def __init__(self, hue_factor):
        super(AdjustHue, self).__init__()
        self.hue_factor = hue_factor
    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = F_t.adjust_hue(noised_image, self.hue_factor)
        return  noised_and_cover

class AdjustSaturation(nn.Module):
    def __init__(self, sat_factor):
        super(AdjustSaturation, self).__init__()
        self.sat_factor = sat_factor
    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = F_t.adjust_saturation(noised_image, self.sat_factor)
        return  noised_and_cover

class AdjustBrightness(nn.Module):
    def __init__(self, bri_factor):
        super(AdjustBrightness, self).__init__()
        self.bri_factor = bri_factor
    
    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = (noised_image+1)/2
        batch_encoded_image = [ToPILImage()(x_) for x_ in noised_image]
        # batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        # batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(len(batch_encoded_image)):
            encoded_image = batch_encoded_image[idx]
            enhancer = ImageEnhance.Brightness(encoded_image)
            noise_image = enhancer.enhance(self.bri_factor)
            noise_image = ToTensor()(noise_image).type(torch.FloatTensor).cuda()
            # noise_image = (noise_image*2-1).type(torch.FloatTensor).cuda()
            # noise_image = cv2.blur(encoded_image, (self.kernel, self.kernel))
            # noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)
        return noised_and_cover

class AdjustContrast(nn.Module):
    def __init__(self, con_factor):
        super(AdjustContrast, self).__init__()
        self.con_factor = con_factor

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = (noised_image+1)/2
        batch_encoded_image = [ToPILImage()(x_) for x_ in noised_image]
        # batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        # batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(len(batch_encoded_image)):
            encoded_image = batch_encoded_image[idx]
            enhancer = ImageEnhance.Contrast(encoded_image)
            noise_image = enhancer.enhance(self.con_factor)
            noise_image = ToTensor()(noise_image).type(torch.FloatTensor).cuda()
            # noise_image = (noise_image*2-1).type(torch.FloatTensor).cuda()
            # noise_image = cv2.blur(encoded_image, (self.kernel, self.kernel))
            # noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)
        return noised_and_cover

class AdjustColor(nn.Module):
    def __init__(self, col_factor):
        super(AdjustColor, self).__init__()
        self.col_factor = col_factor
    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = (noised_image+1)/2
        batch_encoded_image = [ToPILImage()(x_) for x_ in noised_image]
        # batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        # batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(len(batch_encoded_image)):
            encoded_image = batch_encoded_image[idx]
            enhancer = ImageEnhance.Color(encoded_image)
            noise_image = enhancer.enhance(self.col_factor)
            noise_image = ToTensor()(noise_image).type(torch.FloatTensor).cuda()
            # noise_image = (noise_image*2-1).type(torch.FloatTensor).cuda()
            # noise_image = cv2.blur(encoded_image, (self.kernel, self.kernel))
            # noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)
        return noised_and_cover

class AdjustSharpness(nn.Module):
    def __init__(self, sha_factor):
        super(AdjustSharpness, self).__init__()
        self.sha_factor = sha_factor
    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = (noised_image+1)/2
        batch_encoded_image = [ToPILImage()(x_) for x_ in noised_image]
        # batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        # batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(len(batch_encoded_image)):
            encoded_image = batch_encoded_image[idx]
            enhancer = ImageEnhance.Sharpness(encoded_image)
            noise_image = enhancer.enhance(self.sha_factor)
            noise_image = ToTensor()(noise_image).type(torch.FloatTensor).cuda()
            # noise_image = (noise_image*2-1).type(torch.FloatTensor).cuda()
            # noise_image = cv2.blur(encoded_image, (self.kernel, self.kernel))
            # noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)
        return noised_and_cover



def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v= torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)

class Compression(nn.Module):
    """
    This uses the DCT to produce a differentiable approximation of JPEG compression.
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, yuv=False, min_pct=0.0, max_pct=0.5):
        super(Compression, self).__init__()
        self.yuv = yuv
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, y):
        z = y
        N, _, H, W = z.size()

        H = int(z.size(2) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        W = int(z.size(3) * (random() * (self.max_pct - self.min_pct) + self.min_pct))

        if self.yuv:
            z = torch.stack([
                (0.299 * z[:, 2, :, :] +
                 0.587 * z[:, 1, :, :] +
                 0.114 * z[:, 0, :, :]),
                (- 0.168736 * z[:, 2, :, :] -
                 0.331264 * z[:, 1, :, :] +
                 0.500 * z[:, 0, :, :]),
                (0.500 * z[:, 2, :, :] -
                 0.418688 * z[:, 1, :, :] -
                 0.081312 * z[:, 0, :, :]),
            ], dim=1)

        z = dct_3d(z)

        if H > 0:
            z[:, :, -H:, :] = 0.0

        if W > 0:
            z[:, :, :, -W:] = 0.0

        z = idct_3d(z)

        if self.yuv:
            z = torch.stack([
                (1.0 * z[:, 0, :, :] +
                 1.772 * z[:, 1, :, :] +
                 0.000 * z[:, 2, :, :]),
                (1.0 * z[:, 0, :, :] -
                 0.344136 * z[:, 1, :, :] -
                 0.714136 * z[:, 2, :, :]),
                (1.0 * z[:, 0, :, :] +
                 0.000 * z[:, 1, :, :] +
                 1.402 * z[:, 2, :, :]),
            ], dim=1)
        y= z
        return y

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



#################################
### Compression Approximation ###
#################################

class UNet(nn.Module):
    def __init__(self, n_channels = 3 , n_classes = 3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, y):
        x = y
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        y = logits
        return y



def transform(tensor, target_range):
    source_min = tensor.min()
    source_max = tensor.max()

    # normalize to [0, 1]
    tensor_target = (tensor - source_min)/(source_max - source_min)
    # move to target range
    tensor_target = tensor_target * (target_range[1] - target_range[0]) + target_range[0]
    return tensor_target


class Quantization(nn.Module):
    def __init__(self, device=None):
        super(Quantization, self).__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.min_value = 0.0
        self.max_value = 255.0
        self.N = 10
        self.weights = torch.tensor([((-1) ** (n + 1)) / (np.pi * (n + 1)) for n in range(self.N)]).to(device)
        self.scales = torch.tensor([2 * np.pi * (n + 1) for n in range(self.N)]).to(device)
        for _ in range(4):
            self.weights.unsqueeze_(-1)
            self.scales.unsqueeze_(-1)


    def fourier_rounding(self, tensor):
        shape = tensor.shape
        z = torch.mul(self.weights, torch.sin(torch.mul(tensor, self.scales)))
        z = torch.sum(z, dim=0)
        return tensor + z


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = transform(noised_image, (0, 255))
        # noised_image = noised_image.clamp(self.min_value, self.max_value).round()
        noised_image = self.fourier_rounding(noised_image.clamp(self.min_value, self.max_value))
        noised_and_cover[0] = transform(noised_image, (noised.min(), noised.max()))
        return noised_and_cover

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio = resize_ratio
        self.interpolation_method = interpolation_method


    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        _, _, H, W = noised_image.shape
        # print(H,W)
        # print(self.resize_ratio)
        noised = F.interpolate(noised_image,
                                    scale_factor=(self.resize_ratio, self.resize_ratio),
                                    mode=self.interpolation_method)

        _, _, new_H, new_W = noised.shape

        # print(new_H, new_W)

        if new_H > H or new_W > W:
            # Center crop to original size
            h_start = np.random.randint(new_H - H)
            w_start = np.random.randint(new_W - W)
            noised = noised[:, :, h_start:h_start+H, w_start:w_start+W]
        else:
            # Resize and center in original size with padding
            pad_h = (H - new_H) // 2
            pad_w = (W - new_W) // 2

            noised = torch.nn.functional.pad(noised, (pad_w, pad_w, pad_h, pad_h))
            noised = noised[:, :, :H, :W]

        noised_and_cover[0] = noised

        return noised_and_cover


#################################
### Compression Approximation ###
#################################

def generate_random_key(l=30):
    s=string.ascii_lowercase+string.digits
    return ''.join(random.sample(s,l))

class JpegCompression2(nn.Module):
    def __init__(self, quality=50):
        super(JpegCompression2, self).__init__()
        self.quality = quality
        # self.key = generate_random_key()

    def forward(self, noised_and_cover):
        # jpeg_folder_path = "./jpeg_" + str(self.quality) + "/" + self.key
        # if not os.path.exists(jpeg_folder_path):
        #     os.makedirs(jpeg_folder_path)

        noised_image = noised_and_cover[0]
        container_img_copy = noised_image.clone()
        containers_ori = container_img_copy.detach().cpu().numpy()
        
        containers = np.transpose(containers_ori, (0, 2, 3, 1))
        N, _, _, _ = containers.shape
        # containers = (containers + 1) / 2 # transform range of containers from [-1, 1] to [0, 1]
        containers = (np.clip(containers, 0.0, 1.0)*255).astype(np.uint8)

        # containers = (np.clip(containers, 0.0, 1.0)*255).astype(np.uint8)

        for i in range(N):
            img = cv2.cvtColor(containers[i], cv2.COLOR_RGB2BGR)
            # folder_imgs = jpeg_folder_path + "/jpg_" + str(i).zfill(2) + ".jpg"
            folder_imgs = str(i).zfill(2) + ".jpg"
            cv2.imwrite(folder_imgs, img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])

        containers_loaded = np.copy(containers)
        
        for i in range(N):
            # folder_imgs = jpeg_folder_path + "/jpg_" + str(i).zfill(2) + ".jpg"
            folder_imgs = str(i).zfill(2) + ".jpg"
            img = cv2.imread(folder_imgs)
            containers_loaded[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # containers_loaded = np.transpose(containers_loaded, (0, 3, 1, 2)).astype(np.float32) / 255

        containers_loaded = containers_loaded.astype(np.float32) / 255
        # containers_loaded = containers_loaded * 2 - 1 # transform range of containers from [0, 1] to [-1, 1]
        containers_loaded = np.transpose(containers_loaded, (0, 3, 1, 2))

        container_gap = containers_loaded - containers_ori
        container_gap = torch.from_numpy(container_gap).float().cuda()

        container_img_noised_jpeg = noised_image + container_gap

        noised_and_cover[0] = container_img_noised_jpeg

        return noised_and_cover

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels = 3 , n_classes = 3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, y):
        x = y
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        y = logits
        return y


class RealComp(nn.Module):
    def __init__(self, comp_type, quality, device):
        super(RealComp, self).__init__()
        self.comp_type = comp_type
        self.device = device 
        self.quality = quality
    def forward(self, imgs):
        x = imgs[0]

        comp_imgs = torch.zeros((x.shape))
        
        for i in range(x.shape[0]):
            # x[i] = x[i]*0.5+0.5
            pil_img = transforms.ToPILImage()(x[i])
            # pil_img = imgs[i]

            
            if self.comp_type == "WebP":
                pil_img.save(str(i+1) + ".webp", format="webp", quality=self.quality)
                comp_img = Image.open(str(i+1) + ".webp").convert('RGB')
            elif self.comp_type == "Compression":
                # compression_quality = np.random.choice([50, 60, 70, 80, 90], 1)[0]
                # print(compression_quality)
                pil_img.save(str(i+1) + ".jpg", format = "JPEG", quality=self.quality)
                comp_img = Image.open(str(i+1)+".jpg").convert('RGB')
                # pil_img.save(str(i+1) + ".png", format = "PNG")
                # comp_img = Image.open(str(i+1)+".png").convert('RGB')

            comp_img = transforms.PILToTensor()(comp_img).float()
            comp_img = transforms.CenterCrop(x.shape[2])(comp_img)
            comp_img = comp_img/255.0
            # comp_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(comp_img)
            
            comp_imgs[i] = comp_img
        imgs[0] = comp_imgs.to(self.device)
        return imgs


########################
#   Distortion Layer   #
########################
def val_resize(image):
    return F_t.resize(image, (c.cropsize_val, c.cropsize_val))
class DistortionLayer(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers, attack_layers, infoDict, device, phase = "train"):
        super(DistortionLayer, self).__init__()
        self.noise_layers = []
        self.attack_layers = []
        self.noise_layers_string = noise_layers
        self.attack_layers_string = attack_layers
        self.phase = phase
        self.coeff = 0
        self.infoDict = infoDict
        self.mask = [None]
        self.device = device
        self.files = sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train))
        self.generator = Generator(checkpoint="./model/states_pt_places2.pth", return_flow=True).to(device)
        # print(self.files)
        if self.phase == "train":
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomCrop(c.cropsize),
                T.ToTensor()
            ])
        else:
            self.transform = T.Compose([
                T.Lambda(val_resize),
                T.ToTensor()
            ])

    def forward(self, encoded_and_cover, no_identity = False):
        if self.phase == "train":
            self.noise_layers = [Identity()]
        else:
            self.noise_layers = []
        # print(encoded_and_cover[0].shape)
        # print(encoded_and_cover[1].shape)
        for layer in self.noise_layers_string:
            if type(layer) is str:
                self.coeff = 0
                #Post-processing
                if layer == 'Compression':
                    if self.phase == "train" or self.phase == "val":
                        self.coeff = int(np.random.choice([80, 85, 90, 95], 1)[0])
                        # self.coeff = 90
                        self.noise_layers.append(DiffJPEG(differentiable = True, quality = self.coeff).to(self.device))
                    else:
                        net = RealComp(layer, int(self.infoDict['JPEGQ']), self.device)
                        net.to(self.device)
                        self.noise_layers.append(net)

                elif layer == 'WebP' or self.phase == "val":
                    # self.noise_layers.append(DiffJPEG().to(device))
                    if self.phase == "train" or self.phase == "val":
                        self.coeff = int(np.random.choice([80, 85, 90, 95], 1)[0])
                        self.noise_layers.append(DiffJPEG(quality = self.coeff).to(self.device))
                    else:
                        net = RealComp(layer, int(self.infoDict['JPEGQ']), self.device)
                        net.to(self.device)
                        self.noise_layers.append(net)

                elif layer == 'GaussianNoise':
                    if self.phase == "train" or self.phase == "val":
                        self.coeff = float(np.random.choice([0.03, 0.04, 0.05], 1)[0])
                    else:
                        self.coeff = self.infoDict['Standard_deviation']
                    self.noise_layers.append(GaussianNoise(self.coeff))

                elif layer == "DropOut" or self.phase == "val":
                    if self.phase == "train":
                        self.coeff = float(np.random.choice([0.8, 0.85, 0.9, 0.95], 1)[0])
                        # self.coeff = self.infoDict['prob']
                    else:
                        self.coeff = self.infoDict['prob']
                    self.noise_layers.append(DropOut(self.coeff))
                
                elif layer == 'GaussianBlur':
                    if self.phase == "train":
                        self.coeff = int(np.random.choice([1, 1.5, 3], 1)[0])
                    else:
                        self.coeff = self.infoDict['sigma']
                    self.noise_layers.append(GaussianBlur(self.coeff))
                elif layer == 'MedianFilter':
                    if self.phase == "train":
                        # self.coeff = int(np.random.choice([13, 31, 33], 1)[0])
                        self.coeff = 33
                    else:
                        self.coeff = self.infoDict['med_kernel']
                    self.noise_layers.append(MedianFilter(self.coeff))
                elif layer == 'Rescaling':
                    # if self.phase == "train":
                    self.coeff = float(np.random.choice([0.25, 0.5, 1.5, 2], 1)[0])
                    # else:
                        # self.coeff = self.infoDict['resize_ratio']
                    self.noise_layers.append(Resize(self.coeff))
                elif layer == 'Crop':
                    if self.phase == "train":
                        self.coeff = float(np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7], 1)[0])
                    else:
                        self.coeff = self.infoDict['crop_ratio']
                    self.noise_layers.append(Crop(self.coeff))
                elif layer == "AdjustSaturation":
                    if self.phase == "train":
                        self.coeff = float(np.random.choice([0.5, 1, 1.5, 2], 1)[0])
                    else:
                        self.coeff = self.infoDict['sat_factor']
                    self.noise_layers.append(AdjustSaturation(self.coeff))

                elif layer == "AdjustBrightness":
                    if self.phase == "train":
                        self.coeff = float(np.random.choice([0.5, 1, 1.2, 1.5], 1)[0])
                    else:
                        self.coeff = self.infoDict['bri_factor']
                    self.noise_layers.append(AdjustBrightness(self.coeff))
                
                elif layer == "AdjustContrast":
                    if self.phase == "train":
                        self.coeff = float(np.random.choice([0.33, 0.44, 0.55, 0.66], 1)[0])
                    else:
                        self.coeff = self.infoDict['con_factor']
                    self.noise_layers.append(AdjustContrast(self.coeff))

                elif layer == "AdjustColor":
                    if self.phase == "train":
                        self.coeff = float(np.random.choice([0.5, 1, 1.5], 1)[0])
                    else:
                        self.coeff = self.infoDict['col_factor']
                    self.noise_layers.append(AdjustColor(self.coeff))

                elif layer == "AverageFilter":
                    if self.phase == "train":
                        # self.coeff = float(np.random.choice([0.33, 0.44, 0.55, 0.66], 1)[0])
                        self.coeff = 33
                    else:
                        self.coeff = self.infoDict['ave_kernel']
                    self.noise_layers.append(BoxFilter(self.coeff))

                elif layer =="Identity":
                    self.noise_layers = [Identity()]
                else:
                    raise ValueError(f'Wrong layer placeholder string in DistortionLayer.__init__().'
                                     f' Expected distortion layers but got {layer} instead')
                # print("Coeff: ", self.coeff)
        if self.phase == "train":
            self.attack_layers = [IdentityAttack()]
        else:
            self.attack_layers = []
        # if no_identity:
        #     self.attack_layers = []
        for layer in self.attack_layers_string:
            if type(layer) is str:
                # Tampering
                if layer == "Splicing":
                    index = random.randint(0,len(self.files)-1)
                    # print("index: ", index)
                    image = Image.open(self.files[index]).convert("RGB")
                    cover = self.transform(image)
                    self.attack_layers.append(Splicing(cover.to(self.device), self.phase, self.device))
                
                elif layer == "CopyMove":
                    self.attack_layers.append(CopyMove(self.phase, self.device))
                
                elif layer == "Inpainting":
                    self.attack_layers.append(Inpainting(self.generator, self.phase, self.device))

                elif layer == "ObjectAddition":
                    self.attack_layers.append(ObjectAddition(self.device, "C:/Users/maarab/Forensics/Datasets/COCO/val_seg/"))
                else:
                    raise ValueError(f'Wrong layer placeholder string in DistortionLayer.__init__().'
                                     f' Expected distortion layers but got {layer} instead')
        
        if self.attack_layers == []:
            self.attack_layers = [IdentityAttack()]

        if self.phase == "train" or self.phase == "val":
            random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
            random_attack_layer = np.random.choice(self.attack_layers, 1)[0]
            
            mask = None
            attacked_image = None
            [attacked_image, mask] = random_attack_layer(encoded_and_cover)
            
            # if self.phase == "val":
            #     print(mask)
            #     print(type(random_attack_layer).__name__)
            # print("attacked_image[0]: ", attacked_image[0].shape)
            # print("attacked_image[1]: ", attacked_image[1].shape)
            # print("mask: ", mask.shape)
            return [random_noise_layer(attacked_image), mask]

        elif self.phase == "test" or self.phase == "distort":
            if self.noise_layers == []:
                self.noise_layers = [Identity()]
            # random_noise_layer = self.noise_layers[-1]
            random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
            mask = None
            attacked_image = None
            if self.phase == "distort":
                random_attack_layer = IdentityAttack()
                [attacked_image, _] = random_attack_layer(encoded_and_cover)
            else:
                if self.attack_layers == []:
                    random_attack_layer = IdentityAttack()
                else:
                    # random_attack_layer = self.attack_layers[-1]
                    random_attack_layer = np.random.choice(self.attack_layers, 1)[0]
                [attacked_image, mask] = random_attack_layer(encoded_and_cover)
            # print(mask)
            return [random_noise_layer(attacked_image), mask]
#!/usr/bin/env python

import torch
import torchvision
import torch.nn
import torch.optim
import math
import numpy as np
import cv2
from math import exp
from torch.autograd import Variable

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


from distortions import DistortionLayer
import config as c

from model import *
# from tensorboardX import SummaryWriter
from invertible_net import ForgeryDetector
import datasets
import viz
import modules.Unet_common as common
import torch.nn.functional as F
import warnings
from tqdm import tqdm
from itertools import chain

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DeepLabv3 with MobileNetV3 Large backbone for semantic segmentation
class DeepLabv3Segmentation(nn.Module):
    def __init__(self):
        super(DeepLabv3Segmentation, self).__init__()

        self.model = deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)

        self.model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
        
    def forward(self, x):
        # print(self.model(x)["out"].shape)
        return self.model(x)["out"]

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        # self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

def uniform_noise(shape):
        noise = torch.zeros(shape).to(device)
        for i in range(noise.shape[0]):
            noise[i] = (torch.rand(noise[i].shape)*2-1).to(device)
        return noise

def gauss_noise(shape):
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)

    return noise


#Loss Functions

# SSIM Loss Function
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + C1) * (mu2_sq + C2) * (sigma1_sq + C2) * (sigma2_sq + C2))
    
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

# Custom Hybrid Loss Function
# SSIM Loss Function
def gaussian(window_size, sigma, device):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).to(device)
    return gauss / gauss.sum()

def create_window(window_size, channel, device):
    _1D_window = gaussian(window_size, 1.5, device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous()).to(device)
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel, img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + C1) * (mu2_sq + C2) * (sigma1_sq + C2) * (sigma2_sq + C2))
    
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


def calculate_psnr(img1, img2, max_pixel_value=1.0):
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10(max_pixel_value**2 / mse)
    return psnr

def hybrid_l1Loss(input1, input2, limit=None):
    psnr_value = calculate_psnr(input1, input2)
    l1_loss = nn.L1Loss(reduce=True, size_average=False)
    loss = l1_loss(input1, input2)
    if limit == None:
        return loss
    else:
        return loss + 1000*torch.relu(limit - psnr_value)

def hybrid_bceLoss(input1, input2, limit=None):
    psnr_value = calculate_psnr(input1, input2)
    bce_loss = nn.BCELoss(reduce=True, size_average=False)
    loss = bce_loss(input1, input2)
    if limit == None:
        return loss
    else:
        return loss + 1000*torch.relu(limit - psnr_value)

def hybrid_l2Loss(input1, input2, limit=None):
    psnr_value = calculate_psnr(input1, input2)
    l2_loss = nn.MSELoss(reduce=True, size_average=False)
    loss = l2_loss(input1, input2)
    if limit == None:
        return loss
    else:
        return loss + 1000*torch.relu(limit - psnr_value)


def perc_loss(output, bicubic_image):
    loss_fn = VGGPerceptualLoss()
    loss = loss_fn(output.to("cpu"), bicubic_image.to("cpu"))
    return loss.to(device)

def ncs_loss(image1, image2):
    # Normalize the image tensors
    image1_normalized = F.normalize(image1, p=2, dim=1)
    image2_normalized = F.normalize(image2, p=2, dim=1)
    
    # Compute the cosine similarity
    cosine_similarity = torch.sum(image1_normalized * image2_normalized, dim=1)
    
    # Negative cosine similarity loss
    loss = 1 - cosine_similarity.mean()
    
    return loss.to(device)

def localization_loss(output, bicubic_image):
    loss_fn = torch.nn.BCELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)

def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
    # loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)

def noise_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)

def cover_reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
    # loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)






# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


def load(net, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)


def load_with_forgery(net, fd, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)

    network_state_dict_fd = {k: v for k, v in state_dicts['fd'].items() if 'tmp_var' not in k}
    fd.load_state_dict(network_state_dict_fd)



def rgb_to_yuv(image):
    r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.stack((y, u, v), dim=1)


def image_mix(img1, img2):
    # Convert images from RGB to YUV
    yuv1 = rgb_to_yuv(img1)
    yuv2 = rgb_to_yuv(img2)

    # Extract Y channels
    y1 = yuv1[:, 0, :, :]
    y2 = yuv2[:, 0, :, :]

    # Calculate the difference between Y channels
    diff_y = torch.abs(y1 - y2)

    # Merge Y channels
    fused_y = torch.stack((y1, diff_y, y2), dim=1)

    return fused_y

def DeBlocking(im):
    N, _, W, H = im.shape
    if c.BW:
        # Split the image into four quadrants
        img = im[:, 0, :, :].squeeze(1)  # secret image

        # Initialize a list to store all the patches
        R_patches = []
        G_patches = []
        B_patches = []

        if c.AA:
            code_patches = []

        # Calculate the height and width of each patch. Each path is a 2X2 pattern of R,G,B, and hash code.
        patch_height = W // (c.thumbnail_number*2)
        patch_width = H // (c.thumbnail_number*2)

        for i in range(c.thumbnail_number):
            for j in range(c.thumbnail_number):
                R_patches.append(img[:, (j*2*patch_width) + 0: (j*2*patch_width) + patch_width,
                                    (i*2*patch_height) + 0: (i*2*patch_height) + patch_height])
                G_patches.append(img[:, (j*2*patch_width) + patch_width: (j*2*patch_width) + (2 * patch_width),
                                    (i*2*patch_height) + 0: (i*2*patch_height) + patch_height])
                B_patches.append(img[:, (j*2*patch_width) + 0: (j*2*patch_width) + patch_width,
                                    (i*2*patch_height) + patch_height: (i*2*patch_height) + (2 * patch_height)])
                if c.AA:
                    code_patches.append(img[:, (j*2*patch_width) + patch_width: (j*2*patch_width) + (2 * patch_width),
                                            (i*2*patch_height) + patch_height: (i*2*patch_height) + (2 * patch_height)])

        r_avg = sum(R_patches) / len(R_patches)
        g_avg = sum(G_patches) / len(G_patches)
        b_avg = sum(B_patches) / len(B_patches)
        if c.AA:
            code_avg = sum(code_patches) / len(code_patches)

        # print("b_avg: ", b_avg.shape)     

        img_rgb = torch.stack([r_avg, g_avg, b_avg], dim=1)

        if c.AA:
            return img_rgb, code_avg 

        return img_rgb


    elif c.YCC:
        # Split the image into four quadrants
        Y = im[:, 0, :, :].squeeze(1)  # Y channel

        CbCr = im[:, 1, :, :].squeeze(1)  # Combined CbCr channel

        # Calculate the height and width of each patch
        patch_height = W // c.thumbnail_number
        patch_width = H // c.thumbnail_number

        # Initialize a list to store all the patches
        patches = []
        if c.AA:
            code_patches = []

        # Loop over each patch and extract it
        for i in range(c.thumbnail_number):
            for j in range(c.thumbnail_number):
                if c.AA:
                    if (i + j) % 2 == 0:
                        code_patch = im[:, :, i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
                        code_patches.append(code_patch)
                    else:
                        patch = Y[:, i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
                        patches.append(patch)
                else:
                    patch = Y[:, i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
                    patches.append(patch)

        # Calculate the average of all the patches
        y_avg = sum(patches) / len(patches)
        if c.AA:
            code_avg = sum(code_patches) / len(code_patches)

        CbCr = im[:, 1, :, :].squeeze(1)  # Combined CbCr channel
        # Calculate the height and width of each quadrant
        quad_height = W // c.thumbnail_number
        quad_width = H // c.thumbnail_number

        # Initialize lists to store Cb and Cr patches
        cb_patches = []
        cr_patches = []

        # Loop through each quadrant
        for i in range(c.thumbnail_number):
            for j in range(c.thumbnail_number):
                if c.AA and (i + j) % 2 == 0:
                    continue
                # Extract each quadrant
                quadrant = CbCr[:, i * quad_height:(i + 1) * quad_height, j * quad_width:(j + 1) * quad_width]
                
                # Calculate the height and width of the sub-patches (for Cb and Cr)
                patch_height = quad_height // 2
                patch_width = quad_width // 2

                # Extract the Cb and Cr sub-patches following the chessboard pattern
                cb_patches.append(quadrant[:, :patch_height, :patch_width])        # Top-left Cb
                cb_patches.append(quadrant[:, patch_height:, patch_width:])        # Bottom-right Cb
                cr_patches.append(quadrant[:, patch_height:, :patch_width])        # Bottom-left Cr
                cr_patches.append(quadrant[:, :patch_height, patch_width:])        # Top-right Cr

        # Calculate the average of all Cb and Cr patches
        cb_avg = sum(cb_patches) / len(cb_patches)
        cr_avg = sum(cr_patches) / len(cr_patches)

        # Resize cb_avg and cr_avg to W//L and H//L
        cb_avg = F.interpolate(cb_avg.unsqueeze(1), size=(W // c.thumbnail_number, H // c.thumbnail_number), mode=c.Upsample_mode).squeeze(1)
        cr_avg = F.interpolate(cr_avg.unsqueeze(1), size=(W // c.thumbnail_number, H // c.thumbnail_number), mode=c.Upsample_mode).squeeze(1)

        # Combine Y, Cb, and Cr to form the final image
        img_ycbcr = torch.stack([y_avg, cb_avg, cr_avg], dim=1)

        if c.AA:
            return img_ycbcr, code_avg 

        return img_ycbcr
    
    elif c.ND2:
        # Split the image into four quadrants
        Y = im[:, 0, :, :].squeeze(1)  # Y channel

        CbCr = im[:, 1, :, :].squeeze(1)  # Combined CbCr channel

        # Calculate the height and width of each quadrant
        quad_height = W // c.thumbnail_number
        quad_width = H // c.thumbnail_number

        # Initialize a list to store all the patches
        patches = []
        if c.AA:
            code_patches = []

        # Loop over each patch and extract it
        for i in range(c.thumbnail_number):
            for j in range(c.thumbnail_number):
                patch = Y[:, i * quad_height:(i + 1) * quad_height, j * quad_width:(j + 1) * quad_width]
                patches.append(patch)

        # Calculate the average of all the patches
        y_avg = sum(patches) / len(patches)

        CbCr = im[:, 1, :, :].squeeze(1)  # Combined CbCr channel
        # Calculate the height and width of each quadrant
        patch_height = W // 2 * c.thumbnail_number
        patch_width = H // 2 * c.thumbnail_number

        # Initialize lists to store Cb and Cr patches
        cb_patches = []
        cr_patches = []

        # Loop through each quadrant
        for i in range(c.thumbnail_number):
            for j in range(c.thumbnail_number):

                # Calculate the height and width of the sub-patches (for Cb and Cr)
                patch_height = quad_height // 2
                patch_width = quad_width // 2

                # Extract each quadrant
                quadrant = CbCr[:, i * quad_height:(i + 1) * quad_height, j * quad_width:(j + 1) * quad_width]
                


                code_patches.append(quadrant[:, patch_height:, :patch_width])
                code_patches.append(quadrant[:, :patch_height, patch_width:])
                # Extract the Cb and Cr sub-patches following the chessboard pattern
                cb_patches.append(quadrant[:, :patch_height, :patch_width])        # Top-left Cb

                cr_patches.append(quadrant[:, patch_height:, patch_width:])        # Bottom-right Cr

        if c.AA:
            code_avg = sum(code_patches) / len(code_patches)

        # Calculate the average of all Cb and Cr patches
        cb_avg = sum(cb_patches) / len(cb_patches)
        cr_avg = sum(cr_patches) / len(cr_patches)

        # Resize cb_avg and cr_avg to W//L and H//L
        cb_avg = F.interpolate(cb_avg.unsqueeze(1), size=(W // c.thumbnail_number, H // c.thumbnail_number), mode=c.Upsample_mode).squeeze(1)
        cr_avg = F.interpolate(cr_avg.unsqueeze(1), size=(W // c.thumbnail_number, H // c.thumbnail_number), mode=c.Upsample_mode).squeeze(1)

        # Combine Y, Cb, and Cr to form the final image
        img_ycbcr = torch.stack([y_avg, cb_avg, cr_avg], dim=1)

        if c.AA:
            return img_ycbcr, code_avg 

        return img_ycbcr

    elif c.ND3:
        # Split the image into four quadrants
        Y = im[:, 0, :, :].squeeze(1)  # Y channel

        CbCr = im[:, 1, :, :].squeeze(1)  # Combined CbCr channel
        CbCr2 = im[:, 2, :, :].squeeze(1)  # Combined CbCr channel

        # Calculate the height and width of each quadrant
        quad_height = W // c.thumbnail_number
        quad_width = H // c.thumbnail_number

        # Initialize a list to store all the patches
        patches = []
        if c.AA:
            code_patches = []

        # Loop over each patch and extract it
        for i in range(c.thumbnail_number):
            for j in range(c.thumbnail_number):
                patch = Y[:, i * quad_height:(i + 1) * quad_height, j * quad_width:(j + 1) * quad_width]
                patches.append(patch)

        # Calculate the average of all the patches
        y_avg = sum(patches) / len(patches)

        # Calculate the height and width of each quadrant
        patch_height = W // 2 * c.thumbnail_number
        patch_width = H // 2 * c.thumbnail_number

        # Initialize lists to store Cb and Cr patches
        cb_patches = []
        cr_patches = []

        # Loop through each quadrant
        for i in range(c.thumbnail_number):
            for j in range(c.thumbnail_number):

                # Calculate the height and width of the sub-patches (for Cb and Cr)
                patch_height = quad_height // 2
                patch_width = quad_width // 2

                # Extract each quadrant
                quadrant = CbCr[:, i * quad_height:(i + 1) * quad_height, j * quad_width:(j + 1) * quad_width]
                quadrant2 = CbCr2[:, i * quad_height:(i + 1) * quad_height, j * quad_width:(j + 1) * quad_width]
                


                code_patches.append(quadrant[:, patch_height:, :patch_width])
                code_patches.append(quadrant[:, :patch_height, patch_width:])

                code_patches.append(quadrant2[:, patch_height:, :patch_width])
                code_patches.append(quadrant2[:, :patch_height, patch_width:])

                # Extract the Cb and Cr sub-patches following the chessboard pattern
                cb_patches.append(quadrant[:, :patch_height, :patch_width])        # Top-left Cb
                cb_patches.append(quadrant2[:, :patch_height, :patch_width])        # Top-left Cb

                cr_patches.append(quadrant[:, patch_height:, patch_width:])        # Bottom-right Cr
                cr_patches.append(quadrant2[:, patch_height:, patch_width:])        # Bottom-right Cr

        if c.AA:
            code_avg = sum(code_patches) / len(code_patches)

        # Calculate the average of all Cb and Cr patches
        cb_avg = sum(cb_patches) / len(cb_patches)
        cr_avg = sum(cr_patches) / len(cr_patches)

        # Resize cb_avg and cr_avg to W//L and H//L
        cb_avg = F.interpolate(cb_avg.unsqueeze(1), size=(W // c.thumbnail_number, H // c.thumbnail_number), mode=c.Upsample_mode).squeeze(1)
        cr_avg = F.interpolate(cr_avg.unsqueeze(1), size=(W // c.thumbnail_number, H // c.thumbnail_number), mode=c.Upsample_mode).squeeze(1)

        # Combine Y, Cb, and Cr to form the final image
        img_ycbcr = torch.stack([y_avg, cb_avg, cr_avg], dim=1)

        if c.AA:
            return img_ycbcr, code_avg 

        return img_ycbcr
        

def ycbcr_to_rgb(ycbcr_tensor):
    """
    Convert a batch of images from YCbCr to RGB.
    
    Args:
        ycbcr_tensor (torch.Tensor): Tensor of shape (N, C, H, W) in YCbCr color space.
        
    Returns:
        torch.Tensor: Tensor of shape (N, C, H, W) in RGB color space.
    """
    # Ensure the input tensor has the correct shape
    assert ycbcr_tensor.shape[1] == 3, "Input tensor must have 3 channels (Y, Cb, Cr)"
    
    Y = ycbcr_tensor[:, 0, :, :]
    Cb = ycbcr_tensor[:, 1, :, :]
    Cr = ycbcr_tensor[:, 2, :, :]
    
    # Allocate space for the output tensor
    rgb_tensor = torch.empty_like(ycbcr_tensor)
    
    rgb_tensor[:, 0, :, :] = Y + 1.402 * (Cr - 0.5)  # R
    rgb_tensor[:, 1, :, :] = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)  # G
    rgb_tensor[:, 2, :, :] = Y + 1.772 * (Cb - 0.5)  # B
    
    # Clip the values to be in the range [0, 1]
    rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)
    
    return rgb_tensor

#####################
# Model initialize: #
#####################
def main():
    visualizer = viz.Visualizer(c.trained_epoch, c.loss_names)

    def show_loss(losses, logscale=False):
        visualizer.update_losses(losses)

    def show_imgs(*imgs):
        visualizer.update_images(*imgs)

    def show_hist(data):
        visualizer.update_hist(data.data)

    def signal_start():
        visualizer.update_running(True)

    def signal_stop():
        visualizer.update_running(False)

    def close():
        visualizer.close()

    
    if c.TFD:
        stop_f_train = False
        if c.BW:
            forgery_detector = ForgeryDetector(in_channels=4)
        elif c.YCC:
            forgery_detector = ForgeryDetector(in_channels=6)
        # forgery_detector = DeepLabv3Segmentation().to(device)
        forgery_detector.to(device)
        init_model(forgery_detector)
        forgery_detector = torch.nn.DataParallel(forgery_detector, device_ids=c.device_ids)
        
    net = Model()
    net.to(device)
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    
    if c.TFD and c.TE:
        all_trainable_params = chain(net.parameters(), forgery_detector.parameters())
        print("Preparing to train the model as well as the forgery detector.")
    elif c.TE:
        params_trainable_net = list(filter(lambda p: p.requires_grad, net.parameters()))
        all_trainable_params = params_trainable_net
        print("Preparing to train the model.")
    elif c.TFD:
        params_trainable_net = list(filter(lambda p: p.requires_grad, forgery_detector.parameters()))
        all_trainable_params = params_trainable_net
        print("Preparing to the forgery detector.")

    optim = torch.optim.Adam(all_trainable_params, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, c.milestone, gamma=c.gamma)

    if c.TF == 'DCT':
        dwt = common.DCT()
        iwt = common.IDCT()
    else:
        dwt = common.DWT()
        iwt = common.IWT()

    
    print("attacks: ",c.attacks)
    print("noises: ",c.noises)
    distortion_layers = DistortionLayer(c.noises, c.attacks, c.infoDict, device)
    distortion_layers.to(device)

    distortion_layers_test = DistortionLayer(c.noises, c.attacks, c.infoDict, device, "val")
    distortion_layers_test.to(device)

    if c.continue_train == 1:
        load(net, c.MODEL_PATH + c.suffix)
        print("Model in "+c.MODEL_PATH+" is loaded.")
    elif c.continue_train == 2:
        load_with_forgery(net, forgery_detector, c.MODEL_PATH + c.suffix)
        print("Model in "+c.MODEL_PATH+" along with the forgery detector are loaded.")
    # elif c.continue_train == 3:

    if c.continue_train > 0:
        for _ in range(c.trained_epoch):
            weight_scheduler.step()
    
    try:
        # writer = SummaryWriter(comment='hinet', filename_suffix="steg")

        for i_epoch in range(c.epochs):
            i_epoch = i_epoch + c.trained_epoch + 1
            if c.TE:
                net.train()
            else:
                net.eval()

            if c.TFD:
                if stop_f_train == False:
                    forgery_detector.train()
                else:
                    forgery_detector.eval()

            loss_history = []

            l_loss_history = []
            loc_loss_history = []
            n_loss_history = []
            g_loss_history = []
            p_loss_history = []
            n_loss_history = []
            e_loss_history = []
            gr_loss_history = []
            gr_loss2_history = []
            nr_loss_history = []
            d_loss_history = []

            r_loss_history = []
            cr_loss_history = []

            MaskPSNR = 0
            SecPSNR = 0
            #################
            #     train:    #
            #################

            for i, batch in enumerate(tqdm(datasets.trainloader)):
                if c.AA:
                    cover, secret, code_block, message = batch
                    cover = cover.to(device)
                    secret = secret.to(device)
                    code_block = code_block.to(device)
                    message = message.to(device)
                else:
                    cover, secret = batch
                    cover = cover.to(device)
                    secret = secret.to(device)
                if c.BW:
                    secret = torch.mean(secret, dim=1, keepdim=True)


                cover_input = dwt(cover)
                secret_input = dwt(secret)

                # print("cov: ", cover.shape)
                # print("sec: ", secret.shape)

                input_img = torch.cat((cover_input, secret_input), 1)

                #################
                #    forward:   #
                #################
                output = net(input_img)
                
                if c.TF == 'DCT':
                    output_steg = output.narrow(1, 0, c.channels_in)
                    output_z = output.narrow(1, c.channels_in, output.shape[1] - c.channels_in)
                else:
                    output_steg = output.narrow(1, 0, 4 * c.channels_in)
                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)

                steg_img = iwt(output_steg)

                # print("steg: ", steg_img.shape)
                real_mask = None
                if i_epoch > 300:
                    [noised_img, real_mask] = distortion_layers([steg_img, cover])
                else:
                    [noised_img, real_mask] = distortion_layers([steg_img, cover], True)
                noised_img = noised_img[0]
                # print(noised_img.shape)
                # print(real_mask.shape)
                real_mask = real_mask.to(device)
                

                if real_mask == None:
                    real_mask = torch.zeros((cover[0].shape[0], 1, cover[0].shape[2], cover[0].shape[3])).to(device)

                if real_mask.shape[1] == 1:
                    real_mask = real_mask.repeat(1, 3, 1, 1)

                fused_tensor = noised_img

                real_mask = real_mask/255

                # if c.TFD:
                #     if MaskPSNR>25:
                #         tensor_mask_3c = ret_mask
                #     else:
                #         tensor_mask_3c = real_mask
                # else:
                tensor_mask_3c = real_mask.to(device)
                # print("real_mask: ", real_mask.shape)

                if c.TFD:
                    inv_input = noised_img * (1 - tensor_mask_3c)
                else:
                    inv_input = noised_img
                
                output_steg = dwt(inv_input)

                #################
                #   backward:   #
                #################
                if c.NOISE_TYPE == "Gaussian":
                    output_z_backward = gauss_noise(output_z.shape)
                elif c.NOISE_TYPE == "Uniform":
                    output_z_backward = uniform_noise(output_z.shape)
                elif c.NOISE_TYPE == "Constant":
                    output_z_backward = 0.5 * torch.ones(output_z.shape).to(device)

                output_rev = torch.cat((output_steg, output_z_backward), 1)
                output_image = net(output_rev, rev=True)
                
                if c.TF == 'DCT':
                    cover_rev = output_image.narrow(1, 0, c.channels_in)
                    secret_rev = output_image.narrow(1, c.channels_in, c.channels_in)
                else:
                    cover_rev = output_image.narrow(1, 0, 4 * c.channels_in)
                    secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] -  4 * c.channels_in)
                        

                secret_rev = iwt(secret_rev)
                secret_rev = torch.clamp(secret_rev, 0 , 1)
                cover_rev = iwt(cover_rev)
                cover_rev = torch.clamp(cover_rev, 0 , 1)
                
                ##
                if c.TFD:
                    secret_rev = torch.clamp(secret_rev, 0 , 1)
                    
                    # print(secret_rev.shape)

                    extracted_sec = DeBlocking(secret_rev)
                    # print(extracted_sec.shape)
                    if c.YCC:
                        extracted_sec = ycbcr_to_rgb(extracted_sec)
                    
                    extracted_sec = F.interpolate(extracted_sec, size=(c.cropsize, c.cropsize), mode=c.Upsample_mode)

                    sec_rgb = DeBlocking(secret)
                    if c.YCC:
                        sec_rgb = ycbcr_to_rgb(sec_rgb)
                    sec_rgb = F.interpolate(sec_rgb, size=(c.cropsize, c.cropsize), mode=c.Upsample_mode)
                    
                    extracted_sec_3c = sec_rgb

                    # fused_tensor = image_mix(extracted_sec_3c, noised_img)
                    fused_tensor = torch.cat((noised_img, extracted_sec_3c), 1)
                    # print(fused_tensor.shape)
                    ret_mask = forgery_detector(fused_tensor)
                    ret_mask = torch.sigmoid(ret_mask)
                    ret_mask = ret_mask.repeat(1, 3, 1, 1).to(device)

                #################
                #     loss:     #
                #################
                optim.zero_grad()
                if c.HLoss:
                    coef = 0
                    if c.TFD:
                        if stop_f_train == False:
                            loc_loss = hybrid_bceLoss(ret_mask, real_mask, 12)
                            coef = 1
                        else:
                            loc_loss = 0
                    else:
                        loc_loss = 0


                    if c.TE:
                        g_loss = hybrid_l2Loss(steg_img.cuda(), cover.cuda(), 35)
                        # g_loss = perc_loss(steg_img.cuda(), cover.cuda())

                        if c.AA:
                            secret_rev_avg = torch.clamp(secret_rev, 0 , 1)

                            secret_rev_avg, code_avg_ext = DeBlocking(secret_rev_avg)

                            # print(tensor_mask_3c.shape)

                            secret_rev_avg = ycbcr_to_rgb(secret_rev_avg) # Only when BW and we use YCbCr instead of RGB
                            secret_rev_avg = secret_rev_avg.to(device)
                           
                            secret_rev_avg = F.interpolate(secret_rev_avg, size=(c.cropsize, c.cropsize), mode=c.Upsample_mode)

                            secret_avg, code_avg = DeBlocking(secret)

                            secret_avg = ycbcr_to_rgb(secret_avg) # Only when BW and we use YCbCr instead of RGB
                            secret_avg = secret_avg.to(device)
                            secret_avg = F.interpolate(secret_avg, size=(c.cropsize, c.cropsize), mode=c.Upsample_mode)
                        
                        
                        steg_low = output_steg.narrow(1, 0, c.channels_in)
                        cover_low = cover_input.narrow(1, 0, c.channels_in)
                        l_loss = hybrid_l2Loss(steg_low, cover_low)
                        cr_loss = hybrid_l1Loss((cover * (1 - tensor_mask_3c)).cuda(), cover_rev * (1 - tensor_mask_3c))

                        n_loss = hybrid_l2Loss(output_z_backward, output_z)
                        if c.YCC or c.ND2:
                            tensor_mask_3c = tensor_mask_3c[:,:2,:,:]
                        elif c.BW:
                            tensor_mask_3c = tensor_mask_3c[:,:1,:,:]

                        r_loss = hybrid_l1Loss(secret_rev * (1 - tensor_mask_3c), secret * (1 - tensor_mask_3c), 21)
                        
                else:
                    coef = 0
                    if c.TFD:
                        if stop_f_train == False:
                            loc_loss = localization_loss(ret_mask, real_mask)
                            coef = 1
                        else:
                            loc_loss = 0
                    else:
                        loc_loss = 0


                    if c.TE:
                        g_loss = guide_loss(steg_img.cuda(), cover.cuda())
                        # g_loss = perc_loss(steg_img.cuda(), cover.cuda())
                        
                        r_loss = reconstruction_loss(secret_rev * (1 - tensor_mask_3c[:,:2,:,:]), secret * (1 - tensor_mask_3c[:,:2,:,:]))
                        steg_low = output_steg.narrow(1, 0, c.channels_in)
                        cover_low = cover_input.narrow(1, 0, c.channels_in)
                        l_loss = low_frequency_loss(steg_low, cover_low)

                        n_loss = noise_loss(output_z_backward, output_z)
                        cr_loss = cover_reconstruction_loss((cover * (1 - tensor_mask_3c)).cuda(), cover_rev * (1 - tensor_mask_3c))

                if c.TE:
                    # total_loss = 8 * r_loss + 16 * g_loss + n_loss + cr_loss + l_loss
                    if i_epoch < 20:
                        total_loss = 8 * r_loss + 16 * g_loss + n_loss + cr_loss + l_loss + 16 * coef * loc_loss
                    elif i_epoch < 200:
                        total_loss = 8 * r_loss + 16 * g_loss + n_loss + cr_loss + l_loss + 4 * coef * loc_loss
                    elif i_epoch < 400:
                        total_loss = 4 * r_loss + 6 * g_loss + n_loss + cr_loss + l_loss + coef * loc_loss
                    else:
                        total_loss = 4 * r_loss + 4 * g_loss + n_loss + cr_loss + l_loss +  coef * loc_loss
                # if c.TE:
                #     if i_epoch < 20:
                #         total_loss = 16 * r_loss + g_loss + n_loss + cr_loss + l_loss + 16 * coef * loc_loss
                #     elif i_epoch < 200:
                #         total_loss = 16 * r_loss + 16 * g_loss + n_loss + cr_loss + l_loss + 4 * coef * loc_loss
                #     elif i_epoch < 400:
                #         total_loss = 16 * r_loss + 8 * g_loss + n_loss + cr_loss + l_loss + coef * loc_loss
                #     else:
                #         total_loss = 16 * r_loss + 4 * g_loss + n_loss + cr_loss + l_loss +  coef * loc_loss
                else:
                    total_loss = coef * loc_loss

                if c.TE:
                    n_loss_history.append([n_loss.item(), 0.])
                    g_loss_history.append([g_loss.item(), 0.])
                    r_loss_history.append([r_loss.item(), 0.])
                    l_loss_history.append([l_loss.item(), 0.])
                    # loc_loss_history.append([loc_loss.item(), 0.])
                    cr_loss_history.append([cr_loss.item(), 0.])

                
                total_loss.backward()
                optim.step()
                

                loss_history.append([total_loss.item(), 0.])

                    

            epoch_losses = np.mean(np.array(loss_history), axis=0)

            if c.TE:
                n_epoch_losses = np.mean(np.array(n_loss_history), axis=0)
                g_epoch_losses = np.mean(np.array(g_loss_history), axis=0)
                r_epoch_losses = np.mean(np.array(r_loss_history), axis=0)
                l_epoch_losses = np.mean(np.array(l_loss_history), axis=0)
                # loc_epoch_losses = np.mean(np.array(loc_loss_history), axis=0)
                cr_epoch_losses = np.mean(np.array(cr_loss_history), axis=0)

            epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])

            #################
            #     val:    #
            #################
            if i_epoch % c.val_freq == 0 or i_epoch == 1:
                with torch.no_grad():
                    psnr_s = []
                    psnr_c = []
                    psnr_m = []
                    net.eval()
                    if c.TFD:
                        forgery_detector.eval()
                    val_counter = 0
                    MaskPSNR = 0
                    SecPSNR = 0
                    for i, batch in enumerate(tqdm(datasets.testloader)):
                        if c.AA:
                            cover, secret, code_block, message = batch
                            # Process the 4 values
                        else:
                            cover, secret = batch
                        if c.BW:
                            secret = torch.mean(secret, dim=1, keepdim=True)
                        cover_input = dwt(cover)
                        secret_input = dwt(secret)
                        
                        input_img = torch.cat((cover_input, secret_input), 1)

                        #################
                        #    forward:   #
                        #################

                        output = net(input_img)
                        if c.TF == 'DCT':
                            output_steg = output.narrow(1, 0, c.channels_in)
                            output_z = output.narrow(1, c.channels_in, output.shape[1] - c.channels_in)
                        else:
                            output_steg = output.narrow(1, 0, 4 * c.channels_in)
                            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                        
                        steg = iwt(output_steg)
                            
                        steg = torch.clamp(steg, 0 , 1)

                        [noised, mask] = distortion_layers_test([steg, cover])
                        noised = noised[0]
                        mask = mask.to(device)
                        if mask == None:
                            # print("mask is none!")
                            mask = torch.zeros((cover[0].shape[0], 1, cover[0].shape[2], cover[0].shape[3])).to(device)
                        
                        if mask.shape[1] == 1:
                            mask = mask.repeat(1, 3, 1, 1)

                        # print(mask)
                        mask = mask/255


                        # if c.TFD:
                        #     inv_input = noised * (1 - tensor_mask_3c)
                        # else:
                        inv_input = noised
                        
                        output_steg = dwt(inv_input)
                            
                        if c.NOISE_TYPE == "Gaussian":
                            output_z_backward = gauss_noise(output_z.shape)
                        elif c.NOISE_TYPE == "Uniform":
                            output_z_backward = uniform_noise(output_z.shape)
                        elif c.NOISE_TYPE == "Constant":
                            output_z_backward = 0.5 * torch.ones(output_z.shape).to(device)

                        #################
                        #   backward:   #
                        #################
                        output_steg = output_steg.to(device)

                        output_rev = torch.cat((output_steg, output_z_backward), 1)
                        output_image = net(output_rev, rev=True)

                        if c.TF == 'DCT':
                            secret_rev = output_image.narrow(1, c.channels_in, c.channels_in)
                        else:    
                            secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] -  4 * c.channels_in)

                        secret_rev = iwt(secret_rev)

                        if c.TFD:
                            secret_rev = torch.clamp(secret_rev, 0 , 1)
                            # print(secret_rev.shape)
                            
                            extracted_sec = DeBlocking(secret_rev)
                            if c.YCC:
                                extracted_sec = ycbcr_to_rgb(extracted_sec)
                            extracted_sec = F.interpolate(extracted_sec, size=(c.cropsize_val, c.cropsize_val), mode=c.Upsample_mode)
                            


                            # fused_tensor = image_mix(extracted_sec, noised)
                            fused_tensor = torch.cat((noised, extracted_sec), 1)

                            ret_mask = forgery_detector(fused_tensor)
                            ret_mask = torch.sigmoid(ret_mask)
                            ret_mask = ret_mask.repeat(1, 3, 1, 1).to(device)

                        if val_counter % 30 == 5:
                            torchvision.utils.save_image(cover, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_1cover.png' % (i_epoch, val_counter))
                            torchvision.utils.save_image(steg, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_2steg.png' % (i_epoch, val_counter))
                            torchvision.utils.save_image(noised, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_3noised.png' % (i_epoch, val_counter))

                            torchvision.utils.save_image(mask, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_4mask.png' % (i_epoch, val_counter))
                            if c.TFD:
                                torchvision.utils.save_image(ret_mask, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_5ret_mask.png' % (i_epoch, val_counter))
                                # torchvision.utils.save_image(fused_tensor, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_6mix_image.png' % (i_epoch, val_counter))

                            if c.AA:
                                extracted_sec, code_avg_ext = DeBlocking(secret_rev.detach().cpu())
                            else:
                                extracted_sec = DeBlocking(secret_rev.detach().cpu())
                            extracted_sec = ycbcr_to_rgb(extracted_sec)

                            if c.AA:
                                sec_rgb, code_avg = DeBlocking(secret.detach().cpu())
                                # print(code_avg)
                            else:
                                sec_rgb = DeBlocking(secret.detach().cpu())
                            sec_rgb = ycbcr_to_rgb(sec_rgb)

                            
                            torchvision.utils.save_image(sec_rgb, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_6th.png' % (i_epoch, val_counter))
                            torchvision.utils.save_image(extracted_sec, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_7th-rev.png' % (i_epoch, val_counter))
                            torchvision.utils.save_image(secret, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_8secret.png' % (i_epoch, val_counter))
                            torchvision.utils.save_image(secret_rev, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_9secret-rev.png' % (i_epoch, val_counter))
                            if c.AA:
                                # print(code_avg.shape)
                                if c.BW or c.ND2 or c.ND3:
                                    torchvision.utils.save_image(code_avg[:,:,:], c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_acode.png' % (i_epoch, val_counter))
                                    torchvision.utils.save_image(code_avg_ext[:,:,:], c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_bcode-ext.png' % (i_epoch, val_counter))
                                elif c.YCC:
                                    torchvision.utils.save_image(code_avg[:,1,:,:], c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_acode.png' % (i_epoch, val_counter))
                                    torchvision.utils.save_image(code_avg_ext[:,1,:,:], c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_bcode-ext.png' % (i_epoch, val_counter))
                            # else:
                            #     torchvision.utils.save_image(secret, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_7secret.png' % (i_epoch, val_counter))
                            #     torchvision.utils.save_image(secret_rev, c.VAL_IMAGE_PATH + '%.3d_epoch_%.5d_8secret-rev.png' % (i_epoch, val_counter))
                            
                        val_counter = val_counter + 1

                        secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                        np.clip(secret_rev, 0, 255)

                        secret = secret.cpu().numpy().squeeze() * 255
                        np.clip(secret, 0, 255)

                        
                        mask = mask.cpu().numpy().squeeze() * 255
                        np.clip(mask, 0, 255)

                        if c.TFD:
                            ret_mask = ret_mask.cpu().numpy().squeeze() * 255
                            # ret_mask = ret_mask.transpose((1, 2, 0))
                            np.clip(ret_mask, 0, 255)

                        cover = cover.cpu().numpy().squeeze() * 255
                        np.clip(cover, 0, 255)

                        steg = steg.cpu().numpy().squeeze() * 255
                        np.clip(steg, 0, 255)

                        psnr_temp = computePSNR(secret_rev, secret)
                        psnr_s.append(psnr_temp)

                        psnr_temp_c = computePSNR(cover, steg)
                        psnr_c.append(psnr_temp_c)
                        if c.TFD:
                            psnr_m.append(computePSNR(ret_mask, mask))

                    SecPSNR = np.mean(psnr_s)
                    # writer.add_scalars("PSNR", {"psnr_s": SecPSNR}, i_epoch)

                    print("PSNR of cover & steg:", np.mean(psnr_c))
                    print("PSNR of secret & retrieved:", SecPSNR)
                    if c.TFD:
                        MaskPSNR = np.mean(psnr_m)
                        print("PSNR of mask & retrieved mask:", MaskPSNR)
                        # writer.add_scalars("PSNR", {"psnr_m": MaskPSNR}, i_epoch)
                        # if MaskPSNR >= 22:
                        #     stop_f_train = True
                        # if MaskPSNR <=18:
                        #     stop_f_train = False
                    # writer.add_scalars("PSNR", {"psnr_c": np.mean(psnr_c)}, i_epoch)

            show_loss(epoch_losses)

            # writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)

            # if c.TE:
            #     writer.add_scalars("Encoder", {"Noise_Loss": n_epoch_losses[0]}, i_epoch)
            #     writer.add_scalars("Encoder", {"Guide_Reconstruction_Loss": g_epoch_losses[0]}, i_epoch)
            #     writer.add_scalars("Encoder", {"Low_Frequency_Loss": l_epoch_losses[0]}, i_epoch)
            #     writer.add_scalars("Decoder", {"Reconstruction_Reveal_Loss": r_epoch_losses[0]}, i_epoch)
            #     writer.add_scalars("Decoder", {"Cover_Reconstruction_Reveal_Loss": cr_epoch_losses[0]}, i_epoch)


            if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
                if c.TFD:
                    torch.save({'opt': optim.state_dict(),
                                'net': net.state_dict(),
                                'fd': forgery_detector.state_dict()}, c.MODEL_PATH + 'model_forgery_%.5i' % i_epoch + '.pt')
                else:
                    torch.save({'opt': optim.state_dict(),
                                'net': net.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i' % i_epoch + '.pt')
            weight_scheduler.step()
        
        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, c.MODEL_PATH + 'model' + '.pt')
        writer.close()

    except:
        if c.checkpoint_on_error:
        
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, c.MODEL_PATH + 'model_ERROR' + '.pt')
        raise

    finally:
        signal_stop()

if __name__ == '__main__':
    main()

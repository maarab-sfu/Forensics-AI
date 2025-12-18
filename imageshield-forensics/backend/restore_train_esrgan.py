import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import math

import argparse
import os
import itertools
import config as c
import sys

from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, transforms_=None):

        self.transform = transforms.Compose(transforms_)
       
        # List of secret-restored image paths for future expansion
        self.sec_restored_paths = [
            sorted(glob.glob(c.RESTORE_PATH + "JPEG/secret-rev/*.png")),
            sorted(glob.glob(c.RESTORE_PATH + "GN/secret-rev/*.png")),
            sorted(glob.glob(c.RESTORE_PATH + "GB/secret-rev/*.png")),
        ]
        self.files = sorted(glob.glob(c.RESTORE_PATH + "cover/*.png"))
        self.img_size = c.cropsize_val
        

    def __getitem__(self, index):

        image = Image.open(self.files[index])
        image = to_rgb(image)
        image = self.transform(image)

        # Randomly choose from the secret-restored options
        secrev_paths_at_index = [paths[index] for paths in self.sec_restored_paths]
        secrev_path = random.choice(secrev_paths_at_index)

        # Load the selected secret restored image
        secrev = Image.open(secrev_path)
        secrev = to_rgb(secrev)  # Ensure it's RGB
        secrev = self.transform(secrev)

        return {"hr": image, "lr": secrev}

    def __len__(self):
        return len(self.files)

def disect_secrev(im):
    if im.dim() == 3:
        # If there's no batch dimension, add it
        im = im.unsqueeze(0)
    N, _, W, H = im.shape

    # print(im.shape)
    img = im[:, 0, :, :].squeeze(1)  # Y channel
    # print(img.shape)
    quadrant_size_w = W // c.thumbnail_number
    quadrant_size_h = H // c.thumbnail_number

    images = []

    # Calculate the height and width of each patch. Each path is a 2X2 pattern of R,G,B, and hash code.
    patch_height = W // (c.thumbnail_number*2)
    patch_width = H // (c.thumbnail_number*2)

    for i in range(c.thumbnail_number):
        for j in range(c.thumbnail_number):
            Y = img[:, (j*2*patch_width) + 0: (j*2*patch_width) + patch_width,
                                (i*2*patch_height) + 0: (i*2*patch_height) + patch_height]
            Cb = img[:, (j*2*patch_width) + patch_width: (j*2*patch_width) + (2 * patch_width),
                                (i*2*patch_height) + 0: (i*2*patch_height) + patch_height]
            Cr = img[:, (j*2*patch_width) + 0: (j*2*patch_width) + patch_width,
                                (i*2*patch_height) + patch_height: (i*2*patch_height) + (2 * patch_height)]
            
            imgg = torch.stack([Y, Cb, Cr], dim=1)
            # print(imgg.shape)
            images.append(imgg)

    return images



class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()

        # Update input channels to handle 4 low-resolution images concatenated
        self.conv1 = nn.Conv2d(channels * 4, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x1, x2, x3, x4):
        # Concatenate the 4 low-resolution images along the channel dimension
        x = torch.cat((x1, x2, x3, x4), dim=1)

        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


def main():
    os.makedirs("images/training", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=512, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=512, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
    parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=5000, help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize generator and discriminator
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
    feature_extractor = FeatureExtractor().to(device)

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    transforms_ = [
        transforms.Resize((opt.hr_height, opt.hr_height), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    dataloader = DataLoader(
        ImageDataset(transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, imgs in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i

            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            imgs_lr1, imgs_lr2, imgs_lr3, imgs_lr4 = disect_secrev(imgs_lr)

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_hr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_hr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr1, imgs_lr2, imgs_lr3, imgs_lr4)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            if batches_done < opt.warmup_batches:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
                )

                if batches_done % opt.sample_interval == 0:
                    # Save image grid with upsampled inputs and ESRGAN outputs
                    # imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                    img_grid = denormalize(torch.cat((imgs_hr, gen_hr), -1))
                    save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)

                if batches_done % opt.checkpoint_interval == 0:
                    # Save model checkpoints
                    torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
                    torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" %epoch)
                continue

            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr).detach()
            loss_content = criterion_content(gen_features, real_features)

            # Total generator loss
            loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_content.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                )
            )

            if batches_done % opt.sample_interval == 0:
                # Save image grid with upsampled inputs and ESRGAN outputs
                # imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                img_grid = denormalize(torch.cat((imgs_hr, gen_hr), -1))
                save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)

            if batches_done % opt.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
                torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" %epoch)

if __name__ == '__main__':
    main()
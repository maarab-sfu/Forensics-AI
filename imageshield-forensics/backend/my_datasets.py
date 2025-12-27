import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
import torch
from natsort import natsorted
import numpy as np
import cv2
import os
import math
import random
from torchvision.transforms.functional import crop,resize

def val_resize(image):
    return resize(image, (c.cropsize_val, c.cropsize_val))

def edit_resize(image):
    # Resize the image to a random shape equal to or smaller than c.cropsize_val//4
    random_size = random.randint(c.cropsize_val // 4, c.cropsize_val // 1.5)
    resized_image = resize(image, random_size)
    
    # Generate a random position to place the resized image within a canvas of size c.cropsize_val
    canvas_size = c.cropsize_val
    x_offset = random.randint(0, canvas_size - random_size)
    y_offset = random.randint(0, canvas_size - random_size)

    # Create a blank canvas
    canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))

    # Place the resized image onto the canvas at the random position
    canvas.paste(resized_image, (x_offset, y_offset))

    return canvas


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

def to_rgba(image):
    rgb_image = Image.new("RGBA", image.size)
    rgb_image.paste(image)
    return rgb_image


def data_aug(img, mode=0):  # img: W*H
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img, axes=(0, 1))
    elif mode == 3:
        return np.flipud(np.rot90(img, axes=(0, 1)))
    elif mode == 4:
        return np.rot90(img, k=2, axes=(0, 1))
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2, axes=(0, 1)))
    elif mode == 6:
        return np.rot90(img, k=3, axes=(0, 1))
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3, axes=(0, 1)))

class Forgery_Dataset(Dataset):
    def __init__(self, mode='train', root='./dataset/', transforms_=None):
        self.root = root
        self.mode = mode
        self.transforms = transforms_

        # self.image_files = sorted(glob.glob(os.path.join(self.root, self.mode, 'mix', "*.png")))
        # self.label_files = sorted(glob.glob(os.path.join(self.root, self.mode, 'localization-map', "*.png")))

        self.image_files = sorted(glob.glob(os.path.join(c.IMAGE_PATH, 'mix', "*.png")))
        self.label_files = sorted(glob.glob(os.path.join(c.IMAGE_PATH, 'localization-map', "*.png")))


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.image_files[idx], cv2.IMREAD_COLOR)
        img_bgr = cv2.resize(img_bgr, (512, 512))
        img_rgb = img_bgr[:, :, ::-1] / 255

        label = cv2.imread(self.label_files[idx], cv2.IMREAD_GRAYSCALE)
        label = label[:,:]/255

        if self.mode == "train" or self.mode == "val":
            random_dataaug = random.randint(0, 7)
            img_rgb = data_aug(img_rgb, mode=random_dataaug).transpose((2, 0, 1))  # C*W*H
            label = data_aug(label, mode=random_dataaug)
        elif self.mode == "test":
            img_rgb = img_rgb.transpose((2, 0, 1))  # C*W*H

        tensor_img = torch.from_numpy(img_rgb.astype(np.float32))
        tensor_label = torch.from_numpy(label.astype(np.float32))
        tensor_label = tensor_label.unsqueeze(0)

        return tensor_img, tensor_label


def make_secret(shapes, data_dim):

    # Ensure shapes has three dimensions, adding channels if necessary
    if c.BW:
        shapes = (1,) + shapes  # Adding a channel dimension for RGB
    else:
        shapes = (2,) + shapes  # Adding a channel dimension for RGB
    # print(shapes)
    message_bits = math.log2(data_dim)
    message_bytes = message_bits - 3
    if message_bytes % 2 == 0:
        message_shape = int(2 ** (message_bytes // 2))
    else:
        raise ValueError("Message bytes should be even for this calculation.")  # Optional error handling

    secret = np.zeros(shapes)
    message = np.zeros((message_shape, message_shape))

    for aa in range(message_shape):
        for bb in range(message_shape):
            rand_message = np.random.rand(1,1,1)
            # print(rand_message)
            message[aa,bb] = rand_message.squeeze()
            pad_size = int(shapes[-1]//message_shape)
            # one_channel = nn.ReplicationPad2d((pad_size-1, 0, pad_size-1, 0))(rand_message)
            one_channel = np.pad(rand_message, ((0, 0), (pad_size-1, 0), (pad_size-1, 0)), mode='edge')
            if c.BW:
                secret[:,aa*pad_size:(aa+1)*pad_size, bb*pad_size:(bb+1)*pad_size] = np.expand_dims(one_channel, axis=0)
            else:
                secret[:,aa*pad_size:(aa+1)*pad_size, bb*pad_size:(bb+1)*pad_size] = np.concatenate([one_channel, one_channel], axis=0)

    # print("secret: ", secret[0])
    # Normalize "secret"
    secret = (secret * 255).astype(int)
    # print("secret: (normalized)", secret[0])

    message = (message * 255).astype(np.int32)
    # print("message: ", message)
    message_binary = np.unpackbits(message.astype(np.uint8), axis=-1)
    message = message_binary.reshape((message_shape * message_shape * 8))
    # print("message bits: ", message)

    return secret, message

class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            self.files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
            self.img_size = c.cropsize
        elif mode == 'val':
            self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))
            self.img_size = c.cropsize_val
        elif mode == 'makenoise':
            self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))
            self.img_size = c.cropsize_val
        elif mode == 'restore':
            # List of secret-restored image paths for future expansion
            self.sec_restored_paths = [
                sorted(glob.glob(c.IMAGE_PATH + "JPEG/secret-rev/*.png")),
                sorted(glob.glob(c.IMAGE_PATH + "GN/secret-rev/*.png")),
                sorted(glob.glob(c.IMAGE_PATH + "GB/secret-rev/*.png")),
            ]
            self.files = sorted(glob.glob(c.IMAGE_PATH + "cover/*.png"))
            self.img_size = c.cropsize_val
        elif mode == 'restorecode':
            self.sec_restored = sorted(glob.glob(c.IMAGE_PATH + "secret-rev/*.png"))
            self.files = sorted(glob.glob(c.IMAGE_PATH + "secret/*.png"))
            self.img_size = c.cropsize_val
        elif mode == 'hash':
            self.distorted_files = glob.glob(c.IMAGE_PATH + "distorted/*.png")
            self.files = glob.glob(c.IMAGE_PATH + "steg/*.png")
            self.tampered_files = glob.glob(c.IMAGE_PATH + "noised/*.png")

            random.shuffle(self.distorted_files)
            random.shuffle(self.files)
            random.shuffle(self.tampered_files)

            self.img_size = c.cropsize_val
        elif mode == 'localize':
            self.files = sorted(glob.glob(c.IMAGE_PATH_noised + "*.png"))
            self.restored = sorted(glob.glob(c.IMAGE_PATH_restored + "*.png"))
            self.masks = sorted(glob.glob(c.IMAGE_PATH_localization_map + "*.png"))
            self.img_size = c.cropsize_val
        elif mode == 'imagemix':
            self.files = sorted(glob.glob(c.IMAGE_PATH + "noised/*.png"))
            self.restored = sorted(glob.glob(c.IMAGE_PATH + "restored/*.png"))
            self.masks = sorted(glob.glob(c.IMAGE_PATH + "extracted-map/*.png"))
            self.covers = sorted(glob.glob(c.IMAGE_PATH + "cover/*.png"))
            self.img_size = c.cropsize_val


        elif mode == 'edit':
            self.edit_transform = transform_edit
            self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))
            self.objects = natsorted(sorted(glob.glob(c.OBJECT_PATH + "/*." + "png")))
            self.img_size = c.cropsize_val

        elif mode == 'extract':
            self.files = sorted(glob.glob(c.IMAGE_PATH_double_WM + "/*.png"))
            self.old_sec = sorted(glob.glob(c.IMAGE_PATH + "secret-rev/*.png"))
            self.img_size = c.cropsize_val
        elif mode == 'tile-rev':
            self.files = sorted(glob.glob(c.IMAGE_PATH_secret_rev + "/*." + c.format_val))
            self.img_size = c.cropsize_val
        elif mode == 'tile':
            self.files = sorted(glob.glob(c.IMAGE_PATH_secret + "/*." + c.format_val))
            self.img_size = c.cropsize_val

    def __getitem__(self, index):

        if self.mode == 'makenoise':
            image = Image.open(self.files[index])
            image = to_rgb(image)
            cover = self.transform(image)
            return cover

        if self.mode == 'hash':
            anchor = Image.open(self.files[index])
            anchor = to_rgb(anchor)
            anchor = self.transform(anchor)

            positive = Image.open(self.files[index])
            positive = to_rgb(positive)
            positive = self.transform(positive)

            negative = Image.open(self.files[index])
            negative = to_rgb(negative)
            negative = self.transform(negative)

            return anchor, positive, negative

        if self.mode == 'restore' or self.mode == 'restorecode':
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

            return image, secrev
        
        if self.mode == 'localize':
            noised = Image.open(self.files[index])
            noised = to_rgb(noised)
            noised = self.transform(noised)

            restored = Image.open(self.restored[index])
            restored = to_rgb(restored)
            restored = self.transform(restored)

            mask = Image.open(self.masks[index]).convert('L')
            mask = self.transform(mask)

            return noised, restored, mask

        if self.mode == 'imagemix':
            cover = Image.open(self.covers[index])
            cover = to_rgb(cover)
            cover = self.transform(cover)

            noised = Image.open(self.files[index])
            noised = to_rgb(noised)
            noised = self.transform(noised)

            restored = Image.open(self.restored[index])
            restored = to_rgb(restored)
            restored = self.transform(restored)

            mask = Image.open(self.masks[index]).convert('L')
            mask = self.transform(mask)

            return cover, noised, restored, mask
        
        image = Image.open(self.files[index])
        image = to_rgb(image)

        if self.mode == 'train' or self.mode == 'val' or self.mode == 'test':
            if c.BW:
                if self.mode == 'train':
                    channel1 = np.zeros((c.cropsize, c.cropsize), dtype=np.uint8)
                else:
                    channel1 = np.zeros((c.cropsize_val, c.cropsize_val), dtype=np.uint8)

                img_ycbcr = image.convert('YCbCr')

                # Get the Y, Cb, and Cr channels
                r, g, b = img_ycbcr.split()

                # Resize R, G, and B channels according to the c.thumbnail_number value
                r_resized = np.array(r.resize((self.img_size // (c.thumbnail_number*2), self.img_size // (c.thumbnail_number*2)), Image.LANCZOS))
                g_resized = np.array(g.resize((self.img_size // (c.thumbnail_number*2), self.img_size // (c.thumbnail_number*2)), Image.LANCZOS))
                b_resized = np.array(b.resize((self.img_size // (c.thumbnail_number*2), self.img_size // (c.thumbnail_number*2)), Image.LANCZOS))

                
                if self.mode == 'train':
                    code_block, message = make_secret((c.cropsize//(c.thumbnail_number*2), c.cropsize//(c.thumbnail_number*2)), c.secret_length)
                else:
                    code_block, message = make_secret((c.cropsize_val//(c.thumbnail_number*2), c.cropsize_val//(c.thumbnail_number*2)), c.secret_length)
                N, H , W = code_block.shape
                # print("H: ", H)
                # print("W: ", W)
                # print("code_block: ", code_block)
                # print("channel1: ", channel1)
                for i in range(c.thumbnail_number):
                    for j in range(c.thumbnail_number):
                        channel1[(j*2*H) + 0: (j*2*H) + H, (i*2*W) + 0: (i*2*W) + W] = r_resized
                        channel1[(j*2*H) + H:(j*2*H) + (2 * H), (i*2*W) + 0: (i*2*W) + W] = g_resized
                        channel1[(j*2*H) + 0: (j*2*H) + H, (i*2*W) + W:(i*2*W) + (2 * W)] = b_resized
                        if c.AA:
                            channel1[(j*2*H) + H:(j*2*H) + (2 * H), (i*2*W) + W:(i*2*W) + (2 * W)] = code_block[0,:,:]
                        else:
                            channel1[(j*2*H) + H:(j*2*H) + (2 * H), (i*2*W) + W:(i*2*W) + (2 * W)] = r_resized

                # Convert the numpy arrays back to images
                channel1_image = Image.fromarray(channel1)
                # print("channel 1 size: ", channel1_image.size)
                

                # Apply transformations if needed (replace with your transform method)
                secret = self.transform(channel1_image)
                cover = self.transform(image)

                if c.AA:
                    return cover, secret, code_block, message
                return cover, secret

            elif c.YCC:
                img_ycbcr = image.convert('YCbCr')

                # Get the Y, Cb, and Cr channels
                y, cb, cr = img_ycbcr.split()

                # Resize Y, Cb, and Cr channels according to the c.thumbnail_number value
                y_resized = y.resize((self.img_size // c.thumbnail_number, self.img_size // c.thumbnail_number), Image.LANCZOS)
                cb_resized = cb.resize((self.img_size // (2 * c.thumbnail_number), self.img_size // (2 * c.thumbnail_number)), Image.LANCZOS)
                cr_resized = cr.resize((self.img_size // (2 * c.thumbnail_number), self.img_size // (2 * c.thumbnail_number)), Image.LANCZOS)

                # Repeat the Y channel to form an LxL grid
                channel1 = np.array(y_resized)
                channel1 = np.tile(channel1, (c.thumbnail_number, c.thumbnail_number))


                # Create the chessboard pattern for the Cb and Cr channels
                cb_array = np.array(cb_resized)
                cr_array = np.array(cr_resized)

                # Initialize an empty array for the combined CbCr pattern
                combined_shape = (2 * c.thumbnail_number * cb_array.shape[0], 2 * c.thumbnail_number * cb_array.shape[1])
                channel2 = np.zeros(combined_shape, dtype=np.uint8)

                for i in range(2 * c.thumbnail_number):
                    for j in range(2 * c.thumbnail_number):
                        if (i + j) % 2 == 0:
                            # Place Cb in even (i, j) positions
                            channel2[i * cb_array.shape[0]:(i + 1) * cb_array.shape[0],
                                    j * cb_array.shape[1]:(j + 1) * cb_array.shape[1]] = cb_array
                        else:
                            # Place Cr in odd (i, j) positions
                            channel2[i * cr_array.shape[0]:(i + 1) * cr_array.shape[0],
                                    j * cr_array.shape[1]:(j + 1) * cr_array.shape[1]] = cr_array

                if c.AA:
                    if self.mode == 'train':
                        code_block, message = make_secret((c.cropsize//c.thumbnail_number, c.cropsize//c.thumbnail_number), c.secret_length)
                    else:
                        code_block, message = make_secret((c.cropsize_val//c.thumbnail_number, c.cropsize_val//c.thumbnail_number), c.secret_length)
                    # print("code_block: ", code_block)
                    # print("channel1: ", channel1)
                    for i in range(c.thumbnail_number):
                        for j in range(c.thumbnail_number):
                            if (i + j) % 2 == 0:
                                channel1[i * code_block.shape[1]:(i + 1) * code_block.shape[1],
                                    j * code_block.shape[2]:(j + 1) * code_block.shape[2]] = code_block[0,:,:]
                                channel2[i * code_block.shape[1]:(i + 1) * code_block.shape[1],
                                    j * code_block.shape[2]:(j + 1) * code_block.shape[2]] = code_block[1,:,:]


                # Convert the numpy arrays back to images
                channel1_image = Image.fromarray(channel1)
                channel2_image = Image.fromarray(channel2)
                # print("channel 1 size: ", channel1_image.size)
                # print("channel 2 size: ", channel2_image.size)

                # Merge the two channels (channel1 as luminance and channel2 as alpha)
                two_channel_image = Image.merge('LA', (channel1_image, channel2_image))
                
                # Apply transformations if needed (replace with your transform method)
                secret = self.transform(two_channel_image)
                cover = self.transform(image)

                if c.AA:
                    return cover, secret, code_block, message
                return cover, secret

            elif c.ND1:
                img_ycbcr = image.convert('YCbCr')

                # Get the Y, Cb, and Cr channels
                y, cb, cr = img_ycbcr.split()

                # Resize Y, Cb, and Cr channels according to the c.thumbnail_number value
                y_resized = y.resize((self.img_size // c.thumbnail_number, self.img_size // c.thumbnail_number), Image.LANCZOS)
                cb_resized = cb.resize((self.img_size // (2 * c.thumbnail_number), self.img_size // (2 * c.thumbnail_number)), Image.LANCZOS)
                cr_resized = cr.resize((self.img_size // (2 * c.thumbnail_number), self.img_size // (2 * c.thumbnail_number)), Image.LANCZOS)

                # Repeat the Y channel to form an LxL grid
                channel1 = np.array(y_resized)
                channel1 = np.tile(channel1, (c.thumbnail_number, c.thumbnail_number))


                # Create the chessboard pattern for the Cb and Cr channels
                cb_array = np.array(cb_resized)
                cr_array = np.array(cr_resized)

                # Initialize an empty array for the combined CbCr pattern
                combined_shape = (2 * c.thumbnail_number * cb_array.shape[0], 2 * c.thumbnail_number * cb_array.shape[1])
                channel2 = np.zeros(combined_shape, dtype=np.uint8)

                for i in range(2 * c.thumbnail_number):
                    for j in range(2 * c.thumbnail_number):
                        if (i + j) % 2 == 0:
                            # Place Cb in even (i, j) positions
                            channel2[i * cb_array.shape[0]:(i + 1) * cb_array.shape[0],
                                    j * cb_array.shape[1]:(j + 1) * cb_array.shape[1]] = cb_array
                        else:
                            # Place Cr in odd (i, j) positions
                            channel2[i * cr_array.shape[0]:(i + 1) * cr_array.shape[0],
                                    j * cr_array.shape[1]:(j + 1) * cr_array.shape[1]] = cr_array

                if c.AA:
                    if self.mode == 'train':
                        code_block, message = make_secret((c.cropsize//c.thumbnail_number, c.cropsize//c.thumbnail_number), c.secret_length)
                    else:
                        code_block, message = make_secret((c.cropsize_val//c.thumbnail_number, c.cropsize_val//c.thumbnail_number), c.secret_length)
                    # print("code_block: ", code_block)
                    # print("channel1: ", channel1)
                    for i in range(c.thumbnail_number):
                        for j in range(c.thumbnail_number):
                            if (i + j) % 2 == 0:
                                channel1[i * code_block.shape[1]:(i + 1) * code_block.shape[1],
                                    j * code_block.shape[2]:(j + 1) * code_block.shape[2]] = code_block[0,:,:]
                                channel2[i * code_block.shape[1]:(i + 1) * code_block.shape[1],
                                    j * code_block.shape[2]:(j + 1) * code_block.shape[2]] = code_block[1,:,:]


                # Convert the numpy arrays back to images
                channel1_image = Image.fromarray(channel1)
                channel2_image = Image.fromarray(channel2)
                # print("channel 1 size: ", channel1_image.size)
                # print("channel 2 size: ", channel2_image.size)

                # Merge the two channels (channel1 as luminance and channel2 as alpha)
                two_channel_image = Image.merge('LA', (channel1_image, channel2_image))
                
                # Apply transformations if needed (replace with your transform method)
                secret = self.transform(two_channel_image)
                cover = self.transform(image)

                if c.AA:
                    return cover, secret, code_block, message
                return cover, secret
            
            elif c.ND2:
                img_ycbcr = image.convert('YCbCr')

                # Get the Y, Cb, and Cr channels
                y, cb, cr = img_ycbcr.split()

                # Resize Y, Cb, and Cr channels according to the c.thumbnail_number value
                y_resized = y.resize((self.img_size // c.thumbnail_number, self.img_size // c.thumbnail_number), Image.LANCZOS)
                cb_resized = cb.resize((self.img_size // (2 * c.thumbnail_number), self.img_size // (2 * c.thumbnail_number)), Image.LANCZOS)
                cr_resized = cr.resize((self.img_size // (2 * c.thumbnail_number), self.img_size // (2 * c.thumbnail_number)), Image.LANCZOS)

                # Repeat the Y channel to form an LxL grid
                channel1 = np.array(y_resized)
                channel1 = np.tile(channel1, (c.thumbnail_number, c.thumbnail_number))


                # Create the chessboard pattern for the Cb and Cr channels
                cb_array = np.array(cb_resized)
                cr_array = np.array(cr_resized)

                # Initialize an empty array for the combined CbCr pattern
                combined_shape = (2 * c.thumbnail_number * cb_array.shape[0], 2 * c.thumbnail_number * cb_array.shape[1])
                channel2 = np.zeros(combined_shape, dtype=np.uint8)

                if self.mode == 'train':
                    code_block, message = make_secret((c.cropsize//(2 * c.thumbnail_number), c.cropsize//(2 * c.thumbnail_number)), c.secret_length)
                else:
                    code_block, message = make_secret((c.cropsize_val//(2 * c.thumbnail_number), c.cropsize_val//(2 * c.thumbnail_number)), c.secret_length)

                for i in range(2 * c.thumbnail_number):
                    for j in range(2 * c.thumbnail_number):
                        if (i + j) % 2 == 1:
                            channel2[i * code_block.shape[1]:(i + 1) * code_block.shape[1],
                                    j * code_block.shape[2]:(j + 1) * code_block.shape[2]] = code_block[1,:,:]
                        else:
                            if i % 2 == 0:
                                # Place Cb in even (i, j) positions
                                # print("yo")
                                channel2[i * cb_array.shape[0]:(i + 1) * cb_array.shape[0],
                                        j * cb_array.shape[1]:(j + 1) * cb_array.shape[1]] = cb_array
                            else:
                                # Place Cr in odd (i, j) positions
                                channel2[i * cr_array.shape[0]:(i + 1) * cr_array.shape[0],
                                        j * cr_array.shape[1]:(j + 1) * cr_array.shape[1]] = cr_array

                # Convert the numpy arrays back to images
                channel1_image = Image.fromarray(channel1)
                channel2_image = Image.fromarray(channel2)
                # print("channel 1 size: ", channel1_image.size)
                # print("channel 2 size: ", channel2_image.size)

                # Merge the two channels (channel1 as luminance and channel2 as alpha)
                two_channel_image = Image.merge('LA', (channel1_image, channel2_image))
                
                # Apply transformations if needed (replace with your transform method)
                secret = self.transform(two_channel_image)
                cover = self.transform(image)

                if c.AA:
                    return cover, secret, code_block, message
                return cover, secret 

            elif c.ND3:
                img_ycbcr = image.convert('YCbCr')

                # Get the Y, Cb, and Cr channels
                y, cb, cr = img_ycbcr.split()

                # Resize Y, Cb, and Cr channels according to the c.thumbnail_number value
                y_resized = y.resize((self.img_size // c.thumbnail_number, self.img_size // c.thumbnail_number), Image.LANCZOS)
                cb_resized = cb.resize((self.img_size // (2 * c.thumbnail_number), self.img_size // (2 * c.thumbnail_number)), Image.LANCZOS)
                cr_resized = cr.resize((self.img_size // (2 * c.thumbnail_number), self.img_size // (2 * c.thumbnail_number)), Image.LANCZOS)

                # Repeat the Y channel to form an LxL grid
                channel1 = np.array(y_resized)
                channel1 = np.tile(channel1, (c.thumbnail_number, c.thumbnail_number))


                # Create the chessboard pattern for the Cb and Cr channels
                cb_array = np.array(cb_resized)
                cr_array = np.array(cr_resized)

                # Initialize an empty array for the combined CbCr pattern
                combined_shape = (2 * c.thumbnail_number * cb_array.shape[0], 2 * c.thumbnail_number * cb_array.shape[1])
                channel2 = np.zeros(combined_shape, dtype=np.uint8)

                if self.mode == 'train':
                    code_block, message = make_secret((c.cropsize//(2 * c.thumbnail_number), c.cropsize//(2 * c.thumbnail_number)), c.secret_length)
                else:
                    code_block, message = make_secret((c.cropsize_val//(2 * c.thumbnail_number), c.cropsize_val//(2 * c.thumbnail_number)), c.secret_length)

                for i in range(2 * c.thumbnail_number):
                    for j in range(2 * c.thumbnail_number):
                        if (i + j) % 2 == 1:
                            channel2[i * code_block.shape[1]:(i + 1) * code_block.shape[1],
                                    j * code_block.shape[2]:(j + 1) * code_block.shape[2]] = code_block[1,:,:]
                        else:
                            if i % 2 == 0:
                                # Place Cb in even (i, j) positions
                                # print("yo")
                                channel2[i * cb_array.shape[0]:(i + 1) * cb_array.shape[0],
                                        j * cb_array.shape[1]:(j + 1) * cb_array.shape[1]] = cb_array
                            else:
                                # Place Cr in odd (i, j) positions
                                channel2[i * cr_array.shape[0]:(i + 1) * cr_array.shape[0],
                                        j * cr_array.shape[1]:(j + 1) * cr_array.shape[1]] = cr_array

                # Convert the numpy arrays back to images
                channel1_image = Image.fromarray(channel1)
                channel2_image = Image.fromarray(channel2)
                channel3_image = Image.fromarray(channel2)
                # print("channel 1 size: ", channel1_image.size)
                # print("channel 2 size: ", channel2_image.size)

                # Merge the two channels (channel1 as luminance and channel2 as alpha)
                three_channel_image = Image.merge('RGB', (channel1_image, channel2_image, channel3_image))
                
                # Apply transformations if needed (replace with your transform method)
                secret = self.transform(three_channel_image)
                cover = self.transform(image)

                if c.AA:
                    return cover, secret, code_block, message
                return cover, secret 

        elif self.mode == 'extract':
            # print(self.old_sec)

            item = Image.open(self.old_sec[index])
            item = to_rgb(item)
            item = self.transform(item)

            doubleWM = self.transform(image)


            return item, doubleWM
        elif self.mode == 'edit':
            theobject = Image.open(self.objects[index%len(self.objects)])
            theobject = to_rgba(theobject)

            random_index = random.randint(0, len(self.files) - 1)
            fakesecret = Image.open(self.files[random_index])
            fakesecret = to_rgb(fakesecret)
            if c.YCC:
                img_ycbcr = image.convert('YCbCr')
                if c.ChB:
                    img_ch = Image.open('ch.png')
                    img_ch = img_ch.convert('L')
                # Get the Y, Cb, and Cr channels
                y, cb, cr = img_ycbcr.split()

                # Apply chroma subsampling to Cb and Cr channels
                y_resized = y.resize((self.img_size // 2, self.img_size // 2), Image.LANCZOS)
                cb_resized = cb.resize((self.img_size // 4, self.img_size // 4), Image.LANCZOS)
                cr_resized = cr.resize((self.img_size // 4, self.img_size // 4), Image.LANCZOS)

                channel1 = np.hstack([np.array(y_resized)] * 2)
                channel1 = np.vstack([channel1, channel1])

                channel2 = np.hstack([np.array(cb_resized), np.array(cr_resized)])
                channel2_ = np.hstack([np.array(cr_resized), np.array(cb_resized)])
                channel2 = np.vstack([channel2, channel2_])

                channel2 = np.vstack([channel2, channel2])
                channel2 = np.hstack([channel2, channel2])

                channel1_image = Image.fromarray(channel1)
                channel2_image = Image.fromarray(channel2)

                if c.ChB:
                    three_channel_image = Image.merge('RGB', (channel1_image, channel2_image, img_ch))
                    secret = self.transform(three_channel_image)
                else:
                    two_channel_image = Image.merge('LA', (channel1_image, channel2_image))
                    secret = self.transform(two_channel_image)


                cover = self.transform(image)
                theobject = self.edit_transform(theobject)
                bin_map = theobject[3,:,:]
                theobject = theobject[0:3,:,:]




                img_ycbcr = fakesecret.convert('YCbCr')

                # Get the Y, Cb, and Cr channels
                y, cb, cr = img_ycbcr.split()

                # Apply chroma subsampling to Cb and Cr channels
                y_resized = y.resize((self.img_size // 2, self.img_size // 2), Image.LANCZOS)
                cb_resized = cb.resize((self.img_size // 4, self.img_size // 4), Image.LANCZOS)
                cr_resized = cr.resize((self.img_size // 4, self.img_size // 4), Image.LANCZOS)

                channel1 = np.hstack([np.array(y_resized)] * 2)
                channel1 = np.vstack([channel1, channel1])

                channel2 = np.hstack([np.array(cb_resized), np.array(cr_resized)])
                channel2_ = np.hstack([np.array(cr_resized), np.array(cb_resized)])
                channel2 = np.vstack([channel2, channel2_])

                channel2 = np.vstack([channel2, channel2])
                channel2 = np.hstack([channel2, channel2])

                channel1_image = Image.fromarray(channel1)
                channel2_image = Image.fromarray(channel2)


                if c.ChB:
                    three_channel_image = Image.merge('RGB', (channel1_image, channel2_image, img_ch))
                    fakesecret = self.transform(three_channel_image)
                else:
                    two_channel_image = Image.merge('LA', (channel1_image, channel2_image))
                    fakesecret = self.transform(two_channel_image)
                
                return cover, secret, fakesecret, theobject, bin_map.squeeze()

            else:
            
                thumbnail = image.resize((self.img_size//c.thumbnail_number, self.img_size//c.thumbnail_number))
                secret = Image.new('RGB', (self.img_size, self.img_size), color='darkgray')

                for i in range(c.thumbnail_number):
                    for j in range(c.thumbnail_number):
                        secret.paste(thumbnail, (i * thumbnail.size[0], j * thumbnail.size[1]))

                secret = to_rgb(secret)

                cover = self.transform(image)
                theobject = self.edit_transform(theobject)
                secret = T.ToTensor()(secret)

                return cover, secret, theobject


            files = os.listdir(folder_path)
            # Filter only image files
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
            if not image_files:
                print("No image files found in the folder.")
                return None
            
            # Select a random image file
            random_image = random.choice(image_files)
            # Form the full path to the image file
            image_path = os.path.join(folder_path, random_image)
            
            # Open the image using PIL
            image = Image.open(image_path)
            random_size = (random.randint(1, c.cropsize_val//4), random.randint(1, c.cropsize_val//4))
            resized_image = image.resize(random_size)

            

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)


transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    T.Lambda(val_resize),
    T.ToTensor(),
])

transform_hash = T.Compose([
    T.Lambda(val_resize),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] range
])

transform_edit = T.Compose([
    T.Lambda(edit_resize),
    T.ToTensor(),
])

# Forgery Detector datasets
# f_trainloader = DataLoader(
#     Forgery_Dataset(mode="train", transforms_ = transform_val),
#     batch_size=16,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=1,
#     drop_last=True
# )

# f_valloader = DataLoader(
#     Forgery_Dataset(mode="val", transforms_ = transform_val),
#     batch_size=1,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=1,
# )

# f_testloader = DataLoader(
#     Forgery_Dataset(mode="test", transforms_ = transform_val),
#     batch_size=1,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=1,
# )


# Training data loader
# trainloader = DataLoader(
#     Hinet_Dataset(transforms_=transform, mode="train"),
#     batch_size=c.batch_size,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=1,
#     drop_last=True
# )
# # Test data loader
# distortloader = DataLoader(
#     Hinet_Dataset(transforms_=transform_val, mode="makenoise"),
#     batch_size=c.batchsize_val,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=1,
# )

# restoreloader = DataLoader(
#     Hinet_Dataset(transforms_=transform_val, mode="restore"),
#     batch_size=c.batchsize_val,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=1,
# )

# restorecodeloader = DataLoader(
#     Hinet_Dataset(transforms_=transform_val, mode="restorecode"),
#     batch_size=c.batchsize_val,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=1,
# )

# hashloader = DataLoader(
#     Hinet_Dataset(transforms_=transform_hash, mode="hash"),
#     batch_size=c.batchsize_val,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=1,
# )

# localizeloader = DataLoader(
#     Hinet_Dataset(transforms_=transform_val, mode="localize"),
#     batch_size=c.batchsize_val,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=1,
# )

testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

# testloader_ex = DataLoader(
#     Hinet_Dataset(transforms_=transform_val, mode="extract"),
#     batch_size=c.batchsize_val,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=1,
#     drop_last=True
# )

# testloader_ed = DataLoader(
#     Hinet_Dataset(transforms_=transform_val, mode="edit"),
#     batch_size=c.batchsize_val,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=1,
#     drop_last=True
# )

# tileloader_rev = DataLoader(
#         Hinet_Dataset(transforms_=transform_val, mode="tile-rev"),
#         batch_size = 1,
#         shuffle = False,
#         pin_memory = True,
#         num_workers = 1,
#         drop_last = True
#     )

# tileloader = DataLoader(
#         Hinet_Dataset(transforms_=transform_val, mode="tile"),
#         batch_size = 1,
#         shuffle = False,
#         pin_memory = True,
#         num_workers = 1,
#         drop_last = True
#     )
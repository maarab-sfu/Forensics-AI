import math
import torch
import torch.nn

import torch.optim
import torchvision
import numpy as np
from PIL import Image
from distortions import *
import config as c
from invertible_net import ForgeryDetector
from tqdm import tqdm
from train import DeBlocking
from torchvision.transforms.functional import to_pil_image
import ecc
import bchecc


import imagehash

from model import *
import torch.nn.functional as F
import my_datasets
import os
import os.path
import modules.Unet_common as common
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchvision.transforms as transforms
from itertools import chain
import lpips

import metrics
if c.WM_MODEL == "InvisMark":
    import noise
    import train_watermark
elif c.WM_MODEL == "TrustMark":
    from tmark_code import TrustMark
elif c.WM_MODEL == "RedMark":
    from redmark import buildModel, embed_watermark, extract_watermark
import utils


# from polarecc import initialize_polar, embed_polar, extract_polar


import torchvision.utils as vutils

from PIL import Image
import kornia.augmentation as aug
from torchvision.utils import save_image
import torchvision.transforms as T
# from datasets import load_dataset
import io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
# from torchvision.transforms import v2


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load(net, optim, name):
    if torch.cuda.is_available():
        state_dicts = torch.load(name)
    else:
        state_dicts = torch.load(name, map_location=torch.device('cpu'))
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

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


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


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

def val_resize(image):
    return resize(image, (c.cropsize_val, c.cropsize_val))

def hex_to_binary_string(hex_string):
    # # Pad the string with a leading zero if the length is odd
    # if len(hex_string) % 2 != 0:
    #     hex_string = '0' + hex_string
    
    # Create a mapping of hex characters to their 4-bit binary representation
    hex_to_bin_map = {
        '0': '0000', '1': '0001', '2': '0010', '3': '0011', '4': '0100', '5': '0101',
        '6': '0110', '7': '0111', '8': '1000', '9': '1001', 'a': '1010', 'b': '1011',
        'c': '1100', 'd': '1101', 'e': '1110', 'f': '1111'
    }

    # Convert hex string to binary string (each hex digit becomes 4 binary bits)
    binary_string = ''.join(hex_to_bin_map[c] for c in hex_string.lower())
    
    # Convert binary string to a list of bits (0 or 1)
    bits = [int(bit) for bit in binary_string]
    
    # Convert the list of bits to a PyTorch tensor of type float32
    # bits_tensor = torch.tensor(bits, dtype=torch.int32)
    
    return bits

def main():
    loss_fn_alex = lpips.LPIPS(net='alex')

    # Thumbnail Embedding Model
    net = Model().to(device)
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    params_trainable_net = list(filter(lambda p: p.requires_grad, net.parameters()))
    all_trainable_params = params_trainable_net
    optim = torch.optim.Adam(all_trainable_params, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    load(net, optim, c.MODEL_PATH + c.suffix)
    net.eval()

    # Watermarking Model
    if c.WM_MODEL == 'InvisMark':
        ckpt_path = './model/paper.ckpt'
        if torch.cuda.is_available():
            state_dicts = torch.load(ckpt_path)
        else:
            state_dicts = torch.load(ckpt_path, map_location=torch.device('cpu'))
        cfg = state_dict['config']
        wm_model = train_watermark.Watermark(cfg)
        wm_model.load_model(ckpt_path)
    elif c.WM_MODEL == 'TrustMark':
        flag_Use_ECC = False
        if c.ECCMethod == None:
            flag_Use_ECC = True
        tm = TrustMark(use_ECC=flag_Use_ECC, verbose=True, model_type='Q', encoding_type=TrustMark.Encoding.BCH_SUPER)
        capacity=tm.schemaCapacity()
    elif c.WM_MODEL == 'RedMark':
        model_path = "./model/weights_final.h5"
        embedding_net, _ = buildModel(model_path)

    if c.TF == 'DCT':
        dwt = common.DCT()
        iwt = common.IDCT()
    else:
        dwt = common.DWT()
        iwt = common.IWT()

    # Making the needed directories
    if not(os.path.isdir(c.IMAGE_PATH)):
            os.mkdir(c.IMAGE_PATH)

    for path in c.embedding_paths:
        os.makedirs(path, exist_ok=True)

    transform_val = T.Compose([
        T.Lambda(val_resize),
        T.ToTensor(),
    ])

    
    psnr_total = 0
    psnr_total_distort = 0
    ssim_total = 0

    psnr_total_s = 0
    ssim_total_s = 0

    psnr_total_r = 0

    if c.ECCMethod == 'BCH':
        D = bchecc.DataLayer(100,False,1)
    elif c.ECCMethod == 'polar':
        codec = initialize_polar()

    with torch.no_grad():

        psnr_cs = []
        psnr_ce = []

        lpips_arr_cs = []
        lpips_arr_ce = []
        
        ssim_arr_cs = []
        ssim_arr_ce = []

        # image_files = [f for f in os.listdir(c.VAL_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files = [fi for fi in os.listdir(c.IMAGE_PATH_cover) if fi.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Folders in embedding phase
        # IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
        # IMAGE_PATH_secret = IMAGE_PATH + 'secret/'

        with open(c.IMAGE_PATH + "watermarks.txt", "w") as f, open(c.IMAGE_PATH + "hashs.txt", "w") as f2:

            for i, batch in enumerate(my_datasets.testloader):
                if i>c.Counter2 - c.Counter:
                    break
            # for image_file in image_files:
                """ Prepare input """ 
                cover, secret = batch
                pil_cover = to_pil_image(cover[0])
                cover = cover.to(device)
                secret = secret.to(device)

                if c.BW:
                    secret = torch.mean(secret, dim=1, keepdim=True)

                cover_input = dwt(cover)
                secret_input = dwt(secret)

                input_img = torch.cat((cover_input, secret_input), 1)
                

                '''Thumbnail Embedding'''
                output = net(input_img)
                if c.TF == 'DCT':
                    output_steg = output.narrow(1, 0, c.channels_in)
                    output_z = output.narrow(1, c.channels_in, output.shape[1] - c.channels_in)
                else:
                    output_steg = output.narrow(1, 0, 4 * c.channels_in)
                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)

                steg = iwt(output_steg)
                    
                steg = torch.clamp(steg, 0 , 1)

                pil_steg = to_pil_image(steg[0]).convert("RGB")
                ''' Watermarking'''
                if c.HashEmbedding:
                    hash_value = imagehash.phash(pil_cover, hash_size = c.hash_length)
                    hash_value_bin = hex_to_binary_string(str(hash_value))
                    # print(len(hash_value_bin))
                    
                    if c.ECCMethod == 'RS':
                        if len(hash_value_bin) % 8 != 0:
                            padding_length = 8 - (len(hash_value_bin) % 8)
                            hash_value_bin += [0] * padding_length  # Pad with zeros

                        encoded_hash_hex = ecc.encode_message(hash_value_bin, c.ecc_symbol-1)
                        encoded_hash_bits = ecc.byte_to_bits(encoded_hash_hex)
                        
                        if len(encoded_hash_bits)  != c.wm_cap:
                            padding_length = c.wm_cap - (len(encoded_hash_bits))
                            encoded_hash_bits += '0' * padding_length  # Pad with zeros if the final length of watermark is shorter than the capacity
                        
                        watermark = ecc.string_to_tensor(encoded_hash_bits).unsqueeze(0).to(device)
                        # print(watermark)
                    elif c.ECCMethod == 'BCH':
                        if len(hash_value_bin) < c.WM_SIZE:
                            padding_length = c.WM_SIZE - len(hash_value_bin)
                            hash_value_bin += [0] * padding_length  # Pad with zeros
                        hash_value_bin_string = ''.join(str(bit) for bit in hash_value_bin)
                        watermark = D.encode_binary([hash_value_bin_string])   
                        watermark = torch.from_numpy(watermark).to(device)
                        # print(len(watermark))
                    elif c.ECCMethod == 'polar':
                        if len(hash_value_bin) < c.WM_SIZE:
                            padding_length = c.WM_SIZE - len(hash_value_bin)
                            hash_value_bin += [0] * padding_length  # Pad with zeros
                        hash_value_bin_string = ''.join(str(bit) for bit in hash_value_bin)
                        hash_value_bin_numpy = np.array([int(bit) for bit in hash_value_bin_string], dtype=np.uint8)
                        watermark_numpy = embed_polar(codec, hash_value_bin_numpy)
                        watermark = torch.from_numpy(watermark_numpy).to(device)
                        # print(watermark)

                else:
                    secrets = [''.join(random.choice(['0', '1']) for _ in range(100)) for _ in range(steg.shape[0])]
                    bit_array = np.array([[int(b) for b in secret] for secret in secrets], dtype=np.uint8)
                    watermark = torch.tensor(bit_array, dtype=torch.float32).to(device)

                    # watermark, _ = utils.uuid_to_bits(steg.shape[0])
                    # watermark = watermark[:, :100].to(device)
                    # print(watermark)
                
                # Encode the watermark
                if c.WM_MODEL == 'InvisMark':
                    embedded, enc_input, enc_output = wm_model._encode(steg.cpu(), watermark)
                elif c.WM_MODEL == 'TrustMark':
                    if c.ECCMethod == None:
                        if len(hash_value_bin) < c.WM_SIZE:
                            padding_length = c.WM_SIZE - len(hash_value_bin)
                            hash_value_bin += [0] * padding_length  # Pad with zeros
                        hash_value_bin_string = ''.join(str(bit) for bit in hash_value_bin)
                        embedded, watermark = tm.encode(pil_steg, hash_value_bin_string, MODE='binary')
                        watermark = torch.tensor(watermark, dtype = torch.int32).to(device)

                    else:
                        watermark_string = ''.join(map(str, watermark.int().squeeze(0).tolist()))
                        embedded, _ = tm.encode(pil_steg, watermark_string, MODE='binary')
                    # print(np.asarray(embedded))
                    
                    the_transform = transforms.Compose([transforms.ToTensor()])
                    embedded_tensor = the_transform(embedded)
                elif c.WM_MODEL == 'RedMark':
                    if c.ECCMethod == None:
                        hash_value_bin_string = ''.join(str(bit) for bit in hash_value_bin)
                        
                        watermark_numpy = np.array([int(bit) for bit in hash_value_bin_string], dtype=np.uint8).squeeze()
                        watermark = torch.from_numpy(watermark_numpy).to(device)
                    else:
                        watermark_numpy = watermark.cpu().numpy().squeeze()
                    steg_np = steg.cpu().squeeze(0).permute(1, 2, 0).numpy()
                    steg_np = (steg_np - steg_np.min()) / (steg_np.max() - steg_np.min()) * 255.0

                    # Convert to uint8 for image format
                    steg_np = np.uint8(steg_np)
                    # print("hash_value_bin: ",hash_value_bin)
                    
                    embedded = embed_watermark(embedding_net, steg_np, watermark_numpy, 1)
                    # print(embedded.shape)
                    embedded = Image.fromarray(embedded)
                    the_transform = transforms.Compose([transforms.ToTensor()])
                    embedded_tensor = the_transform(embedded)

                    # watermark = torch.tensor(watermark, dtype = torch.int32).to(device)
                    # watermark = embedded_tensor[0]

                    
                if c.HashEmbedding:
                    watermark_numpy = watermark.cpu().numpy().reshape(1,-1)  # Convert to numpy array
                    # print(watermark_numpy.shape)
                    np.savetxt(f, watermark_numpy, fmt='%d', header=f"Watermark for image {i:05d}", comments='')  # Append with a header

                
                    hash_numpy = np.expand_dims(np.asarray(hash_value_bin), 0)
                    np.savetxt(f2, hash_numpy, fmt='%d', header=f"Hash for image {i:05d}", comments='')  # Append with a header


                if c.save_img:
                    torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' %(i+c.Counter))
                    if not c.HashEmbedding:
                        torchvision.utils.save_image(steg, c.IMAGE_PATH_steg + '%.5d.png' %(i+c.Counter))
                    pil_steg.save(c.IMAGE_PATH_steg + '%.5d.png' %(i+c.Counter))
                    # torchvision.utils.save_image(embedded, c.IMAGE_PATH_double_WM + '%.5d.png' %(i+c.Counter))
                    if c.HashEmbedding:
                        embedded.save(c.IMAGE_PATH_double_WM + '%.5d.png' %(i+c.Counter))
                    torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' %(i+c.Counter))

                    sec_rgb = DeBlocking(secret)
                    sec_rgb = ycbcr_to_rgb(sec_rgb)

                    torchvision.utils.save_image(sec_rgb, c.IMAGE_PATH_secret_converted + '%.5d.png' %(i+c.Counter))

                
                # Calculate metrics
                lpips_temp = loss_fn_alex(cover.to("cpu"), steg.to("cpu"))
                lpips_arr_cs.append(lpips_temp)
                if c.HashEmbedding:
                    lpips_temp2 = loss_fn_alex(cover.to("cpu"), embedded_tensor.to("cpu"))
                    lpips_arr_ce.append(lpips_temp2)
                # print(lpips_arr_ce)

                cover_np = cover.cpu().numpy().squeeze() * 255
                np.clip(cover_np, 0, 255)

                steg_np = steg.cpu().numpy().squeeze() * 255
                np.clip(steg_np, 0, 255)

                embedded_np = embedded_tensor.cpu().numpy().squeeze() * 255
                np.clip(embedded_np, 0, 255)

                psnr_temp = computePSNR(cover_np, steg_np)
                psnr_cs.append(psnr_temp)

                psnr_temp2 = computePSNR(cover_np, embedded_np)
                psnr_ce.append(psnr_temp2)
                
                
                # Move tensors to CPU (if on GPU) and convert to numpy arrays
                # cover_np = cover.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert [1, 3, 512, 512] -> [512, 512, 3]
                # steg_np = steg.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert [1, 3, 512, 512] -> [512, 512, 3]

                # Compute SSIM between img1 and img2 (for multichannel images, set multichannel=True)
                # print(cover_np.shape)
                # print(steg_np.shape)
                # ssim_temp, _ = structural_similarity (cover_np, steg_np, data_range=1.0, channel_axis = 2)
                # ssim_arr.append(ssim_temp)
    
                # ssim_temp = structural_similarity(cover_np, steg_img_np, multichannel=True)
                # ssim_temp_s = structural_similarity(secret_rev, secret, multichannel=True)
                # ssim_total = ssim_total + ssim_temp

                # print("Cover SSIM: ", ssim_temp)
                # print("Secret SSIM: ", ssim_temp_s)
                print(f"Index: {i:05d}, LPIPS cover vs. embedded: {lpips_temp2}, PSNR cover vs. embedded: {psnr_temp2}")


            coverstegPSNR = np.mean(psnr_cs)
            coverembeddedPSNR = np.mean(psnr_ce)
            # covSSIM = np.mean(ssim_arr)
            if c.HashEmbedding:
                stacked_tensors = torch.stack(lpips_arr_ce)
                lpips_ave_ce = torch.mean(stacked_tensors, dim=0)

            stacked_tensors = torch.stack(lpips_arr_cs)
            lpips_ave_cs = torch.mean(stacked_tensors, dim=0)

            

            print("Total PSNR for cover and steg: ", coverstegPSNR)
            print("Total PSNR for cover and embedded: ", coverembeddedPSNR)
            
            print("LPIPS for cover and steg: ", lpips_ave_cs)
            if c.HashEmbedding:
                print("LPIPS for cover and embedded: ", lpips_ave_ce)

if __name__ == '__main__':
    main()
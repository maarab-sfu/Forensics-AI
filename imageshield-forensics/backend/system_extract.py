import math
import torch
import torch.nn
import torch.optim
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as T
from torch.autograd import Variable

import numpy as np
from PIL import Image
from distortions import *
import config as c
from restore_train_esrgan import ImageDataset, disect_secrev, GeneratorRRDB, denormalize, mean, std
import ecc
import bchecc

# from polarecc import initialize_polar, embed_polar, extract_polar

from system_embed import hex_to_binary_string
from torchvision.transforms.functional import to_pil_image
import imagehash
from train import DeBlocking

from model import *
import torch.nn.functional as F
import my_datasets
import os
import os.path
import modules.Unet_common as common

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

import metrics
if c.WM_MODEL == "InvisMark":
    import train_watermark
elif c.WM_MODEL == "TrustMark":
    import tmark_code
elif c.WM_MODEL == "RedMark":
    from redmark import buildModel, embed_watermark, extract_watermark
import utils

import matplotlib.pyplot as plt




from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.models import vgg19
from torch.utils.data import DataLoader, random_split
from my_datasets import Hinet_Dataset
import config as c
import my_datasets
from PIL import Image
import numpy as np
import argparse
from torchvision.transforms.functional import resize

import os
import random
from torchvision import models
from torchvision.utils import save_image
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score  # Correct library
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric  # For PSNR and SSIM

from tqdm import tqdm
import warnings
from fd_train import TamperingLocalizationNet, TamperingLocalizationNet2


warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def analyze_4d_array(arr):
    """
    Analyze a 4D NumPy array by computing statistics and plotting a histogram of its values.
    
    Parameters:
    - arr: 4D NumPy array
    
    Returns:
    - stats: Dictionary containing statistical information
    """
    # Flatten the 4D array to 1D for analysis
    flattened = arr.flatten()

    # Compute statistics
    stats = {
        "mean": np.mean(flattened),
        "median": np.median(flattened),
        "min": np.min(flattened),
        "max": np.max(flattened),
        "q1": np.percentile(flattened, 25),
        "q3": np.percentile(flattened, 75),
        "std": np.std(flattened),
    }

    # Print statistics
    print("Array Statistics:")
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value:.3f}")
    
    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flattened, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Value Frequency Histogram")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    return stats


def dec2bin(x, bits=8):
    mask = 2**torch.arange(bits-1,-1,-1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def bin2dec(b, bits=8):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def load(net, optim, name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

def load_with_forgery(net, fd, optim, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)

    network_state_dict_fd = {k: v for k, v in state_dicts['fd'].items() if 'tmp_var' not in k}
    fd.load_state_dict(network_state_dict_fd)
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


def YCbCr2RGB(im, BW = False, mask = None):
    N, _, W, H = im.shape

    # Split the image into four quadrants
    Y = im[:, 0, :, :].squeeze(1)  # Y channel
    
    L1 = Y[:, :W//2, :H//2]
    L2 = Y[:, :W//2, H//2:]
    L3 = Y[:, W//2:, :H//2]
    L4 = Y[:, W//2:, H//2:]

    y_avg = (L1 + L2 + L3 + L4)/4

    if not BW:
        CbCr = im[:, 1, :, :].squeeze(1)  # Combined CbCr channel

        A1 = CbCr[:, :W//2, :H//2]
        A2 = CbCr[:, :W//2, H//2:]
        A3 = CbCr[:, W//2:, :H//2]
        A4 = CbCr[:, W//2:, H//2:]

        cb11 = A1[:, :W//4,:H//4]
        cb12 = A1[:, W//4:W//2,H//4:H//2]
        cr11 = A1[:, W//4:W//2,:H//4]
        cr12 = A1[:, :W//4,H//4:H//2]

        cb21 = A2[:, :W//4,:H//4]
        cb22 = A2[:, W//4:W//2,H//4:H//2]
        cr21 = A2[:, W//4:W//2,:H//4]
        cr22 = A2[:, :W//4,H//4:H//2]

        cb31 = A3[:, :W//4,:H//4]
        cb32 = A3[:, W//4:W//2,H//4:H//2]
        cr31 = A3[:, W//4:W//2,:H//4]
        cr32 = A3[:, :W//4,H//4:H//2]

        cb41 = A4[:, :W//4,:H//4]
        cb42 = A4[:, W//4:W//2,H//4:H//2]
        cr41 = A4[:, W//4:W//2,:H//4]
        cr42 = A4[:, :W//4,H//4:H//2]

        cb_avg = (cb11 + cb12 + cb21 + cb22 + cb31 + cb32 + cb41 + cb42)/8
        cr_avg = (cr11 + cr12 + cr21 + cr22 + cr31 + cr32 + cr41 + cr42)/8

        cb_avg = F.interpolate(cb_avg.unsqueeze(1), size=(W//2, W//2), mode = c.Upsample_mode, align_corners=False).squeeze(1)
        cr_avg = F.interpolate(cr_avg.unsqueeze(1), size=(W//2, W//2), mode = c.Upsample_mode, align_corners=False).squeeze(1)

    

        img_ycbcr = torch.stack([y_avg, cb_avg, cr_avg], dim=1)

        return img_ycbcr
    else:
        return y_avg.unsqueeze(1)
    
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

def main():
    to_pil = T.ToPILImage()
    loss_fn_alex = lpips.LPIPS(net='alex')
    net = Model()
    net.to(device)
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)

    watermarks = {}
    hashs = {}
    counter = 0


    """ The Watermarking Module """
    if c.WM_MODEL == 'InvisMark':
        ckpt_path = './model/paper.ckpt'
        state_dict = torch.load(ckpt_path)
        cfg = state_dict['config']
        wm_model = train_watermark.Watermark(cfg)
        wm_model.load_model(ckpt_path)
    elif c.WM_MODEL == 'TrustMark':
        flag_Use_ECC = False
        if c.ECCMethod == None:
            flag_Use_ECC = True
        tm = tmark_code.TrustMark(use_ECC = flag_Use_ECC, verbose = True, secret_len = 100, device = device, model_type = 'Q', encoding_type = tmark_code.TrustMark.Encoding.BCH_5)
        print(tm)
        capacity=tm.schemaCapacity()
    elif c.WM_MODEL == 'RedMark':
        model_path = "./model/weights_final.h5"
        _, extractor_net = buildModel(model_path)

    """ Restoration Module """
    generator = GeneratorRRDB(3, filters=64, num_res_blocks=23).to(device)
    generator.load_state_dict(torch.load('./saved_models/generator_2.pth'))
    generator.eval()

    """ Tampering Detection Module """
    TD_model = TamperingLocalizationNet2()
    TD_model = TD_model.to(device)

    if os.path.exists("./model/best_localize_model.pth"):
        checkpoint = torch.load("./model/best_localize_model.pth")
        TD_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Tampering Localization Model is loaded.")
    
    TD_model.eval()

    

    with open(c.IMAGE_PATH + "watermarks.txt", "r") as f:
        content = f.read().split("Watermark for image")  # Split by header
        for entry in content[1:]:  # Skip the first split (before the first header)
            lines = entry.strip().split("\n")
            secret_numpy = np.array([list(map(int, line.split())) for line in lines[1:]])
            watermarks[counter] = torch.tensor(secret_numpy, dtype=torch.int32).to(device)
            counter = counter + 1
    
    counter = 0
    with open(c.IMAGE_PATH + "hashs.txt", "r") as f:
        content = f.read().split("Hash for image")  # Split by header
        for entry in content[1:]:  # Skip the first split (before the first header)
            lines = entry.strip().split("\n")
            secret_numpy = np.array([list(map(int, line.split())) for line in lines[1:]])
            hashs[counter] = torch.tensor(secret_numpy, dtype=torch.int32).to(device)
            counter = counter + 1

    params_trainable_net = list(filter(lambda p: p.requires_grad, net.parameters()))
    all_trainable_params = params_trainable_net
    optim = torch.optim.Adam(all_trainable_params, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    
    counter = 0

    load(net, optim, c.MODEL_PATH + c.suffix)
    net.eval()

    if c.TF == 'DCT':
        dwt = common.DCT()
        iwt = common.IDCT()
    else:
        dwt = common.DWT()
        iwt = common.IWT()

    # Making the needed directories
    if not(os.path.isdir(c.IMAGE_PATH + c.DIRPATH)):
            os.mkdir(c.IMAGE_PATH + c.DIRPATH)

    for path in c.extraction_paths:
        os.makedirs(path, exist_ok=True)

    transform_val = T.Compose([
        T.Lambda(val_resize),
        T.ToTensor(),
    ])
    
    rand_wat = watermarks[0].to(device)
    rand_hash = hashs[0].to(device)
    if c.MIXED == False and c.DISTORT == False:
        print(c.noises)
        print(c.attacks)
        distortion_layers_test = DistortionLayer(c.noises, c.attacks, c.infoDict, device, 'test')
        distortion_layers_test.to(device)

        distortion_layers_distort = DistortionLayer(c.noises, [], c.infoDict, device, 'distort')
        distortion_layers_distort.to(device)
    
    with torch.no_grad():

        psnr_fc = [] # final vs. cover image
        lpips_fc_arr = []
        ssim_fc_arr = []

        psnr_rc = [] # restored vs. cover image
        lpips_rc_arr = []
        ssim_rc_arr = []
        
        bit_acc_list = []
        noised_bit_acc_list = []
        num_batches = 0

        precision_values = []
        recall_values = []
        f1_values = []
        iou_values = []

        Hash_detected = True
        missed_watermarks = 0
        low_acc_count = 0

        if c.ECCMethod == 'BCH':
            D=bchecc.DataLayer(100,False,1)
        elif c.ECCMethod == 'polar':
            codec = initialize_polar()

        if c.noises == []:
            image_files = sorted([f for f in os.listdir(c.IMAGE_PATH_edited) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        else:
            image_files = sorted([f for f in os.listdir(c.IMAGE_PATH_cover) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        num_edited_image_files = len([f for f in os.listdir(c.IMAGE_PATH_edited) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        with open(c.IMAGE_PATH + c.DIRPATH + "watermarks_extracted.txt", "w") as f, open(c.IMAGE_PATH + c.DIRPATH + "hashs_extracted.txt", "w") as f2:
            for counter, image_file in enumerate(image_files):
                if c.noises != [] and (counter<c.Counter or counter>=c.Counter2):
                    continue


                """ Prepare input """ 
                cover_path = os.path.join(c.IMAGE_PATH_cover, image_file)
                secret_path = os.path.join(c.IMAGE_PATH_secret, image_file)
                steg_path = os.path.join(c.IMAGE_PATH_steg, image_file)
                # converted_path = os.path.join(c.IMAGE_PATH_converted, image_file)
                # sec_converted_path = os.path.join(c.IMAGE_PATH_secret_converted, image_file)
                doubleWM_path = os.path.join(c.IMAGE_PATH_double_WM, image_file)

                cover = Variable(transform_val(Image.open(cover_path).convert("RGB"))).to(device).unsqueeze(0)
                secret = Variable(transform_val(Image.open(secret_path).convert("RGB"))).to(device).unsqueeze(0)
                steg = Variable(transform_val(Image.open(steg_path).convert("RGB"))).to(device).unsqueeze(0)
                # converted = Variable(transform_val(Image.open(converted_path).convert("RGB"))).to(device).unsqueeze(0)
                # sec_converted = Variable(transform_val(Image.open(sec_converted_path).convert("RGB"))).to(device).unsqueeze(0)
                if c.HashEmbedding:
                    doubleWM = Variable(transform_val(Image.open(doubleWM_path).convert("RGB"))).to(device).unsqueeze(0)
                else:
                    doubleWM = steg

                cover = cover.to(device)
                secret = secret.to(device)
                steg = steg.to(device)
                # converted = converted.to(device)
                # sec_converted = sec_converted.to(device)
                doubleWM = doubleWM.to(device)

                try:
                    watermark = watermarks[num_batches].to(device)
                    hash_value = hashs[num_batches].to(device)
                except:
                    watermark = rand_wat
                    hash_value = rand_hash


                """Adding distortions to doubly watermarked image """
                if c.noises != []:
                    [noised, mask] = distortion_layers_test([doubleWM,doubleWM])
                    noised = noised[0]
                    mask = mask.to(device)
                    if mask == None:
                        # print("mask is none!")
                        mask = torch.zeros((cover[0].shape[0], 1, cover[0].shape[2], cover[0].shape[3])).to(device)

                    if mask.shape[1] == 1:
                        mask = mask.repeat(1, 3, 1, 1)
                    mask = mask/255
                    
                else:
                    edited_path = os.path.join(c.IMAGE_PATH_edited, image_file)
                    edited = Variable(transform_val(Image.open(edited_path).convert("RGB"))).to(device).unsqueeze(0)
                    noised = edited.to(device)

                    mask_path = os.path.join(c.IMAGE_PATH_mask, image_file)
                    mask = Variable(transform_val(Image.open(mask_path).convert("RGB"))).to(device).unsqueeze(0)
                    mask = mask.to(device)

                mask = F.interpolate(mask, size=(256, 256), mode="bilinear", align_corners=False)

                if c.HashEmbedding:
                    """ Extracting the watermark """
                    if c.WM_MODEL == 'InvisMark':
                        extracted_watermark = wm_model._decode(noised)

                        extracted_watermark_bits = (extracted_watermark >= 0.5).int()
                        extracted_watermark_bits_numpy = extracted_watermark_bits.cpu().numpy()

                        # print(extracted_watermark, type(extracted_watermark))
                        # print(extracted_watermark_bits, type(extracted_watermark_bits))
                        # print(extracted_watermark_bits_numpy, type(extracted_watermark_bits_numpy))
                    elif c.WM_MODEL == 'TrustMark':
                        pil_noised = to_pil_image(noised[0])
                        # pil_noised = to_pil_image(doubleWM[0])
                        if c.ECCMethod == None:
                            watermark_extracted, extracted_hash_bits, Hash_detected, wm_schema = tm.decode(pil_noised, MODE='binary')
                            extracted_hash_bits_numpy = np.array(list(extracted_hash_bits), dtype=int).reshape(1, -1)
                            extracted_hash_tensor = torch.from_numpy(extracted_hash_bits_numpy).to(device)

                            extracted_watermark = ''.join(str(int(x)) for x in watermark_extracted[0])
                            extracted_watermark_bits_numpy = np.array(list(extracted_watermark), dtype=int).reshape(1, -1)
                            extracted_watermark_bits = torch.from_numpy(extracted_watermark_bits_numpy)
                            extracted_watermark = extracted_watermark_bits.to(device)
                        else:
                            extracted_watermark, wm_present, wm_schema = tm.decode(pil_noised, MODE='binary')
                            extracted_watermark_bits_numpy = np.array(list(extracted_watermark), dtype=int).reshape(1, -1)
                            extracted_watermark_bits = torch.from_numpy(extracted_watermark_bits_numpy)
                            extracted_watermark = extracted_watermark_bits.to(device)
                    
                    elif c.WM_MODEL == 'RedMark':
                        noised_np = noised.cpu().squeeze(0).permute(1, 2, 0).numpy()
                        noised_np = (noised_np - noised_np.min()) / (noised_np.max() - noised_np.min()) * 255.0

                        # Convert to uint8 for image format
                        noised_np = np.uint8(noised_np)

                        extracted_watermark = extract_watermark(extractor_net, noised_np)
                        
                        # Convert the array to integers
                        arr_int = extracted_watermark.astype(int)

                        # Flatten the array to 1D
                        arr_flat = arr_int.flatten()

                        # Convert the array to a string
                        extracted_watermark_bits_string = ''.join(arr_flat.astype(str))
                        # extracted_hash_bits = ''.join(extracted_watermark.astype(int).astype(str))
                        # print(extracted_watermark)
                        extracted_watermark_bits_numpy = np.array(list(extracted_watermark), dtype=int).reshape(1, -1)
                        extracted_watermark_bits = torch.from_numpy(extracted_watermark_bits_numpy)
                        extracted_watermark = extracted_watermark_bits.to(device)

                        # ecc.print_comparison(ecc.tensor_to_string(hash_value[0])[:c.HASH_SIZE], extracted_hash_bits[:c.HASH_SIZE])

                    
                    if c.ECCMethod != None:
                        # print("Saving Watermark...")
                        # print(extracted_watermark_bits_numpy)
                        np.savetxt(f, extracted_watermark_bits_numpy, fmt='%d', header=f"Watermark for image {num_batches:05d}", comments='')  # Append with a header

                    ''' Calculating Hash '''
                    pil_noised = to_pil_image(noised[0])
                    hash_noised = imagehash.phash(pil_noised, hash_size = c.hash_length)
                    hash_noised_bin = hex_to_binary_string(str(hash_noised))
                    
                    watermark_string = ecc.tensor_to_string(watermark[0])

                    if c.ECCMethod == 'RS':
                        if len(hash_noised_bin) % 8 != 0:
                            padding_length = 8 - (len(hash_noised_bin) % 8)
                            hash_noised_bin += [0] * padding_length  # Pad with zeros
                        extracted_watermark_bits = extracted_watermark_bits[:,:96]

                        ''' Extracting the Embedded Hash Values '''
                        extracted_watermark_bits = ecc.tensor_to_string(extracted_watermark_bits[0])
                        # print(extracted_watermark_bits)
                        extracted_watermark_bytes = ecc.bits_to_bytearray(extracted_watermark_bits)
                        extracted_hash_bits = ecc.decode_message(extracted_watermark_bytes, c.ecc_symbol-1)
                        if extracted_hash_bits == None:
                            Hash_detected = False
                            missed_watermarks += 1
                        else:
                            Hash_detected = True
                        # print(extracted_hash_bits)

                        # ecc.print_comparison(extracted_watermark_bits, ecc.tensor_to_string(watermark[0])[:96])
                        if Hash_detected:
                            extracted_hash_bits = ecc.string_to_tensor(extracted_hash_bits)

                        # ecc.print_comparison(watermark_string, extracted_watermark_bits)

                    elif c.ECCMethod == 'BCH':
                        if len(hash_noised_bin) < c.WM_SIZE:
                            padding_length = c.WM_SIZE - len(hash_noised_bin)
                            hash_noised_bin += [0] * padding_length  # Pad with zeros
                        
                        extracted_watermark_bits = ecc.tensor_to_string(extracted_watermark_bits[0])
                        extracted_hash_bits, Hash_detected, _ = D.decode_bitstream(extracted_watermark_bits_numpy, MODE='binary')[0]
                        
                        ecc.print_comparison(ecc.tensor_to_string(watermark[0])[:96], extracted_watermark_bits[:96])
                        ecc.print_comparison(ecc.tensor_to_string(hash_value[0])[:c.HASH_SIZE], extracted_hash_bits[:c.HASH_SIZE]) # the hash that has been decoded from the original watermark and that has been saved in the original file. They basically should be exactly the same, which they are.
                        
                        if (ecc.tensor_to_string(hash_value[0])[:c.HASH_SIZE] ==  extracted_hash_bits[:c.HASH_SIZE]):
                            Hash_detected = True
                        # print(ecc.tensor_to_string(watermark[0])[:96])
                        # print(extracted_watermark_bits[:96])
                    elif c.ECCMethod == 'polar':
                        if len(hash_noised_bin) < c.WM_SIZE:
                            padding_length = c.WM_SIZE - len(hash_noised_bin)
                            hash_noised_bin += [0] * padding_length  # Pad with zeros
                        # print(extracted_watermark_bits_numpy.shape)
                        ex_w = extracted_watermark_bits_numpy.squeeze()
                        # print(type(ex_w))
                        # print(ex_w)
                        err_rate = 0.09
                        llr_0 = math.log((1 - err_rate) / err_rate)
                        llr_1 = math.log(err_rate / (1 - err_rate))

                        ex_w = np.where(ex_w == 0, llr_0, llr_1)
                        extracted_hash_bits_numpy = extract_polar(codec, ex_w)
                        # print(extracted_hash_bits_numpy)
                        extracted_hash_bits = torch.from_numpy(extracted_hash_bits_numpy).to(device)
                        hash_noised_numpy = np.array(hash_noised_bin).reshape(1, -1)
                        noised_hash_bits = torch.from_numpy(hash_noised_numpy).to(device)
                        if (ecc.tensor_to_string(hash_value[0])[:c.HASH_SIZE] ==  extracted_hash_bits[:c.HASH_SIZE]):
                            Hash_detected = True
                    else:
                        # extracted_watermark_bits = ecc.tensor_to_string(extracted_watermark_bits[0])
                        # ecc.print_comparison(ecc.tensor_to_string(watermark[0])[:96], extracted_watermark_bits[:96])
                        extracted_hash_bits = watermark[0]
                        if (ecc.tensor_to_string(hash_value[0])[:c.HASH_SIZE] ==  extracted_hash_bits[:c.HASH_SIZE]):
                            Hash_detected = True
                        # ecc.print_comparison(ecc.tensor_to_string(hash_value[0])[:c.HASH_SIZE], extracted_hash_bits[:c.HASH_SIZE]) 
                        

                    if Hash_detected:
                        binary_list = [int(bit) for bit in extracted_hash_bits]
                        hash_numpy = np.array(binary_list).reshape(1, -1)
                        np.savetxt(f2, hash_numpy, fmt='%d', header=f"Hash for image {num_batches:05d}", comments='')  # Append with a header
                    else:
                        print("Hash was not detected!")
                        missed_watermarks += 1
                        zero_hash = np.array([0] * hash_value.shape[1]).reshape(1, -1)
                        extracted_hash_tensor = torch.from_numpy(zero_hash).to(device)
                        # zero_hash = ''.join(str(bit) for bit in zero_hash)
                        np.savetxt(f2, zero_hash, fmt='%d', header=f"Hash for image {num_batches:05d}", comments='')  # Append with a header


                """ Extracting the thumbnails """
                output_steg = dwt(noised)
                    
                if c.NOISE_TYPE == "Gaussian":
                    output_z_backward = gauss_noise(torch.Size([1, 24, c.cropsize_val//2, c.cropsize_val//2]))
                elif c.NOISE_TYPE == "Uniform":
                    output_z_backward = uniform_noise(torch.Size([1, 24, c.cropsize_val//2, c.cropsize_val//2]))
                elif c.NOISE_TYPE == "Constant":
                    output_z_backward = 0.5 * torch.ones(torch.Size([1, 24, c.cropsize_val//2, c.cropsize_val//2])).to(device)

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

                # secret_rev = secret_rev.float()

                # secret_rev_01 = (secret_rev - secret_rev.min()) / (secret_rev.max() - secret_rev.min() + 1e-8)
                secret_rev_01 = torch.clamp(secret_rev, 0 , 1)
                secret_rev_normalized = (secret_rev_01 - torch.tensor(mean).view(1, -1, 1, 1).to(secret_rev.device)) / \
                                        torch.tensor(std).view(1, -1, 1, 1).to(secret_rev.device)

                """Restoration"""
                imgs_lr1, imgs_lr2, imgs_lr3, imgs_lr4 = disect_secrev(secret_rev_normalized.float())
                with torch.no_grad():
                    restored = denormalize(generator(imgs_lr1, imgs_lr2, imgs_lr3, imgs_lr4))

                """ Tampering Localization """
                extracted_mask = TD_model(noised, restored)

                if extracted_mask.shape[1] == 1:
                    extracted_mask = extracted_mask.repeat(1, 3, 1, 1)
                # Convert tensors to numpy arrays
                extracted_mask_np = extracted_mask.cpu().numpy().transpose(0, 2, 3, 1)  # Convert from NCHW to NHWC
                # print(output_np)
                mask_np = mask.cpu().numpy().transpose(0, 2, 3, 1)  # Convert from NCHW to NHWC
                # stats = analyze_4d_array(extracted_mask_np)
                mask_np = (mask_np > 0.5).astype(np.uint8)

                # Apply thresholding to output and convert to binary
                extracted_mask_binary_np = (extracted_mask_np >= 0.5).astype(np.uint8)  # Use threshold directly

                for j in range(extracted_mask_binary_np.shape[0]):  # Loop over the batch size

                    # Flatten arrays for pixel-level classification metrics
                    mask_flat = mask_np[j].flatten()
                    extracted_mask_flat = extracted_mask_binary_np[j].flatten()

                    # print("m: ", np.unique(mask_flat).size)
                    # print("e: ", np.unique(extracted_mask_flat).size)
                    
                    if  np.unique(extracted_mask_flat).size == 2:
                        extracted_mask_flat = (extracted_mask_flat > 0.5).astype(int)
                        mask_flat = (mask_flat > 0.5).astype(int)

                        precision = precision_score(mask_flat, extracted_mask_flat, average='binary')
                        recall = recall_score(mask_flat, extracted_mask_flat, average='binary')
                        f1 = f1_score(mask_flat, extracted_mask_flat, average='binary')
                        iou = jaccard_score(mask_flat, extracted_mask_flat, average='binary')

                        precision_values.append(precision)
                        recall_values.append(recall)
                        f1_values.append(f1)
                        iou_values.append(iou)
                    else:
                        f1 = 0.0
                        precision_values.append(np.nan)
                        recall_values.append(np.nan)
                        f1_values.append(np.nan)
                        iou_values.append(np.nan)

                    # Ensure the output array is in the correct format for saving
                    extracted_mask_img_np = np.squeeze(extracted_mask_binary_np[j])  # Remove the last channel if it is 1
                    extracted_mask_img_np = np.clip(extracted_mask_img_np * 255, 0, 255).astype(np.uint8)  # Scale to [0, 255]
                    extracted_mask_img = Image.fromarray(extracted_mask_img_np, mode='RGB')  # 'L' mode for grayscale
                    if c.cropsize_val != 512:
                        # Resize the image to c.cropsize_val
                        extracted_mask_img = extracted_mask_img.resize((c.cropsize_val, c.cropsize_val), Image.ANTIALIAS)


                """ Final Results """
                extracted_mask = F.interpolate(extracted_mask, size=(c.cropsize_val, c.cropsize_val), mode="bilinear", align_corners=False)
                final_nr = torch.where(extracted_mask < 0.0000000001, noised, restored)


                """ Save Images """
                if c.save_img:
                    extracted_sec = DeBlocking(secret_rev)
                    extracted_sec = ycbcr_to_rgb(extracted_sec)

                    print(image_file)
                    torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + image_file)
                    if c.noises != []:
                        torchvision.utils.save_image(mask, c.IMAGE_PATH_localization_map + image_file)
                        torchvision.utils.save_image(noised, c.IMAGE_PATH_noised + image_file)

                    # extracted_mask_img.save(c.extracted_folder + image_file)
                    save_image(extracted_mask, c.extracted_folder + image_file)
                    torchvision.utils.save_image(extracted_sec, c.IMAGE_PATH_ex + image_file)
                    torchvision.utils.save_image(restored, c.IMAGE_PATH_restored + image_file)
                    torchvision.utils.save_image(final_nr, c.IMAGE_PATH_final_nr + image_file)
                    

                final_nr_np = final_nr.cpu().numpy().squeeze() * 255
                np.clip(final_nr_np, 0, 255)

                cover_np = cover.cpu().numpy().squeeze() * 255
                np.clip(cover_np, 0, 255)

                restored_np = restored.cpu().numpy().squeeze() * 255
                np.clip(restored_np, 0, 255)

                
                psnr_temp1 = computePSNR(final_nr_np, cover_np)
                psnr_fc.append(psnr_temp1)

                psnr_temp2 = computePSNR(restored_np, cover_np)
                psnr_rc.append(psnr_temp2)
                if c.HashEmbedding:
                    if c.WM_MODEL == 'RedMark':
                        # equal_elements = (W == w_extracted)
                        # print(extracted_hash_bits.shape)
                        # print(hash_value.shape)
                        bitAcc = metrics.bit_accuracy(extracted_hash_bits.reshape(1,-1), hash_value)
                        noised_bitAcc = metrics.bit_accuracy(noised_hash_bits.reshape(1,-1), hash_value)
                    else:
                        bitAcc = metrics.bit_accuracy(extracted_watermark[:96], watermark[:96])
                    if bitAcc < c.MIN_ACC/100:
                        low_acc_count += 1
                    bit_acc_list.append(bitAcc.item())
                    noised_bit_acc_list.append(noised_bitAcc.item())

                    print(f"Index: {num_batches:05d}, BitAcc: {bitAcc}, noised BitAcc: {noised_bitAcc}, F1 Score: {f1}, PSNR final vs. cover: {psnr_temp1}, PSNR restored vs. cover: {psnr_temp2}\n")
                else:
                    print(f"Index: {num_batches:05d}, F1 Score: {f1}, PSNR final vs. cover: {psnr_temp1}, PSNR restored vs. cover: {psnr_temp2}\n")

                num_batches += 1
                if c.noises == [] and num_batches == num_edited_image_files:
                    break

        ''' Final Results'''
        avg_precision = np.nanmean(precision_values)
        avg_recall = np.nanmean(recall_values)
        avg_f1 = np.nanmean(f1_values)
        avg_iou = np.nanmean(iou_values)

        PSNR_fc = np.mean(psnr_fc)
        PSNR_rc = np.mean(psnr_rc)
        # Compute average metrics
        # Compute statistics
        # Convert list to tensor
        if c.HashEmbedding:
            noised_bit_acc_list = torch.tensor(noised_bit_acc_list)
            bit_acc_list = torch.tensor(bit_acc_list)

            average_noised_bit_acc = noised_bit_acc_list.mean()
            average_bit_acc = bit_acc_list.mean()
            min_bit_acc = bit_acc_list.min()
            max_bit_acc = bit_acc_list.max()
            std_bit_acc = bit_acc_list.std(unbiased=False)  # Use unbiased=False for population std
            q1 = torch.quantile(bit_acc_list, 0.25)  # First quartile (Q1)
            q3 = torch.quantile(bit_acc_list, 0.75)  # Third quartile (Q3)

            # Print results
            print(f"Average Bit Accuracy: {average_bit_acc.item():.4f}")
            print(f"Average Bit Accuracy for noised image: {average_noised_bit_acc.item():.4f}")
            print(f"Min Bit Accuracy: {min_bit_acc.item():.4f}")
            print(f"Max Bit Accuracy: {max_bit_acc.item():.4f}")
            print(f"Standard Deviation: {std_bit_acc.item():.4f}")
            print(f"Q1 (First Quartile): {q1.item():.4f}")
            print(f"Q3 (Third Quartile): {q3.item():.4f}")
            if c.ECCMethod == None:
                print(f"Average Detection Accuracy: {(1 - missed_watermarks/num_batches)*100:.4f}\%")
        

        print("Average PSNR of the final image vs. the cover image is: ", PSNR_fc)
        print("Average PSNR of the restored thumbnail vs. cover image is: ", PSNR_rc)
        # Print average metrics

        # print(f"Average Bit Accuracy: {average_bit_acc.item():.4f}")

        

        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f}")
        print(f"Average IOU: {avg_iou:.4f}")
        if c.HashEmbedding:
            print(f"Number of Missed Watermarks: {missed_watermarks:2d}")
            print(f"Number of Low Accuracy Watermarks Detectoion: {low_acc_count:2d}")

if __name__ == '__main__':
    main()


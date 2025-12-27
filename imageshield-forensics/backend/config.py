# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = - 4.5
lr = 10 ** log10_lr
epochs = 30000
weight_decay = 1e-5
init_scale = 0.01
save_img = True
Counter = 13000
Counter2 = 14000



WM_MODEL = "InvisMark" # InvisMark or TrustMark or RedMark
HashEmbedding = False
ECCMethod = 'None'  # None or 'BCH' or 'RS' or 'polar'

hash_length = 8 #^2. The hash length is power of two of hash_length.
if hash_length == 7:
    HASH_SIZE = 52
    WM_SIZE = 61
    MIN_ACC = 94
elif hash_length == 6:
    HASH_SIZE = 36
    WM_SIZE = 40
    MIN_ACC = 92
else:
    HASH_SIZE = hash_length * hash_length
    WM_SIZE = HASH_SIZE
    MIN_ACC = 99

wm_cap = 100 # Capacity of the watermarking system
ecc_symbol = (wm_cap-hash_length*hash_length)//8  # (embedding capacity - hash length)//8

if WM_MODEL == 'RedMark' and ECCMethod == 'polar':
    wm_cap = 256
elif WM_MODEL == 'RedMark' and ECCMethod == None:
    wm_cap = 64


USE_HUMAN_MASKS = 2 # 0 means only automatic mask, 1 means only human mask and 2 means mixed.




IMAGE_PATH = './results/'

# RESTORE_PATH = 'C:/Users/maarab/Forensics/SHIELD/image/'

TFD = False #Train with Forgery Detector
TE = True # Train the embedding module
HLoss = True #Hybrid Loss

AA = False # Advanced Attacker that is able to remove the watermark, tamper with the image, and add the watermark later.
secret_length = 512



num_hidden = 1

lambda_e = 1
lambda_d = 1
Upsample_mode = 'nearest'

# lamda_reconstruction = 5
# lamda_guide = 1
# lamda_low_frequency = 1
device_ids = [0]
BW = True # One channel secret images will be used.
YCC = False # Using YCbCr instead of RGB. Two channel secret images will be used.
ND1 = False # New design for the embedding of Secret Image and the code blocks. The secret channel has 1 channel.
ND2 = False # New design for the embedding of Secret Image and the code blocks. The secret channel has 2 channels.
ND3 = False # New design for the embedding of Secret Image and the code blocks. The secret channel has 3 channels.

ChB = False # Using YCbCr and a CheckerBoard image for tampering detection. 3 channel secret images are used.

if BW:
    thumbnail_number = 2 # for the case of YCC thumbnail number=4 will be 8 thumbnails and 8 codes, if AA is present and 16 thumbnails otherwise. For the case of BW, this number should be 2 for a 4X4 thumbanail and code pattern.
elif YCC:
    thumbnail_number = 4 # for the case of YCC thumbnail number=4 will be 8 thumbnails and 8 codes, if AA is present and 16 thumbnails otherwise. For the case of BW, this number should be 2 for a 4X4 thumbanail and code pattern.
elif ND1:
    thumbnail_number = 4 
elif ND2:
    thumbnail_number = 2
elif ND3:
    thumbnail_number = 2

# Train:

code_size = 4 # The code block will be [code_size X code_size]
bit_redundancy = 6
batch_size = 16
cropsize = 128
betas = (0.5, 0.999)
weight_step = 80
gamma = 0.5
milestone = [100, 200, 300, 400]
noises = []
attacks = []
# noises = ["Identity"]
# noises = ['Compression', 'CropOut',  'DropOut', 'GaussianBlur', 'GaussianNoise', 'SaltPepper', 'AdjustHue', 'AdjustSaturation', 'AdjustBrightness','AdjustContrast','AdjustColor','AdjustSharpness','JPEG2000', 'WebP', 'MedianFilter', 'AverageFilter', 'PixelElimination']

# noises = ['Compression', 'DropOut', 'GaussianBlur', 'GaussianNoise', 'AdjustSaturation', 'AdjustBrightness','AdjustContrast','AdjustColor', 'WebP', 'MedianFilter', 'AverageFilter']
# noises = ['Compression', 'WebP', 'GaussianNoise', 'SaltPepper'] #model_checkpoint_00970_mixed_denoiser
# noises = ['GaussianNoise']
# noises = ['Compression', 'GaussianNoise', 'DropOut', 'WebP']

infoDict = {'JPEGQ':80 ,'resize_ratio': 0.5, 'pixel_ratio': 0.1 ,'cropout_ratio': (0.54,0.55),'cropout_ratio_area':0.2, 'copymove_ratio': 0.2,'crop_ratio':0.7, 'prob': 0.8, 'sigma':1.5, 'Standard_deviation':0.03, 'Amount': 0.03, 'hue_factor':0.05, 'sat_factor':2, 'bri_factor':1.2,'con_factor':0.66,'sha_factor':1.5,'col_factor':2, 'ave_kernel':33, 'med_kernel':33}

# noises = ['Compression','GaussianBlur', 'MedianFilter','GaussianNoise','DropOut', 'Rescaling', 'Crop'] # Distortions for fullmodel420
# noises = ['Compression','GaussianBlur', 'MedianFilter','GaussianNoise','DropOut'] # Distortions for modified470 and 1230
# attacks = ['Splicing', 'CopyMove', 'Inpainting', 'ObjectAddition'] # Tampering

# attacks = ['Inpainting']
# noises = ['GaussianNoise']




# Val:
cropsize_val = 512
batchsize_val = 1
shuffle_val = False
val_freq = 10


# Dataset
OBJECT_PATH = 'C:/Users/maarab/Forensics/Datasets/object'


# TRAIN_PATH = 'C:/Users/maarab/Forensics/Datasets/Mixed-1000'
# VAL_PATH = 'C:/Users/maarab/Forensics/Datasets/Mixed-1000'
# format_train = 'png'
# format_val = 'png'

# TRAIN_PATH = 'C:/Users/maarab/Forensics/Datasets/DIV2K/train'
# VAL_PATH = 'C:/Users/maarab/Forensics/Datasets/DIV2K/val'
# format_train = 'png'
# format_val = 'png'


# TRAIN_PATH = 'C:/Users/maarab/Forensics/Datasets/COCO/train'
# VAL_PATH = 'C:/Users/maarab/Forensics/Datasets/COCO/val'
# format_train = 'jpg'
# format_val = 'jpg'

# VAL_PATH = 'C:/Users/maarab/Forensics/Datasets/ImageNet'
# format_val = 'JPEG'

# VAL_PATH = 'C:/Users/maarab/Forensics/Datasets/Places'
# format_val = 'jpeg'

TRAIN_PATH = 'C:/Users/maarab/Forensics/Datasets/Flickr/train'
VAL_PATH = '/home/ec2-user/Forensics-AI/test-imgs'
format_train = 'png'
format_val = 'png'

# TRAIN_PATH = 'C:/Users/maarab/Forensics/Datasets/celebA/train'
# VAL_PATH = 'C:/Users/maarab/Forensics/Datasets/celebA/val'
# format_train = 'jpg'
# format_val = 'jpg'


# TRAIN_PATH = 'C:/Users/maarab/Forensics/Datasets/Bentham/train'
# VAL_PATH = 'C:/Users/maarab/Forensics/Datasets/Bentham/val'
# format_train = 'jpg'
# format_val = 'jpg'



MIXED = False
DISTORT = False




# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = './model/'
checkpoint_on_error = False
SAVE_freq = 10


DIRPATH = ""
if noises != []:
    if noises[0] == "Compression":
        DIRPATH = "JPEG/"
    elif noises[0] == "GaussianNoise":
        DIRPATH = "GN/"
    elif noises[0] == "GaussianBlur":
        DIRPATH = "GB/"

# Folders in embedding phase
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_converted = IMAGE_PATH + 'converted/'
IMAGE_PATH_secret_converted = IMAGE_PATH + 'sec_converted/'
IMAGE_PATH_double_WM = IMAGE_PATH + 'doubleWM/'
# Folders after manual edit
IMAGE_PATH_edited = IMAGE_PATH + 'edited/'
IMAGE_PATH_mask = IMAGE_PATH + 'localization_map/'

# Folders in extraction phase
IMAGE_PATH_secret_rev = IMAGE_PATH + DIRPATH + 'secret_rev/'
IMAGE_PATH_localization_map = IMAGE_PATH + DIRPATH + 'localization_map/'
extracted_folder = IMAGE_PATH + DIRPATH + 'extracted_map/'
IMAGE_PATH_distorted = IMAGE_PATH + DIRPATH + 'distorted/'
IMAGE_PATH_noised = IMAGE_PATH + DIRPATH + 'noised/'
IMAGE_PATH_ex = IMAGE_PATH + DIRPATH + 'converted_doubleWM/'
IMAGE_PATH_restored = IMAGE_PATH + DIRPATH + 'restored/'
IMAGE_PATH_final_nr = IMAGE_PATH + DIRPATH + 'final_nr/'
IMAGE_PATH_final_nn = IMAGE_PATH + DIRPATH + 'final_nn/'
IMAGE_PATH_final_cr = IMAGE_PATH + DIRPATH + 'final_cr/'




# List of paths to check and create
embedding_paths = [
    IMAGE_PATH_cover,
    IMAGE_PATH_secret,
    IMAGE_PATH_steg,
    IMAGE_PATH_converted,
    IMAGE_PATH_secret_converted,
    IMAGE_PATH_double_WM,
    IMAGE_PATH_edited,
    IMAGE_PATH_mask
]


extraction_paths = [
    IMAGE_PATH_secret_rev,
    IMAGE_PATH_localization_map,
    extracted_folder,
    IMAGE_PATH_distorted,
    IMAGE_PATH_noised,
    IMAGE_PATH_ex,
    IMAGE_PATH_restored,
    IMAGE_PATH_final_nr,
    IMAGE_PATH_final_nn,
    IMAGE_PATH_final_cr
]






# IMAGE_PATH_residual = IMAGE_PATH + 'residual/'
# IMAGE_PATH_denoised = IMAGE_PATH + 'denoised/'


# IMAGE_PATH_code_retrieved = IMAGE_PATH + 'code-ret/'
# IMAGE_PATH_code = IMAGE_PATH + 'code/'
# IMAGE_PATH_localization = IMAGE_PATH + 'map/'








# IMAGE_PATH_ex2 = IMAGE_PATH + 'secret-rev-ex2/'

# EX_PATH = 'C:/Users/maarab/Forensics/SHIELD/image/edited/'
# IMAGE_PATH_ex = 'C:/Users/maarab/Forensics/SHIELD/image/extracted/'


VAL_IMAGE_PATH = 'C:/Users/maarab/Forensics/SHIELD/val_image/'


# IMAGE_PATH_tiled = IMAGE_PATH + 'tiled/'
# IMAGE_PATH_tiled_rev = IMAGE_PATH + 'tiled-rev/'
# IMAGE_PATH_LR = 'C:/Users/maarab/Forensics/SwinIR/testsets/tiled/LR/'
# IMAGE_PATH_HR = 'C:/Users/maarab/Forensics/SwinIR/testsets/tiled/HR/'

save_SR = True

TF = 'Haar' # Haar, DCT

USE_Denoiser = False
USE_Deblur = False

NOISE_TYPE = "Constant" # Gaussian, Uniform, Constant

# Load:
# suffix = 'model_checkpoint_00650_noTFD_HLoss_fulltrain.pt' # The full model that has been trained on thumbnails
suffix = 'model_checkpoint_01110_1channel_JPEG.pt'
# suffix = 'model_YCBCR_00700.pt'

continue_train = 1  #[0,1,2]

# 0:not continuing training
# 1:only data-embedding is trained and will be loaded
# 2:both data-embedding and forgery detector are trained and get loaded

if continue_train>0:
    trained_epoch = 299
else:
    trained_epoch = 0




MultiModel = False
suffix2 = 'model_forgery_00340.pt'
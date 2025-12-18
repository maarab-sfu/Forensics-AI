import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import torch
import torch.nn as nn 
import random
import numpy as np
import skimage
from torch.autograd import Variable 
import cv2
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image, ImageEnhance

import os
from inpainting import Generator
from kornia.filters import MedianBlur, GaussianBlur2d, BoxBlur
import torchvision.transforms as transforms
import torchvision

###############
### Attacks ###
###############
class Splicing(nn.Module):
    def __init__(self, device):
        super(Splicing, self).__init__()
        self.device = device

    def forward(self, noised, cover, mask):
        noised_image = noised.clone()
        tampered_image = noised_image.clone()
        tampered_image[mask > 0] = cover[mask > 0]
        noised_image = tampered_image
        return noised_image

class CopyMove(nn.Module):
    def __init__(self, device):
        super(CopyMove, self).__init__()
        self.device = device

    def forward(self, noised, mask):
        noised_image = noised.clone()
        channels, height, width = noised_image.shape

        tiled_image = torch.zeros(channels, 2 * height, 2 * width).to(self.device)
        tiled_image[:, 0:height, 0:width] = noised_image
        tiled_image[:, 0:height, width:2*width] = noised_image
        tiled_image[:, height:2*height, 0:width] = noised_image
        tiled_image[:, height:2*height, width:2*width] = noised_image

        start_x = random.randint(0, width)
        start_y = random.randint(0, height)

        extracted_image = tiled_image[:, start_y:start_y + height, start_x:start_x + width]

        tampered_image = noised_image.clone()
        tampered_image[mask > 0] = extracted_image[mask > 0]

        return tampered_image

def DeepFillV2(generator, image, mask):
    output = generator.infer(image, mask, return_vals=['inpainted'])
    return output

class Inpainting(nn.Module):
    def __init__(self, generator, device):
        super(Inpainting, self).__init__()
        self.generator = generator
        self.device = device

    def forward(self, noised, mask):
        noised_image = noised.clone()
        tampered_image = DeepFillV2(self.generator, noised_image, mask) / 255.0
        noised_image = torch.from_numpy(np.transpose(tampered_image, (2, 0, 1))).to(self.device)
        return noised_image

class ObjectAddition(nn.Module):
    def __init__(self, device, object_dir):
        super(ObjectAddition, self).__init__()
        self.device = device
        self.object_dir = object_dir
        self.transform = ToTensor()

    def forward(self, noised, mask):
        noised_image = noised.clone()
        mask_image = mask.clone()
        channels, height, width = noised_image.shape

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

        # Generate a random position for placing the object
        max_x, max_y = width - new_size[0], height - new_size[1]
        start_x, start_y = random.randint(0, max_x), random.randint(0, max_y)

        # Extract object RGB and Alpha channel
        obj_rgb = obj_tensor[:3]  # Shape: (3, H, W)
        obj_alpha = obj_tensor[3].unsqueeze(0)  # Shape: (1, H, W)

        
        
        # Place the object onto the image using alpha blending
        end_x, end_y = start_x + new_size[0], start_y + new_size[1]
        noised_image[:, start_y:end_y, start_x:end_x] = (
            noised_image[:, start_y:end_y, start_x:end_x] * (1 - obj_alpha) + obj_rgb * obj_alpha
        )

        # Update the mask (mark new edited area with 0s)
        mask_image[:, start_y:end_y, start_x:end_x] = mask_image[:, start_y:end_y, start_x:end_x] * (1 - obj_alpha) + obj_alpha

        # other_np = mask_image.cpu().numpy()
        # other_np = other_np[0]
        # other_pil = Image.fromarray((other_np* 255).astype(np.uint8)) 
        # # print("Mask image min/max:", mask_image.min().item(), mask_image.max().item())
        # other_pil.show()

        # print("Mask image min/max:", mask_image.min().item(), mask_image.max().item())
        # print("Alpha min/max:", obj_alpha.min().item(), obj_alpha.max().item())
        
        # # Convert mask_tensor to a NumPy image
        # mask_np = mask.cpu().numpy()  # Move to CPU if necessary
        # mask_np = mask_np[0]  # Remove the channel dimension if needed

        # # Assuming `other_tensor` is another image tensor of the same shape
        # other_np = mask_image.cpu().numpy()
        # other_np = other_np[0]  # Remove the channel dimension if needed

        # # Stack images side by side
        # side_by_side = np.hstack((mask_np, other_np))

        # # Convert to PIL Image for visualization
        # before = mask.cpu().numpy()[0]  # Original
        # after = mask_image.cpu().numpy()[0]  # Modified
        # diff = before - after
        # Image.fromarray((diff * 255).astype(np.uint8)).show()
        # side_by_side_image = Image.fromarray(side_by_side.astype(np.uint8))

        # # Show the image
        # side_by_side_image.show()
        
        return noised_image, mask_image

###################
### Distortions ###
###################

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, noised):
        return noised

class GaussianNoise(nn.Module):
    def __init__(self, Standard_deviation):
        super(GaussianNoise, self).__init__()
        self.Standard_deviation = Standard_deviation

    def forward(self, noised):
        noised_image = noised
        batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy()
        batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(batch_encoded_image.shape[0]):
            encoded_image = batch_encoded_image[idx]
            noise_image = skimage.util.random_noise(encoded_image, mode= 'gaussian',clip = False, var = (self.Standard_deviation) ** 2 )
            noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised = 2*batch_noise_image - 1
        return noised

class SaltPepper(nn.Module):
    def __init__(self,Amount):
        super(SaltPepper, self).__init__()
        self.Amount = Amount

    def forward(self, noised):
        noised_image = noised
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
        noised = batch_noise_image
        return noised

class GaussianBlur(nn.Module):
    def __init__(self, sigma = 1.5):
        super(GaussianBlur, self).__init__()
        self.sigma = sigma
        self.gaussian_filters = {
			1: GaussianBlur2d((3,3), (1, 1.5)),
		    1.5: GaussianBlur2d((3,3), (1.5, 1)),
		    3: GaussianBlur2d((3,3), (1.5, 1.5)),
		}

    def forward(self, noised):
        noised_image = noised
        blur_result = self.gaussian_filters[self.sigma](noised_image)
        noised = blur_result
        return noised

class MedianFilter(nn.Module):
    def __init__(self, kernel = 7):
        super(MedianFilter, self).__init__()
        self.kernel = kernel
        self.median_filters = {
			1: MedianBlur((1, 3)),
		    2: MedianBlur((3, 1)),
		    3: MedianBlur((3, 3)),
		}

    def forward(self, noised):
        noised_image = noised
        blur_result = self.median_filters[self.kernel](noised_image)
        noised = blur_result
        return noised

class BoxFilter(nn.Module):
    def __init__(self, kernel = 7):
        super(BoxFilter, self).__init__()
        self.kernel = kernel
        self.box_filters = {
			1: BoxBlur((1, 3)),
		    2: BoxBlur((3, 1)),
		    3: BoxBlur((3, 3)),
		}

    def forward(self, noised):
        noised_image = noised
        blur_result = self.box_filters[self.kernel](noised_image)
        noised = blur_result
        return noised

class AverageFilter(nn.Module):
    def __init__(self, kernel = 5):
        super(AverageFilter, self).__init__()
        self.kernel = kernel

    def forward(self, noised):
        noised_image = noised
        batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        # batch_encoded_image = batch_encoded_image.transpose((1,2,0))
        batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        # for idx in range(batch_encoded_image.shape[0]):

        encoded_image = batch_encoded_image[0]
        noise_image = cv2.blur(encoded_image, (self.kernel, self.kernel))
        noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
        # if (idx == 0):
        batch_noise_image = noise_image.unsqueeze(0)
        # else:
        #     batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C

        # batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised = (2*batch_noise_image - 1)/255
        return noised

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
 
        return noised_and_cover[0]

class PixelElimination(nn.Module):
    def __init__(self, pixel_ratio):
        super(PixelElimination, self).__init__()
        self.pixel_ratio = pixel_ratio

    def forward(self, noised):
        noised_image = noised
        _,_,H,W = noised_image.shape

        elimination_mask = torch.ones_like(noised_image)

        idx_H = np.random.randint(H, size=(int(self.pixel_ratio*H)))
        idx_W = np.random.randint(W, size=(int(self.pixel_ratio*W)))

        elimination_mask[:, :, :, idx_W] = 0
        elimination_mask[:, :, idx_H, :] = 0

        noised = noised_image * elimination_mask
        return noised

class AdjustBrightness(nn.Module):
    def __init__(self, bri_factor):
        super(AdjustBrightness, self).__init__()
        self.bri_factor = bri_factor
    
    def forward(self, noised):
        noised_image = noised
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
        noised = (2*batch_noise_image - 1)
        return noised

class AdjustContrast(nn.Module):
    def __init__(self, con_factor):
        super(AdjustContrast, self).__init__()
        self.con_factor = con_factor

    def forward(self, noised):
        noised_image = noised
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
        noised = (2*batch_noise_image - 1)
        return noised

class AdjustColor(nn.Module):
    def __init__(self, col_factor):
        super(AdjustColor, self).__init__()
        self.col_factor = col_factor
    def forward(self, noised):
        noised_image = noised
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
        noised = (2*batch_noise_image - 1)
        return noised

class AdjustSharpness(nn.Module):
    def __init__(self, sha_factor):
        super(AdjustSharpness, self).__init__()
        self.sha_factor = sha_factor
    def forward(self, noised):
        noised_image = noised
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
        noised = (2*batch_noise_image - 1)
        return noised

class RealComp(nn.Module):
    def __init__(self, comp_type, quality, device):
        super(RealComp, self).__init__()
        self.comp_type = comp_type
        self.device = device 
        self.quality = quality
    def forward(self, imgs):
        x = imgs.squeeze(0)

        # x[i] = x[i]*0.5+0.5
        pil_img = transforms.ToPILImage()(x)
        # pil_img = imgs[i]

        
        if self.comp_type == "WebP":
            pil_img.save("temp.webp", format="webp", quality=self.quality)
            comp_img = Image.open("temp.webp").convert('RGB')

        elif self.comp_type == "Compression":
            pil_img.save("temp.jpg", format = "JPEG", quality=self.quality)
            comp_img = Image.open("temp.jpg").convert('RGB')

        comp_img = transforms.PILToTensor()(comp_img).float()
        comp_img = transforms.CenterCrop(x.shape[2])(comp_img)
        comp_img = comp_img/255.0


        imgs = comp_img.to(self.device)
        return imgs.unsqueeze(0)

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Image Editor")

        self.values_map = None
        

        self.brush_color = "white"  # Brush color is set to white for the mask
        self.brush_size = 15

        

        self.dist_level = 0

        self.image = None
        self.cover = None
        self.tk_image = None
        self.painted_image = None
        self.mask2 = None

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.edit_mode = tk.StringVar(value="copymove")
        self.distortion_mode = tk.StringVar(value="NoDistortion")

        self.setup_ui()

    def setup_ui(self):
        self.root.state('zoomed')

        # Define discrete values and resolution for different distortion modes
        self.values_map = {
            "JPEG": {"values": [80, 85, 90, 95], "resolution": 5},
            "DropOut": {"values": [0.8, 0.85, 0.9, 0.95], "resolution": 0.05},
            # "GaussianBlur": {"values": [1, 2, 3, 4, 5], "resolution": 1},
            "GaussianNoise": {"values": [0.01, 0.02, 0.03], "resolution": 0.01},
            # "SaltPepper": {"values": [0.01, 0.02, 0.03, 0.04, 0.05], "resolution": 0.01},
            # "AdjustHue": {"values": [0.01, 0.03, 0.05, 0.07, 0.09], "resolution": 0.02},
            # "AdjustSaturation": {"values": [0.5, 1.0, 1.5, 2], "resolution": 0.5},
            # "AdjustBrightness": {"values": [0.5, 1.0, 1.5, 2], "resolution": 0.5},
            # "AdjustContrast": {"values": [0.5, 1.0, 1.5, 2], "resolution": 0.5},
            # "AdjustColor": {"values": [0.5, 1.0, 1.5, 2], "resolution": 0.5},
            # "AdjustSharpness": {"values": [0.5, 1.0, 1.5, 2], "resolution": 0.5},
            "WebP": {"values": [80, 85, 90, 95], "resolution": 5},
            # "MedianFilter": {"values": [1, 2, 3], "resolution": 1},
            # "AverageFilter": {"values": [1, 2, 3], "resolution": 1},
            # "PixelElimination": {"values": [0.05, 0.1 ,0.15, 0.2], "resolution": 0.05},
        }
        
        # Left Side for the Image
        left_frame = tk.Frame(self.root)
        left_frame.pack(anchor="n", side=tk.LEFT, padx=20, pady=20)

        # Create a canvas to display the image
        self.canvas = tk.Canvas(root, width=1024, height=1024)
        self.canvas.pack(side=tk.LEFT, expand=True)

        # Right Side for Controls
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, padx=20, pady=20, anchor="n")

        # Load Image button
        self.load_button = tk.Button(right_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(anchor="n", pady=10)

        # Frame for edit mode radio buttons (CopyMove, Splicing, Inpainting)
        edit_mode_frame = tk.Frame(right_frame)
        edit_mode_frame.pack(anchor="n", pady=10)

        tk.Label(edit_mode_frame, text="Edit Mode:").grid(row=0, column=0, columnspan=3)

        self.radio_copymove = tk.Radiobutton(edit_mode_frame, text="CopyMove", variable=self.edit_mode, value="copymove")
        self.radio_copymove.grid(row=1, column=0, padx=5)

        self.radio_splicing = tk.Radiobutton(edit_mode_frame, text="Splicing", variable=self.edit_mode, value="splicing")
        self.radio_splicing.grid(row=1, column=1, padx=5)

        self.radio_inpainting = tk.Radiobutton(edit_mode_frame, text="Inpainting", variable=self.edit_mode, value="inpainting")
        self.radio_inpainting.grid(row=1, column=2, padx=5)

        self.radio_objectadd = tk.Radiobutton(edit_mode_frame, text="ObjectAddition", variable=self.edit_mode, value="object addition")
        self.radio_objectadd.grid(row=1, column=3, padx=5)

        # Frame for brush size slider, apply edit button, and image counter label
        control_frame = tk.Frame(right_frame)
        control_frame.pack(anchor="n", pady=10)

        # Brush size slider

        # Brush size selection using OptionMenu
        self.brush_size_var = tk.StringVar(value="Small")
        brush_sizes = {"Small": 12, "Medium": 21, "Big": 30}

        self.brush_size_menu = tk.OptionMenu(
            control_frame, self.brush_size_var, *brush_sizes.keys(),  # No need to repeat "Small"
            command=lambda choice: self.change_brush_size(brush_sizes[choice])
        )
        self.brush_size_menu.grid(row=0, column=0, padx=10)

        # Create an invisible preview circle
        self.preview_circle = self.canvas.create_oval(0, 0, 0, 0, outline="red")

        # self.brush_size_slider = tk.Scale(control_frame, from_=15, to=50, orient=tk.HORIZONTAL, label="Brush Size")
        # self.brush_size_slider.set(self.brush_size)
        # self.brush_size_slider.grid(row=0, column=0, padx=10)

        # Apply edit button
        self.apply_button = tk.Button(control_frame, text="Apply Edit", command=self.apply_edit)
        self.apply_button.grid(row=0, column=1, padx=10)

        # Image counter label
        # self.image_counter_label = tk.Label(control_frame, text="Image 0 of 0")
        # self.image_counter_label.grid(row=0, column=2, padx=10)

        # Replace Label with Entry
        self.image_counter_entry = tk.Entry(control_frame, width=10)
        self.image_counter_entry.grid(row=0, column=2, padx=10)
        self.image_counter_entry.insert(0, "Image 0 of 0")

        # Bind the Enter key to update the image when user enters a new index
        self.image_counter_entry.bind("<Return>", self.change_image_by_entry)

        # Frame for distortion mode radio buttons (JPEG, DropOut, etc.)
        distortion_frame = tk.Frame(right_frame)
        distortion_frame.pack(anchor="n", pady=10)

        tk.Label(distortion_frame, text="Distortion Mode:").grid(row=0, column=0, columnspan=4)

        self.radio_jpeg = tk.Radiobutton(distortion_frame, text="JPEG", variable=self.distortion_mode, value="JPEG", command=self.update_distortion_level_slider)
        self.radio_jpeg.grid(row=1, column=0, padx=5, pady=5)

        self.radio_dropout = tk.Radiobutton(distortion_frame, text="DropOut", variable=self.distortion_mode, value="DropOut", command=self.update_distortion_level_slider)
        self.radio_dropout.grid(row=1, column=1, padx=5, pady=5)

        self.radio_gaussian_noise = tk.Radiobutton(distortion_frame, text="Gaussian Noise", variable=self.distortion_mode, value="GaussianNoise", command=self.update_distortion_level_slider)
        self.radio_gaussian_noise.grid(row=1, column=2, padx=5, pady=5)

        self.radio_webp = tk.Radiobutton(distortion_frame, text="WebP", variable=self.distortion_mode, value="WebP", command=self.update_distortion_level_slider)
        self.radio_webp.grid(row=1, column=3, padx=5, pady=5)

        self.radio_no_distortion = tk.Radiobutton(distortion_frame, text="No Distortion", variable=self.distortion_mode, value="NoDistortion", command=self.update_distortion_level_slider)
        self.radio_no_distortion.grid(row=2, column=2, padx=5, pady=5)

        # Distortion Level slider
        self.dist_level_slider = tk.Scale(control_frame, from_=0, to=0, orient=tk.HORIZONTAL, label="Distortion Level")
        self.dist_level_slider.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Save and Next button
        self.save_button = tk.Button(right_frame, text="Save and Go to Next", command=self.save_and_next)
        self.save_button.pack(anchor="n", pady=10)

        self.canvas.bind("<Motion>", self.update_preview)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def update_distortion_level_slider(self):
        

        # Get the selected distortion mode
        mode = self.distortion_mode.get()

        # If mode is "NoDistortion", disable the slider
        if mode == "NoDistortion":
            self.dist_level_slider.config(state=tk.DISABLED)
            self.dist_level_slider.set(0)
        else:
            # Enable the slider and set the values and resolution
            self.dist_level_slider.config(state=tk.NORMAL)
            
            # Set the values and resolution for the selected distortion mode
            if mode in self.values_map:
                values = self.values_map[mode]["values"]
                resolution = self.values_map[mode]["resolution"]
                self.dist_level_slider.config(
                    from_=min(values),
                    to=max(values),
                    resolution=resolution
                )
                self.dist_level_slider.set(values[0])


    def load_image(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_folder = folder_path
            self.image_files = sorted([f for f in os.listdir(os.path.join(folder_path, "doubleWM")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.total_images = len(self.image_files)
            self.current_image_index = 0
            self.load_current_image()
            self.update_image_counter()

    def load_current_image(self):
        if self.current_image_index < self.total_images:
            file_path = os.path.join(self.image_folder, "doubleWM", self.image_files[self.current_image_index])
            cover_file_path = os.path.join(self.image_folder, "cover", self.image_files[self.current_image_index])
            self.image = Image.open(file_path)
            self.cover = Image.open(cover_file_path)
            
            # Select a random cover image from the "doubleWM" folder
            random_cover_filename = random.choice(self.image_files)
            cover_image_path = os.path.join(self.image_folder, "doubleWM", random_cover_filename)
            self.cover_image = Image.open(cover_image_path)

            self.painted_image = Image.new("L", self.image.size, 0)  # Black mask image
            self.mask2 = Image.new("L", self.image.size, 0)  # Initialize mask2 as well
            self.update_canvas()
    
    # def update_image_counter(self):
    #     self.image_counter_label.config(text=f"Image {self.current_image_index + 1} of {self.total_images}")
    def update_image_counter(self):
        """Update the entry text based on the current image index."""
        self.image_counter_entry.delete(0, tk.END)
        self.image_counter_entry.insert(0, f"Image {self.current_image_index + 1} of {self.total_images}")

    def change_image_by_entry(self, event):
        """Handle user input to change the image based on entered index."""
        try:
            new_index = int(self.image_counter_entry.get().split()[1]) - 1  # Extract number and convert to 0-based index
            if 0 <= new_index < self.total_images:
                self.current_image_index = new_index
                self.load_current_image()
                self.update_image_counter()
            else:
                self.image_counter_entry.delete(0, tk.END)
                self.image_counter_entry.insert(0, f"Image {self.current_image_index + 1} of {self.total_images}")
        except (ValueError, IndexError):
            # Reset to valid value if invalid input
            self.update_image_counter()
        
    def update_preview_circle(self):
        """Update the preview circle to match the selected brush size."""
        x, y = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx(), self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        self.canvas.coords(self.preview_circle, x - self.brush_size, y - self.brush_size, x + self.brush_size, y + self.brush_size)
        self.canvas.itemconfig(self.preview_circle, outline="red")
        self.canvas.tag_raise(self.preview_circle)

    def update_preview(self, event):
        """Move the preview circle with the cursor."""
        x, y = event.x, event.y
        self.canvas.coords(self.preview_circle, x - self.brush_size, y - self.brush_size, x + self.brush_size, y + self.brush_size)
        self.canvas.tag_raise(self.preview_circle)  # Ensure it's visible on top


    def update_canvas(self):
        if self.image and self.painted_image:
            # Create a combined image with original image and painted mask
            mask_image = Image.new("RGB", self.image.size)
            mask_image.paste(self.image)
            mask_image.paste(Image.new("RGB", self.image.size, "black"), mask=self.painted_image)

            self.tk_image = ImageTk.PhotoImage(mask_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            # self.canvas.itemconfig(self.image_on_canvas, image=self.image_tk)
            self.canvas.tag_raise(self.preview_circle)  # Ensure the ring is always visible

    def paint(self, event):
        """Paint on the canvas with the selected brush size."""
        if self.painted_image:
            x, y = event.x, event.y
            draw = ImageDraw.Draw(self.painted_image)
            draw.ellipse((x - self.brush_size, y - self.brush_size, 
                          x + self.brush_size, y + self.brush_size), 
                         fill=self.brush_color)
            self.update_canvas()

    def reset(self, event):
        self.update_canvas()
    
    def change_brush_size(self, value):
        self.brush_size = int(value)  # Update brush size based on slider value
        self.update_preview_circle()
        
    def save_mask(self):
        if self.mask2:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
            if file_path:
                self.mask2.save(file_path)

    def save_and_next(self):
        # if self.edit_mode.get() != "object addition":
        #     self.apply_edit()
        if self.mask2 and self.image:
            # Save the mask
            mask_filename = os.path.join(self.image_folder, "localization_map", self.image_files[self.current_image_index])
            self.mask2.save(mask_filename)

            noised_image = torch.from_numpy(np.array(self.image).transpose(2, 0, 1)).float().to(self.device) / 255.0
            noised_image = noised_image.unsqueeze(0)
            cover_image = torch.from_numpy(np.array(self.cover).transpose(2, 0, 1)).float().to(self.device) / 255.0
            cover_image = cover_image.unsqueeze(0)

            if self.distortion_mode.get() == "NoDistortion":
                distort = Identity()
                edited_image = distort(noised_image)
            elif self.distortion_mode.get() == "JPEG":
                distort = RealComp("Compression", self.dist_level_slider.get(), self.device)
                edited_image = distort(noised_image)
            elif self.distortion_mode.get() == "DropOut":
                distort = DropOut(self.dist_level_slider.get())
                edited_image = distort([noised_image, cover_image])
            elif self.distortion_mode.get() == "GaussianNoise":
                distort = GaussianNoise(self.dist_level_slider.get())
                edited_image = distort(noised_image)
            elif self.distortion_mode.get() == "WebP":
                distort = RealComp("WebP", self.dist_level_slider.get(), self.device)
                edited_image = distort(noised_image)


            edited_image_filename = os.path.join(self.image_folder, "edited", self.image_files[self.current_image_index])
            torchvision.utils.save_image(edited_image, edited_image_filename)

            self.current_image_index += 1
            if self.current_image_index < self.total_images:
                self.load_current_image()
                self.update_image_counter()
            else:
                self.image_counter_label.config(text="All images processed")
                self.canvas.delete("all")  # Clear the canvas when done

    def apply_edit(self):
        if not self.painted_image or not self.image:
            return

        # Convert PIL image to torch tensor
        noised_image = torch.from_numpy(np.array(self.image).transpose(2, 0, 1)).float().to(self.device) / 255.0
        cover_image = torch.from_numpy(np.array(self.cover_image).transpose(2, 0, 1)).float().to(self.device) / 255.0

        
        # Modify the mask to have 3 channels
        mask_tensor = torch.from_numpy(np.array(self.painted_image)).unsqueeze(0).repeat(3, 1, 1).float().to(self.device)
        # print(self.painted_image)
        # Choose the editing function
        if self.edit_mode.get() == "splicing":
            splicing = Splicing(self.device)
            edited_image = splicing(noised_image, cover_image, mask_tensor)
        elif self.edit_mode.get() == "copymove":
            copymove = CopyMove(self.device)
            edited_image = copymove(noised_image, mask_tensor)
        elif self.edit_mode.get() == "inpainting":
            # You need to pass a pre-trained generator to Inpainting
            generator = Generator(checkpoint="./model/inpainting_model.pth", return_flow=True).to(self.device)
            inpainting = Inpainting(generator, self.device)
            edited_image = inpainting(noised_image, mask_tensor)
        elif self.edit_mode.get() == "object addition":
            objectadd = ObjectAddition(self.device, "C:/Users/maarab/Forensics/Datasets/COCO/test_seg/")
            edited_image, mask_tensor = objectadd(noised_image, mask_tensor)
            # Step 1: Remove the extra dimension
            mask_tensor = mask_tensor.squeeze(0)

            # Step 2: Convert to a NumPy array
            np_image = mask_tensor.cpu().numpy()  # Use .cpu() if mask_tensor is on a GPU

            # Step 3: Convert from CHW to HWC format if necessary
            np_image = np.transpose(np_image, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

            # Step 4: Convert to a PIL image
            self.painted_image = Image.fromarray((np_image*255).astype(np.uint8))
            self.painted_image = self.painted_image.convert('L')
            # print(self.painted_image)

        # Convert torch tensor back to PIL image
        edited_image_pil = Image.fromarray((edited_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

        # Update the original image with the edited one
        self.image = edited_image_pil

        # Update mask2 to keep track of all edits
        draw = ImageDraw.Draw(self.mask2)
        draw.bitmap((0, 0), self.painted_image, fill=255)

        # Clear the painted mask to display the edited image clearly
        self.painted_image = Image.new("L", self.image.size, 0)

        # Update the canvas to show the edited image
        self.update_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    editor = ImageEditor(root)
    root.mainloop()

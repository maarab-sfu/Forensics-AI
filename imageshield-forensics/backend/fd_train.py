import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import vgg19
from torch.utils.data import DataLoader, random_split
import config as c
import numpy as np
import my_datasets
import os
import random
from torchvision import models
from torchvision.utils import save_image
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score  # Correct library
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class TamperingLocalizationNet(nn.Module):
    def __init__(self):
        super(TamperingLocalizationNet, self).__init__()

        # Shared backbone (e.g., ResNet)
        backbone = models.resnet34(pretrained=True)
        self.shared_features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
        # Separate encoders for tampered and low-quality images
        self.tampered_encoder = self._make_encoder()
        self.low_quality_encoder = self._make_encoder()

        # Feature alignment (e.g., cross-attention)
        self.alignment = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Difference module (calculates feature difference)
        self.difference_module = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Decoder to generate the tampering mask
        self.decoder = self._make_decoder()

    def _make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def _make_decoder(self):
        return nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # Upsample to 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Upsample to 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # Upsample to 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # Upsample to 256x256
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),    # Upsample to 512x512
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),                        # Final 1-channel output
            nn.Sigmoid()
        )

    def forward(self, tampered_img, low_quality_img):
        # Shared feature extraction
        shared_features_tampered = self.shared_features(tampered_img)
        shared_features_low_quality = self.shared_features(low_quality_img)

        # Separate encoders
        encoded_tampered = self.tampered_encoder(shared_features_tampered)
        encoded_low_quality = self.low_quality_encoder(shared_features_low_quality)

        # Feature alignment
        aligned_low_quality = self.alignment(encoded_low_quality)

        # Calculate difference
        feature_difference = torch.abs(encoded_tampered - aligned_low_quality)
        difference_features = self.difference_module(feature_difference)

        # Decode the tampering mask
        mask = self.decoder(difference_features)

        return mask

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv3_6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3_12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv1x1_output = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3_1(x)
        x3 = self.conv3_6(x)
        x4 = self.conv3_12(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv1x1_output(out)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape
        weights = self.global_avg_pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights

class TamperingLocalizationNet2(nn.Module):
    def __init__(self):
        super(TamperingLocalizationNet2, self).__init__()
        backbone = models.resnet50(pretrained=True)
        self.shared_features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
        self.tampered_encoder = ASPP(2048, 512)
        self.low_quality_encoder = ASPP(2048, 512)
        self.alignment = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.difference_module = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(512)
        )
        
        self.decoder = self._make_decoder()
    
    def _make_decoder(self):
        return nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # Upsample to 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Upsample to 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # Upsample to 256x256
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # Upsample to 512x512
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Final 1-channel output
            nn.Sigmoid()
        )

    
    def forward(self, tampered_img, low_quality_img):
        shared_features_tampered = self.shared_features(tampered_img)
        shared_features_low_quality = self.shared_features(low_quality_img)
        
        encoded_tampered = self.tampered_encoder(shared_features_tampered)
        encoded_low_quality = self.low_quality_encoder(shared_features_low_quality)
        
        aligned_low_quality = self.alignment(encoded_low_quality.flatten(2).permute(2, 0, 1),
                                             encoded_tampered.flatten(2).permute(2, 0, 1),
                                             encoded_tampered.flatten(2).permute(2, 0, 1))[0]
        aligned_low_quality = aligned_low_quality.permute(1, 2, 0).view_as(encoded_tampered)
        
        feature_difference = torch.abs(encoded_tampered - aligned_low_quality)
        difference_features = self.difference_module(feature_difference)
        
        mask = self.decoder(difference_features)
        return mask

class VGGPerceptualLoss(nn.Module):
    def __init__(self, vgg):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = vgg
        self.vgg.eval()  # Set to evaluation mode
        
        # Disable gradients for VGG layers
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        # Layers to extract features from
        self.feature_layers = ['0', '5', '10', '19', '28']  # corresponding to conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
        
    def forward(self, x, y):
        x_features = self.extract_features(x)
        y_features = self.extract_features(y)
        loss = 0
        for x_f, y_f in zip(x_features, y_features):
            loss += F.mse_loss(x_f, y_f)
        return loss
    
    def extract_features(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.feature_layers:
                features.append(x)
        return features

def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def main():

    if not os.path.exists('./localize_images/'):
        os.makedirs('./localize_images/')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vgg = vgg19(pretrained=True).features.to(device)
    perceptual_loss_fn = VGGPerceptualLoss(vgg)

    # Initialize the model
    model = TamperingLocalizationNet2()
    model = model.to(device)

    # model = UNet()
    # model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Optionally load a model to continue training
    start_epoch = 0
    best_PSNR = 0.0
    if os.path.exists("best_localize_model.pth"):
        checkpoint = torch.load("best_localize_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch'] + 1
        start_epoch = 10
        # best_PSNR = checkpoint['best_PSNR']
        best_PSNR = 14
        print(f"Loaded model from epoch {start_epoch} with PSNR Score: {best_PSNR:.4f}")

    # Divide dataset into 80% training and 20% validation
    dataset_size = len(my_datasets.localizeloader.dataset)
    # Define the number of images you want to sample per epoch
    num_samples_per_epoch = 100

    indices = list(range(dataset_size))
    split = int(0.2 * dataset_size)
    random.shuffle(indices)

    val_indices = indices[:split]
    train_pool_indices = indices[split:]  # Indices available for training (excluding validation set)

    # Define a function to create a custom sampler ensuring no overlap with the validation set
    def custom_sampler(train_pool_indices, num_samples):
        # Get a random subset of indices from the training pool (excluding validation set)
        indices = random.sample(train_pool_indices, num_samples)
        return torch.utils.data.SubsetRandomSampler(indices)

    # Create the samplers using the custom function
    train_sampler = custom_sampler(train_pool_indices, num_samples_per_epoch)
    # train_sampler = torch.utils.data.SubsetRandomSampler(train_pool_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    # val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(my_datasets.localizeloader.dataset, sampler=train_sampler, batch_size=4)
    val_loader = torch.utils.data.DataLoader(my_datasets.localizeloader.dataset, sampler=val_sampler, batch_size=4)

    threshold = 0.00000

    # Training loop
    num_epochs = 200
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for i_batch, (noised, restored, mask) in enumerate(tqdm(train_loader)):
            noised = noised.to(device)
            restored = restored.to(device)
            mask = mask.to(device)

            mask = F.interpolate(mask, size=(256, 256), mode="bilinear", align_corners=False)

            # Forward pass
            output = model(noised, restored)

            output_binary = (output > 0.5).byte()
            mask_binary = (mask > 0.5).byte()

            output = torch.sigmoid(output)
            bce_loss = F.binary_cross_entropy(output.squeeze(), mask.squeeze())
            perceptual_loss = perceptual_loss_fn(output.repeat(1, 3, 1, 1), mask.repeat(1, 3, 1, 1))
            d_loss = dice_loss(output_binary, mask_binary)

            total_loss = bce_loss + perceptual_loss + d_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        f1_values = []
        # Validation
        if (epoch + 1) % 5 == 0 or (epoch == 0):
            model.eval()
            with torch.no_grad():
                psnr_values = []
                ssim_values = []
                for i_batch, (noised, restored, mask) in enumerate(tqdm(val_loader)):
                    noised = noised.to(device)
                    restored = restored.to(device)
                    mask = mask.to(device)

                    mask = F.interpolate(mask, size=(256, 256), mode="bilinear", align_corners=False)

                    # Forward pass
                    output = model(noised, restored)

                    # Convert tensors to numpy arrays
                    output_np = output.cpu().numpy().transpose(0, 2, 3, 1).squeeze()  # Convert from NCHW to NHWC
                    mask_np = mask.cpu().numpy().transpose(0, 2, 3, 1).squeeze()  # Convert from NCHW to NHWC

                    # Apply thresholding to output and convert to binary
                    output_binary_np = (output_np > threshold).astype(np.uint8)  # Use threshold directly

                    # Calculate PSNR and SSIM for each image in the batch
                    for j in range(output_np.shape[0]):  # Loop over the batch size
                        psnr_value = psnr_metric(mask_np[j], output_np[j])

                        
                        mask_flat = mask_np[j].flatten()
                        output_flat = output_binary_np[j].flatten()

                        # print("np.unique(mask_flat).size: ", np.unique(mask_flat).size)
                        # print("np.unique(output_flat).size: ", np.unique(output_flat).size)

                        

                        # if np.unique(mask_flat).size == 2 and np.unique(output_flat).size == 2:
                        #     print("hello!")
                        output_binary = (output_flat > 0.5).astype(int)
                        mask_binary = (mask_flat > 0.5).astype(int)

                        # print("mask_flat.type: ", type(mask_binary))
                        # print("output_binary.type: ", type(output_binary))

                        f1 = f1_score(mask_binary, output_binary, average='binary')
                        f1_values.append(f1)
                        # else:
                            # f1_values.append(np.nan)

                        # Handle infinite PSNR (due to identical images) by setting a maximum value, e.g., 100
                        if np.isinf(psnr_value):
                            psnr_value = 100.0  # You can choose a reasonable high value to cap the PSNR

                        ssim_value = ssim_metric(mask_np[j], output_np[j])
                        psnr_values.append(psnr_value)
                        ssim_values.append(ssim_value)

                    if i_batch<5:
                        save_image(output, f"./localize_images/{epoch+1}_img_{i_batch}_output.png")
                        save_image(mask, f"./localize_images/{epoch+1}_img_{i_batch}_mask.png")

                avg_f1 = np.nanmean(f1_values)
                avg_psnr = sum(psnr_values) / len(psnr_values)
                avg_ssim = sum(ssim_values) / len(ssim_values)

                print(f"PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, F1 Score: {avg_f1:.4f}")

                

                # Save the best model based on PSNR
                if avg_psnr > best_PSNR:
                    best_PSNR = avg_psnr
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_PSNR': best_PSNR,
                    }, "best_localize_model.pth")
                    print(f"Saved model at epoch {epoch+1} with PSNR Score: {best_PSNR:.4f}")


if __name__ == '__main__':
    main()
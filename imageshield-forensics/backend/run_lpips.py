import os
import torch
import lpips
from PIL import Image
from torchvision import transforms

# Initialize LPIPS model
loss_fn_alex = lpips.LPIPS(net='alex')

# Function to calculate average LPIPS for two folders
def calculate_average_lpips(folder1, folder2):
    # List all PNG files in the first folder
    file_names = [f for f in os.listdir(folder1) if f.endswith('.png')]
    # Limit to only the first 1000 files (or fewer if there are less than 100)
    file_names = file_names[:1000]

    # Initialize the list to store LPIPS values
    lpips_arr = []
    # Loop over each image file
    for file_name in file_names:
        # Load both images from folder1 and folder2
        img1 = Image.open(os.path.join(folder1, file_name))
        img2 = Image.open(os.path.join(folder2, file_name))

        # Transform images to tensors
        img1_tensor = transforms.ToTensor()(img1).unsqueeze(0)  # Add batch dimension
        img2_tensor = transforms.ToTensor()(img2).unsqueeze(0)  # Add batch dimension

        # Calculate LPIPS
        lpips_temp = loss_fn_alex(img1_tensor.to("cpu"), img2_tensor.to("cpu"))
        lpips_arr.append(lpips_temp)

    # Stack the LPIPS results and calculate the average
    stacked_tensors = torch.stack(lpips_arr)
    lpips_ave = torch.mean(stacked_tensors, dim=0)

    return lpips_ave.item()

# Main function to loop over subfolders and calculate LPIPS
def main():
    # folder_path = './results/DIV2K/Inpainting/'  # Splicing, CopyMove, Inpainting
    folder_path = './results_diff_size/'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # Loop through each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        restored_folder = os.path.join(subfolder_path, "restored")
        cover_folder = os.path.join(subfolder_path, "cover")


        print(f'{subfolder}:')
        # # Calculate LPIPS between restored and cover
        lpips_value = calculate_average_lpips(restored_folder, cover_folder)
        print(f'restored vs. cover: {lpips_value:.6f}\n')

        # Uncomment this section if you need to compare more folders
        final_nr_folder = os.path.join(subfolder_path, "final_nr")
        final_nn_folder = os.path.join(subfolder_path, "final_nn")
        final_cr_folder = os.path.join(subfolder_path, "final_cr")
        distorted_folder = os.path.join(subfolder_path, "distorted")

        lpips_value = calculate_average_lpips(distorted_folder, final_nr_folder)
        print(f'distorted vs. final_nr: {lpips_value:.6f}')

        lpips_value = calculate_average_lpips(distorted_folder, final_nn_folder)
        print(f'distorted vs. final_nn: {lpips_value:.6f}')

        lpips_value = calculate_average_lpips(distorted_folder, final_cr_folder)
        print(f'distorted vs. final_cr: {lpips_value:.6f}')

        lpips_value = calculate_average_lpips(cover_folder, final_nr_folder)
        print(f'cover vs. final_nr: {lpips_value:.6f}')

        lpips_value = calculate_average_lpips(cover_folder, final_nn_folder)
        print(f'cover vs. final_nn: {lpips_value:.6f}')

        lpips_value = calculate_average_lpips(cover_folder, final_cr_folder)
        print(f'cover vs. final_cr: {lpips_value:.6f}')

if __name__ == "__main__":
    main()

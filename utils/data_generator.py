import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    """
    PyTorch-compatible Dataset for BraTS 2020 preprocessed .npy data.

    Each image: (128, 128, 128, 3) → 3 MRI modalities (T2, T1ce, FLAIR)
    Each mask:  (128, 128, 128, 4) → One-hot encoded, 4 tumor classes

    Returns:
        image tensor: shape (3, 128, 128, 128)  - channels first
        label tensor: shape (128, 128, 128)     - class index per voxel
    """
    def __init__(self, img_dir, mask_dir, file_list):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        # Handle the naming difference: images are "image_XX.npy", masks are "mask_XX.npy"
        self.valid_files = []
        self.mask_files = {}  # Dictionary to map from image filename to mask filename
        
        for file_name in file_list:
            img_path = os.path.join(img_dir, file_name)
            
            # Convert "image_XX.npy" to "mask_XX.npy"
            if file_name.startswith("image_"):
                mask_file = file_name.replace("image_", "mask_")
            else:
                # For other naming patterns, extract number and create mask filename
                # Assuming format like "XXX_61.npy"
                parts = file_name.split("_")
                if len(parts) >= 2 and parts[-1].endswith(".npy"):
                    num = parts[-1]  # This would be "61.npy"
                    mask_file = f"mask_{num}"
                else:
                    print(f"Warning: Couldn't determine mask filename for {file_name}")
                    continue
                    
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.valid_files.append(file_name)
                self.mask_files[file_name] = mask_file
            else:
                # If files are missing, print detailed information
                if not os.path.exists(img_path):
                    print(f"Missing image file: {img_path}")
                if not os.path.exists(mask_path):
                    print(f"Missing mask file: {mask_path}")
        
        # Report on file validation
        print(f"Found {len(self.valid_files)} valid image-mask pairs out of {len(file_list)} files")
        if len(self.valid_files) == 0:
            raise ValueError("No valid image-mask pairs found! Check your directory paths and file naming.")
        
        self.file_list = self.valid_files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_file = self.file_list[idx]
        mask_file = self.mask_files[img_file]
        
        # Load image and mask from .npy files
        image = np.load(os.path.join(self.img_dir, img_file))
        mask = np.load(os.path.join(self.mask_dir, mask_file))

        # Convert image to float32 and rearrange axes to (C, D, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(3, 0, 1, 2)

        # Convert mask to long type: (D, H, W), class indices via argmax
        mask = torch.tensor(np.argmax(mask, axis=-1), dtype=torch.long)

        return image, mask
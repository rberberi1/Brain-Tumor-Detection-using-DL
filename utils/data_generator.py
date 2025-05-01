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
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load image and mask from .npy files
        image = np.load(os.path.join(self.img_dir, self.file_list[idx]))
        mask = np.load(os.path.join(self.mask_dir, self.file_list[idx]))

        # Convert image to float32 and rearrange axes to (C, D, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(3, 0, 1, 2)

        # Convert mask to long type: (D, H, W), class indices via argmax
        mask = torch.tensor(np.argmax(mask, axis=-1), dtype=torch.long)

        return image, mask

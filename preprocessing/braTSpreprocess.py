import os
import numpy as np
import nibabel as nib
import glob
from sklearn.preprocessing import MinMaxScaler
import splitfolders
import matplotlib.pyplot as plt
import random

# --------- CONFIG ---------
DATASET_PATH = 'BrainTS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
SAVE_PATH = 'BrainTS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/input_data_3channels/'
CROP_SHAPE = (128, 128, 128)
USE_MODALITIES = ['flair', 't1ce', 't2']
THRESHOLD_PERCENT = 0.01  # 1%
os.makedirs(SAVE_PATH + 'images/', exist_ok=True)
os.makedirs(SAVE_PATH + 'masks/', exist_ok=True)

scaler = MinMaxScaler()

# --------- NumPy version of one-hot encoding ---------
def one_hot_encode(mask, num_classes=4):
    shape = mask.shape
    one_hot = np.zeros(shape + (num_classes,), dtype=np.uint8)
    for c in range(num_classes):
        one_hot[..., c] = (mask == c).astype(np.uint8)
    return one_hot

# --------- GLOB ALL MODALITY FILES ---------
t2_list = sorted(glob.glob(os.path.join(DATASET_PATH, '*/*t2.nii')))
t1ce_list = sorted(glob.glob(os.path.join(DATASET_PATH, '*/*t1ce.nii')))
flair_list = sorted(glob.glob(os.path.join(DATASET_PATH, '*/*flair.nii')))
mask_list = sorted(glob.glob(os.path.join(DATASET_PATH, '*/*seg.nii')))

# --------- MAIN LOOP ---------
for idx in range(len(t2_list)):
    print(f"[INFO] Processing sample {idx+1}/{len(t2_list)}")
    
    def load_and_scale(path):
        img = nib.load(path).get_fdata()
        return scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

    t2 = load_and_scale(t2_list[idx])
    t1ce = load_and_scale(t1ce_list[idx])
    flair = load_and_scale(flair_list[idx])

    mask = nib.load(mask_list[idx]).get_fdata().astype(np.uint8)
    mask[mask == 4] = 3  # Reassign label 4 → 3

    image = np.stack([flair, t1ce, t2], axis=3)
    image = image[56:184, 56:184, 13:141]
    mask = mask[56:184, 56:184, 13:141]

    val, counts = np.unique(mask, return_counts=True)
    if (1 - (counts[0]/counts.sum())) > THRESHOLD_PERCENT:
        print(f"✅ Saved: {1 - (counts[0]/counts.sum()):.2%} labeled")
        mask_encoded = one_hot_encode(mask, num_classes=4)
        np.save(f"{SAVE_PATH}images/image_{idx}.npy", image)
        np.save(f"{SAVE_PATH}masks/mask_{idx}.npy", mask_encoded)
    else:
        print("⛔ Skipped: Not enough labeled voxels")

# --------- SPLIT TRAIN/VAL ---------
print("[INFO] Splitting dataset into train/val folders...")
splitfolders.ratio(SAVE_PATH, output="BraTS2020_TrainingData/input_data_128/",
                   seed=42, ratio=(0.75, 0.25), group_prefix=None)

print("[DONE] Preprocessing complete without TensorFlow.")

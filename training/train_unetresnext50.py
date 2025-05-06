import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_generator import BraTSDataset
from models.unet_resnext50 import UNetResNeXt50  # Your model file
from metrics.metrics import dice_score, calculate_metrics  # Already provided by user

# ================= Training Script =================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # === Paths ===
    train_img_dir = 'BraTS2020_TrainingData/input_data_128/train/images/'
    train_mask_dir = 'BraTS2020_TrainingData/input_data_128/train/masks/'
    val_img_dir = 'BraTS2020_TrainingData/input_data_128/val/images/'
    val_mask_dir = 'BraTS2020_TrainingData/input_data_128/val/masks/'

    train_files = sorted(os.listdir(train_img_dir))
    val_files = sorted(os.listdir(val_img_dir))

    # === Datasets & Loaders ===
    train_dataset = BraTSDataset(train_img_dir, train_mask_dir, train_files)
    val_dataset = BraTSDataset(val_img_dir, val_mask_dir, val_files)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    # === Model ===
    model = UNetResNeXt50(in_channels=3, out_channels=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 20
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        model.train()
        train_loss, train_dice = 0.0, 0.0

        for images, masks in tqdm(train_loader, desc="Training"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_score(outputs, masks).item()

        train_losses.append(train_loss / len(train_loader))
        train_dices.append(train_dice / len(train_loader))

        # === Validation ===
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks).item()
                all_preds.append(outputs)
                all_targets.append(masks)

        val_losses.append(val_loss / len(val_loader))
        val_dices.append(val_dice / len(val_loader))

        print(f"[Train] Loss: {train_losses[-1]:.4f} | Dice: {train_dices[-1]:.4f}")
        print(f"[Val]   Loss: {val_losses[-1]:.4f} | Dice: {val_dices[-1]:.4f}")

    # === Save model ===
    torch.save(model.state_dict(), "brats3d_unetresnext50.pth")
    print("[INFO] Model saved.")

    # === Calculate metrics on final validation set ===
    model.eval()
    final_preds, final_targets = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            final_preds.append(outputs)
            final_targets.append(masks)

    y_pred = torch.cat(final_preds, dim=0)
    y_true = torch.cat(final_targets, dim=0)

    print("\n📊 Final Validation Metrics:")
    metrics = calculate_metrics(y_pred, y_true)
    for k, v in metrics.items():
        print(f"{k.title()}: {v:.4f}")

    # === Plot combined Loss & Dice ===
    epochs = range(1, EPOCHS+1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.plot(epochs, train_dices, label='Train Dice')
    plt.plot(epochs, val_dices, label='Val Dice')
    plt.title("Loss and Dice Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("combined_loss_dice_plot.png")
    plt.show()

if __name__ == "__main__":
    train()

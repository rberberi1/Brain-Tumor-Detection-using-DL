import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_generator import BraTSDataset
from models.unet_3d import UNet3D
from metrics.metrics import calculate_metrics

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # === Directories ===
    train_img_dir = 'BraTS2020_TrainingData/input_data_128/train/images/'
    train_mask_dir = 'BraTS2020_TrainingData/input_data_128/train/masks/'
    val_img_dir = 'BraTS2020_TrainingData/input_data_128/val/images/'
    val_mask_dir = 'BraTS2020_TrainingData/input_data_128/val/masks/'

    train_files = sorted(os.listdir(train_img_dir))
    val_files = sorted(os.listdir(val_img_dir))

    # === Dataset & DataLoaders ===
    train_dataset = BraTSDataset(train_img_dir, train_mask_dir, train_files)
    val_dataset = BraTSDataset(val_img_dir, val_mask_dir, val_files)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

    # === Model, Loss, Optimizer ===
    model = UNet3D(in_channels=3, out_channels=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 20  # Reduce for speed
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_dices, val_dices = [], []

    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        model.train()
        running_loss = 0
        dice_total = 0
        acc_total = 0

        for images, masks in tqdm(train_loader, desc="Training"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            metrics = calculate_metrics(outputs, masks)
            acc_total += metrics["accuracy"]
            dice_total += metrics["dice"]

        avg_loss = running_loss / len(train_loader)
        avg_dice = dice_total / len(train_loader)
        avg_acc = acc_total / len(train_loader)

        train_losses.append(avg_loss)
        train_dices.append(avg_dice)
        train_accs.append(avg_acc)
        print(f"[Train] Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | Acc: {avg_acc:.4f}")

        # === Validation ===
        model.eval()
        val_loss, val_dice, val_acc = 0, 0, 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                metrics = calculate_metrics(outputs, masks)
                val_acc += metrics["accuracy"]
                val_dice += metrics["dice"]

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)
        val_accs.append(avg_val_acc)
        print(f"[Val]   Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} | Acc: {avg_val_acc:.4f}")

    # === Save Model ===
    torch.save(model.state_dict(), "brats3d_unet.pth")
    print("[INFO] Model saved as brats3d_unet.pth")

    # === Combined Plot ===
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.plot(train_dices, label='Train Dice')
    plt.plot(val_dices, label='Val Dice')
    plt.title("Loss, Accuracy, and Dice Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("combined_plot.png")
    plt.show()

if __name__ == "__main__":
    train()

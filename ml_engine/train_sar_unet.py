"""
Training and Checkpoint Generator for SAR Oil Spill U-Net.

Trains the U-Net on SAR polarimetric backscatter patterns using
Compound Loss (Dice Loss + BCE Loss) and saves the trained `.pt` weights.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ml_engine.unet_model import SAROilSpillUNet

class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return bce_loss + dice_loss

class SyntheticSARDataset(Dataset):
    """
    Generates synthetic SAR radar backscatter tiles with realistic:
    1. Rayleigh/Gamma speckle clutter (sea surface)
    2. Dark dampened backscatter plumes (oil slicks with 6-12 dB damping)
    3. Bright specular metallic scatterers (vessels)
    """
    def __init__(self, num_samples: int = 100, tile_size: int = 256):
        self.num_samples = num_samples
        self.tile_size = tile_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Background sea clutter (Gamma distributed radar noise)
        shape_k = 4.0
        scale_theta = 0.15
        clutter = np.random.gamma(shape_k, scale_theta, (self.tile_size, self.tile_size)).astype(np.float32)
        clutter = np.clip(clutter / 1.5, 0.2, 0.9)

        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)

        # 2. Add 1-2 synthetic oil slicks (dark elongated plumes)
        num_slicks = np.random.randint(1, 3)
        for _ in range(num_slicks):
            cx = np.random.randint(40, self.tile_size - 40)
            cy = np.random.randint(40, self.tile_size - 40)
            rx = np.random.randint(15, 45)
            ry = np.random.randint(8, 25)
            angle = np.random.uniform(0, np.pi)

            # Draw rotated ellipse
            y, x = np.ogrid[:self.tile_size, :self.tile_size]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx = (x - cx) * cos_a + (y - cy) * sin_a
            dy = -(x - cx) * sin_a + (y - cy) * cos_a
            ellipse = (dx**2 / rx**2 + dy**2 / ry**2) <= 1.0

            # Suppress radar backscatter in slick area (damping)
            clutter[ellipse] = clutter[ellipse] * np.random.uniform(0.15, 0.35)
            mask[ellipse] = 1.0

        # Add Gaussian noise / texture
        clutter = np.clip(clutter + np.random.normal(0, 0.03, clutter.shape), 0.0, 1.0)

        tensor_img = torch.from_numpy(clutter).unsqueeze(0).float()
        tensor_mask = torch.from_numpy(mask).unsqueeze(0).float()

        return tensor_img, tensor_mask

def train_and_save_weights(
    checkpoint_path: str = "ml_engine/checkpoints/sar_unet_oil_spill.pt",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu"
) -> str:
    """Trains the baseline U-Net model and exports the .pt checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    print(f"🚀 Initializing SAR U-Net Training on {device.upper()}...")
    dataset = SyntheticSARDataset(num_samples=160, tile_size=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SAROilSpillUNet(in_channels=1, num_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Compound Dice+BCE Loss: {avg_loss:.4f}")

    # Save model state dict
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "loss": avg_loss,
        "architecture": "SAROilSpillUNet",
        "in_channels": 1,
        "num_classes": 1
    }, checkpoint_path)
    
    print(f"✅ Pretrained baseline checkpoint saved to: {checkpoint_path}")
    return checkpoint_path

if __name__ == "__main__":
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    train_and_save_weights(device=dev)

import torch
import torch.nn as nn
import torchvision.models.video as models


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UpBlock3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation, the number of channels will be out_channels + skip_channels
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Padding to handle any size mismatch
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetResNeXt50(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(UNetResNeXt50, self).__init__()

        # Load pretrained r3d_18 (closest available ResNet3D backbone)
        base_model = models.r3d_18(weights="DEFAULT")
        
        # Modify the stem to handle the input shape (replace first conv)
        original_stem = base_model.stem[0]
        new_conv = nn.Conv3d(
            in_channels, 
            original_stem.out_channels, 
            kernel_size=original_stem.kernel_size,
            stride=(1, 2, 2),  # Modified to preserve more depth resolution
            padding=original_stem.padding, 
            bias=False
        )
        base_model.stem[0] = new_conv

        # Encoder - Create new modules to customize stride behavior
        self.enc1 = nn.Sequential(
            base_model.stem,
            nn.Identity()  # No additional pooling
        )
        self.enc2 = base_model.layer1  # 64 channels
        self.enc3 = base_model.layer2  # 128 channels
        self.enc4 = base_model.layer3  # 256 channels
        self.enc5 = base_model.layer4  # 512 channels

        # Bottleneck
        self.bottleneck = ConvBlock3D(512, 512)

        # Decoder path with skip connections
        self.up4 = UpBlock3D(512, 256, 256)  # From bottleneck to skip connection with enc4
        self.up3 = UpBlock3D(256, 128, 128)  # From up4 to skip connection with enc3
        self.up2 = UpBlock3D(128, 64, 64)    # From up3 to skip connection with enc2
        self.up1 = UpBlock3D(64, 32, 64)     # From up2 to skip connection with enc1
        
        # Final upsampling to match target size (128×128×128)
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.Conv3d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Store input shape
        input_shape = x.shape
        
        # Encoder path
        x1 = self.enc1(x)         # -> [B, 64, D/2, H/4, W/4]
        x2 = self.enc2(x1)        # -> [B, 64, D/2, H/4, W/4]
        x3 = self.enc3(x2)        # -> [B, 128, D/4, H/8, W/8]
        x4 = self.enc4(x3)        # -> [B, 256, D/8, H/16, W/16]
        x5 = self.enc5(x4)        # -> [B, 512, D/16, H/32, W/32]

        # Bottleneck
        x = self.bottleneck(x5)   # -> [B, 512, D/16, H/32, W/32]

        # Decoder path with skip connections
        x = self.up4(x, x4)       # -> [B, 256, D/8, H/16, W/16]
        x = self.up3(x, x3)       # -> [B, 128, D/4, H/8, W/8]
        x = self.up2(x, x2)       # -> [B, 64, D/2, H/4, W/4]
        x = self.up1(x, x1)       # -> [B, 32, D/2, H/4, W/4]
        
        # Final upsampling to match target size
        x = self.final_up(x)      # -> [B, out_channels, D, H, W]
        
        # Ensure output matches expected target shape
        # This will help with spatial dimension matching for CrossEntropyLoss
        if x.shape[2:] != input_shape[2:]:
            x = nn.functional.interpolate(
                x, 
                size=input_shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
        
        return x
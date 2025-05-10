import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, features=[32, 64, 128]):
        super(UNet3D, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = self.conv_block(features[0], features[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.conv_block(features[1], features[2])

        # Decoder
        self.upconv2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(features[1]*2, features[1])

        self.upconv1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(features[0]*2, features[0])

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)

        enc2 = self.encoder2(x)
        x = self.pool2(enc2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.upconv2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.decoder2(x)

        x = self.upconv1(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.decoder1(x)

        return self.final_conv(x)

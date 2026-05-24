from network.encoder.pvtv2 import pvt_v2_b0
import torch
import torch.nn as nn

class ResUNetV3(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ResUNetV3, self).__init__()
        self.encoder = pvt_v2_b0(pretrained=pretrained)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
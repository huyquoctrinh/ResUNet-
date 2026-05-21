import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, encoder_channels, num_classes, deep_supervision=False):
        super().__init__()
        ch = encoder_channels  # e.g. [64, 128, 320, 512]
        self.deep_supervision = deep_supervision
        self.dec4 = DecoderBlock(ch[3], ch[2], ch[2])
        self.dec3 = DecoderBlock(ch[2], ch[1], ch[1])
        self.dec2 = DecoderBlock(ch[1], ch[0], ch[0])
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[0], kernel_size=2, stride=2),
            nn.Conv2d(ch[0], ch[0], 3, padding=1),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch[0], ch[0] // 2, kernel_size=2, stride=2),
        )
        self.seg_head = nn.Conv2d(ch[0] // 2, num_classes, 1)

        if deep_supervision:
            self.aux_head = nn.Conv2d(ch[2], num_classes, 1)

    def forward(self, mhsa4_out, mhsa3_out, enc2, enc1):
        d4 = self.dec4(mhsa4_out, mhsa3_out)
        d3 = self.dec3(d4, enc2)
        d2 = self.dec2(d3, enc1)
        d1 = self.dec1(d2)
        logits = self.seg_head(d1)
        if self.deep_supervision and self.training:
            aux_logits = self.aux_head(d4)
            return logits, aux_logits
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeNet(nn.Module):
    """Lightweight learned edge branch producing features at two encoder scales."""

    def __init__(self, in_ch=3, mid_ch=64, enc3_ch=320, enc4_ch=512,
                 enc3_stride=16, enc4_stride=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )  # output: H/4

        self.proj3 = nn.Sequential(
            nn.Conv2d(mid_ch, enc3_ch, 3, stride=enc3_stride // 4, padding=1),
            nn.BatchNorm2d(enc3_ch),
        )
        self.proj4 = nn.Sequential(
            nn.Conv2d(mid_ch, enc4_ch, 3, stride=enc4_stride // 4, padding=1),
            nn.BatchNorm2d(enc4_ch),
        )

    def forward(self, x):
        e = self.stem(x)
        e3 = self.proj3(e)
        e4 = self.proj4(e)
        return e3, e4


class SobelEdgeDetector(nn.Module):
    """Fixed Sobel gradient-magnitude edge detector with learned projection."""

    def __init__(self, enc3_ch=320, enc4_ch=512):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        self.proj3 = nn.Sequential(
            nn.Conv2d(1, enc3_ch, 3, stride=16, padding=1),
            nn.BatchNorm2d(enc3_ch),
        )
        self.proj4 = nn.Sequential(
            nn.Conv2d(1, enc4_ch, 3, stride=32, padding=1),
            nn.BatchNorm2d(enc4_ch),
        )

    def _to_gray(self, x):
        if x.shape[1] == 1:
            return x
        return x.mean(dim=1, keepdim=True)

    def forward(self, x):
        gray = self._to_gray(x)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return self.proj3(edge), self.proj4(edge)


class CannyEdgeDetector(nn.Module):
    """Canny-style edge detector (Sobel + NMS approximation) with learned projection.

    Uses a simplified non-maximum suppression via max-pool to approximate Canny.
    Not fully differentiable through NMS but the projection layers are learned.
    """

    def __init__(self, enc3_ch=320, enc4_ch=512, low_thresh=0.1, high_thresh=0.3):
        super().__init__()
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        self.proj3 = nn.Sequential(
            nn.Conv2d(1, enc3_ch, 3, stride=16, padding=1),
            nn.BatchNorm2d(enc3_ch),
        )
        self.proj4 = nn.Sequential(
            nn.Conv2d(1, enc4_ch, 3, stride=32, padding=1),
            nn.BatchNorm2d(enc4_ch),
        )

    def _to_gray(self, x):
        if x.shape[1] == 1:
            return x
        return x.mean(dim=1, keepdim=True)

    def forward(self, x):
        gray = self._to_gray(x)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        mag_max = mag.max()
        if mag_max > 0:
            mag = mag / mag_max

        # Simplified NMS: keep pixels that are local maxima in a 3x3 window
        local_max = F.max_pool2d(mag, 3, stride=1, padding=1)
        nms = (mag == local_max).float() * mag

        # Double threshold
        strong = (nms >= self.high_thresh).float()
        weak = ((nms >= self.low_thresh) & (nms < self.high_thresh)).float()
        # Connect weak edges adjacent to strong edges
        strong_dilated = F.max_pool2d(strong, 3, stride=1, padding=1)
        edge = strong + weak * strong_dilated
        edge = edge.clamp(0, 1)

        return self.proj3(edge), self.proj4(edge)


class ConcatFusion(nn.Module):
    """Concatenate edge and encoder features, then reduce with 1x1 conv."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, enc_feat, edge_feat):
        return self.conv(torch.cat([enc_feat, edge_feat], dim=1))

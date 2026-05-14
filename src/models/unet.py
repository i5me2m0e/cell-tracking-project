from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv2d + ReLU layers used in U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Lightweight 2D U-Net for binary cell segmentation.

    Input shape: [B, 1, H, W]
    Output shape: [B, 1, H, W]

    The output is logits. Apply sigmoid outside the model.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.up3(bottleneck)
        dec3 = self._match_size(dec3, enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = self._match_size(dec2, enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = self._match_size(dec1, enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)

    @staticmethod
    def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Pad or crop x so it has the same H,W as reference."""
        target_h, target_w = reference.shape[-2:]
        h, w = x.shape[-2:]

        diff_h = target_h - h
        diff_w = target_w - w

        if diff_h > 0 or diff_w > 0:
            pad_left = max(diff_w // 2, 0)
            pad_right = max(diff_w - pad_left, 0)
            pad_top = max(diff_h // 2, 0)
            pad_bottom = max(diff_h - pad_top, 0)
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        h, w = x.shape[-2:]

        if h > target_h:
            start_h = (h - target_h) // 2
            x = x[..., start_h : start_h + target_h, :]

        if w > target_w:
            start_w = (w - target_w) // 2
            x = x[..., :, start_w : start_w + target_w]

        return x


def main() -> None:
    model = UNet(in_channels=1, out_channels=1, base_channels=32)

    x1 = torch.randn(2, 1, 256, 256)
    y1 = model(x1)
    print("test 1 input:", tuple(x1.shape))
    print("test 1 output:", tuple(y1.shape))

    x2 = torch.randn(1, 1, 443, 512)
    y2 = model(x2)
    print("test 2 input:", tuple(x2.shape))
    print("test 2 output:", tuple(y2.shape))

    assert y1.shape == x1.shape
    assert y2.shape == x2.shape

    print("UNet shape test passed.")


if __name__ == "__main__":
    main()
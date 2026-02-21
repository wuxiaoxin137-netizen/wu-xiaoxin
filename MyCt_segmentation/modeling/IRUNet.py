
import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidualsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(InvertedResidualsBlock, self).__init__()
        channels = expansion * in_channels
        self.stride = stride

        self.basic_block = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # The shortcut operation does not affect the number of channels
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    # def forward(self, x):
    #     out = self.basic_block(x)
    #     if self.stride == 1:
    #         # print("With shortcut!")
    #         out = out + self.shortcut(x)
    #     else:
    #         print("No shortcut!")
    #     print(out.size())
    #
    #     return out

    def forward(self, x):
        out = self.basic_block(x)
        out = out + self.shortcut(x)
        return out

# if __name__ == "__main__":
#     x = torch.randn(16, 3, 32, 32)
#     # no shortcut
#     net1 = InvertedResidualsBlock(3, 6, 6, 2)
#     # with shortcut
#     net2 = InvertedResidualsBlock(3, 6, 6, 1)
#     y1, y2 = net1(x), net2(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (InvertedResidualsBlock(64, 128, expansion=4, stride=1))
        self.down2 = (InvertedResidualsBlock(128, 256, expansion=4, stride=1))
        self.down3 = (InvertedResidualsBlock(256, 512, expansion=4, stride=1))
        factor = 2 if bilinear else 1
        self.down4 = (InvertedResidualsBlock(512, 1024, expansion=4, stride=1))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        print(f":x1 {x1.size()}")

        x2 = self.down1(x1)
        print(f":x2 {x2.size()}")

        x3 = self.down2(x2)
        print(f":x3 {x3.size()}")

        x4 = self.down3(x3)
        print(f":x4 {x4.size()}")

        x5 = self.down4(x4)
        print(f":x5 {x5.size()}")

        x = self.up1(x5, x4)
        print(f":x {x.size()}")

        x = self.up2(x, x3)
        print(f":x {x.size()}")

        x = self.up3(x, x2)
        print(f":x {x.size()}")
        x = self.up4(x, x1)
        print(f":x {x.size()}")
        logits = self.outc(x)
        print(f":x {logits.size()}")

        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

model = UNet(n_channels=3, n_classes=3)  # Example: 3 input channels, 2 output classes
input_tensor = torch.randn(1, 3, 512, 512)  # Example input
output = model(input_tensor)
# print(output.shape)  # Output shape should be (1, 2, 256, 256)


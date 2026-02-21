import torch
import torch.nn as nn
import torch.nn.functional as F


# from tensorboardX import SummaryWriter


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


# ==============================================================================
# 1. 添加 CARAFE 模块的定义 (必须步骤)
# ==============================================================================
class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(inC, inC // 4, 1)
        self.encoder = nn.Conv2d(inC // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(inC, outC, 1)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # Kernel Prediction Module
        kernel_tensor = self.down(in_tensor)
        kernel_tensor = self.encoder(kernel_tensor)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)

        # Content-aware Reassembly Module
        in_tensor_unfolded = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                                   self.kernel_size // 2, self.kernel_size // 2),
                                   mode='constant', value=0)
        in_tensor_unfolded = in_tensor_unfolded.unfold(2, self.kernel_size, step=1)
        in_tensor_unfolded = in_tensor_unfolded.unfold(3, self.kernel_size, step=1)
        in_tensor_unfolded = in_tensor_unfolded.reshape(N, C, H, W, -1)
        in_tensor_unfolded = in_tensor_unfolded.permute(0, 2, 3, 1, 4)

        out_tensor = torch.matmul(in_tensor_unfolded, kernel_tensor)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        return out_tensor


# ==============================================================================
# 2. 修改 EUCB 模块以使用 CARAFE
# ==============================================================================
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 步骤 1: 使用 CARAFE 进行上采样
        # CARAFE的输入和输出通道数保持为 in_channels，因为它只负责上采样，不改变通道数以供后续层处理
        self.carafe_up = CARAFE(inC=self.in_channels, outC=self.in_channels, kernel_size=kernel_size, up_factor=2)

        # 步骤 2: 定义上采样后的处理层 (深度卷积、BN、ReLU)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )

        # 步骤 3: 逐点卷积 (Pointwise Convolution)
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # 步骤 4: 最终的融合卷积
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 将原有的 self.up_dwc(x1) 分解为两步
        x1_up = self.carafe_up(x1)  # 使用 CARAFE 上采样
        x1_processed = self.dw_conv(x1_up)  # 对上采样后的结果进行处理

        x1_shuffled = channel_shuffle(x1_processed, self.in_channels)
        x1_final = self.pwc(x1_shuffled)

        x = torch.cat([x2, x1_final], dim=1)
        x = self.conv(x)

        return x


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

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
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


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up1 = EUCB(512, 256)
        self.up2 = EUCB(256, 128)
        self.up3 = EUCB(128, 64)
        self.up4 = EUCB(64, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(f":x1 {x1.size()}")
        x2 = self.down1(x1)
        # print(f":x2 {x2.size()}")
        x3 = self.down2(x2)
        # print(f":x3 {x3.size()}")
        x4 = self.down3(x3)
        # print(f":x4{x4.size()}")
        x5 = self.down4(x4)
        # print(f":x5 {x5.size()}")
        up1 = self.up1(x5, x4)
        # print(f":up1 {up1.size()}")
        up2 = self.up2(up1, x3)
        # print(f":up2 {up2.size()}")
        up3 = self.up3(up2, x2)
        # print(f":up3 {up3.size()}")
        up4 = self.up4(up3, x1)
        # print(f":up4 {up4.size()}")
        logits = self.outc(up4)
        # print(f":result {logits.size()}")
        return logits


# model = Unet(n_channels=3, n_classes=3, bilinear=False)  # Example: 3 input channels, 2 output classes
# input_tensor = torch.randn(1, 3, 512, 512)  # Example input
# output = model(input_tensor)
# # print(output.shape)  # Output shape should be (1, 2, 256, 256)
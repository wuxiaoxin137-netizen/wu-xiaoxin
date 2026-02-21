import torch
import torch.nn as nn
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MLLABlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, L, C = x.shape
        H = self.input_resolution
        assert L == H, "input feature has wrong size"

        x = x + self.cpe1(x.permute(0, 2, 1)).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).permute(0, 2, 1)
        x = self.act(self.dwc(x)).permute(0, 2, 1)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class ResNetSegmentation(nn.Module):
    def __init__(self, block, blocks_num, num_classes=3):
        super(ResNetSegmentation, self).__init__()
        self.in_channel = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder layers
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # Decoder layers with upsampling and concatenation
        self.upsample1 = nn.ConvTranspose2d(512 * block.expansion, 256, kernel_size=2, stride=2)
        self.conv_cat3 = nn.Conv2d(512, 256, kernel_size=1)

        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_cat2 = nn.Conv2d(256, 128, kernel_size=1)

        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_cat1 = nn.Conv2d(128, 64, kernel_size=1)

        self.upsample4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upsample5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

        # Final convolution
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoding path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Decoding path with concatenation and channel reduction
        x = self.upsample1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_cat3(x)

        x = self.upsample2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_cat2(x)

        x = self.upsample3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_cat1(x)

        x = self.upsample4(x)
        x = self.upsample5(x)

        # Final output
        x = self.final_conv(x)
        return x


# Creating different ResNet configurations for segmentation tasks
def resnet34_segmentation(num_classes=3):
    return ResNetSegmentation(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


# # Example usage
# if __name__ == "__main__":
#     model = resnet34_segmentation(num_classes=3)
#     inputs = torch.randn(1, 3, 512, 512)  # Input image of size 512x512
#     outputs = model(inputs)
#     print(outputs.shape)  # Output shape should be (B, num_classes, 512, 512)

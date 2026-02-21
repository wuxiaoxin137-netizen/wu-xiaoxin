import torch
import torch.nn as nn
import logging
from .cswin_unet import CSWinTransformer

logger = logging.getLogger(__name__)

class CSwinUnet(nn.Module):
    def __init__(self, img_size=512, num_classes=3, config=None):
        super(CSwinUnet, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.cswin_unet = CSWinTransformer(
            img_size=img_size,
            patch_size=16,
            in_chans=3,
            num_classes=self.num_classes,
            embed_dim=64,
            depth=[1, 2, 9, 1],
            split_size=[1, 2, 7, 7],
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0,
            norm_layer=nn.LayerNorm,
            use_chk=False
        )

    def forward(self, x):
        if x.size(1) == 1:  # If input image has only 1 channel, replicate it to 3 channels.
            x = x.repeat(1, 3, 1, 1)
        logits = self.cswin_unet(x)  # Perform forward pass through the CSWinTransformer
        return logits

    def load_from(self, config):
        # Removed pretrained model loading as per your request
        print("No pretrained weights to load.")
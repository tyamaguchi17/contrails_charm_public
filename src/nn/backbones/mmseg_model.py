from logging import getLogger

try:
    from timm.models._manipulate import adapt_input_conv
except ModuleNotFoundError:
    adapt_input_conv = None
from torch import Tensor, nn

from .base import BackboneBase

logger = getLogger(__name__)

mmseg_ckpt_map = {
    "upernet_convnext_base_fp16_640x640_160k_ade20k": "upernet_convnext_base_fp16_640x640_160k_ade20k_20220227_182859-9280e39b.pth",
    "upernet_convnext_large_fp16_640x640_160k_ade20k": "upernet_convnext_large_fp16_640x640_160k_ade20k_20220226_040532-e57aa54d.pth",
    "upernet_convnext_xlarge_fp16_640x640_160k_ade20k": "upernet_convnext_xlarge_fp16_640x640_160k_ade20k_20220226_080344-95fc38c2.pth",
    "upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K": "upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K_20210531_132020-05b22ea4.pth",
    "upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k": "upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k_20220318_091743-9ba68901.pth",
}


class mmSegModelBackbone(BackboneBase):
    def __init__(self, model, arch="swin", in_channels=3, num_classes=1) -> None:
        super().__init__()
        self.model = model
        if arch == "swin":
            first_conv = model.backbone.patch_embed.projection
            first_conv_name = "backbone.patch_embed.projection"
        elif arch == "convnext":
            first_conv = model.backbone.downsample_layers[0][0]
            first_conv_name = "backbone.downsample_layers.0.0"
        if first_conv.in_channels != in_channels:
            state_dict = first_conv.state_dict()
            state_dict["weight"] = adapt_input_conv(in_channels, state_dict["weight"])
            _, out_channels, kernel_size, stride = (
                first_conv.in_channels,
                first_conv.out_channels,
                first_conv.kernel_size,
                first_conv.stride,
            )

            first_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            logger.info(
                f"Converted input conv {first_conv_name} pretrained weights from 3 to {in_channels} channel(s)"
            )
            first_conv.load_state_dict(state_dict)
            if arch == "swin":
                model.backbone.patch_embed.projection = first_conv
            elif arch == "convnext":
                model.backbone.downsample_layers[0][0] = first_conv
        self.conv = nn.Conv2d(150, num_classes, 3, 1, 1)
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        x = self.model.encode_decode(x, None)
        x = self.conv(x)
        return x

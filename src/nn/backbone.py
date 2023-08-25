from typing import Optional

import segmentation_models_pytorch as smp

try:
    from mmseg.apis import init_segmentor
except ModuleNotFoundError:
    init_segmentor = None

from .backbones.base import BackboneBase
from .backbones.mmseg_model import mmseg_ckpt_map, mmSegModelBackbone
from .backbones.seg_model import SegModelBackbone, SegModelMid25dBackbone


def load_backbone(
    base_model: str,
    pretrained: bool,
    in_chans: int = 3,
    output_dim: int = 3,
    use_25d: bool = False,
    n_frames: int = 1,
    config_path: Optional[str] = None,
) -> BackboneBase:
    if "convnext" in base_model or "swin" in base_model:
        if config_path is None:
            config_path = f"../mmsegmentation_models/{base_model}.py"
        ckpt_path = f"../mmsegmentation_models/{mmseg_ckpt_map[base_model]}"
        if pretrained:
            model = init_segmentor(config_path, ckpt_path, device="cpu")
        else:
            model = init_segmentor(config_path, device="cpu")
        if "convnext" in base_model:
            arch = "convnext"
        elif "swin" in base_model:
            arch = "swin"
        backbone = mmSegModelBackbone(model, arch, in_chans, output_dim)
    elif use_25d:
        model = smp.Unet(
            encoder_name=f"tu-{base_model}",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_chans,
            classes=output_dim,
        )
        backbone = SegModelMid25dBackbone(model, n_frames)
    else:
        model = smp.Unet(
            encoder_name=f"tu-{base_model}",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_chans,
            classes=output_dim,
        )
        backbone = SegModelBackbone(model)

    return backbone

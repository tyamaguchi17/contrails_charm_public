from logging import getLogger

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.nn.backbone import load_backbone
from src.nn.backbones.base import BackboneBase
from src.utils.checkpoint import get_weights_to_load

logger = getLogger(__name__)


def init_model_from_config(cfg: DictConfig, pretrained: bool):
    model = nn.Sequential()
    backbone = init_backbone(cfg.base_model, cfg, pretrained=pretrained)
    forward_seg = nn.Sequential()
    forward_seg.add_module("backbone", backbone)
    model.add_module("forward_seg", forward_seg)

    if cfg.restore_path is not None and cfg.restore_path != ".":
        logger.info(f'Loading weights from "{cfg.restore_path}"...')
        ckpt = torch.load(cfg.restore_path, map_location="cpu")
        model_dict = get_weights_to_load(model, ckpt)
        model.load_state_dict(model_dict, strict=True)

    return model


def init_backbone(base_model: str, cfg: DictConfig, pretrained: bool) -> BackboneBase:
    in_chans = cfg.in_chans
    output_dim = cfg.output_dim
    if cfg.use_label_aux_min_max:
        output_dim += 2
    use_25d = cfg.use_25d
    n_frames = cfg.n_frames_before + cfg.n_frames_after + 1
    backbone = load_backbone(
        base_model=base_model,
        pretrained=pretrained,
        in_chans=in_chans,
        output_dim=output_dim,
        use_25d=use_25d,
        n_frames=n_frames,
    )
    if cfg.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    return backbone

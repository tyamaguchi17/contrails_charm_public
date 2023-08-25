from typing import List, Tuple

import torch
from torch import Tensor

from .base import BackboneBase


class SegModelBackbone(BackboneBase):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x


class Conv3dBlock(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        padding: Tuple[int, int, int],
    ):
        super().__init__(
            torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                padding_mode="replicate",
            ),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
        )


class SegModelMid25dBackbone(BackboneBase):
    def __init__(self, model, n_frames=3) -> None:
        super().__init__()
        self.model = model
        self.n_frames = n_frames
        conv3ds = [
            torch.nn.Sequential(
                Conv3dBlock(ch, ch, (2, 3, 3), (0, 1, 1)),
                Conv3dBlock(ch, ch, (2, 3, 3), (0, 1, 1)),
            )
            for ch in self.model.encoder.out_channels[1:]
        ]
        self.conv3ds = torch.nn.ModuleList(conv3ds)

    def _to2d(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        ret = []
        for conv3d, feature in zip(self.conv3ds, features):
            total_batch, ch, H, W = feature.shape
            feat_3d = feature.reshape(
                total_batch // self.n_frames, self.n_frames, ch, H, W
            ).transpose(1, 2)
            feat_2d = conv3d(feat_3d).squeeze(2)
            ret.append(feat_2d)
        return ret

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, n_frame, in_ch, H, W = x.shape
        x = x.reshape(n_batch * n_frame, in_ch, H, W)

        self.model.check_input_shape(x)

        features = self.model.encoder(x)
        features[1:] = self._to2d(features[1:])
        decoder_output = self.model.decoder(*features)

        masks = self.model.segmentation_head(decoder_output)
        return masks

from typing import List, Optional, Union

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from omegaconf import DictConfig


class Preprocessing:
    def __init__(
        self,
        aug_cfg: Optional[DictConfig],
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        h_resize_to: int,
        w_resize_to: int,
    ):
        self.aug_cfg = aug_cfg.copy()
        if isinstance(mean, float):
            self.mean = mean
            assert isinstance(std, float)
            self.std = std
        else:
            self.mean = mean.copy()
            self.std = std.copy()
        self.h_resize_to = h_resize_to
        self.w_resize_to = w_resize_to
        additional_targets = {f"mask{i+1}": "mask" for i in range(5)}
        for i in range(
            (aug_cfg.in_chans // 3)
            * (1 + aug_cfg.n_frames_before + aug_cfg.n_frames_after)
            - 1
        ):
            additional_targets[f"image{i+1}"] = "image"
        self.additional_targets = additional_targets

    def get_train_transform(self) -> A.Compose:
        cfg = self.aug_cfg

        if cfg.use_light_aug:
            transforms = [
                A.Resize(self.h_resize_to, self.w_resize_to),
                A.ShiftScaleRotate(0.15, 0.15, 15, p=0.2),
                A.RandomResizedCrop(
                    self.h_resize_to,
                    self.w_resize_to,
                    scale=(cfg.crop_scale, 1.0),
                    ratio=(cfg.crop_l, cfg.crop_r),
                    p=0.2,
                ),
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.2),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(transpose_mask=True),
            ]
        elif cfg.use_light_aug2:
            transforms = [
                A.HorizontalFlip(p=0.1),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.4),
                A.Resize(self.h_resize_to, self.w_resize_to),
                A.ShiftScaleRotate(0.05, 0.1, 15, p=0.3),
                A.RandomResizedCrop(
                    self.h_resize_to,
                    self.w_resize_to,
                    scale=(0.75, 1.0),
                    ratio=(0.9, 1.1111111111111),
                    p=0.7,
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(transpose_mask=True),
            ]
        elif cfg.use_light_aug3:
            transforms = [
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Resize(self.h_resize_to, self.w_resize_to),
                A.ShiftScaleRotate(0.05, 0.1, 15, p=0.5),
                A.RandomResizedCrop(
                    self.h_resize_to,
                    self.w_resize_to,
                    scale=(0.75, 1.0),
                    ratio=(0.9, 1.1111111111111),
                    p=0.7,
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(transpose_mask=True),
            ]
        elif cfg.use_aug:
            transforms = [
                A.Resize(self.h_resize_to, self.w_resize_to),
                A.ShiftScaleRotate(0.3, 0.3, 90, p=0.5),
                A.RandomResizedCrop(
                    self.h_resize_to,
                    self.w_resize_to,
                    scale=(cfg.crop_scale, 1.0),
                    ratio=(cfg.crop_l, cfg.crop_r),
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(transpose_mask=True),
            ]
        elif cfg.use_heavy_aug:
            transforms = [
                A.Resize(self.h_resize_to, self.w_resize_to),
                A.Affine(
                    rotate=(-cfg.rotate, cfg.rotate),
                    translate_percent=(0.0, cfg.translate),
                    shear=(-cfg.shear, cfg.shear),
                    p=cfg.p_affine,
                ),
                A.RandomResizedCrop(
                    self.h_resize_to,
                    self.w_resize_to,
                    scale=(cfg.crop_scale, 1.0),
                    ratio=(cfg.crop_l, cfg.crop_r),
                ),
                # A.CLAHE(clip_limit=(1, 4), p=0.5),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.0),
                        A.ElasticTransform(alpha=3),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=[10, 50]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                        A.MedianBlur(),
                    ],
                    p=0.2,
                ),
                A.ToGray(p=cfg.p_gray),
                A.GaussianBlur(blur_limit=(3, 7), p=cfg.p_blur),
                A.GaussNoise(p=cfg.p_noise),
                A.OneOf(
                    [
                        A.JpegCompression(),
                        A.Downscale(scale_min=0.1, scale_max=0.15),
                    ],
                    p=0.2,
                ),
                # A.Downscale(scale_min=0.1, scale_max=0.15, p=0.2),
                A.PiecewiseAffine(p=0.2),
                A.Sharpen(p=0.2),
                A.RandomGridShuffle(grid=(2, 2), p=cfg.p_shuffle),
                A.Posterize(p=cfg.p_posterize),
                A.RandomBrightnessContrast(p=cfg.p_bright_contrast),
                A.Cutout(
                    max_h_size=int(self.h_resize_to * 0.1),
                    max_w_size=int(self.w_resize_to * 0.1),
                    num_holes=5,
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(transpose_mask=True),
            ]
        else:
            transforms = [
                # Targets: image, mask, bboxes, keypoints
                A.Resize(self.h_resize_to, self.w_resize_to, p=1),
                # Targets: image
                A.Normalize(mean=self.mean, std=self.std),
                # Targets: image, mask
                ToTensorV2(transpose_mask=True),
            ]

        aug = A.Compose(transforms, additional_targets=self.additional_targets)
        return aug

    def get_val_transform(self) -> A.Compose:
        # cfg = self.aug_cfg
        transforms = [
            # Targets: image, mask, bboxes, keypoints
            A.Resize(self.h_resize_to, self.w_resize_to, p=1),
            # Targets: image
            A.Normalize(mean=self.mean, std=self.std),
            # Targets: image, mask
            ToTensorV2(transpose_mask=True),
        ]
        aug = A.Compose(transforms, additional_targets=self.additional_targets)
        return aug

    def get_test_transform(self) -> A.Compose:
        # cfg = self.aug_cfg
        transforms = [
            # Targets: image, mask, bboxes, keypoints
            A.Resize(self.h_resize_to, self.w_resize_to, p=1),
            # Targets: image
            A.Normalize(mean=self.mean, std=self.std),
            # Targets: image, mask
            ToTensorV2(transpose_mask=True),
        ]
        aug = A.Compose(transforms, additional_targets=self.additional_targets)
        return aug

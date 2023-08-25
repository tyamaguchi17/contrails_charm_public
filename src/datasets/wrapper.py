from typing import Callable

import torch
from torch.utils.data import Dataset


class WrapperDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        transform: Callable,
    ):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def apply_transform(self, data):
        image = data.pop("image")
        kwargs = {}
        in_chans = image[0].shape[-1]
        n_frames = len(image)
        for frame in range(len(image)):
            image_frame = image[frame]
            for i in range(in_chans // 3):
                if frame == 0 and i == 0:
                    kwargs["image"] = image_frame[:, :, 3 * i : 3 * (i + 1)]
                else:
                    kwargs[f"image{i+frame*(in_chans // 3)}"] = image_frame[
                        :, :, 3 * i : 3 * (i + 1)
                    ]
        assert in_chans // 3 * len(image) == len(kwargs)
        label = data.pop("label")
        kwargs["mask"] = label.copy()
        kwargs["mask1"] = data.pop("label_aux")
        kwargs["mask2"] = data.pop("label_aux_min")
        kwargs["mask3"] = data.pop("label_aux_max")
        kwargs["mask4"] = data.pop("label_2")
        kwargs["mask5"] = data.pop("label_3")
        transformed = self.transform(**kwargs)
        image = []
        for frame in range(n_frames):
            image_frame = []
            for i in range(in_chans // 3):
                if frame == 0 and i == 0:
                    image_frame.append(transformed["image"])
                else:
                    image_frame.append(transformed[f"image{i+frame*(in_chans // 3)}"])
            image_frame = torch.cat(image_frame, dim=0).unsqueeze(0)
            image.append(image_frame)
        image = torch.cat(image, dim=0)
        if len(image) == 1:
            image = image[0]
        data["image"] = image
        data["label"] = transformed["mask"]
        data["label_aux"] = transformed["mask1"]
        data["label_aux_min"] = transformed["mask2"]
        data["label_aux_max"] = transformed["mask3"]
        data["label_2"] = transformed["mask4"]
        data["label_3"] = transformed["mask5"]

        return data

    def __getitem__(self, index: int):
        data: dict = self.base[index]
        data = self.apply_transform(data)
        return data

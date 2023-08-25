import glob
import random
import warnings
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


BAND11_MEAN_STD = (272.9, 20.0)
BAND13_MEAN_STD = (275.2, 19.5)
BAND14_MEAN_STD = (273.7, 21.5)
BAND15_MEAN_STD = (270.9, 21.1)


def normalize_mean_std(data, mean, std):
    return (data - mean) / std


class ContrailsDataset(Dataset):
    """Dataset class for kaggle contrails."""

    @classmethod
    def create_dataframe(
        cls,
        num_folds: int = 5,
        seed: int = 2023,
        num_records: int = 0,
        data_path: str = "",
    ) -> pd.DataFrame:
        logger.info(f"loading {data_path}")
        df = pd.DataFrame({"data_path": glob.glob(f"{data_path}/*")})
        df["image_id"] = df["data_path"].map(lambda x: Path(x).name)

        if num_folds > 0:
            n_splits = num_folds
            shuffle = True

            kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
            X = df["image_id"].values
            fold = -np.ones(len(df))
            for i, (_, indices) in enumerate(kfold.split(X)):
                fold[indices] = i

            df["fold"] = fold
        else:
            df["fold"] = -1

        if num_records:
            df = df[::num_records]

        return df

    def __init__(
        self,
        df: pd.DataFrame,
        phase="train",
        cfg=None,
    ) -> None:
        self.df = df.copy()
        self.df["original_index"] = df.index

        self.df.reset_index(inplace=True)
        self.phase = phase

        self.image_ch = 6
        self.normalize_method = cfg.normalize_method
        self.n_frames_before = cfg.n_frames_before
        self.n_frames_after = cfg.n_frames_after
        self.cfg_aug = cfg.augmentation
        self.pl_path = Path(cfg.pl_path)
        self.use_round = cfg.use_round
        self.use_pred_crop = cfg.use_pred_crop
        self.thr = cfg.thr
        if self.use_pred_crop:
            self.preds = np.load(cfg.pred_mask_path)

    def __len__(self) -> int:
        return len(self.df)

    def _read_image_cached(self, file_path: str):
        img = self._read_from_storage(file_path)

        return img

    def _read_from_storage(self, file_path: str):
        img = np.load(file_path)
        return img

    def get_ash_color_images(self, path, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{path}/band_11.npy"  # 8.4-μm
        band13_path = f"{path}/band_13.npy"  # 10.3-μm
        band14_path = f"{path}/band_14.npy"  # 11.2-μm
        band15_path = f"{path}/band_15.npy"  # 12.3-μm

        band11 = self._read_image_cached(band11_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        band11 = band11[:, :, frame]
        band13 = band13[:, :, frame]
        band14 = band14[:, :, frame]
        band15 = band15[:, :, frame]

        if self.normalize_method == "mean_std":
            band11 = normalize_mean_std(band11, BAND11_MEAN_STD[0], BAND11_MEAN_STD[1])
            band13 = normalize_mean_std(band13, BAND13_MEAN_STD[0], BAND13_MEAN_STD[1])
            band14 = normalize_mean_std(band14, BAND14_MEAN_STD[0], BAND14_MEAN_STD[1])
            band15 = normalize_mean_std(band15, BAND15_MEAN_STD[0], BAND15_MEAN_STD[1])

            r = band15 - band14
            g = band14 - band11
            b = band14
            false_color = np.stack([r, g, b], axis=2)

            bands = np.stack([band11, band13, band15], axis=2)

            false_color = np.concatenate([false_color, bands], axis=2)
        else:
            r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
            g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
            b = normalize_range(band14, _T11_BOUNDS)

            false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

            band11 = normalize_range(band11, _T11_BOUNDS)
            band13 = normalize_range(band13, _T11_BOUNDS)
            band15 = normalize_range(band15, _T11_BOUNDS)

            bands = np.clip(np.stack([band11, band13, band15], axis=2), 0, 1)

            false_color = np.concatenate([false_color, bands], axis=2)

        return false_color

    def get_file_path(self, index):
        path = self.df.at[index, "data_path"]
        return path

    def read_image(self, index, frame=4):
        path = self.get_file_path(index)
        image = self.get_ash_color_images(path, frame=frame)
        return image

    def read_label(self, index, frame=4, use_pl=False, use_round=False):
        if frame != 4 or use_pl:
            path = self.pl_path / self.df.at[index, "image_id"] / f"frame{frame}.npy"
        else:
            path = self.get_file_path(index)
            path = f"{path}/human_pixel_masks.npy"
        if use_round:
            label = np.round(self._read_image_cached(path) * 4.0) / 4.0
        else:
            label = (self._read_image_cached(path) > 0.5).astype(int)
        return label

    def read_label_individual(self, index, mean=True):
        path = self.get_file_path(index)
        path = f"{path}/human_individual_masks.npy"
        label = self._read_image_cached(path)
        if mean:
            label = label.mean(axis=-1)
        return label

    def augmentation(self, image, label, label_aux, label_aux_min, label_aux_max):
        cfg_aug = self.cfg_aug
        p_crop = cfg_aug.p_crop
        ratio_max = cfg_aug.scale_max
        ratio_min = cfg_aug.scale_min
        if np.random.uniform() < p_crop:
            components = cv2.connectedComponentsWithStats(
                label.astype(np.uint8), connectivity=8
            )
            idx = np.random.randint(components[0])
            _, _, w, h, _ = components[2][idx]
            x, y = components[3][idx]
            scale = np.random.uniform(ratio_min, ratio_max)
            w *= scale
            w = max(64, w)
            h *= scale
            h = max(64, h)
            x_min = max(int(x - w / 2), 0)
            x_max = min(int(x + w / 2), 256)
            y_min = max(int(y - h / 2), 0)
            y_max = min(int(y + h / 2), 256)
            image = [_image[y_min:y_max, x_min:x_max] for _image in image]
            label = label[y_min:y_max, x_min:x_max]
            label_aux = label_aux[y_min:y_max, x_min:x_max]
            label_aux_min = label_aux_min[y_min:y_max, x_min:x_max]
            label_aux_max = label_aux_max[y_min:y_max, x_min:x_max]
        return image, label, label_aux, label_aux_min, label_aux_max

    def __getitem__(self, index: int):
        use_pl = False
        path = self.get_file_path(index)
        if (
            self.phase == "train"
            and np.random.uniform() < self.cfg_aug.p_frame
            and "train" in path
        ):
            frame = random.choice([2, 3, 4, 5, 6, 7])
            use_pl = True
        else:
            frame = 4
        frames = [frame - i - 1 for i in range(self.n_frames_before)][::-1]
        frames.append(frame)
        frames += [frame + i + 1 for i in range(self.n_frames_after)]
        image = [self.read_image(index, i) for i in frames]
        image_id = self.df.at[index, "image_id"]
        label = self.read_label(index, frame, use_pl)

        if self.phase == "train" and "train" in path:
            if frame == 4 and not use_pl:
                label_aux_all = self.read_label_individual(index, mean=False)
                label_aux = label_aux_all.mean(axis=-1)
                label_aux_min = label_aux_all.min(axis=-1)
                label_aux_max = label_aux_all.max(axis=-1)
                label_2 = self.read_label(index, frame - 2)
                label_3 = self.read_label(index, frame - 1)
            else:
                label_aux = self.read_label(
                    index, frame, use_pl, use_round=self.use_round
                )
                label_aux_min = label_aux
                label_aux_max = label_aux
                label_2 = self.read_label(index, max(frame - 2, 2))
                label_3 = self.read_label(index, max(frame - 1, 2))
        else:
            label_aux = label
            label_aux_min = label
            label_aux_max = label
            label_2 = label
            label_3 = label

        aux_mask = "train" in path and frame == 4
        label_origin = label.copy()
        if self.phase == "train":
            image, label, label_aux, label_aux_min, label_aux_max = self.augmentation(
                image, label, label_aux, label_aux_min, label_aux_max
            )

        y_min, y_max, x_min, x_max = 0, 256, 0, 256

        res = {
            "original_index": self.df.at[index, "original_index"],
            "image_id": int(image_id),
            "image": image,
            "label": label.astype(float),
            "label_2": label_2.astype(float),
            "label_3": label_3.astype(float),
            "label_aux": label_aux.astype(float),
            "label_aux_min": label_aux_min.astype(float),
            "label_aux_max": label_aux_max.astype(float),
            "label_origin": label_origin.astype(float),
            "aux_mask": aux_mask,
            "y_min": y_min,
            "y_max": y_max,
            "x_min": x_min,
            "x_max": x_max,
        }

        return res

from logging import getLogger
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader

from run.init.dataset import init_datasets_from_config
from run.init.forwarder import Forwarder
from run.init.model import init_model_from_config
from run.init.optimizer import init_optimizer_from_config
from run.init.preprocessing import Preprocessing
from run.init.scheduler import init_scheduler_from_config
from src.datasets.wrapper import WrapperDataset

logger = getLogger(__name__)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calc_val_score(pred_binary, target):
    tp = (pred_binary * target).sum()
    pred_positive = pred_binary.sum()
    target_positive = target.sum()

    dice_score = 2 * tp / (pred_positive + target_positive)
    return dice_score


def search_best_thresh(y_pred, y_true, thresh_min=0.00, thresh_max=1.01):
    val_score_on_best_thresh = -100
    best_thresh = 0.0

    for i, thresh in enumerate(np.arange(thresh_min, thresh_max, 0.01)):
        y_pred_binary = (y_pred > thresh).to(dtype=torch.int32)
        val_score = calc_val_score(y_pred_binary, y_true)
        if val_score_on_best_thresh < val_score:
            val_score_on_best_thresh = val_score
            best_thresh = thresh

    return val_score_on_best_thresh, best_thresh


class PLModel(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg.copy()
        self.save_embed = self.cfg.training.save_embed

        pretrained = False if cfg.training.debug else True

        logger.info("creating model")
        model = init_model_from_config(cfg.model, pretrained=pretrained)
        self.forwarder = Forwarder(cfg.forwarder, model)
        if cfg.training.use_gradient_checkpointing:
            self.forwarder.model.forward_seg.backbone.model.encoder.model.set_grad_checkpointing(
                True
            )

        logger.info("loading metadata")
        raw_datasets = init_datasets_from_config(cfg.dataset)

        preprocessing = Preprocessing(cfg.augmentation, **cfg.preprocessing)
        self.datasets = {}
        transforms = {
            "train": preprocessing.get_train_transform(),
            "val": preprocessing.get_val_transform(),
            "test": preprocessing.get_test_transform(),
        }
        for phase in ["train", "val", "test"]:
            if phase == "train":
                train_dataset = WrapperDataset(
                    raw_datasets["train"],
                    transforms["train"],
                )
                self.datasets["train"] = train_dataset
                logger.info(f"{phase}: {len(self.datasets[phase])}")
            else:
                self.datasets[phase] = WrapperDataset(
                    raw_datasets[phase], transforms[phase]
                )
                logger.info(f"{phase}: {len(self.datasets[phase])}")

        logger.info(
            f"training steps per epoch: {len(self.datasets['train'])/cfg.training.batch_size}"
        )
        self.cfg.scheduler.num_steps_per_epoch = (
            len(self.datasets["train"]) / cfg.training.batch_size
        )

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        additional_info = {}
        _, loss = self.forwarder.forward(
            batch, phase="train", epoch=self.current_epoch, **additional_info
        )

        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        sch = self.lr_schedulers()
        sch.step()
        self.log(
            "lr",
            sch.get_last_lr()[0],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )

        return loss

    def _end_process(self, outputs: List[Dict[str, Tensor]], phase: str):
        # Aggregate results
        epoch_results: Dict[str, np.ndarray] = {}
        outputs = self.all_gather(outputs)

        for key in [
            "original_index",
            "image_id",
            "labels",
            "preds",
        ]:
            if isinstance(outputs[0][key], Tensor):
                result = torch.cat([torch.atleast_1d(x[key]) for x in outputs], dim=1)
                result = torch.flatten(result, end_dim=1)
                epoch_results[key] = result.detach().cpu().numpy()
                if key == "preds":
                    epoch_results[key] = epoch_results[key].astype(np.float16)
                if key == "labels":
                    epoch_results[key] = epoch_results[key].astype(int)
            else:
                result = np.concatenate([x[key] for x in outputs])
                epoch_results[key] = result

        preds = torch.cat([torch.atleast_1d(x["preds"]) for x in outputs], dim=1)
        preds = torch.flatten(preds, end_dim=1)
        labels = torch.cat([torch.atleast_1d(x["labels"]) for x in outputs], dim=1)
        labels = torch.flatten(labels, end_dim=1)

        dice_score, best_thr = search_best_thresh(preds, labels)

        if phase == "test" and self.trainer.global_rank == 0:
            # Save test results ".npz" format
            test_results_filepath = Path(self.cfg.out_dir) / "test_results"
            if not test_results_filepath.exists():
                test_results_filepath.mkdir(exist_ok=True)
            if self.cfg.save_results:
                np.savez_compressed(
                    str(test_results_filepath / "test_results.npz"),
                    **epoch_results,
                )

            df = pd.DataFrame(
                data={
                    "original_index": epoch_results["original_index"]
                    .reshape(-1)
                    .astype(int),
                }
            )
            df["threshold"] = best_thr

            df.to_csv(test_results_filepath / "test_results.csv", index=False)

        loss = (
            torch.cat([torch.atleast_1d(x["loss"]) for x in outputs])
            .detach()
            .cpu()
            .numpy()
        )
        mean_loss = np.mean(loss)

        if phase != "test" and self.trainer.global_rank == 0:
            test_results_filepath = Path(self.cfg.out_dir) / "test_results"
            if not test_results_filepath.exists():
                test_results_filepath.mkdir(exist_ok=True)

            weights_filepath = Path(self.cfg.out_dir) / "weights"
            if not weights_filepath.exists():
                weights_filepath.mkdir(exist_ok=True)
            weights_path = str(weights_filepath / "model_weights.pth")
            logger.info(f"Extracting and saving weights: {weights_path}")
            torch.save(self.forwarder.model.state_dict(), weights_path)

        # Log items
        self.log(f"{phase}/loss", mean_loss, prog_bar=True)
        self.log(f"{phase}/dice", dice_score, prog_bar=True)
        self.log(f"{phase}/threshold", best_thr, prog_bar=True)

    def _evaluation_step(self, batch: Dict[str, Tensor], phase: Literal["val", "test"]):
        preds, loss = self.forwarder.forward(
            batch, phase=phase, epoch=self.current_epoch
        )
        preds = F.resize(img=preds.detach(), size=(256, 256))
        output = {
            "loss": loss,
            "original_index": batch["original_index"],
            "image_id": batch["image_id"],
            "preds": preds.sigmoid(),
            "labels": batch["label_origin"].squeeze(-1),
        }
        return output

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        return self._evaluation_step(batch, phase="val")

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        self._end_process(outputs, "val")

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self._evaluation_step(batch, phase="test")

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        self._end_process(outputs, "test")

    def configure_optimizers(self):
        model = self.forwarder.model
        opt_cls, kwargs = init_optimizer_from_config(
            self.cfg.optimizer, model.forward_seg.parameters()
        )

        optimizer = opt_cls([kwargs])

        scheduler = init_scheduler_from_config(self.cfg.scheduler, optimizer)

        if scheduler is None:
            return [optimizer]
        return [optimizer], [scheduler]
        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_before_zero_grad(self, *args, **kwargs):
        self.forwarder.ema.update(self.forwarder.model.parameters())

    def _dataloader(self, phase: str) -> DataLoader:
        logger.info(f"{phase} data loader called")
        dataset = self.datasets[phase]

        batch_size = self.cfg.training.batch_size
        num_workers = self.cfg.training.num_workers

        num_gpus = self.cfg.training.num_gpus
        if phase != "train":
            batch_size = self.cfg.training.batch_size_test
        batch_size //= num_gpus
        num_workers //= num_gpus

        drop_last = True if self.cfg.training.drop_last and phase == "train" else False
        shuffle = phase == "train"

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(phase="train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(phase="val")

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(phase="test")

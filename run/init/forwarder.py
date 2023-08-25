from typing import Dict, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch_ema import ExponentialMovingAverage


class Forwarder(nn.Module):
    def __init__(self, cfg: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # workaround for device inconsistency of ExponentialMovingAverage
        self.ema = None
        self.cfg = cfg
        self.dice_func = smp.losses.DiceLoss(mode="binary")
        self.bce_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.cos_sim = nn.CosineSimilarity()
        self.use_label_aux_min_max = cfg.use_label_aux_min_max
        self.use_amp = cfg.use_amp

    def loss(
        self,
        logits,
        labels,
    ):
        if len(logits) == 0:
            return 0
        loss = self.dice_func(logits, labels)

        return loss

    def dice_2_loss(
        self,
        logits,
        labels,
        numerator=700000,
        denominator=1000000,
        smooth=0,
    ):
        if len(logits) == 0:
            return 0
        pred = torch.sigmoid(logits).flatten()
        label = labels.flatten()
        intersection = (label * pred).sum()
        return 1 - (2.0 * intersection + smooth + numerator) / (
            label.sum() + pred.sum() + smooth + denominator
        )

    def bce_loss(
        self,
        logits,
        labels,
    ):
        if len(logits) == 0:
            return 0
        loss = self.bce_func(logits, labels)

        return loss

    def cossim_loss(
        self,
        logits,
        labels,
    ):
        loss = 1 - self.cos_sim(logits, labels)
        return loss.mean()

    def forward(
        self, batch: Dict[str, Tensor], phase: str, epoch=None
    ) -> Tuple[Tensor, Tensor]:
        # workaround for device inconsistency of ExponentialMovingAverage
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)

        # inputs: Input tensor.
        inputs = batch["image"]

        # labels
        if self.use_amp:
            labels = batch["label"].to(torch.float16)
            labels_aux = batch["label_aux"].to(torch.float16)
            labels_aux_min = batch["label_aux_min"].to(torch.float16)
            labels_aux_max = batch["label_aux_max"].to(torch.float16)
        else:
            labels = batch["label"].to(torch.float32)
            labels_aux = batch["label_aux"].to(torch.float32)
            labels_aux_min = batch["label_aux_min"].to(torch.float32)
            labels_aux_max = batch["label_aux_max"].to(torch.float32)
        aux_mask = batch["aux_mask"]

        if phase == "train":
            with torch.set_grad_enabled(True):
                logits = self.model.forward_seg(inputs)
        else:
            if phase == "test":
                with self.ema.average_parameters():
                    logits = self.model.forward_seg(inputs)
            elif phase == "val":
                logits = self.model.forward_seg(inputs)

        loss = (
            self.loss(
                logits=logits[:, 0],
                labels=labels,
            )
            * self.cfg.loss.dice_loss_weight
            + self.dice_2_loss(
                logits=logits[:, 0],
                labels=labels,
            )
            * self.cfg.loss.dice_2_loss_weight
            + self.loss(
                logits=logits[:, 1],
                labels=1 - labels,
            )
            * self.cfg.loss.dice_inv_loss_weight
            + self.loss(
                logits=logits[aux_mask, 2],
                labels=labels_aux[aux_mask],
            )
            * self.cfg.loss.dice_aux_loss_weight
            + self.dice_2_loss(
                logits=logits[aux_mask, 2],
                labels=labels_aux[aux_mask],
            )
            * self.cfg.loss.dice_2_aux_loss_weight
            + self.loss(
                logits=logits[aux_mask, 0],
                labels=labels_aux[aux_mask],
            )
            * self.cfg.loss.dice_aux_loss_0_weight
            + self.bce_loss(
                logits=logits[:, 0],
                labels=labels[:, 0],
            )
            * self.cfg.loss.bce_loss_weight
            + self.bce_loss(
                logits=logits[aux_mask, 2],
                labels=labels_aux[:, 0][aux_mask],
            )
            * self.cfg.loss.bce_aux_loss_weight
        )

        if self.use_label_aux_min_max:
            loss += (
                self.loss(
                    logits=logits[aux_mask, 3],
                    labels=labels_aux_min[aux_mask],
                )
                * self.cfg.loss.dice_aux_min_max_loss_weight
            )
            loss += self.loss(
                logits=logits[aux_mask, 4],
                labels=labels_aux_max[aux_mask],
            ) * (
                self.cfg.loss.dice_aux_min_max_loss_weight
                + self.cfg.loss.dice_aux_max_loss_weight
            )

        return (
            logits[:, 0],
            loss,
        )

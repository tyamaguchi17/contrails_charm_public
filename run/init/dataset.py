from logging import getLogger
from typing import Dict

import pandas as pd
from omegaconf import DictConfig

from src.datasets.contrails import ContrailsDataset

logger = getLogger(__name__)


def init_datasets_from_config(cfg: DictConfig):
    if cfg.type == "contrails":
        datasets = get_contrails_dataset(
            num_folds=cfg.num_folds,
            test_fold=cfg.test_fold,
            val_fold=cfg.val_fold,
            seed=cfg.seed,
            num_records=cfg.num_records,
            phase=cfg.phase,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown dataset type: {cfg.type}")

    return datasets


def get_contrails_dataset(
    num_folds: int,
    test_fold: int,
    val_fold: int,
    seed: int = 2023,
    num_records: int = 0,
    phase: str = "train",
    cfg=None,
) -> Dict[str, ContrailsDataset]:
    logger.info("creating dataframe...")
    df = ContrailsDataset.create_dataframe(
        num_folds,
        seed,
        num_records,
        data_path=f"{cfg.data_path}/train",
    )

    test_df = ContrailsDataset.create_dataframe(
        num_folds,
        seed,
        num_records,
        data_path=f"{cfg.data_path}/validation",
    )

    train_df = df[(df["fold"] != val_fold) & (df["fold"] != test_fold)]
    val_df = df[df["fold"] == val_fold]

    if cfg.use_valid:
        train_df = pd.concat([train_df, test_df])

    if phase == "train":
        train_dataset = ContrailsDataset(train_df, phase="train", cfg=cfg)
        val_dataset = ContrailsDataset(val_df, phase="test", cfg=cfg)
        test_dataset = ContrailsDataset(test_df, phase="test", cfg=cfg)
    elif phase == "valid":
        train_dataset = ContrailsDataset(df, phase="train", cfg=cfg)
        val_dataset = ContrailsDataset(test_df, phase="test", cfg=cfg)
        test_dataset = ContrailsDataset(test_df, phase="test", cfg=cfg)
    elif phase == "test":
        train_dataset = ContrailsDataset(test_df, phase="test", cfg=cfg)
        val_dataset = ContrailsDataset(test_df, phase="test", cfg=cfg)
        test_dataset = ContrailsDataset(test_df, phase="test", cfg=cfg)
    elif phase == "all":
        train_dataset = ContrailsDataset(
            pd.concat([df, test_df]), phase="train", cfg=cfg
        )
        val_dataset = ContrailsDataset(test_df, phase="test", cfg=cfg)
        test_dataset = ContrailsDataset(test_df, phase="test", cfg=cfg)

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    return datasets

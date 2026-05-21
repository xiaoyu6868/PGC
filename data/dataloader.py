import argparse
import logging

import torch
from torch.utils.data import DataLoader

from .train_dataset import RealFakeDataset

logger = logging.getLogger(__name__)


def create_dataloader(opt: argparse.Namespace) -> DataLoader:

    dataset = RealFakeDataset(opt)
    logger.info("Dataset size: %d", len(dataset))

    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_threads,
        pin_memory=True,
        drop_last=True,
    )

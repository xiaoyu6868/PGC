"""Data pipeline package for the PGC project.

Re-exports the symbols that other parts of the codebase rely on:

- Constants and transforms from :mod:`data.transforms`
- Datasets from :mod:`data.train_dataset` and :mod:`data.eval_dataset`
- The training :func:`create_dataloader` factory from :mod:`data.dataloader`
"""

from .dataloader import create_dataloader
from .eval_dataset import UniversalFakeDetectDataset
from .train_dataset import RealFakeDataset
from .transforms import (
    MEAN,
    STD,
    RESIDUAL_MEAN,
    RESIDUAL_STD,
    AppendResidual,
    PadCenterCrop,
    PadRandomCrop,
    create_eval_transforms,
    create_train_transforms,
    get_list,
    recursively_read,
)

__all__ = [
    # Constants
    "MEAN",
    "STD",
    "RESIDUAL_MEAN",
    "RESIDUAL_STD",
    # Transforms
    "PadRandomCrop",
    "PadCenterCrop",
    "AppendResidual",
    "create_train_transforms",
    "create_eval_transforms",
    "recursively_read",
    "get_list",
    # Datasets
    "RealFakeDataset",
    "UniversalFakeDetectDataset",
    # Loaders
    "create_dataloader",
]

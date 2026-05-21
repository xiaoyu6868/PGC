import logging
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .transforms import get_list

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class _PerSubsetEvalDataset(Dataset):
    """Common implementation shared by the per-subset evaluation dataset.

    Subclasses only differ in the names of the *real* and *fake* leaf
    directories below ``root_dir/<subset>/``.
    """

    # Override these in subclasses.
    REAL_DIR_NAME: str = ""
    FAKE_DIR_NAME: str = ""
    BENCHMARK_NAME: str = "Eval"

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Each entry: ``(path, label, subset_name)``.
        self.samples: List[Tuple[str, int, str]] = []
        # Mapping ``subset_name -> [global indices]`` (post-sort).
        self.subset_indices: Dict[str, List[int]] = {}

        # First-level sub-directories under ``root_dir`` are subsets.
        subsets = [
            d
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]

        for subset in subsets:
            subset_path = os.path.join(root_dir, subset)
            real_list = get_list(subset_path, self.REAL_DIR_NAME)
            fake_list = get_list(subset_path, self.FAKE_DIR_NAME)

            logger.info(
                "Subset '%s': %d real images (%s), %d fake images (%s)",
                subset,
                len(real_list),
                self.REAL_DIR_NAME,
                len(fake_list),
                self.FAKE_DIR_NAME,
            )

            for p in real_list:
                self.samples.append((p, 0, subset))  # 0 = real
            for p in fake_list:
                self.samples.append((p, 1, subset))  # 1 = fake

        # Sort so iteration order is stable across runs / machines.
        self.samples.sort(key=lambda x: x[0])

        # Rebuild subset->indices mapping after the global sort.
        self.subset_indices = {}
        for idx, (_, _, subset) in enumerate(self.samples):
            self.subset_indices.setdefault(subset, []).append(idx)

        logger.info(
            "Loaded %d images from %d %s subsets",
            len(self.samples),
            len(subsets),
            self.BENCHMARK_NAME,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, subset = self.samples[idx]

        # Corrupt-image fallback (only allowed try/except per ROLE.md).
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:  # noqa: BLE001 - intentional broad catch
            logger.warning("Failed to load image %s: %s", path, e)
            img = Image.new("RGB", (224, 224))

        if self.transform is not None:
            img = self.transform(img)

        return img, label, subset

    # ---- helpers used by test.py ------------------------------------
    def get_subset_names(self) -> List[str]:
        """Names of every subset discovered under ``root_dir``."""
        return list(self.subset_indices.keys())

    def get_subset_indices(self, subset_name: str) -> List[int]:
        """Sample indices (within this Dataset) belonging to ``subset_name``."""
        return self.subset_indices.get(subset_name, [])


class UniversalFakeDetectDataset(_PerSubsetEvalDataset):
    """UniversalFakeDetect benchmark (paper Table 3).

    Layout::

        root/
          subset_a/
            0_real/   (label = 0)
            1_fake/   (label = 1)
          subset_b/
            0_real/
            1_fake/
          ...
    """

    REAL_DIR_NAME = "0_real"
    FAKE_DIR_NAME = "1_fake"
    BENCHMARK_NAME = "UniversalFakeDetect"

    def __init__(self, root_dir: str, transform=None):
        super().__init__(root_dir=root_dir, transform=transform)


# Re-export for convenience.
__all__ = [
    "UniversalFakeDetectDataset",
]


# Type-narrowing alias used by some external imports.
DatasetType = Optional[_PerSubsetEvalDataset]

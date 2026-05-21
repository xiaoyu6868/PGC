import argparse
import logging
import random
from typing import Dict, List

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .transforms import MEAN, STD, create_train_transforms, get_list

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class RealFakeDataset(Dataset):


    def __init__(self, opt: argparse.Namespace):
        self.opt = opt
        self.data_list: List[Dict] = []
        self.dataset_sources: Dict[str, int] = {}

        if not opt.real_image_dir:
            raise ValueError(
                "--real_image_dir must be provided with at least one "
                "directory (use space-separated paths for multiple sources)."
            )
        if not opt.fake_image_dir:
            raise ValueError(
                "--fake_image_dir must be provided with at least one "
                "directory (use space-separated paths for multiple sources)."
            )

        primary_samples = self._load_dataset(
            real_image_dirs=opt.real_image_dir,
            fake_image_dirs=opt.fake_image_dir,
            source_name="primary",
        )
        self.data_list.extend(primary_samples)

        for source, count in self.dataset_sources.items():
            logger.info("Loaded %d samples from %s dataset", count, source)

        random.shuffle(self.data_list)

        real_count = sum(1 for s in self.data_list if s["label"] == 0)
        fake_count = sum(1 for s in self.data_list if s["label"] == 1)
        total_count = len(self.data_list)
        real_ratio = real_count / total_count if total_count > 0 else 0
        fake_ratio = fake_count / total_count if total_count > 0 else 0

        logger.info("=" * 60)
        logger.info("DATASET CLASS DISTRIBUTION:")
        logger.info("  Total samples: %d", total_count)
        logger.info(
            "  Real (label=0): %d (%.2f%%)", real_count, real_ratio * 100
        )
        logger.info(
            "  Fake (label=1): %d (%.2f%%)", fake_count, fake_ratio * 100
        )
        logger.info(
            "  Real/Fake ratio: %.3f",
            real_count / fake_count if fake_count > 0 else float("inf"),
        )
        logger.info("=" * 60)

        stat_from = "dino"
        logger.info("Mean and std stats are from: %s", stat_from)

        self.transform = create_train_transforms(
            image_size=opt.cropSize,
            mean=MEAN[stat_from],
            std=STD[stat_from],
            is_crop=True,
        )

    # ------------------------------------------------------------------
    def _load_dataset(
        self,
        real_image_dirs: List[str],
        fake_image_dirs: List[str],
        source_name: str,
    ) -> List[Dict]:
        samples: List[Dict] = []
        self.dataset_sources[source_name] = 0

        # Real images (potentially multiple cohorts in ``real_image_dirs``).
        for real_dir in real_image_dirs:
            logger.info("Scanning real images from: %s", real_dir)
            real_list = get_list(real_dir)
            logger.info(
                "Loaded %d real images from %s", len(real_list), real_dir
            )

            for real_path in tqdm(
                real_list, desc=f"Loading {source_name} real ({real_dir})"
            ):
                samples.append(
                    {
                        "path": real_path,
                        "label": 0,  # 0 = real
                        "source": source_name,
                    }
                )
                self.dataset_sources[source_name] += 1

        # Fake images (potentially multiple generators in fake_image_dirs).
        for fake_dir in fake_image_dirs:
            logger.info("Scanning fake images from: %s", fake_dir)
            fake_list = get_list(fake_dir)
            logger.info(
                "Loaded %d fake images from %s", len(fake_list), fake_dir
            )

            for fake_path in tqdm(
                fake_list, desc=f"Loading {source_name} fake ({fake_dir})"
            ):
                samples.append(
                    {
                        "path": fake_path,
                        "label": 1,  # 1 = fake
                        "source": source_name,
                    }
                )
                self.dataset_sources[source_name] += 1

        logger.info(
            "Total samples loaded from %s: %d (real + fake across %d real "
            "dir(s) and %d fake dir(s))",
            source_name,
            len(samples),
            len(real_image_dirs),
            len(fake_image_dirs),
        )
        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data_list[idx]

        try:
            img = Image.open(sample["path"]).convert("RGB")
        except Exception as e:  
            logger.warning("Failed to load image %s: %s", sample["path"], e)
            img = Image.new("RGB", (self.opt.cropSize, self.opt.cropSize))

        img_tensor: torch.Tensor = self.transform(img)

        return {
            "image": img_tensor,
            "label": sample["label"],
            "source": sample["source"],
        }

import logging
import os
from typing import List

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

logger = logging.getLogger(__name__)

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
    "dino": [0.485, 0.456, 0.406],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
    "dino": [0.229, 0.224, 0.225],
}

RESIDUAL_MEAN: List[float] = [0.0, 0.0, 0.0]
RESIDUAL_STD: List[float] = [0.5, 0.5, 0.5]


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def recursively_read(
    rootdir: str,
    must_contain: str,
    exts=("png", "jpg", "jpeg", "JPEG", "bmp", "webp"),
) -> List[str]:
    """Recursively scan ``rootdir`` for image files.

    Args:
        rootdir: Root of the search.
        must_contain: Optional substring filter applied to the full path.
        exts: Allowed file extensions (case-insensitive on the lowercase form).

    Returns:
        List of absolute / joined paths to the matching files.
    """
    out: List[str] = []
    exts_lc = {e.lower() for e in exts}
    for r, _, files in os.walk(rootdir, followlinks=True):
        for file in files:
            # Defensive split: files without an extension are skipped instead
            # of raising IndexError.
            parts = file.split(".")
            if len(parts) < 2:
                continue
            ext = parts[-1].lower()
            full = os.path.join(r, file)
            if ext in exts_lc and must_contain in full:
                out.append(full)
    return out


def get_list(path: str, must_contain: str = "") -> List[str]:
    """Convenience wrapper around :func:`recursively_read` with no extras."""
    return recursively_read(path, must_contain)


# ---------------------------------------------------------------------------
# Geometry transforms
# ---------------------------------------------------------------------------
class PadRandomCrop:
    """Pad the image to ``size`` if needed, then take a random crop.

    Used at training time to preserve original pixel patterns (no resize
    distortion that would erase the per-pixel quantization residual).
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)

        if pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2,
            )
            img = F.pad(img, padding, fill=0)

        return transforms.RandomCrop(self.size)(img)


class PadCenterCrop:
    """Pad the image to ``size`` if needed, then take a deterministic centre crop.

    Mirrors :class:`PadRandomCrop` exactly (same padding strategy) so train and
    test pipelines differ only in the crop position.  Critical for
    forensic/PRNU-style features that can be wiped by ``Resize``.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)

        if pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2,
            )
            img = F.pad(img, padding, fill=0)

        return transforms.CenterCrop(self.size)(img)


# ---------------------------------------------------------------------------
# Residual transform
# ---------------------------------------------------------------------------
class AppendResidual:
    """Append the quantization residual as 3 extra channels.

    The residual is computed on the RGB tensor in ``[0, 1]`` (BEFORE
    normalization).  Normalization is then applied to the resulting 6-channel
    tensor, with different mean/std for the RGB block and the residual block.

    Output: ``[6, H, W]`` where channels ``0..2`` are RGB and channels
    ``3..5`` are the residual.

    The residual extractor used here is exactly the one used inside the
    model (``models.encoder.residual_extractor``).  We import it lazily in
    ``__init__`` to keep this module importable when the model package is
    not yet on the import path (e.g. during unit tests).
    """

    def __init__(self):
        # Lazy import so this module can still be imported when the
        # ``models`` package is absent (used by light-weight smoke tests).
        from models.encoder.residual_extractor import get_residual_extractor

        self.residual_extractor = get_residual_extractor()

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Compute the residual and concatenate with the RGB tensor.

        Args:
            img_tensor: ``[3, H, W]`` RGB tensor in ``[0, 1]``.

        Returns:
            ``[6, H, W]`` tensor: ``[RGB ∥ residual]``.
        """
        assert img_tensor.size(0) == 3, (
            f"Expected 3 channels, got {img_tensor.size(0)}"
        )
        assert img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0, (
            f"Input must be in [0, 1], got "
            f"range [{img_tensor.min():.3f}, {img_tensor.max():.3f}]"
        )

        residual = self.residual_extractor(img_tensor.unsqueeze(0)).squeeze(0)
        residual = residual.to(dtype=img_tensor.dtype)
        return torch.cat([img_tensor, residual], dim=0)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(input=[0,1])"


# ---------------------------------------------------------------------------
# Compose helpers
# ---------------------------------------------------------------------------
def create_train_transforms(
    image_size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    is_crop: bool = True,
) -> transforms.Compose:

    if is_crop:
        resize_func = PadRandomCrop(image_size)
    else:
        logger.info("Using Resize to %d x %d", image_size, image_size)
        resize_func = transforms.Resize((image_size, image_size))

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.15,
                        hue=0.05,
                    )
                ],
                p=0.2,
            ),
            resize_func,
            transforms.ToTensor(),
            AppendResidual(),
            transforms.Normalize(
                mean=list(mean) + RESIDUAL_MEAN,
                std=list(std) + RESIDUAL_STD,
            ),
        ]
    )


def create_eval_transforms(
    image_size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> transforms.Compose:

    logger.info(
        "[Eval] Using deterministic PadCenterCrop to %d x %d",
        image_size,
        image_size,
    )

    return transforms.Compose(
        [
            PadCenterCrop(image_size),
            transforms.ToTensor(),
            AppendResidual(),
            transforms.Normalize(
                mean=list(mean) + RESIDUAL_MEAN,
                std=list(std) + RESIDUAL_STD,
            ),
        ]
    )

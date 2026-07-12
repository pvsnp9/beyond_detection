from __future__ import annotations

import random
from typing import Optional

from PIL import Image


def normalize_image(image) -> Image.Image:
    if image is None:
        raise ValueError("image is None")
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(image)
        except Exception as exc:
            shape = getattr(image, "shape", None)
            raise ValueError(
                f"invalid image type for PIL conversion: {type(image)} shape={shape}"
            ) from exc
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resize_cap(
    pil: Image.Image,
    max_side: int = 1536,
    max_pixels: int = 6_000_000,
) -> Image.Image:
    pil = pil.convert("RGB")
    w, h = pil.size
    scale_side = min(1.0, max_side / float(max(w, h)))
    area = w * h
    scale_area = (max_pixels / float(area)) ** 0.5 if area > max_pixels else 1.0
    scale = min(scale_side, scale_area)
    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pil = pil.resize((nw, nh), resample=Image.BICUBIC)
    return pil


def random_crop_fraction(
    pil: Image.Image,
    frac_min: float,
    frac_max: float,
    rng: Optional[random.Random] = None,
) -> Image.Image:
    """mDPO image corruption: random crop retaining U(frac_min, frac_max) of the
    area, resized back to the original dims so processor vision grids and image
    placeholder token counts stay identical to the uncorrupted image."""
    rng = rng or random
    w, h = pil.size
    frac = rng.uniform(frac_min, frac_max)
    side_scale = max(frac, 1e-4) ** 0.5
    cw = max(1, int(w * side_scale))
    ch = max(1, int(h * side_scale))
    left = rng.randint(0, w - cw)
    top = rng.randint(0, h - ch)
    crop = pil.crop((left, top, left + cw, top + ch))
    return crop.resize((w, h), resample=Image.BICUBIC)

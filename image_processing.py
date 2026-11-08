"""Image validation, tiling, and classification-map assembly."""

from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from config import CLASS_COLORS, MAX_IMAGE_PIXELS, SUPPORTED_IMAGE_FORMATS


@dataclass(frozen=True)
class ImageTile:
    index: int
    x: int
    y: int
    width: int
    height: int
    pixels: np.ndarray


def decode_uploaded_image(data: bytes, filename: str) -> np.ndarray:
    """Validate common raster image content and return its first frame as RGB."""
    if not data:
        raise ValueError("No image data was pasted.")

    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in SUPPORTED_IMAGE_FORMATS:
                supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                raise ValueError(
                    f"Unsupported image format {image.format!r}. "
                    f"Supported raster formats: {supported}."
                )
            image.verify()
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image dimensions must be positive and no larger than "
                    f"{MAX_IMAGE_PIXELS:,} pixels total."
                )
            image.seek(0)
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Could not decode the pasted image: {exc}") from exc


def create_tiles(image: np.ndarray, tile_size: int) -> list[ImageTile]:
    """Split an RGB image into same-sized, edge-padded tiles."""
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a non-empty RGB image.")
    if tile_size <= 0:
        raise ValueError("Tile size must be a positive integer.")

    image_height, image_width = image.shape[:2]
    tiles: list[ImageTile] = []
    for y in range(0, image_height, tile_size):
        for x in range(0, image_width, tile_size):
            height = min(tile_size, image_height - y)
            width = min(tile_size, image_width - x)
            crop = image[y:y + height, x:x + width]
            padding = ((0, tile_size - height), (0, tile_size - width), (0, 0))
            padded = np.pad(crop, padding, mode="edge")
            tiles.append(
                ImageTile(
                    index=len(tiles),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    pixels=np.ascontiguousarray(padded),
                )
            )
    return tiles


def combine_masks(
    image_shape: tuple[int, ...],
    tiles: list[ImageTile],
    masks: list[np.ndarray],
) -> np.ndarray:
    """Crop padded predictions and stitch them into the full-size label map."""
    if len(tiles) != len(masks):
        raise ValueError("The number of prediction masks must match the number of tiles.")
    result = np.zeros(image_shape[:2], dtype=np.uint8)
    for tile, mask in zip(tiles, masks):
        if mask.ndim != 2 or mask.shape[0] < tile.height or mask.shape[1] < tile.width:
            raise ValueError(f"Invalid prediction shape for tile {tile.index}.")
        result[
            tile.y:tile.y + tile.height,
            tile.x:tile.x + tile.width,
        ] = mask[:tile.height, :tile.width]
    return result


def render_classification(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.55,
    visible_classes: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a palette mask and a color overlay on the original image."""
    palette = np.asarray(CLASS_COLORS, dtype=np.uint8)
    if mask.size and int(mask.max()) >= len(palette):
        raise ValueError("Prediction contains an unknown class ID.")
    color_map = palette[mask]
    if visible_classes is not None:
        if any(class_id < 0 or class_id >= len(palette) for class_id in visible_classes):
            raise ValueError("Visible classes must be valid class IDs.")
        hidden_pixels = ~np.isin(mask, list(visible_classes))
        color_map[hidden_pixels] = (35, 42, 52)
    overlay = cv2.addWeighted(image, 1.0 - alpha, color_map, alpha, 0)
    return color_map, overlay


def class_percentages(mask: np.ndarray) -> list[float]:
    """Calculate the share of pixels assigned to each of the four classes."""
    if mask.size == 0:
        raise ValueError("Cannot calculate statistics for an empty mask.")
    counts = np.bincount(mask.ravel(), minlength=len(CLASS_COLORS))[:len(CLASS_COLORS)]
    return (counts / mask.size * 100.0).tolist()


def compare_landcover_masks(
    before: np.ndarray, after: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all-change, water-gain, and water-loss masks for aligned labels."""
    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("Before and after class maps must be two-dimensional.")
    if before.shape != after.shape:
        raise ValueError("Before and after images must have identical dimensions.")
    if before.size == 0:
        raise ValueError("Cannot compare empty class maps.")

    changed = before != after
    before_is_water = before == 2
    after_is_water = after == 2
    water_gain = ~before_is_water & after_is_water
    water_loss = before_is_water & ~after_is_water
    return changed, water_gain, water_loss


def class_transition_counts(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Count per-pixel transitions; rows are before IDs and columns after IDs."""
    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("Before and after class maps must be two-dimensional.")
    if before.shape != after.shape:
        raise ValueError("Before and after images must have identical dimensions.")
    class_count = len(CLASS_COLORS)
    if (
        before.size == 0
        or int(before.max()) >= class_count
        or int(after.max()) >= class_count
        or int(before.min()) < 0
        or int(after.min()) < 0
    ):
        raise ValueError("Class maps must contain valid class IDs.")
    return np.bincount(
        before.ravel().astype(np.int64) * class_count
        + after.ravel().astype(np.int64),
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)

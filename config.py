"""Shared configuration for the satellite image classifier."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "classifier.pt"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CLASS_NAMES = ("Land", "Vegetation", "Water", "Urban")
CLASS_COLORS = (
    (255, 220, 0),   # Land: yellow
    (34, 170, 72),   # Vegetation: green
    (35, 115, 235),  # Water: blue
    (230, 55, 55),   # Urban: red
)

DEFAULT_TILE_SIZE = 256
DEFAULT_BATCH_SIZE = 16
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"}

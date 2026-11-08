"""Four-class tile inference with an explicit, non-trained demo fallback."""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from config import CLASS_NAMES, MODEL_PATH


class ModelLoadError(RuntimeError):
    """Raised when an existing trained model cannot be loaded."""


class SatelliteClassifier:
    """Run one TorchScript segmentation model or the labeled heuristic demo."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        device: torch.device | str | None = None,
    ) -> None:
        self.cuda_available = torch.cuda.is_available()
        requested_device = torch.device(
            device if device is not None
            else ("cuda" if self.cuda_available else "cpu")
        )
        if requested_device.type == "cuda" and not self.cuda_available:
            raise RuntimeError("CUDA was requested but is unavailable.")
        self.device = requested_device
        self.cuda_fallback = False
        self.model = None
        self.demo_mode = not model_path.is_file()
        self.mode_description = "Heuristic demonstration (not a trained model)"

        if not self.demo_mode:
            try:
                self.model = torch.jit.load(str(model_path), map_location="cpu")
                self.model.eval()
                if self.device.type == "cuda":
                    try:
                        self.model.to(self.device)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        self.device = torch.device("cpu")
                        self.cuda_fallback = True
            except (OSError, RuntimeError) as exc:
                raise ModelLoadError(
                    f"Could not load the trained model at {model_path}: {exc}"
                ) from exc
            self.mode_description = f"Trained TorchScript model: {model_path.name}"

    def move_to_cpu(self) -> None:
        """Move the same model and subsequent inference to CPU after CUDA OOM."""
        if self.model is not None:
            self.model.to(torch.device("cpu"))
        self.device = torch.device("cpu")
        self.cuda_fallback = True

    @torch.inference_mode()
    def predict_batch(self, tiles: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Classify RGB uint8 tiles and return one integer mask per tile."""
        if not tiles:
            return []

        batch = np.stack(tiles).astype(np.float32, copy=False) / 255.0
        inputs = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        inputs = inputs.to(self.device)

        if self.demo_mode:
            logits = self._demo_logits(inputs)
        else:
            logits = self.model(inputs)
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("The trained model must return a logits tensor.")

        if logits.ndim != 4 or logits.shape[1] != len(CLASS_NAMES):
            raise RuntimeError(
                "The model output must have shape [batch, 4, height, width]."
            )
        if logits.shape[0] != len(tiles):
            raise RuntimeError("The model returned a different batch size than requested.")
        if logits.shape[2:] != inputs.shape[2:]:
            raise RuntimeError(
                "The model must return logits at the same height and width as its input."
            )

        masks = logits.argmax(dim=1).to("cpu").numpy().astype(np.uint8, copy=False)
        return [mask for mask in masks]

    @staticmethod
    def _demo_logits(inputs: torch.Tensor) -> torch.Tensor:
        """Build rough RGB-index scores; these are not learned predictions."""
        red, green, blue = inputs.unbind(dim=1)
        gray = (red + green + blue) / 3.0
        brightness = red + green + blue + 1e-6
        green_index = (2.0 * green - red - blue) / brightness
        blue_index = (blue - red) / brightness
        red_index = (red - blue) / brightness
        gray_for_texture = functional.pad(
            gray.unsqueeze(1), (2, 2, 2, 2), mode="replicate"
        )
        local_mean = functional.avg_pool2d(
            gray_for_texture, kernel_size=5, stride=1
        ).squeeze(1)
        local_square_mean = functional.avg_pool2d(
            gray_for_texture.square(), kernel_size=5, stride=1
        ).squeeze(1)
        texture = (local_square_mean - local_mean.square()).clamp_min(0).sqrt()

        vegetation = 0.10 + 1.5 * green_index
        water = 0.10 + 1.5 * blue_index
        land = (
            0.12
            + 1.15 * red_index.clamp_min(0)
            + 0.10 * (-green_index).clamp_min(0)
        )
        urban = (
            0.16
            + 1.5 * texture
            - 0.7 * green_index.abs()
            - 0.7 * blue_index.abs()
            - 0.3 * red_index.abs()
        )
        return torch.stack((land, vegetation, water, urban), dim=1)

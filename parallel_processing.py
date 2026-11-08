"""Sequential and batched inference paths over the same image tiles/model."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from classifier import SatelliteClassifier
from image_processing import ImageTile


ProgressCallback = Callable[[int, int], None]


@dataclass
class ProcessingResult:
    masks: list[np.ndarray]
    execution_device: str


def _classify_chunk(
    classifier: SatelliteClassifier, tiles: list[ImageTile]
) -> list[np.ndarray]:
    masks = classifier.predict_batch([tile.pixels for tile in tiles])
    return [
        mask[:tile.height, :tile.width]
        for tile, mask in zip(tiles, masks)
    ]


def _handle_cuda_oom(classifier: SatelliteClassifier, requested_batch: int) -> None:
    if classifier.device.type != "cuda":
        raise
    torch.cuda.empty_cache()
    if requested_batch <= 1:
        classifier.move_to_cpu()


def run_sequential(
    classifier: SatelliteClassifier,
    tiles: list[ImageTile],
    progress: ProgressCallback | None = None,
) -> ProcessingResult:
    """Infer one tile per model call, preserving the original tile order."""
    masks: list[np.ndarray] = []
    used_cuda = classifier.device.type == "cuda"
    for index, tile in enumerate(tiles, start=1):
        while True:
            try:
                masks.extend(_classify_chunk(classifier, [tile]))
                break
            except torch.cuda.OutOfMemoryError:
                _handle_cuda_oom(classifier, 1)
        if progress is not None:
            progress(index, len(tiles))

    device = "CUDA" if used_cuda and classifier.device.type == "cuda" else "CPU"
    if used_cuda and classifier.device.type == "cpu":
        device = "CPU (CUDA out-of-memory fallback)"
    elif classifier.cuda_fallback:
        device = "CPU after CUDA out-of-memory fallback"
    return ProcessingResult(masks, device)


def run_batched(
    classifier: SatelliteClassifier,
    tiles: list[ImageTile],
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> ProcessingResult:
    """Infer batches of independent tiles on CUDA or the CPU fallback."""
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    masks: list[np.ndarray] = []
    used_cuda = classifier.device.type == "cuda"
    cursor = 0
    current_batch_size = batch_size
    while cursor < len(tiles):
        chunk_size = min(current_batch_size, len(tiles) - cursor)
        chunk = tiles[cursor:cursor + chunk_size]
        try:
            masks.extend(_classify_chunk(classifier, chunk))
        except torch.cuda.OutOfMemoryError:
            _handle_cuda_oom(classifier, chunk_size)
            if classifier.device.type == "cpu":
                current_batch_size = 1
            else:
                current_batch_size = max(1, chunk_size // 2)
            continue

        cursor += chunk_size
        if progress is not None:
            progress(cursor, len(tiles))

    device = "CUDA" if used_cuda and classifier.device.type == "cuda" else "CPU"
    if used_cuda and classifier.device.type == "cpu":
        device = "CUDA with CPU fallback after out-of-memory"
    elif classifier.cuda_fallback:
        device = "CPU after CUDA out-of-memory fallback"
    elif classifier.device.type == "cpu":
        device = "CPU batched fallback (CUDA unavailable)"
    return ProcessingResult(masks, device)

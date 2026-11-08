"""Timing helpers and fair sequential-versus-batched performance metrics."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import torch

from parallel_processing import ProcessingResult


@dataclass
class TimedResult:
    result: ProcessingResult
    seconds: float


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_processing(
    operation: Callable[[], ProcessingResult],
) -> TimedResult:
    """Measure wall-clock inference time, including asynchronous CUDA work."""
    _synchronize_cuda()
    started = perf_counter()
    result = operation()
    _synchronize_cuda()
    return TimedResult(result=result, seconds=perf_counter() - started)


def calculate_speedup(sequential_seconds: float, batched_seconds: float) -> float | None:
    """Return sequential/batched speedup, or None when timing is unusable."""
    if sequential_seconds <= 0 or batched_seconds <= 0:
        return None
    return sequential_seconds / batched_seconds

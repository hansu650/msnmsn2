from __future__ import annotations

import io
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, TypeVar

import numpy as np
import torch


T = TypeVar("T")


@dataclass(frozen=True)
class TimingRecord:
    phase: str
    device: str
    wall_seconds: float

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)


class PhaseTimer:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        self._clock = clock
        self._synchronize_override = synchronize
        self.records: list[TimingRecord] = []

    def _sync(self, device: str) -> None:
        if not device.startswith("cuda"):
            return
        if self._synchronize_override is not None:
            self._synchronize_override()
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA timing requested but CUDA is unavailable")
        torch.cuda.synchronize(torch.device(device))

    @contextmanager
    def phase(self, name: str, *, device: str) -> Iterator[None]:
        if not name:
            raise ValueError("Timing phase name must be non-empty")
        normalized_device = str(device).lower()
        if not (normalized_device == "cpu" or normalized_device.startswith("cuda")):
            raise ValueError("Timing device must be cpu or cuda")
        self._sync(normalized_device)
        started = self._clock()
        try:
            yield
        finally:
            self._sync(normalized_device)
            elapsed = self._clock() - started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError("Invalid elapsed time")
            self.records.append(
                TimingRecord(name, normalized_device, float(elapsed))
            )

    def time_call(
        self,
        name: str,
        function: Callable[..., T],
        *args: Any,
        device: str,
        **kwargs: Any,
    ) -> T:
        with self.phase(name, device=device):
            return function(*args, **kwargs)

    def as_dicts(self) -> list[dict[str, str | float]]:
        return [record.as_dict() for record in self.records]


def warm_inference(
    function: Callable[[], Any],
    *,
    device: str,
    warmup: int,
    repetitions: int,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if warmup < 0 or repetitions <= 0:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    timer = PhaseTimer(clock=clock, synchronize=synchronize)
    for _ in range(warmup):
        function()
    timer._sync(device)
    samples_ms: list[float] = []
    for _ in range(repetitions):
        timer._sync(device)
        started = clock()
        function()
        timer._sync(device)
        elapsed = (clock() - started) * 1000.0
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("Invalid inference timing")
        samples_ms.append(float(elapsed))
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "device": str(device).lower(),
        "warmup": int(warmup),
        "repetitions": int(repetitions),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "samples_ms": samples_ms,
    }


def serialized_state_bytes(state: Any) -> int:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return int(buffer.tell())


REQUIRED_PHASES = {
    "apn_load",
    "feature_extraction",
    "cache_read",
    "cache_write",
    "slrh_solve",
    "cfg_solve",
    "validation_selection",
    "serialization",
    "warm_inference",
}


def validate_timing_records(
    records: list[dict[str, Any]],
    *,
    required_phases: set[str] | None = None,
) -> None:
    expected = REQUIRED_PHASES if required_phases is None else required_phases
    observed = {str(record.get("phase")) for record in records}
    missing = expected - observed
    if missing:
        raise ValueError(f"Missing timing phases: {sorted(missing)}")
    for record in records:
        if record.get("device") not in {"cpu", "cuda", "cuda:0"} and not str(
            record.get("device")
        ).startswith("cuda:"):
            raise ValueError("Invalid timing device label")
        elapsed = float(record.get("wall_seconds", -1))
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid timing duration")

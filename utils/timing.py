from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Generator


@dataclass
class TimerResult:
    seconds: float

    @property
    def ms(self) -> float:
        return self.seconds * 1000.0


class Timer:
    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed: float = 0.0

    def start(self) -> None:
        if self._start is not None:
            raise RuntimeError("Timer already running")
        self._start = time.perf_counter()

    def stop(self) -> TimerResult:
        if self._start is None:
            raise RuntimeError("Timer not running")
        elapsed = time.perf_counter() - self._start
        self._elapsed += elapsed
        self._start = None
        return TimerResult(elapsed)

    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return self._elapsed + (time.perf_counter() - self._start)
        return self._elapsed


@contextlib.contextmanager
def time_block() -> Generator[Timer, None, None]:
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


import threading
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Optional

T = TypeVar("T")


class ServiceUnavailableError(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, compare=False, repr=False)

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        with self._lock:
            if self.state == "OPEN":
                if self.last_failure_time is not None and (
                    time.time() - self.last_failure_time > self.recovery_timeout
                ):
                    self.state = "HALF_OPEN"
                else:
                    raise ServiceUnavailableError("Circuit breaker OPEN")

        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
            return result
        except Exception:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
            raise

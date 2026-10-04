"""Training: deadline."""

import signal
import threading
import time


class CollectionBudgetExceeded(TimeoutError):
    pass


class CollectionDeadline:
    def __init__(self, maximum_millis):
        self.expires = time.monotonic() + maximum_millis / 1000.0
        self._handler = None

    def remaining(self):
        value = self.expires - time.monotonic()
        if value <= 0:
            raise CollectionBudgetExceeded("collection wall-clock budget exhausted")
        return value

    def check(self):
        self.remaining()

    def arm(self, *, required):
        supported = (
            hasattr(signal, "setitimer")
            and hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
        )
        if not supported:
            if required:
                raise ValueError("real collection requires a main-thread POSIX deadline guard")
            return
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise ValueError("collection cannot replace another active process timer")
        remaining = self.remaining()
        self._handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)

    def _expired(self, *_):
        raise CollectionBudgetExceeded("collection wall-clock budget exhausted")

    def disarm(self):
        if self._handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self._handler)
            self._handler = None

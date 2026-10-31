"""Small decorators used by the matching run."""

import time
from functools import wraps

TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
}


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return type(exc).__name__ in TRANSIENT_ERROR_NAMES


def retry(attempts: int = 3, backoff: float = 0.5):
    """Retry a call when the provider reports a transient failure.

    Timeouts, connection and OS errors, and provider errors named in
    :data:`TRANSIENT_ERROR_NAMES` are retried with exponential backoff. Other
    exceptions, and the last failure, are raised.

    Args:
        attempts (int): Total attempts, including the first call.
        backoff (float): Seconds to wait before the first retry; doubled
            after each retry.

    Returns:
        Callable: A decorator.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            delay = backoff
            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if not _is_transient(exc) or attempt == attempts - 1:
                        raise
                    time.sleep(delay)
                    delay *= 2
        return wrapper

    return decorator


def timed(listener, bucket: str):
    """Record how long a call takes on a metrics listener.

    The time is recorded even when the call raises.

    Args:
        listener (MetricsListener): Receives
            :meth:`~kroma.observers.MetricsListener.add_timing`.
        bucket (str): Timing bucket name.

    Returns:
        Callable: A decorator.

    Example:
        >>> from kroma.observers import MetricsListener
        >>> listener = MetricsListener()
        >>> timed(listener, "work")(sum)([1, 2])
        3
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                listener.add_timing(bucket, time.perf_counter() - started)
        return wrapper

    return decorator

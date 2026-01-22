import time
import functools
import logging
import inspect
from typing import Any


logger: logging.Logger = logging.getLogger(__name__)


def measure_time() -> Any:
    def decorator(func):
        is_async: bool = inspect.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start: float = time.perf_counter()
                result: Any = await func(*args, **kwargs)
                elapsed: float = time.perf_counter() - start
                logger.info(f"{func.__name__} took {elapsed:.4f}s")
                return result

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start: float = time.perf_counter()
                result: Any = func(*args, **kwargs)
                elapsed: float = time.perf_counter() - start
                logger.info(f"{func.__name__} took {elapsed:.4f}s")
                return result

            return sync_wrapper

    return decorator

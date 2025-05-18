import time
from contextlib import contextmanager

@contextmanager
def elapsed_time(mode: str = "seconds", precision: int = 4):
    """
    Context manager to measure elapsed time with flexible formatting.

    Args:
        mode (str): Output format mode.
            - "seconds": Display total time in seconds (default).
            - "minutes": Display time as minutes and seconds.
        precision (int): Number of decimal places for seconds (default 4).

    Usage:
        with elapsed_time():  # default seconds format, 4 decimal places
            ...
        with elapsed_time(mode="minutes", precision=2):
            ...
    """
    start = time.time()
    yield
    elapsed = time.time() - start

    if mode == "minutes":
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"Elapsed time: {minutes} min {seconds:.{precision}f} sec")
    else:
        # Default to seconds format
        print(f"Elapsed time: {elapsed:.{precision}f} seconds")

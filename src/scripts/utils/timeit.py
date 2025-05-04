import logging
import time


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(
            f"{func.__name__} duration: {duration:.2f} seconds or {duration / 60:.2f} minutes"
        )
        return result

    return wrapper

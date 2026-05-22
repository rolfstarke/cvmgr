import functools
import logging
import time


def _format_elapsed(seconds):
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    elif m > 0:
        return f"{m}m {s:05.2f}s"
    else:
        return f"{s:.2f}s"


def _get_dataset(args, kwargs):
    if "dataset_name" in kwargs:
        return kwargs["dataset_name"]
    if "dataset" in kwargs:
        dataset = kwargs["dataset"]
        return getattr(dataset, "name", str(dataset))
    if args:
        first = args[0]
        return getattr(first, "name", str(first))
    return "n/a"


def util_log(util, success_check=None, success_text=None):
    check = success_check or (lambda result, args, kwargs: result is not False and result is not None)
    detail = success_text or (lambda result, args, kwargs: "result_check")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("cvmgr")
            started = time.time()
            dataset = _get_dataset(args, kwargs)
            logger.info(f"util {util} started with {dataset}")
            try:
                result = func(*args, **kwargs)
                if check(result, args, kwargs):
                    elapsed = time.time() - started
                    logger.info(f"util {util} [{dataset}] successful with {detail(result, args, kwargs)} in {_format_elapsed(elapsed)}")
                else:
                    logger.info(f"util {util} failed")
                return result
            except Exception:
                logger.info(f"util {util} failed")
                raise

        return wrapper

    return decorator
import functools
import logging
import pathlib
import time


class _ErrorMessageOnlyFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if record.levelno >= logging.ERROR and " - " in msg:
            prefix, body = msg.split(" - ", 1)
            return f"{prefix} - \033[31m{body}\033[0m"
        return msg


def configure_cvmgr_logging(full_log="logs/full.log", selective_log="logs/selective.log", level=logging.INFO):
    log_dir = pathlib.Path(full_log).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=full_log, level=level, format=fmt, datefmt=datefmt)

    logger = logging.getLogger("cvmgr")
    logger.setLevel(level)
    selective_path = str(pathlib.Path(selective_log).resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == selective_path:
            return logger

    handler = logging.FileHandler(selective_log)
    handler.setLevel(level)
    handler.setFormatter(_ErrorMessageOnlyFormatter(fmt, datefmt))
    logger.addHandler(handler)
    return logger


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
                    message = f"util {util} failed"
                    logger.error(message)
                return result
            except Exception:
                message = f"util {util} failed"
                logger.error(message)
                raise

        return wrapper

    return decorator
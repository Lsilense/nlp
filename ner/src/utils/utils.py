# src/utils.py
import logging
from pathlib import Path


def init_logger(log_file=None, level=logging.INFO):
    logger = logging.getLogger("myapp")

    # 如果已经配置过（已有 handlers），直接返回，避免重复
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")

    # 添加控制台 handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 添加文件 handler（可选）
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

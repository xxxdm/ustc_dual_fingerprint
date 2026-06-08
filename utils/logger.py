
##训练日志，训练过程中的信息同时显示在终端，并保存到文件，方便后续查看和画图

import logging
import os

def get_logger(log_dir="logs", log_file="train.log"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        fh = logging.FileHandler(os.path.join(log_dir, log_file), encoding="utf-8")
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
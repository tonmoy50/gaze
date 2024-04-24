import logging
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))


def init_logger():
    logger = logging.getLogger(__name__)
    if os.path.exists(os.path.join(ROOT, "logs")) is False:
        os.mkdir(os.path.join(ROOT, "logs"))
    logfile_name = os.path.join(
        ROOT, "logs", f"""{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}_logs.txt"""
    )
    logging.basicConfig(filename=logfile_name, encoding="utf-8", level=logging.DEBUG)
    return logger

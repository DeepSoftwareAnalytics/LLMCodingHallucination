import logging
import sys
from pathlib import Path

def init_log(file=None, level=logging.INFO):
    format = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.getLogger().setLevel(level)
    formatter = logging.Formatter(format)
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(formatter)
    stderr.setLevel(level)
    logging.getLogger().handlers = []
    logging.getLogger().addHandler(stderr)

    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=file, mode="w", encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logging.getLogger().addHandler(file_handler)
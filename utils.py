import colorlog
import logging
from omegaconf import OmegaConf, ValidationError
from typing import List

from conf.algorithms import AlgorithmConfig

LOG_LEVEL = logging.DEBUG


def get_logger(name):
    bold_seq = "\033[1m"
    colorlog_format = (
        f"{bold_seq}"
        "%(log_color)s"
        "%(asctime)s | %(name)s.%(funcName)s | "
        "%(levelname)s:%(reset)s %(message)s"
    )
    colorlog.basicConfig(
        format=colorlog_format, level=logging.DEBUG, datefmt="%d/%m/%Y %H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    return logger


def load_algo_config(algo_names: List[str], conf_path: str):
    algo_configs = []
    for algo_name in algo_names:
        schema = OmegaConf.structured(AlgorithmConfig)
        conf = OmegaConf.load(f"{conf_path}/algorithms/{algo_name}.yaml")
        algo_configs.append(OmegaConf.merge(schema, conf))
    return algo_configs

import logging
import os
import json
from typing import List

import colorlog
from omegaconf import OmegaConf, ValidationError

from conf import Config
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


def save_results(cfg: Config, results: dict, filename: str, is_json: bool = True):
    """
    Save the results dictionary into a json file
    """
    results_path = os.path.join(cfg.output_dir, cfg.exp_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open(os.path.join(results_path, filename), "w") as f:
        if is_json:
            f.write(json.dumps(results))
        else:
            f.write(results)


def get_results(cfg: Config, filename: str, is_json: bool = True):
    """
    Reads the results dictionary from a json file
    """
    with open(os.path.join(cfg.output_dir, cfg.exp_name, filename), "r") as f:
        if is_json:
            results = json.loads(f.read())
        else:
            results = f.read()
    return results

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from typing import List

from conf.mode import TrainerConfig, TunerConfig


@dataclass
class AlgorithmConfig:
    """
    This class provides a "schema" for the configuration of the algorithm that includes
    - Hyperparamaters
    - Discrete/Continuous Actions
    - Multi-Agent
    - Multi-GPU
    """

    name: str = ""

    # support of the algorithm
    discrete_actions: bool = False
    continuous_actions: bool = False
    multi_agent: bool = False
    multi_gpu: bool = False

    # model
    hyperparameters: List[str] = field(default_factory=list)


cs = ConfigStore.instance()
cs.store(name="algo_config", node=AlgorithmConfig)

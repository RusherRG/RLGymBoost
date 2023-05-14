from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore

from conf.algorithms import AlgorithmConfig
from conf.mode import TrainerConfig, TunerConfig


@dataclass
class Config:
    """This class provides a "schema" for the config file, validating types."""

    # gym
    gym_name: str = ""
    has_discrete_actions: bool = False
    multi_agent: bool = False

    # algorithms
    algorithms: List[AlgorithmConfig] = field(default_factory=list)

    # modes
    tuner: TunerConfig = field(default_factory=TunerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # run
    exp_name: str = ""
    overwrite: bool = False  # Overwrite the output directory if it exists
    seed: int = 42
    output_dir: str = "./output"

    # ray config
    use_cluster: bool = False
    ray_cluster_address: str = "auto"
    ray_logging: bool = False


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

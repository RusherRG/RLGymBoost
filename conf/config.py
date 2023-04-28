from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class Config:
    """This class provides a "schema" for the config file, validating types."""

    # gym
    gym_name: str = ""

    # run
    exp_name: str = ""
    overwrite: bool = False # Overwrite the output directory if it exists
    seed: int = 123
    epochs: int = 100       # number of epochs to run for each algorithm

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from conf.mode import TrainerConfig, TunerConfig


@dataclass
class Config:
    """This class provides a "schema" for the config file, validating types."""

    # gym
    gym_name: str = ""
    render_freq: int = 10  # frequency of environment render

    # modes
    tuner: TunerConfig = field(default_factory=TunerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # run
    exp_name: str = ""
    overwrite: bool = False  # Overwrite the output directory if it exists
    seed: int = 42


cs = ConfigStore.instance()
cs.store(name="default_config", node=Config)

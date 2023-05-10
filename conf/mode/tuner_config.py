from dataclasses import dataclass, field


@dataclass
class StoppingCriteria:
    """
    This class provides a "schema" for the parameters used by the Tuner to
    determine when to stop the algorithm tuning
    """

    training_iteration: int = 100
    episode_reward_mean: float = 300.0


@dataclass
class TunerConfig:
    """This class provides a "schema" for the config file, validating types."""

    run: bool = False

    # PBT scheduler configurations
    num_samples: int = 4
    time_attr: str = "training_iteration"
    perturbation_interval: int = 10
    resample_probability: float = 0.5

    # tune config
    metric: str = "episode_reward_mean"
    mode: str = "max"

    # resources used by the Tuner
    num_cpus: int = 5
    num_gpus: int = 1
    num_workers: int = 4

    # run config
    stopping_criteria: StoppingCriteria = field(default_factory=StoppingCriteria)

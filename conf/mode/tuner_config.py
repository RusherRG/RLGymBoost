from dataclasses import dataclass


@dataclass
class TunerConfig:
    """This class provides a "schema" for the config file, validating types."""

    run: bool = False
    epochs: int = 10  # number of epochs to run for each algorithm
    target_episode_reward: int = 100
    num_workers: int = 4
    num_cpus: int = 1  # number of CPUs to use per trial
    num_gpus: int = 1  # number of GPUs to use per trial


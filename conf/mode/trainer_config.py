from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """This class provides a "schema" for the config file, validating types."""

    run: bool = False
    epochs: int = 100  # number of epochs to run for each algorithm
    num_rollout_workers: int = 2
    num_envs_per_worker: int = 1
    num_gpus: int = 1
    train_batch_size: int = 1000
    top_k_algos: int = 1  # k value for top k algorithm results

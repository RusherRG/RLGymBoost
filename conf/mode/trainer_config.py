from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """This class provides a "schema" for the config file, validating types."""

    run: bool = False
    epochs: int = 10  # number of epochs to run for each algorithm

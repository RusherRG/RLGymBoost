from dataclasses import dataclass

@dataclass
class TunerConfig:
    """This class provides a "schema" for the config file, validating types."""
    run: bool = False
    epochs: int = 10       # number of epochs to run for each algorithm

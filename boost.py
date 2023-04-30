import hydra

from conf import Config
from tuner import Tuner
from trainer import Trainer
from utils import get_logger
from validator import validate_gym_environment

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    logger = get_logger(__name__)
    logger.info("Running RLGymBoost with configuration: " + str(cfg))

    if not validate_gym_environment(cfg.gym_name, cfg.seed):
        exit(0)

    if cfg.tuner.run:
        tuner = Tuner(gym_name=cfg.gym_name)
        tuner.run()

    if cfg.trainer.run:
        trainer = Trainer(gym_name=cfg.gym_name)
        trainer.run()


if __name__ == "__main__":
    main()

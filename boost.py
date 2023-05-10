import hydra

from conf import Config
from tuner import Tuner
from trainer import Trainer
from utils import get_logger
from validator import validate_gym_environment
from utils import load_algo_config

logger = get_logger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: Config):
    logger = get_logger(__name__)
    cfg.algorithms = load_algo_config(cfg.algorithms, "conf")
    logger.info("Running RLGymBoost with configuration: " + str(cfg))

    if not validate_gym_environment(cfg.gym_name, cfg.seed):
        exit(0)

    if cfg.tuner.run:
        tuner = Tuner(gym_name=cfg.gym_name, config=cfg.tuner)
        tuner.run(cfg.algorithms)

    # if cfg.trainer.run:
    #     trainer = Trainer(gym_name=cfg.gym_name)
    #     trainer.run()


if __name__ == "__main__":
    main()

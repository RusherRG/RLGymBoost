import hydra
from dotenv import load_dotenv
import ray

from conf import Config
from trainer import Trainer
from tuner import Tuner
from utils import get_logger, load_algo_config
from validator import validate_gym_environment

logger = get_logger(__name__)
load_dotenv()


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: Config):
    logger = get_logger(__name__)
    cfg.algorithms = load_algo_config(cfg.algorithms, "conf")
    logger.info("Running RLGymBoost with configuration: " + str(cfg))

    if not validate_gym_environment(cfg.gym_name, cfg.seed):
        exit(0)

    ray.init(address="auto", configure_logging=False)

    if cfg.tuner.run:
        tuner = Tuner(gym_name=cfg.gym_name, config=cfg.tuner)
        tuner_results: dict = tuner.run(cfg.algorithms)

    if cfg.trainer.run:
        trainer = Trainer(
            gym_name=cfg.gym_name, config=cfg.trainer, tuner_results=tuner_results
        )
        trainer_results: list[dict] = trainer.run()  # return top k algorithm results


if __name__ == "__main__":
    main()

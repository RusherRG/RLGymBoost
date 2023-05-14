import hydra
import ray
from dotenv import load_dotenv

from conf import Config
from trainer import Trainer
from tuner import Tuner
from tuner.utils import filter_algorithms
from utils import get_logger, get_results, load_algo_config, save_results
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

    if cfg.use_cluster:
        ray.init(address=cfg.ray_cluster_address, configure_logging=cfg.ray_logging)

    if cfg.tuner.run:
        tuner = Tuner(gym_name=cfg.gym_name, config=cfg.tuner)
        cfg.algorithms = filter_algorithms(cfg)
        tuner_results: dict = tuner.run(cfg.algorithms)
        save_results(cfg, tuner_results, "tuner_results.json")

    if cfg.trainer.run:
        tuner_results = get_results(cfg, "tuner_results.json")
        trainer = Trainer(
            gym_name=cfg.gym_name, config=cfg.trainer, tuner_results=tuner_results
        )
        trainer_results: list[dict] = trainer.run()  # return top k algorithm results
        save_results(cfg, trainer_results, "trainer_results.json")


if __name__ == "__main__":
    main()

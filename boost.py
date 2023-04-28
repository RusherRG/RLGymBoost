import hydra

from conf import Config
from utils import get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    logger = get_logger(__name__)
    logger.info("Running RLGymBoost with configuration: " + str(cfg))


if __name__ == "__main__":
    main()

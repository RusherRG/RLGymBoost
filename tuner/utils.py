from conf import Config
from utils import get_logger

logger = get_logger(__name__)


def filter_algorithms(config: Config):
    """
    Filter the algorithms based on the game environment specifications
    """
    filtered_algos = []
    for algo in config.algorithms:
        # if the game in multi agent and algorithm doesn't support multi agent
        if config.multi_agent and not algo.multi_agent:
            continue
        if (config.has_discrete_actions and algo.discrete_actions) or (
            not config.has_discrete_actions and algo.continuous_actions
        ):
            filtered_algos.append(algo)
    return filtered_algos

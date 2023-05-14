import gym

from utils import get_logger

logger = get_logger(__name__)


def validate_gym_environment(name: str, seed: int = 42) -> bool:
    logger.debug(f"Validating gym environment: {name}")
    try:
        env = gym.make(name)
        env.reset(seed=seed)
        logger.debug(f"action_space: {env.action_space}")
        logger.debug(f"observation_space: {env.observation_space}")

        done = False
        while not done:
            action = (
                env.action_space.sample()
            )  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

        logger.info("Gym environment validated")
    except Exception as e:
        logger.error(e, exc_info=True)
        return False
    return True

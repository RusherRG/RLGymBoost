import os
import pprint

from ray.air.config import RunConfig, ScalingConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train.rl import RLTrainer

from conf.mode import TrainerConfig
from utils import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(self, gym_name: str, config: TrainerConfig, tuner_results: dict):
        self.gym_name = gym_name
        self.config = config
        self.tuner_results = tuner_results
        self.best_results: list = []

    def train_algorithm(self, algo_name: str):
        trainer = RLTrainer(
            run_config=RunConfig(
                stop={"training_iteration": self.config.epochs},
                callbacks=[
                    WandbLoggerCallback(
                        project="RLGymBoost", api_key=os.getenv("WANDB_API_KEY")
                    )
                ],
            ),
            scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
            algorithm=algo_name,
            config={
                # environment
                "env": self.gym_name,
                # rollouts
                "num_rollout_workers": self.config.num_rollout_workers,
                "num_envs_per_worker": self.config.num_envs_per_worker,
                # framework
                "framework": "torch",
                # resources
                "num_gpus": self.config.num_gpus,
                # training
                "model": {"fcnet_hiddens": [64, 64]},
                "train_batch_size": self.config.train_batch_size,
                "evaluation_interval": self.config.eval_interval,
                "evaluation_config": {"render_env": self.config.render_env},
                **self.tuner_results[algo_name]["hyperparameters"],
            },
        )
        result = trainer.fit()
        metrics = self.get_metrics(result.metrics)

        self.save_checkpoint(self.config.epochs, trainer)

        algo_result: dict = {
            "algo_name": algo_name,
            "hyperparameters": self.tuner_results[algo_name]["hyperparameters"],
            "metrics": metrics,
        }

        return algo_result

    def get_metrics(self, results):
        metrics = [
            "episode_reward_mean",
            "episode_reward_max",
            "episode_reward_min",
            "episode_len_mean",
        ]

        metrics = {k: v for k, v in results.items() if k in metrics}

        pprint.pprint(metrics)

        return metrics

    def save_checkpoint(self, epochs: int, trainer: RLTrainer):
        checkpoint_dir = f"./checkpoints/{self.gym_name}/"

        os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)

        saver = trainer.as_trainable()
        saver().save_checkpoint(checkpoint_dir=checkpoint_dir)

        logger.info(f"Checkpoint saved in directory {checkpoint_dir}")

    def update_best_results(self, result: dict):
        for i in range(min(len(self.best_results), self.config.top_k_algos)):
            self.best_results[i] = max(
                self.best_results[i],
                result,
                key=lambda x: x["metrics"]["episode_reward_mean"],
            )
            result = min(
                result,
                self.best_results[i],
                key=lambda x: x["metrics"]["episode_reward_mean"],
            )

        self.best_results.append(result)

    def run(self):
        for algo_name in self.tuner_results:
            result = self.train_algorithm(algo_name)
            self.update_best_results(result)

        return self.best_results[: self.config.top_k_algos]

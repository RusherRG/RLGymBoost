from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.logger import pretty_print

import ray
from ray.air.config import RunConfig, ScalingConfig
from ray.train.rl import RLTrainer
from ray.rllib.algorithms.bc.bc import BC

import pprint


class Train:
    def train(self, epochs=5):
        # Configure algorithm
        config = (
            DQNConfig()
            .environment(
                env="CartPole-v1",
                observation_space=None,
                action_space=None,
            )
            .rollouts(
                num_rollout_workers=2,
                num_envs_per_worker=1,
            )
            .framework("torch")
            .resources(
                num_gpus=1,
            )
            .training(
                model={"fcnet_hiddens": [64, 64]},
                train_batch_size=32,
                lr=1e-4,
            )
            .evaluation(evaluation_num_workers=1, evaluation_interval=1)
        )

        # build algorithm
        algo = config.build()

        # train algorithm
        for i in range(epochs):
            results = algo.train()
            self.print_metrics(results)

        results = algo.evaluate()
        self.print_metrics(results["evaluation"])

        checkpoint_dir = algo.save("checkpoint/")
        print(f"\nCheckpoint saved in directory {checkpoint_dir}")

    def print_metrics(self, results):
        metrics_to_print = [
            "episode_reward_mean",
            "episode_reward_max",
            "episode_reward_min",
            "episode_len_mean",
        ]

        pprint.pprint({k: v for k, v in results.items() if k in metrics_to_print})

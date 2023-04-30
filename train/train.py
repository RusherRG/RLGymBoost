from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.logger import pretty_print

import ray
from ray.air.config import RunConfig, ScalingConfig
from ray.train.rl import RLTrainer
from ray.rllib.algorithms.bc.bc import BC

import pprint

from ray.air.config import RunConfig, ScalingConfig
from ray.train.rl import RLTrainer


class Train:
    def train(self, epochs=50):
        trainer = RLTrainer(
            run_config=RunConfig(stop={"training_iteration": epochs}),
            scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
            algorithm="PPO",
            config={
                # environment
                "env": "CartPole-v1",
                "observation_space": None,
                "action_space": None,
                # rollouts
                "num_rollout_workers": 2,
                "num_envs_per_worker": 1,
                # framework
                "framework": "torch",
                # resources
                "num_gpus": 1,
                # training
                "model": {"fcnet_hiddens": [64, 64]},
                "train_batch_size": 1000,
                "lr": 1e-4,
                "gamma": 0.99,
                # evaluation
                "evaluation_num_workers": 1,
                "evaluation_interval": 1,
                "evaluation_config": {"input": "sampler"},
            },
        )
        result = trainer.fit()
        self.print_metrics(result.metrics)

        checkpoint_dir = trainer.save("checkpoint/")
        print(f"\nCheckpoint saved in directory {checkpoint_dir}")

    def print_metrics(self, results):
        metrics_to_print = [
            "episode_reward_mean",
            "episode_reward_max",
            "episode_reward_min",
            "episode_len_mean",
        ]

        pprint.pprint({k: v for k, v in results.items() if k in metrics_to_print})

import os
import pprint
from typing import List

import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import PopulationBasedTraining

from conf.algorithms import AlgorithmConfig
from conf.mode import TunerConfig
from utils import get_logger

logger = get_logger(__name__)


class Tuner:
    def __init__(self, gym_name: str, config: TunerConfig):
        self.gym_name = gym_name
        self.config = config

    def get_hyperparameter_mutations(self, hyperparameters: List[str]):
        """
        Return the hyperparameter mutations based on the hyperparameter search space
        defined for each hyperparameter
        """
        all_mutations = {
            "lambda": tune.quniform(0.9, 1.0, 0.01),
            "clip_param": tune.quniform(0.05, 0.5, 0.05),
            "lr": tune.quniform(5e-3, 1e-1, 5e-3),
            "kl_coeff": tune.quniform(0.3, 1, 0.1),
            "gamma": tune.quniform(0.95, 0.99, 0.01),
        }

        mutations = {}
        for hyperparameter in hyperparameters:
            if all_mutations.get(hyperparameter):
                mutations[hyperparameter] = all_mutations.get(hyperparameter)
        return mutations

    def parse_results(self, results, algo_config: AlgorithmConfig):
        """
        Parse the results object to generate the results that could be used by the Trainer
        """

        best_result = results.get_best_result()
        best_hyperparameters = {
            param: best_result.config.get(param, 0)
            for param in algo_config.hyperparameters
        }
        metrics = [
            "episode_reward_mean",
            "episode_reward_max",
            "episode_reward_min",
            "episode_len_mean",
        ]
        best_metrics = {
            metric: best_result.metrics.get(metric, 0) for metric in metrics
        }
        return {"hyperparameters": best_hyperparameters, "metrics": best_metrics}

    def tune_algorithm(self, algo_config: AlgorithmConfig):
        """
        Tune the hyperparameters for an algorithm using Population Based Training Scheduler
        and ray Tune API. Stopping the tuning when the stopping criteria defined in the
        algo_config is met.
        """
        logger.info(f"Tuning algorithm: {algo_config.name}")
        hyperparam_mutations = self.get_hyperparameter_mutations(
            algo_config.hyperparameters
        )

        pbt = PopulationBasedTraining(
            time_attr=self.config.time_attr,
            perturbation_interval=self.config.perturbation_interval,
            resample_probability=self.config.resample_probability,
            hyperparam_mutations=hyperparam_mutations,
            synch=True,
        )

        tuner = tune.Tuner(
            algo_config.name,
            tune_config=tune.TuneConfig(
                metric=self.config.metric,
                mode=self.config.mode,
                scheduler=pbt,
                num_samples=self.config.num_samples,
            ),
            param_space={
                "env": self.gym_name,
                "num_workers": self.config.num_workers,
                "num_gpus_per_worker": self.config.num_gpus,
                "num_cpus_per_worker": self.config.num_cpus,
            },
            run_config=air.RunConfig(
                stop=dict(self.config.stopping_criteria),
                callbacks=[
                    WandbLoggerCallback(
                        project="RLGymBoost", api_key=os.getenv("WANDB_API_KEY")
                    )
                ],
            ),
        )
        try:
            results = tuner.fit()
            best_result = self.parse_results(results, algo_config)
            return best_result
        except Exception as e:
            logger.error(e)
            return {}

    def run(self, algorithms: List[AlgorithmConfig]):
        """
        Runs the Tuner to fine-tune the hyperparameters on each of the algorithms
        on the given game environment to find the algorithm that has the best reward
        """
        logger.info(
            f"Running Tuner to find the best algorithm: {[algo['name'] for algo in algorithms]}"
        )
        results = {}
        for algo in algorithms:
            results[algo.name] = self.tune_algorithm(algo)

        return results

# RLGymBoost
The project aims to train Reinforcement learning algorithms and boost the algorithms to represent the goal of achieving higher rewards through the optimization and tuning of RL models

## Description
Generative AI and LLMs are revolutionizing the gaming industry, offering endless possibilities for in-game content, increasing efficiency in game development, improving accessibility, and providing new opportunities for game innovation. Using deep learning AI algorithms such as Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs), creating new game content, including 3D items, playable areas, and character animations, can be automated using the training dataset. Although in order to test the playability of the game environments there is a need to run agents capable of exploring the game mechanisms and achieving good rewards.
The goal of this project is to develop a tool that tunes and trains multiple reinforcement learning (RL) models to learn game mechanics and optimize them for achieving high rewards in a given gym environment. Researchers can use this tool to test their game environments without having to build RL agents to play the game. They can instead focus on generating game environments.

## Code Structure
- conf: configuration models and YAML input files for setting the parameters of the algorithm, trainer, and tuner models.
- trainer: trainer model to train multiple algorithms with best hyper-parameters and return top k results with max rewards.
- tuner: tuner model to optimize all algorithms with best hyper-parameters for the given environment.
- validator: validator model to perform random actions and run checks for observation space, action space, and number of agents in the environment.
- boost.py: main file to run the tuner and trainer algorithm for any gym environment.
- utils.py: utility functions for logging, saving and fetching results, and loding algorithm configurations.

## Installation

```
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
pip3 install -r requirements.txt
```

## Run tune and train command

```
python boost.py
```
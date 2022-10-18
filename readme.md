# return_transforms

This repository contains code for the following return transformation methods for [Upside Down RL](https://arxiv.org/abs/1912.02875) and [Decision Transformer](https://arxiv.org/abs/2106.01345): 
- ESPER - [You Can't Count on Luck: Why Decision Transformers Fail in Stochastic Environments](https://arxiv.org/abs/2205.15967)

## Installation Instructions

- Install `stochastic_offline_envs` from [here](https://github.com/keirp/stochastic_offline_envs).
- Install dependencies with `pip install -r requirements.txt`.
- Install the package with `pip install -e .`.
- [Optional] Install the included `decision_transformer` package. This is only necessary if you want to use the transformed returns with the included modified decision transformer implementation.

## Usage

`return_transforms` operates on offline RL datasets. It saves a file with the transformed returns in the specified directory.

To use `return_transforms` on a dataset, run the following command:

```python return_transforms/generate.py --env_name tfe --config configs/esper/tfe.yaml --device cuda --n_cpu 10 --ret_file data/tfe.ret```

Then, you can use the included fork of Decision Transformer (in the `decision_transformer` directory) to train on the transformed returns.

```python experiment.py --env tfe --dataset default -w True --max_iters 2 --num_steps_per_iter 25000 --rtg ../data/tfe.ret```

Configurations are included for all included `stochastic_offline_envs` in the `configs/esper` directory.
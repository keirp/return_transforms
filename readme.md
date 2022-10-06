# return_transforms

This repository contains code for the following return transformation methods for [Upside Down RL](https://arxiv.org/abs/1912.02875) and [Decision Transformer](https://arxiv.org/abs/2106.01345): 
- ESPER - [You Can't Count on Luck: Why Decision Transformers Fail in Stochastic Environments](https://arxiv.org/abs/2205.15967)

## Usage

`return_transforms` operates on offline RL datasets stored in the [D4RL](https://arxiv.org/abs/2004.07219) format. It saves a numpy array of the transformed returns in the specified directory.

To use `return_transforms` on a dataset, run the following command:

```python return_transforms.py --method esper --config /path/to/config --env_name halfcheetah-medium-replay-v2 --save_dir /path/to/save/transformed/returns```

Then, you can use the included fork of Decision Transformer to train on the transformed returns.

```example here```
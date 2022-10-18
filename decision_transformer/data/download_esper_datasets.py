import gym
import numpy as np

import collections
import pickle

from stochastic_offline_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
from stochastic_offline_envs.envs.offline_envs.tfe_offline_env import TFEOfflineEnv
from stochastic_offline_envs.envs.offline_envs.gambling_offline_env import GamblingOfflineEnv


def save_esper_dataset(name, offline_env):
    """Gets transforms the traj list format into one that the
    Decision Transformer codebase can understand and saves
    as a pickle."""
    env = offline_env.env_cls()
    n_actions = env.action_space.n

    trajs = offline_env.trajs
    episode_step = 0
    paths = []
    for traj in trajs:
        episode_data = collections.defaultdict(list)
        if 'connect_four' in name:
            episode_data['observations'] = np.array(
                [obs['grid'].reshape(-1) for obs in traj.obs])
        else:
            episode_data['observations'] = np.array(
                [obs.reshape(-1) for obs in traj.obs])

        a = np.array(traj.actions)
        actions = np.zeros((a.size, n_actions))
        actions[np.arange(a.size), a] = 1
        episode_data['actions'] = actions
        episode_data['rewards'] = np.array(traj.rewards)
        terminals = np.array([False] * (len(traj.obs) - 1) + [True])
        episode_data['terminals'] = terminals
        paths.append(episode_data)

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(
        f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(paths, f)


# Gambling task
env_type = 'alias'
name = 'gambling-default-v2'
offline_env = GamblingOfflineEnv()

save_esper_dataset(name, offline_env)

# Connect Four task
name = 'connect_four-default-v2'
offline_env = ConnectFourOfflineEnv()
save_esper_dataset(name, offline_env)

# 2048 task
name = 'tfe-default-v2'
offline_env = TFEOfflineEnv()
save_esper_dataset(name, offline_env)

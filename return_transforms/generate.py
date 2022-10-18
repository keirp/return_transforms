from email import generator
import pickle
from return_transforms.algos.esper.esper import esper
from fire import Fire
import yaml
from pathlib import Path
import numpy as np


def load_config(config_path):
    return yaml.safe_load(Path(config_path).read_text())


def load_env(env_name):
    if env_name == 'connect_four':
        from stochastic_offline_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
        from stochastic_offline_envs.envs.connect_four.connect_four_env import GridWrapper
        # TODO: env should just deal with this automatically
        task = ConnectFourOfflineEnv()
        env = task.env_cls()
        env = GridWrapper(env)
        trajs = task.trajs
        for traj in trajs:
            for i in range(len(traj.obs)):
                traj.obs[i] = traj.obs[i]['grid']
        return env, trajs
    elif env_name == 'tfe':
        from stochastic_offline_envs.envs.offline_envs.tfe_offline_env import TFEOfflineEnv
        task = TFEOfflineEnv()
        env = task.env_cls()
        trajs = task.trajs
        return env, trajs
    elif env_name == 'gambling':
        from stochastic_offline_envs.envs.offline_envs.gambling_offline_env import GamblingOfflineEnv
        task = GamblingOfflineEnv()
        env = task.env_cls()
        trajs = task.trajs
        return env, trajs
    # TODO: implement the rest


def normalize_obs(trajs):
    obs_list = []
    for traj in trajs:
        obs_list.extend(traj.obs)
    obs = np.array(obs_list)
    obs_mean = np.mean(obs, axis=0)
    obs_std = np.std(obs, axis=0) + 1e-8
    for traj in trajs:
        for i in range(len(traj.obs)):
            traj.obs[i] = (traj.obs[i] - obs_mean) / obs_std
    return trajs


def generate(env_name, config, ret_file, device, n_cpu=2):
    print('Loading config...')
    config = load_config(config)

    if config['method'] == 'esper':
        print('Loading offline RL task...')
        env, trajs = load_env(env_name)

        if config['normalize']:
            print('Normalizing observations...')
            trajs = normalize_obs(trajs)

        print('Creating ESPER returns...')
        rets = esper(trajs,
                     env.action_space,
                     config['dynamics_model_args'],
                     config['cluster_model_args'],
                     config['train_args'],
                     device,
                     n_cpu)

        # Save the returns as a pickle
        print('Saving returns...')
        Path(ret_file).parent.mkdir(parents=True, exist_ok=True)
        with open(ret_file, 'wb') as f:
            pickle.dump(rets, f)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    Fire(generate)
